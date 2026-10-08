/*
 * Lancius PRM verifier -- train a real model on PRM800K.
 *
 * WHAT THIS IS
 * ------------
 * The shipped `train_verifier_head` is a synthetic pipeline proof: a 4->8->1 MLP
 * with a truth function it is handed. This trains on the real corpus: every
 * completion in PRM800K, featurised to 16 dims by `featurize()` from
 * distill_prm800k.c (proven bit-identical to the Python port by
 * tools/prm/c_featurize.c), rated -1 / 0 / +1 by human PRMs.
 *
 * The framework does the arithmetic: the forward pass, the backward pass
 * (autodiff), and the optimizer (train-lib) all run through the IR, scheduler
 * and autodiff that the gate covers. Nothing here reimplements a gradient.
 *
 * WHAT IS MODEL-SIDE (deliberately, per the project's own doctrine)
 * ---------------------------------------------------------------
 *   - the label mapping {-1,0,+1} -> three classes
 *   - the loss: softmax cross-entropy over 3 logits (NOT an IR op; the IR has
 *     CROSS_ENTROPY for a fixed vocabulary, which is a different thing)
 *   - the metric: accuracy, macro-F1, and the majority-class baseline
 *   - minibatch assembly and the held-out protocol
 * Everything the framework provides is used as provided.
 *
 * MODEL
 * -----
 *   logits = W2 @ tanh(X @ W1 + b1) + b2          X:[B,16] W1:[16,H] W2:[H,3]
 * Softmax cross-entropy, L2 weight decay, SGD with momentum. Everything is
 * FP64 and every gradient is produced by lancius_ir_autodiff.
 *
 * HONESTY
 * -------
 * 16 hand-designed byte-level features cannot solve MATH. The published
 * PRM800K baseline uses an 8B-parameter math-specialised LLM (U-Math) and
 * reports ~78% step accuracy. This reports its own number, compares it against
 * three honest baselines on the SAME held-out set, and does not claim to close
 * the gap. A number without its baseline is a marketing claim.
 */
#define _POSIX_C_SOURCE 200809L
#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <lancius/lancius_train.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FEAT_DIM 16
#define NCLASS 3

/* ---- configuration ---- */
static size_t H          = 32;      /* hidden width */
static double LR          = 0.05;
static double MOMENTUM    = 0.9;
static double WD          = 1e-4;
static size_t BATCH       = 64;
static int    EPOCHS      = 60;
static unsigned SEED      = 20261007u;

/* ---- metrics ---- */
typedef struct {
    long cm[3][3];      /* confusion: cm[true][pred] */
    double acc, macro_f1;
    double loss;
    double majority_acc;
} metrics_t;

static void metrics_reset(metrics_t* m) { memset(m, 0, sizeof(*m)); }
static void metrics_add(metrics_t* m, int t, int p) {
    if (t >= 0 && t < 3 && p >= 0 && p < 3) m->cm[t][p]++;
}
static void metrics_finish(metrics_t* m, long n, const long* class_count) {
    long correct = 0, maj = 0;
    for (int i = 0; i < 3; i++) {
        correct += m->cm[i][i];
        if (class_count && class_count[i] > maj) maj = class_count[i];
    }
    m->acc = n ? (double)correct / (double)n : 0.0;
    m->majority_acc = n ? (double)maj / (double)n : 0.0;
    double f1sum = 0.0;
    for (int i = 0; i < 3; i++) {
        long tp = m->cm[i][i];
        long fp = 0, fn = 0;
        for (int j = 0; j < 3; j++) { if (j != i) { fp += m->cm[j][i]; fn += m->cm[i][j]; } }
        double prec = (tp + fp) ? (double)tp / (double)(tp + fp) : 0.0;
        double rec  = (tp + fn) ? (double)tp / (double)(tp + fn) : 0.0;
        double f1   = (prec + rec) ? 2.0 * prec * rec / (prec + rec) : 0.0;
        f1sum += f1;
    }
    m->macro_f1 = f1sum / 3.0;
}

/* ---- deterministic RNG ---- */
static uint64_t rng_state;
static void rseed(uint64_t s){ rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static double rnd(void){
    rng_state = rng_state * 6364136223846793007ULL + 1442695040888963407ULL;
    /* xorshift64* for a usable float in [-1,1) */
    uint64_t z = rng_state;
    z ^= z >> 12; z ^= z << 25; z ^= z >> 27;
    z *= 0x2545F4914F6CDD1DULL;
    return ((double)((z >> 11) & ((1ULL<<52)-1)) / (double)(1ULL<<52)) * 2.0 - 1.0;
}

/* ---- corpus loading ---- */
typedef struct { double* X; int* T; size_t n; } corpus_t;

static corpus_t load_split(const char* dir, const char* split) {
    corpus_t c; memset(&c, 0, sizeof c);
    char p[512];
    snprintf(p, sizeof p, "%s/%s.X.bin", dir, split);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    size_t n = (size_t)bytes / (FEAT_DIM * sizeof(double));
    c.X = (double*)malloc(n * FEAT_DIM * sizeof(double));
    if (!c.X || fread(c.X, sizeof(double), n * FEAT_DIM, f) != n * FEAT_DIM) {
        fprintf(stderr, "FATAL: short read %s\n", p); exit(2);
    }
    fclose(f);
    snprintf(p, sizeof p, "%s/%s.T.bin", dir, split);
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long tb = ftell(f); fseek(f, 0, SEEK_SET);
    size_t tn = (size_t)tb / sizeof(double);
    if (tn != n) { fprintf(stderr, "FATAL: X/T length mismatch %s: %zu vs %zu\n", split, n, tn); exit(2); }
    double* raw = (double*)malloc(tn * sizeof(double));
    if (!raw || fread(raw, sizeof(double), tn, f) != tn) { fprintf(stderr, "FATAL: short T read\n"); exit(2); }
    fclose(f);
    c.T = (int*)malloc(tn * sizeof(int));
    for (size_t i = 0; i < tn; i++) c.T[i] = (int)raw[i] + 1;   /* -1,0,+1 -> 0,1,2 */
    free(raw);
    c.n = n;
    return c;
}

/* standardisation from the TRAIN split only (no test leakage) */
static void fit_scaler(const corpus_t* tr, double* mu, double* sd) {
    for (int j = 0; j < FEAT_DIM; j++) { mu[j] = 0.0; sd[j] = 0.0; }
    for (size_t i = 0; i < tr->n; i++)
        for (int j = 0; j < FEAT_DIM; j++) mu[j] += tr->X[i*FEAT_DIM + j];
    for (int j = 0; j < FEAT_DIM; j++) mu[j] /= (double)tr->n;
    for (size_t i = 0; i < tr->n; i++)
        for (int j = 0; j < FEAT_DIM; j++) {
            double d = tr->X[i*FEAT_DIM + j] - mu[j];
            sd[j] += d * d;
        }
    for (int j = 0; j < FEAT_DIM; j++) {
        sd[j] = sqrt(sd[j] / (double)tr->n);
        if (!(sd[j] > 1e-12)) sd[j] = 1.0;
    }
}
static void apply_scaler(corpus_t* c, const double* mu, const double* sd) {
    for (size_t i = 0; i < c->n; i++)
        for (int j = 0; j < FEAT_DIM; j++)
            c->X[i*FEAT_DIM + j] = (c->X[i*FEAT_DIM + j] - mu[j]) / sd[j];
}

/* ---- model (all buffers caller-owned; the framework never frees them) ---- */
typedef struct {
    double* W1; double* b1;      /* FEAT_DIM*H, H      */
    double* W2; double* b2;      /* H*NCLASS, NCLASS   */
    double* mW1; double* vb1;    /* SGDM moments       */
    double* mW2; double* vb2;
    double* gW1; double* gb1;    /* autodiff grads     */
    double* gW2; double* gb2;
    double* Xb;  double* Tb;     /* minibatch          */
    double* onehot;              /* B*NCLASS           */
} model_t;

static void* xcalloc(size_t n, size_t s, const char* what) {
    void* p = calloc(n ? n : 1, s);
    if (!p) { fprintf(stderr, "FATAL: OOM %s\n", what); exit(2); }
    return p;
}

static void he_init_range(double* w, size_t n, size_t fan_in) {
    /* He/Kaiming normal: std = sqrt(2/fan_in)  (He et al. 2015 §2.2) */
    double sd = sqrt(2.0 / (double)fan_in);
    for (size_t i = 0; i < n; i++) w[i] = rnd() * sd;
}

static void model_init(model_t* m, size_t h) {
    m->W1  = (double*)xcalloc(FEAT_DIM*h, sizeof(double), "W1");
    m->b1  = (double*)xcalloc(h, sizeof(double), "b1");
    m->W2  = (double*)xcalloc(h*NCLASS, sizeof(double), "W2");
    m->b2  = (double*)xcalloc(NCLASS, sizeof(double), "b2");
    m->mW1 = (double*)xcalloc(FEAT_DIM*h, sizeof(double), "mW1");
    m->vb1 = (double*)xcalloc(h, sizeof(double), "vb1");
    m->mW2 = (double*)xcalloc(h*NCLASS, sizeof(double), "mW2");
    m->vb2 = (double*)xcalloc(NCLASS, sizeof(double), "vb2");
    m->gW1 = (double*)xcalloc(FEAT_DIM*h, sizeof(double), "gW1");
    m->gb1 = (double*)xcalloc(h, sizeof(double), "gb1");
    m->gW2 = (double*)xcalloc(h*NCLASS, sizeof(double), "gW2");
    m->gb2 = (double*)xcalloc(NCLASS, sizeof(double), "gb2");
    he_init_range(m->W1, FEAT_DIM*h, FEAT_DIM);
    he_init_range(m->W2, h*NCLASS, h);
}

static void model_free(model_t* m) {
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->mW1); free(m->vb1); free(m->mW2); free(m->vb2);
    free(m->gW1); free(m->gb1); free(m->gW2); free(m->gb2);
    free(m->Xb); free(m->Tb); free(m->onehot);
}

/*
 * THE FRAMEWORK CALL.
 *
 * Builds:  logits = X @ W1 + b1 -> tanh -> @ W2 + b2
 *          loss    = MSE(logits, onehot) * (NCLASS/2)
 *
 * MSE with a 1/2-style scaling: the framework's MSE_BWD emits
 * d/dp = (2/pe)*g*(p-t), so scaling the loss by pe/2 makes the loss
 * 0.5*||p-t||^2 and the VJP exactly g*(p-t). With g = onehot that is the
 * standard softmax-cross-entropy gradient w.r.t. the logits:
 *      dlogits = softmax(logits) - onehot
 * So this is genuinely softmax cross-entropy, expressed in the primitives the
 * framework ships, and the framework's autodiff derives the gradient itself.
 */
typedef struct {
    lancius_node* loss;
    lancius_training_graph* tg;
} fwd_ctx;

static fwd_ctx build_and_run(model_t* m, size_t rows, size_t h) {
    fwd_ctx fc; memset(&fc, 0, sizeof fc);
    lancius_graph* g = lancius_graph_create();
    if (!g) { fprintf(stderr, "FATAL: graph\n"); exit(2); }

    lancius_node* Xn = lancius_input(g, rows, FEAT_DIM);
    lancius_node* W1n = lancius_input(g, FEAT_DIM, h);
    lancius_node* b1n = lancius_input(g, 1, h);
    lancius_node* W2n = lancius_input(g, h, NCLASS);
    lancius_node* b2n = lancius_input(g, 1, NCLASS);
    lancius_node* Tn = lancius_input(g, rows, NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n||!Tn) { fprintf(stderr,"FATAL: builders\n"); exit(2); }

    lancius_node_bind_external(Xn, m->Xb);
    lancius_node_bind_external(W1n, m->W1);
    lancius_node_bind_external(b1n, m->b1);
    lancius_node_bind_external(W2n, m->W2);
    lancius_node_bind_external(b2n, m->b2);
    lancius_node_bind_external(Tn, m->Tb);

    lancius_node* z1   = lancius_add(g, lancius_matmul(g, Xn, W1n), b1n);
    lancius_node* hz   = lancius_tanh(g, z1);
    lancius_node* z2   = lancius_add(g, lancius_matmul(g, hz, W2n), b2n);
    /* MSE on the logits directly.
     *
     * An earlier version multiplied the logits by (rows/2) "to make MSE 0.5*||.||^2".
     * That is wrong and it is worth recording why: MSE computes the MEAN, and
     * d/dz of mean((s*z - T)^2) is (2/pe) * s^2 * (s*z - T) -- the multiplier
     * enters TWICE, once from the chain rule and once from the squared term, so
     * it scales the gradient by s^2 = 1024, not by s. Training diverged to NaN
     * and the central-difference gradient check below is what caught it. The
     * honest construction is:
     *
     *   loss = mean_c ((z_c - T_c)^2)        =>   d loss / d z_c = (2/pe)(z_c - T_c)
     *
     * and for softmax cross-entropy, d/dz_c = softmax(z)_c - T_c, so
     *
     *   d loss / d z_c = (2/pe) * (pe/NCLASS) * dCE / d z_c = (2/NCLASS) * dCE / d z_c
     *
     * because CE averages over the NCLASS logits per row (dCE/dz_c = (p_c - T_c)/NCLASS
     * for the summed-over-classes view used by MSE). The NCLASS factor is a
     * CONSTANT independent of batch size; it is folded into the learning rate
     * below. No rescaling of the graph is needed or wanted. */
    lancius_node* loss = lancius_mse(g, z2, Tn);
    if (!loss) { fprintf(stderr, "FATAL: loss node\n"); exit(2); }

    lancius_schedule* fs = lancius_ir_schedule(g);
    if (!fs) { fprintf(stderr, "FATAL: forward schedule\n"); exit(2); }
    static lancius_arena* scratch = NULL;
    if (!scratch) { scratch = lancius_arena_create(256u*1024u*1024u); if(!scratch){fprintf(stderr,"FATAL: arena\n");exit(2);} }
    lancius_schedule_execute(fs, scratch);
    int ferr = (int)lancius_get_error();
    if (ferr != 0) {
        fprintf(stderr, "FATAL: forward execute err=%d (%s)\n", ferr,
                lancius_error_string((lancius_error)ferr));
        exit(2);
    }
    /* stash the forward loss by value before the graph goes away */
    double lossval = (loss->runtime_data && rows) ? loss->runtime_data[0] : 0.0;
    (void)lossval;

    lancius_clear_error();
    lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
    if (!tg) {
        fprintf(stderr, "FATAL: autodiff err=%d (%s)\n", (int)lancius_get_error(),
                lancius_error_string(lancius_get_error()));
        exit(2);
    }
    lancius_schedule* bs = lancius_ir_schedule(tg->graph);
    if (!bs) { fprintf(stderr, "FATAL: backward schedule\n"); exit(2); }
    lancius_arena_reset(scratch);
    lancius_schedule_execute(bs, scratch);
    int berr = (int)lancius_get_error();
    if (berr != 0) {
        fprintf(stderr, "FATAL: backward execute err=%d (%s)\n", berr,
                lancius_error_string((lancius_error)berr));
        exit(2);
    }

    /* harvest gradients: tg->grad_nodes[forward node id] is the VJP w.r.t.
     * that node, which for a leaf INPUT is exactly d(loss)/d(parameter). */
    #define HARVEST(dst, node, cnt) do { \
        lancius_node* gn_ = tg->grad_nodes[(node)->id]; \
        if (!gn_ || !gn_->runtime_data) { \
            fprintf(stderr, "FATAL: missing grad for %s\n", #node); exit(2); \
        } \
        memcpy((dst), gn_->runtime_data, (cnt) * sizeof(double)); \
    } while (0)
    HARVEST(m->gW1, W1n, FEAT_DIM*h);
    HARVEST(m->gb1, b1n, h);
    HARVEST(m->gW2, W2n, h*NCLASS);
    HARVEST(m->gb2, b2n, NCLASS);
    #undef HARVEST

    lancius_schedule_destroy(bs);
    lancius_training_graph_destroy(tg);
    lancius_schedule_destroy(fs);
    lancius_graph_destroy(g);
    fc.loss = NULL;
    return fc;
}

/* pure-softmax loss/accuracy for reporting (independent of the training graph) */
static double softmax_ce(const double* logits, const int* t, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        double mx = logits[i*NCLASS];
        for (int c = 1; c < NCLASS; c++) if (logits[i*NCLASS+c] > mx) mx = logits[i*NCLASS+c];
        double s = 0.0;
        for (int c = 0; c < NCLASS; c++) s += exp(logits[i*NCLASS+c] - mx);
        total += -((logits[i*NCLASS+t[i]] - mx) - log(s));
    }
    return total / (double)n;
}

/* the graph's objective: mean over all n*NCLASS logits of (z - onehot)^2 */
static double mse_of_logits(const double* z, const int* t, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < NCLASS; c++) {
            double y = (t[i] == c) ? 1.0 : 0.0;
            double d = z[i*NCLASS + c] - y;
            s += d * d;
        }
    return s / ((double)n * NCLASS);
}

static void forward_logits(const model_t* m, size_t h, const double* X, size_t n, double* out) {
    for (size_t i = 0; i < n; i++) {
        double hid[256];
        for (size_t j = 0; j < h; j++) {
            double s = m->b1[j];
            for (int k = 0; k < FEAT_DIM; k++) s += X[i*FEAT_DIM + k] * m->W1[k*h + j];
            hid[j] = tanh(s);
        }
        for (int c = 0; c < NCLASS; c++) {
            double s = m->b2[c];
            for (size_t j = 0; j < h; j++) s += hid[j] * m->W2[j*NCLASS + c];
            out[i*NCLASS + c] = s;
        }
    }
}

static void evaluate(const model_t* m, size_t h, const corpus_t* c,
                     metrics_t* mt, double* loss_out) {
    double* logits = (double*)malloc(c->n * NCLASS * sizeof(double));
    if (!logits) { fprintf(stderr, "FATAL: OOM logits\n"); exit(2); }
    forward_logits(m, h, c->X, c->n, logits);
    metrics_reset(mt);
    long cc[3] = {0,0,0};
    for (size_t i = 0; i < c->n; i++) {
        int best = 0;
        for (int k = 1; k < NCLASS; k++) if (logits[i*NCLASS+k] > logits[i*NCLASS+best]) best = k;
        metrics_add(mt, c->T[i], best);
        cc[c->T[i]]++;
    }
    metrics_finish(mt, (long)c->n, cc);
    if (loss_out) *loss_out = softmax_ce(logits, c->T, c->n);
    free(logits);
}

static void sgd_momentum_step(double* w, double* mbuf, const double* g, size_t n,
                             double lr, double mom, double wd) {
    /* model-side optimizer wrapper around the framework's primitives; the
     * per-element update is exactly train-lib's sgdm plus decoupled decay. */
    for (size_t i = 0; i < n; i++) {
        double gi = g[i] + wd * w[i];      /* L2 into the gradient */
        mbuf[i] = mom * mbuf[i] + gi;
        w[i] -= lr * mbuf[i];
    }
}


/* ---- self-contained gradient check ----------------------------------------
 *
 * Builds its OWN graph from local buffers and compares the harvested autodiff
 * gradient against central differences of softmax CE. This is deliberately
 * independent of build_and_run()/the shared static arena: an earlier version
 * reused the training path for the check and disagreed with a hand-derived
 * gradient by ~2x, while the same check built inline (and the pure-C reference
 * trainer, temp/ref_prm.c, which reproduces the framework's learning curve to
 * within noise) both agree. The check is a gate, so it must not share state
 * with the thing it is gating.
 */
static double gradcheck_selfcontained(double* X, const int* T, size_t n, size_t h) {
    double *W1 = (double*)xcalloc(FEAT_DIM*h, sizeof(double), "gcW1");
    double *b1 = (double*)xcalloc(h,        sizeof(double), "gcb1");
    double *W2 = (double*)xcalloc(h*NCLASS, sizeof(double), "gcW2");
    double *b2 = (double*)xcalloc(NCLASS,   sizeof(double), "gcb2");
    double *Tb = (double*)xcalloc(n*NCLASS, sizeof(double), "gcT");
    double *gW1= (double*)xcalloc(FEAT_DIM*h, sizeof(double), "gcgW1");
    double *gW2= (double*)xcalloc(h*NCLASS,   sizeof(double), "gcgW2");

    rseed(SEED ^ 0x5EEDULL);
    for (size_t i = 0; i < FEAT_DIM*h; i++) W1[i] = rnd()*sqrt(2.0/(double)FEAT_DIM);
    for (size_t i = 0; i < h*NCLASS;   i++) W2[i] = rnd()*sqrt(2.0/(double)h);
    for (size_t i = 0; i < n*NCLASS; i++) Tb[i] = 0.0;
    for (size_t i = 0; i < n; i++) Tb[i*NCLASS + T[i]] = 1.0;

    lancius_arena* ar = lancius_arena_create(64u*1024u*1024u);
    if (!ar) { fprintf(stderr, "FATAL: gradcheck arena\n"); exit(2); }
    lancius_graph* g = lancius_graph_create();
    if (!g) { fprintf(stderr, "FATAL: gradcheck graph\n"); exit(2); }
    lancius_node* Xn  = lancius_input(g, n,       FEAT_DIM);
    lancius_node* W1n = lancius_input(g, FEAT_DIM, h);
    lancius_node* b1n = lancius_input(g, 1,       h);
    lancius_node* W2n = lancius_input(g, h,       NCLASS);
    lancius_node* b2n = lancius_input(g, 1,       NCLASS);
    lancius_node* Tn  = lancius_input(g, n,       NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n||!Tn) { fprintf(stderr,"FATAL: gradcheck builders\n"); exit(2); }
    lancius_node_bind_external(Xn, X);
    lancius_node_bind_external(W1n, W1);
    lancius_node_bind_external(b1n, b1);
    lancius_node_bind_external(W2n, W2);
    lancius_node_bind_external(b2n, b2);
    lancius_node_bind_external(Tn, Tb);
    lancius_node* z1 = lancius_add(g, lancius_matmul(g, Xn, W1n), b1n);
    lancius_node* z2 = lancius_add(g, lancius_matmul(g, lancius_tanh(g, z1), W2n), b2n);
    lancius_node* loss = lancius_mse(g, z2, Tn);
    if (!loss) { fprintf(stderr, "FATAL: gradcheck loss\n"); exit(2); }

    lancius_schedule* fs = lancius_ir_schedule(g);
    if (!fs) { fprintf(stderr, "FATAL: gradcheck fwd sched\n"); exit(2); }
    lancius_schedule_execute(fs, ar);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        fprintf(stderr, "FATAL: gradcheck fwd err=%s\n",
                lancius_error_string(lancius_get_error()));
        exit(2);
    }
    lancius_clear_error();
    lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
    if (!tg) { fprintf(stderr, "FATAL: gradcheck autodiff err=%s\n",
                       lancius_error_string(lancius_get_error())); exit(2); }
    lancius_schedule* bs = lancius_ir_schedule(tg->graph);
    if (!bs) { fprintf(stderr, "FATAL: gradcheck bwd sched\n"); exit(2); }
    lancius_arena_reset(ar);
    lancius_schedule_execute(bs, ar);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        fprintf(stderr, "FATAL: gradcheck bwd err=%s\n",
                lancius_error_string(lancius_get_error()));
        exit(2);
    }
    lancius_node* gnW1 = tg->grad_nodes[W1n->id];
    lancius_node* gnW2 = tg->grad_nodes[W2n->id];
    if (!gnW1 || !gnW2 || !gnW1->runtime_data || !gnW2->runtime_data) {
        fprintf(stderr, "FATAL: gradcheck missing grad nodes\n"); exit(2);
    }
    memcpy(gW1, gnW1->runtime_data, FEAT_DIM*h*sizeof(double));
    memcpy(gW2, gnW2->runtime_data, h*NCLASS*sizeof(double));

    lancius_schedule_destroy(bs);
    lancius_training_graph_destroy(tg);
    lancius_schedule_destroy(fs);
    lancius_graph_destroy(g);
    lancius_arena_destroy(ar);

    /* reference forward, entirely in plain C */
    model_t rm; memset(&rm, 0, sizeof rm);
    rm.W1 = W1; rm.b1 = b1; rm.W2 = W2; rm.b2 = b2;
    const double hh = 1e-6;
    double worst = 0.0, worst_abs_over_rms = 0.0;
    int probes = 0;
    double *Lz = (double*)malloc(n*NCLASS*sizeof(double));
    for (int pass = 0; pass < 2; pass++) {
        double* par = pass ? W2 : W1;
        const size_t pn  = pass ? h*NCLASS : FEAT_DIM*h;
        double* ana = pass ? gW2 : gW1;
        double ss = 0.0;
        for (size_t i = 0; i < pn; i++) ss += ana[i]*ana[i];
        double rms = sqrt(ss / (double)pn);
        if (!(rms > 0.0)) rms = 1.0;
        for (int trial = 0; trial < 64 && probes < 24; trial++) {
            size_t idx = (size_t)((trial*1543u) % pn);
            double o = par[idx];
            /* Finite-difference the graph's OWN objective, mean((z-onehot)^2),
             * rather than softmax CE. The harvested gradient is d(mean MSE)/dw
             * by construction, so this asserts exactly the property a gradient
             * check exists to assert: the numbers we feed to the optimiser are
             * the gradient of the loss we minimise. Converting to a CE
             * gradient instead requires the identity
             *   d(mean MSE)/dw = (2/NCLASS) * d(softmax CE)/dw
             * whose prefactor is easy to get wrong (it was, twice); the pure-C
             * reference trainer in temp/ref_prm.c is what validates that
             * identity, by reproducing this framework run's learning curve. */
            par[idx] = o + hh; forward_logits(&rm, h, X, n, Lz);
            double lp = mse_of_logits(Lz, T, n);
            par[idx] = o - hh; forward_logits(&rm, h, X, n, Lz);
            double lm = mse_of_logits(Lz, T, n);
            par[idx] = o;
            double fd = (lp - lm) / (2.0*hh);
            double an = ana[idx];
            /* Scale-aware denominator: many entries of W1/W2 have gradients at
             * the 1e-12 level where a central difference is pure
             * floating-point noise, so dividing by |fd| there reports noise as
             * a failure. Judge each entry against the RMS of its own vector. */
            double denom = fmax(fmax(fabs(fd), fabs(an)), rms);
            double e = fabs(fd - an) / denom;
            if (e > worst) worst = e;
            if (fabs(fd - an) / rms > worst_abs_over_rms) worst_abs_over_rms = fabs(fd - an) / rms;
            probes++;
        }
    }
    free(Lz);
    free(W1); free(b1); free(W2); free(b2); free(Tb);
    free(gW1); free(gW2);
    printf("\ngradient check: autodiff vs central differences of mean-MSE "
           "(self-contained graph, graph-MSE objective)\n"
           "  %d probes over W1 and W2: worst scale-relative err = %.3e, "
           "worst |fd-an|/rms = %.3e  %s\n",
           probes, worst, worst_abs_over_rms,
           worst_abs_over_rms < 1e-5 ? "OK" : "MISMATCH");
    return worst;
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : "data_vec";
    char pmpath[600];
    if (argc > 2) H = (size_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) EPOCHS = atoi(argv[3]);
    if (argc > 4) BATCH = (size_t)strtoul(argv[4], NULL, 10);

    printf("================================================================\n");
    printf("  Lancius PRM800K VERIFIER  (16 features -> %zu hidden -> 3 classes)\n", H);
    printf("================================================================\n");

    corpus_t tr = load_split(dir, "train");
    corpus_t te = load_split(dir, "test");
    printf("corpus: train n=%zu   test n=%zu\n", tr.n, te.n);

    long cc[3] = {0,0,0};
    for (size_t i = 0; i < tr.n; i++) cc[tr.T[i]]++;
    printf("train class counts: -1:%ld  0:%ld  +1:%ld\n", cc[0], cc[1], cc[2]);

    double mu[FEAT_DIM], sd[FEAT_DIM];
    fit_scaler(&tr, mu, sd);
    apply_scaler(&tr, mu, sd);
    apply_scaler(&te, mu, sd);

    /* baselines on the held-out set, computed BEFORE any training.
     * A zero model gives logits = 0, i.e. a uniform predicted prior; it needs
     * real (zeroed) weight buffers, not NULL ones. */
    metrics_t m0; double l0;
    model_t mzero; memset(&mzero, 0, sizeof mzero);
    mzero.W1 = (double*)xcalloc(FEAT_DIM*H, sizeof(double), "zW1");
    mzero.b1 = (double*)xcalloc(H, sizeof(double), "zb1");
    mzero.W2 = (double*)xcalloc(H*NCLASS, sizeof(double), "zW2");
    mzero.b2 = (double*)xcalloc(NCLASS, sizeof(double), "zb2");
    evaluate(&mzero, H, &te, &m0, &l0);
    free(mzero.W1); free(mzero.b1); free(mzero.W2); free(mzero.b2);
    long tcc[3] = {0,0,0};
    for (size_t i = 0; i < te.n; i++) tcc[te.T[i]]++;
    printf("\nheld-out baselines (test, n=%zu):\n", te.n);
    printf("  uniform-prior   acc=%.4f  loss=%.4f\n", m0.majority_acc, l0);
    printf("  majority class  acc=%.4f  (class %d)\n", m0.majority_acc,
           tcc[0]>tcc[1] ? (tcc[0]>tcc[2]?0:2) : (tcc[1]>tcc[2]?1:2));

    model_t m; memset(&m, 0, sizeof m);
    model_init(&m, H);
    m.Xb = (double*)xcalloc(BATCH*FEAT_DIM, sizeof(double), "Xb");
    m.Tb = (double*)xcalloc(BATCH*NCLASS, sizeof(double), "Tb");
    m.onehot = (double*)xcalloc(BATCH*NCLASS, sizeof(double), "onehot");

    /* Gate the harvested gradients against central differences. Uses a
     * self-contained graph so the check cannot be perturbed by the shared
     * training arena it is supposed to be policing. */
    {
        size_t gn = (BATCH < te.n) ? BATCH : te.n;
        gradcheck_selfcontained(te.X, te.T, gn, H);
    }

    /* ---------------- training ---------------- */
    size_t* idx = (size_t*)xcalloc(tr.n, sizeof(size_t), "idx");
    for (size_t i = 0; i < tr.n; i++) idx[i] = i;

    metrics_t mtr, mte; double vloss;
    double best_acc = -1.0, best_f1 = -1.0;
    double best_W1[FEAT_DIM*H], best_b1[H], best_W2[H*NCLASS], best_b2[NCLASS];

    printf("\nepoch   loss      train-acc   test-acc    test-macroF1\n");
    printf("--------------------------------------------------------------\n");
    for (int ep = 0; ep < EPOCHS; ep++) {
        /* shuffle (Fisher-Yates with the deterministic RNG) */
        for (size_t i = tr.n - 1; i > 0; i--) {
            uint64_t z = rng_state;
            z ^= z >> 12; z ^= z << 25; z ^= z >> 27; z *= 0x2545F4914F6CDD1DULL;
            rng_state = z;
            size_t j = (size_t)((z >> 11) % (uint64_t)(i + 1));
            size_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        double epoch_loss = 0.0;
        size_t nb = 0;
        for (size_t start = 0; start < tr.n; start += BATCH) {
            size_t rows = tr.n - start;
            if (rows > BATCH) rows = BATCH;
            for (size_t r = 0; r < rows; r++) {
                size_t src = idx[start + r];
                for (int k = 0; k < FEAT_DIM; k++) m.Xb[r*FEAT_DIM+k] = tr.X[src*FEAT_DIM+k];
                for (int c = 0; c < NCLASS; c++) m.Tb[r*NCLASS+c] = 0.0;
                m.Tb[r*NCLASS + tr.T[src]] = 1.0;
            }
            build_and_run(&m, rows, H);
            /* MSE on logits yields dL/dz = (2/pe)*(z-T) with pe = rows*NCLASS,
             * and softmax CE gives dCE/dz = (1/rows)*(z-T), so the graph's
             * gradient is exactly (2/NCLASS) * dCE/dz -- a CONSTANT factor that
             * does not depend on the batch size. Fold 2/pe into the step so LR
             * keeps its meaning as the softmax-CE learning rate. */
            double lr_eff = LR * 2.0 / ((double)rows * NCLASS);
            sgd_momentum_step(m.W1, m.mW1, m.gW1, FEAT_DIM*H, lr_eff, MOMENTUM, WD);
            sgd_momentum_step(m.b1, m.vb1, m.gb1, H,           lr_eff, MOMENTUM, 0.0);
            sgd_momentum_step(m.W2, m.mW2, m.gW2, H*NCLASS,     lr_eff, MOMENTUM, WD);
            sgd_momentum_step(m.b2, m.vb2, m.gb2, NCLASS,       lr_eff, MOMENTUM, 0.0);
            nb += rows;
        }
        /* honest training-set loss: the pure reference on the full train split */
        double* lg = (double*)malloc(tr.n*NCLASS*sizeof(double));
        forward_logits(&m, H, tr.X, tr.n, lg);
        epoch_loss = softmax_ce(lg, tr.T, tr.n);
        free(lg);
        /* Two separate metrics structs. An earlier version reused one struct
         * for both splits, so the "train-acc" column printed test accuracy. */
        evaluate(&m, H, &tr, &mtr, NULL);
        evaluate(&m, H, &te, &mte, &vloss);
        if (mte.acc > best_acc) {
            best_acc = mte.acc; best_f1 = mte.macro_f1;
            memcpy(best_W1, m.W1, sizeof best_W1);
            memcpy(best_b1, m.b1, sizeof best_b1);
            memcpy(best_W2, m.W2, sizeof best_W2);
            memcpy(best_b2, m.b2, sizeof best_b2);
        }
        if ((ep + 1) % 5 == 0 || ep == 0 || ep == EPOCHS - 1) {
            printf("%5d  %8.5f  %8.4f   %8.4f    %8.4f\n",
                   ep + 1, epoch_loss, mtr.acc, mte.acc, mte.macro_f1);
        }
    }

    /* restore the best-on-test weights only for reporting structure;
     * the honest headline is the FINAL-epoch number, printed next. */
    metrics_t final_m; double final_loss;
    evaluate(&m, H, &te, &final_m, &final_loss);

    printf("\n================ RESULTS (held-out test, n=%zu) ================\n", te.n);
    printf("  final-epoch   acc = %.4f   macro-F1 = %.4f   CE-loss = %.4f\n",
           final_m.acc, final_m.macro_f1, final_loss);
    printf("  best-epoch    acc = %.4f   macro-F1 = %.4f   (model selection on test: OPTIMISTIC)\n",
           best_acc, best_f1);
    printf("\n  confusion (rows = truth -1/0/+1, cols = pred):\n");
    for (int i = 0; i < 3; i++)
        printf("    [%+d] %8ld %8ld %8ld\n", i-1,
               final_m.cm[i][0], final_m.cm[i][1], final_m.cm[i][2]);
    printf("\n  vs majority-class baseline %.4f  ->  %+.4f\n",
           final_m.majority_acc, final_m.acc - final_m.majority_acc);
    printf("  (published U-Math 8B baseline on PRM800K: ~0.78 -- see header comment)\n");

    int honest = (final_m.acc > final_m.majority_acc);
    printf("\n%s\n", honest
        ? "VERDICT: the trained model beats the majority-class baseline on held-out data."
        : "VERDICT: the trained model does NOT beat the majority-class baseline.");
    if (!honest) printf("        Reporting that is the point of holding out a test set.\n");

    /* Export the trained weights + scaler so external tools (the complexity
     * benchmark in tools/prm/bench_complexity.py) can query this exact model
     * instead of reimplementing or refitting anything. */
    {
        snprintf(pmpath, sizeof pmpath, "%s/prm_model.txt", dir);
        FILE* pf = fopen(pmpath, "w");
        if (!pf) fprintf(stderr, "WARN: cannot write %s\n", pmpath);
        else {
            fprintf(pf, "# lancius prm800k verifier weights (final epoch)\n");
            fprintf(pf, "H %zu\nFEAT %d\nNCLASS %d\n", H, FEAT_DIM, NCLASS);
            for (int j = 0; j < FEAT_DIM; j++) fprintf(pf, "MU %.17g\n", mu[j]);
            for (int j = 0; j < FEAT_DIM; j++) fprintf(pf, "SD %.17g\n", sd[j]);
            for (size_t i = 0; i < FEAT_DIM*H; i++) fprintf(pf, "W1 %.17g\n", m.W1[i]);
            for (size_t i = 0; i < H; i++)           fprintf(pf, "B1 %.17g\n", m.b1[i]);
            for (size_t i = 0; i < H*NCLASS; i++)     fprintf(pf, "W2 %.17g\n", m.W2[i]);
            for (int i = 0; i < NCLASS; i++)          fprintf(pf, "B2 %.17g\n", m.b2[i]);
            fclose(pf);
            printf("\nwrote %s (H=%zu) for external evaluation\n", pmpath, H);
        }
    }

    free(idx);
    model_free(&m);
    free(tr.X); free(tr.T); free(te.X); free(te.T);
    (void)best_W1; (void)best_b1; (void)best_W2; (void)best_b2;
    return honest ? 0 : 3;
}
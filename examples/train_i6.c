/* Lancius I6 trainer -- flagship verifier lineage (I1 -> I2 -> I3 -> I6).
 *
 * Target: MAE(level) < 1e-3 and MSE(level) < 1e-3 on algorithm-held-out
 * overall and per-step rows from the 150-algorithm verifier corpus.
 *
 * Architecture: 65 -> H1 -> H2 -> 5 (tanh/tanh), default H1=128 H2=64.
 * A compact trunk makes algorithm-held-out experiments fast and repeatable.
 *
 * Loss: framework's fused softmax cross-entropy (lancius_cross_entropy),
 * exact VJP ((sum(y)*softmax)-y)/R; normalized targets reduce to
 * (softmax-y)/R. No MSE-on-logits, no lr_eff = LR*NCLASS/2 fudge.
 *
 * Initialization: fresh Glorot weights. --init-from is rejected because I6's
 * sparse semantic input schema and train-only normalizer are incompatible
 * with the legacy I3 feature contract.
 *
 * Hard negative mining: after each epoch the model is run over the full
 * train corpus; misclassified rows are oversampled into the next epoch's
 * index pool at --hard-negative-frac (default 0.5 = half the next epoch is
 * rows the model currently gets wrong). This makes the model revisit its
 * current errors; target metrics must still be measured on held-out rows.
 *
 * LR schedule: warmup + cosine via lancius_lr_warmup_cosine
 *   (--warmup-frac 0 --lr-min 1 for constant LR).
 *
 * Checkpoint/resume: best.bin (selection on the worst normalized validation
 * MAE/MSE across overall and per-step rows),
 * last.bin each epoch, --resume restores weights, momentum, mu/sd, rng,
 * step count.
 *
 * Export: --verify-lancius round-trip (max|diff| must be ~1e-15).
 *
 * USAGE
 *   train_i6 <dir> [H1 H2 EPOCHS BATCH]
 *        [--seed N] [--lr L] [--lr-min frac] [--wd f] [--mom f] [--clip f]
 *        [--warmup-frac f] [--eval-frac f] [--pos-repeat n]
 *        [--init-from PATH] [--mu-sd PATH]
 *        [--export-lancius PATH] [--lancius-rows N] [--verify-lancius]
 *        [--resume PATH] [--save-dir D]
 *        [--hard-negative-frac f] [--hard-negative-start EPOCH]
 *
 * Data layout: corpus.hdr + {train,val,test}.{X,L,S,A}.bin (I3 format,
 * FEAT=65 NCLASS=5). Produced by generation_1/I6/gen_data_i6.py or reused
 * from generation_1/I3/corpus.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/stat.h>

#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <lancius/lancius_train.h>

#define CKPT_MAGIC 0x49364334u /* "I6C4": sparse semantic input schema */
#define PI_2 3.14159265358979323846
#define NCMAX 8

static void die(const char* m){ fprintf(stderr, "FATAL: %s\n", m); exit(2); }
static void die2(const char* m, const char* w){
    fprintf(stderr, "FATAL: %s (%s)\n", m, w); exit(2);
}
static void* xcalloc(size_t n, size_t sz, const char* what){
    void* p = calloc(n ? n : 1, sz);
    if (!p) { fprintf(stderr, "FATAL: oom %s\n", what); exit(2); }
    return p;
}

/* ---- deterministic RNG (xorshift64*), state is checkpointed ---- */
static uint64_t rng_state;
static double u01(void){
    rng_state = rng_state*6364136223846793007ULL + 1442695040888963407ULL;
    uint64_t z = rng_state;
    z ^= z>>12; z ^= z<<25; z ^= z>>27; z *= 0x2545F4914F6CDD1DULL;
    rng_state = z;
    return (double)(z>>11)/9007199254740992.0;
}
static double gauss_(void){
    double u1 = u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = u01();
    return sqrt(-2.0*log(u1))*cos(2.0*PI_2*u2);
}

/* ---- configuration (compact trunk, algorithm-held-out CE training) ---- */
static size_t FEAT = 65, NCLASS = 5, H1 = 128, H2 = 64, BATCH = 256;
/* I6 coordinate 57 is the binary any_wrong predicate. */
static double LR = 0.0008, LR_MIN_FRAC = 0.01, MOM = 0.9, WD = 1e-4;
static double CLIP = 5.0, EVAL_FRAC = 0.5, WARMUP_FRAC = 0.03, VALFRAC = 0.15;
static int    EPOCHS = 20;
static unsigned SEED = 20261026u;
static size_t g_step = 0;

/* Hard negative mining */
static double HARD_NEG_FRAC = 0.5;   /* fraction of next epoch that is misclassified rows */
static int    HARD_NEG_START = 2;    /* begin mining after this epoch (1-based) */

/* Sampling repeats.  The corpus is about 18% overall verdict rows and 82%
 * step rows; the verdict is the target, so the overall rows are repeated.
 * Class 1 no longer needs the old 20x repeat: that compensated for the
 * 1-step label ambiguity, which the cleaned corpus removes. */
static size_t POS_REPEAT = 1;
static size_t OVERALL_REPEAT = 4;

static const double LEVEL[NCMAX] = {-1.0, -0.5, 0.0, 0.5, 1.0, 0, 0, 0};

/* ---- corpus ---- */
typedef struct { size_t n; double* X; int* L; int* S; int* A; } corpus;

static long fsize(const char* p){
    struct stat st; return stat(p, &st) == 0 ? (long)st.st_size : -1;
}

/* corpus.hdr: "KEY value" lines (FEAT, NCLASS, NTRAIN, NVAL, NTEST). */
static void read_hdr(const char* dir, size_t *feat, size_t *nclass,
                     size_t *ntrain, size_t *nval, size_t *ntest){
    char p[512]; snprintf(p, sizeof p, "%s/corpus.hdr", dir);
    FILE* f = fopen(p, "r");
    if (!f) die2("corpus.hdr missing (run gen_data_i6.py or point at I3 corpus)", p);
    char line[128]; int seen = 0;
    while (fgets(line, sizeof line, f)) {
        char k[64]; long v;
        if (sscanf(line, "%63s %ld", k, &v) != 2) continue;
        if      (!strcmp(k, "FEAT"))   { *feat = (size_t)v;   seen |= 1; }
        else if (!strcmp(k, "NCLASS")) { *nclass = (size_t)v; seen |= 2; }
        else if (!strcmp(k, "NTRAIN")) { *ntrain = (size_t)v; seen |= 4; }
        else if (!strcmp(k, "NVAL"))   { *nval = (size_t)v;   seen |= 8; }
        else if (!strcmp(k, "NTEST"))  { *ntest = (size_t)v;  seen |= 16; }
    }
    fclose(f);
    /* Required: FEAT|NCLASS|NTRAIN|NTEST = 23; optional NVAL adds 8 -> 31. */
    if ((seen & 23) != 23) die2("corpus.hdr incomplete (need FEAT NCLASS NTRAIN [NVAL] NTEST)", p);
    if (!(seen & 8)) { /* no NVAL: split from train using VALFRAC */
        *nval = (size_t)((double)*ntrain * VALFRAC);
        if (*nval < 1) *nval = 1;
    }
}

static void* read_bin(const char* p, size_t expect_bytes, const char* what){
    long by = fsize(p);
    if (by < 0) die2("cannot stat", p);
    if ((size_t)by != expect_bytes) {
        fprintf(stderr, "FATAL: %s is %ld bytes, header says %zu\n", p, by, expect_bytes);
        exit(2);
    }
    void* buf = xcalloc(expect_bytes ? expect_bytes : 1, 1, what);
    FILE* f = fopen(p, "rb");
    if (!f || fread(buf, 1, expect_bytes, f) != expect_bytes) die2("short read", p);
    fclose(f);
    return buf;
}

static corpus load(const char* dir, const char* split, size_t feat,
                   size_t n_expect, const char* tag){
    corpus c; memset(&c, 0, sizeof c);
    char p[512];
    snprintf(p, sizeof p, "%s/%s.X.bin", dir, split);
    c.n = n_expect;
    size_t xb = c.n*feat*sizeof(double);
    c.X = (double*)read_bin(p, xb, "X");
    /* Poison-row guard: NaN/Inf features kill training silently otherwise. */
    for (size_t i = 0; i < c.n*feat; i++)
        if (!isfinite(c.X[i])) {
            fprintf(stderr, "FATAL: non-finite feature in %s row %zu (idx %zu)\n",
                    p, i/feat, i); exit(2);
        }
    snprintf(p, sizeof p, "%s/%s.L.bin", dir, split);
    c.L = (int*)read_bin(p, c.n*sizeof(int), "L");
    snprintf(p, sizeof p, "%s/%s.S.bin", dir, split);
    c.S = (int*)read_bin(p, c.n*sizeof(int), "S");
    snprintf(p, sizeof p, "%s/%s.A.bin", dir, split);
    c.A = (int*)read_bin(p, c.n*sizeof(int), "A");
    for (size_t i = 0; i < c.n; i++)
        if (c.L[i] < 0 || (size_t)c.L[i] >= NCLASS)
            { fprintf(stderr, "FATAL: label OOB in %s row %zu\n", tag, i); exit(2); }
    return c;
}

/* ---- model ---- */
typedef struct {
    double *W1,*b1,*W2,*b2,*W3,*b3;
    double *mW1,*mb1,*mW2,*mb2,*mW3,*mb3;   /* momentum (incl. biases) */
    double *gW1,*gb1,*gW2,*gb2,*gW3,*gb3;
    double *Xb,*Tb;                          /* bound batch buffers */
} model;

static size_t nparams(void){
    return (size_t)FEAT*H1 + H1 + (size_t)H1*H2 + H2 + (size_t)H2*NCLASS + NCLASS;
}

static void minit(model* m){
    m->W1=(double*)xcalloc((size_t)FEAT*H1,8,"W1");  m->b1=(double*)xcalloc(H1,8,"b1");
    m->W2=(double*)xcalloc((size_t)H1*H2,8,"W2");    m->b2=(double*)xcalloc(H2,8,"b2");
    m->W3=(double*)xcalloc((size_t)H2*NCLASS,8,"W3");m->b3=(double*)xcalloc(NCLASS,8,"b3");
    m->mW1=(double*)xcalloc((size_t)FEAT*H1,8,"mW1");m->mb1=(double*)xcalloc(H1,8,"mb1");
    m->mW2=(double*)xcalloc((size_t)H1*H2,8,"mW2");  m->mb2=(double*)xcalloc(H2,8,"mb2");
    m->mW3=(double*)xcalloc((size_t)H2*NCLASS,8,"mW3");m->mb3=(double*)xcalloc(NCLASS,8,"mb3");
    m->gW1=(double*)xcalloc((size_t)FEAT*H1,8,"gW1");m->gb1=(double*)xcalloc(H1,8,"gb1");
    m->gW2=(double*)xcalloc((size_t)H1*H2,8,"gW2");  m->gb2=(double*)xcalloc(H2,8,"gb2");
    m->gW3=(double*)xcalloc((size_t)H2*NCLASS,8,"gW3");m->gb3=(double*)xcalloc(NCLASS,8,"gb3");
    m->Xb=(double*)xcalloc(BATCH*FEAT,8,"Xb");
    m->Tb=(double*)xcalloc(BATCH*NCLASS,8,"Tb");
    double s1=sqrt(2.0/((double)FEAT+(double)H1));
    double s2=sqrt(2.0/((double)H1+(double)H2));
    double s3=sqrt(2.0/((double)H2+(double)NCLASS));
    for(size_t i=0;i<(size_t)FEAT*H1;i++)     m->W1[i]=gauss_()*s1;
    for(size_t i=0;i<(size_t)H1*H2;i++)       m->W2[i]=gauss_()*s2;
    for(size_t i=0;i<(size_t)H2*NCLASS;i++)   m->W3[i]=gauss_()*s3;
}

static void mfree(model* m){
    free(m->W1);free(m->b1);free(m->W2);free(m->b2);free(m->W3);free(m->b3);
    free(m->mW1);free(m->mb1);free(m->mW2);free(m->mb2);free(m->mW3);free(m->mb3);
    free(m->gW1);free(m->gb1);free(m->gW2);free(m->gb2);free(m->gW3);free(m->gb3);
    free(m->Xb);free(m->Tb);
}

/* ---- persistent training graph (built once, buffers bound by pointer) ---- */
static lancius_arena* ARENA = NULL;
static lancius_graph* TG_FWD = NULL;
static lancius_training_graph* TG = NULL;
static lancius_schedule* BS = NULL;
static lancius_node *NX,*NW1,*NB1,*NW2,*NB2,*NW3,*NB3,*NT,*NLOSS;

static void tgraph_build(model* m){
    TG_FWD = lancius_graph_create(); if(!TG_FWD) die("graph");
    NX  = lancius_input(TG_FWD, BATCH, FEAT);
    NW1 = lancius_input(TG_FWD, FEAT, H1);
    NB1 = lancius_input(TG_FWD, 1, H1);
    NW2 = lancius_input(TG_FWD, H1, H2);
    NB2 = lancius_input(TG_FWD, 1, H2);
    NW3 = lancius_input(TG_FWD, H2, NCLASS);
    NB3 = lancius_input(TG_FWD, 1, NCLASS);
    NT  = lancius_input(TG_FWD, BATCH, NCLASS);
    if(!NX||!NW1||!NB1||!NW2||!NB2||!NW3||!NB3||!NT) die("builders");
    lancius_node_bind_external(NX, m->Xb);
    lancius_node_bind_external(NW1, m->W1);
    lancius_node_bind_external(NB1, m->b1);
    lancius_node_bind_external(NW2, m->W2);
    lancius_node_bind_external(NB2, m->b2);
    lancius_node_bind_external(NW3, m->W3);
    lancius_node_bind_external(NB3, m->b3);
    lancius_node_bind_external(NT, m->Tb);
    lancius_node* h1=lancius_tanh(TG_FWD,lancius_add(TG_FWD,
        lancius_matmul(TG_FWD,NX,NW1),NB1));
    lancius_node* h2=lancius_tanh(TG_FWD,lancius_add(TG_FWD,
        lancius_matmul(TG_FWD,h1,NW2),NB2));
    lancius_node* z3=lancius_add(TG_FWD,lancius_matmul(TG_FWD,h2,NW3),NB3);
    /* Fused softmax cross-entropy (mean over rows), with exact target-sum VJP. */
    NLOSS=lancius_cross_entropy(TG_FWD,z3,NT);
    if(!h1||!h2||!z3||!NLOSS) die("loss");
    TG = lancius_ir_autodiff(TG_FWD, NLOSS); if(!TG) die("autodiff");
    BS = lancius_ir_schedule(TG->graph);     if(!BS) die("bwd sched");
}

static void tgraph_free(void){
    if (BS) lancius_schedule_destroy(BS);
    if (TG) lancius_training_graph_destroy(TG);
    if (TG_FWD) lancius_graph_destroy(TG_FWD);
    BS = NULL; TG = NULL; TG_FWD = NULL;
}

/* grad tensor registry for global-norm clip + guards */
static double* GPTR[6]; static size_t GN[6];
static void grad_registry(model* m){
    GPTR[0]=m->gW1; GN[0]=(size_t)FEAT*H1;   GPTR[1]=m->gb1; GN[1]=H1;
    GPTR[2]=m->gW2; GN[2]=(size_t)H1*H2;     GPTR[3]=m->gb2; GN[3]=H2;
    GPTR[4]=m->gW3; GN[4]=(size_t)H2*NCLASS; GPTR[5]=m->gb3; GN[5]=NCLASS;
}

static double step_model(model* m){
    /* The fixed-size graph uses replayed tail rows with weighted targets, so
     * its mean loss and gradient equal the mean over the real tail rows. */
    lancius_arena_reset(ARENA);
    lancius_schedule_execute(BS, ARENA);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        fprintf(stderr, "train step err=%s\n",
                lancius_error_string(lancius_get_error())); exit(2);
    }
    lancius_clear_error();
    /* Loss value is READ every step. */
    double loss = TG->loss_node->runtime_data[0];
    if (!isfinite(loss)) {
        fprintf(stderr, "FATAL: non-finite loss at step %zu\n", g_step);
        exit(2);
    }
#define HG(dst,node,cnt) do{ lancius_node* g_=TG->grad_nodes[(node)->id]; \
    if(!g_||!g_->runtime_data){ fprintf(stderr,"FATAL: missing grad\n"); exit(2);} \
    memcpy((dst),g_->runtime_data,(cnt)*sizeof(double)); }while(0)
    HG(m->gW1,NW1,(size_t)FEAT*H1); HG(m->gb1,NB1,H1);
    HG(m->gW2,NW2,(size_t)H1*H2);   HG(m->gb2,NB2,H2);
    HG(m->gW3,NW3,(size_t)H2*NCLASS); HG(m->gb3,NB3,NCLASS);
#undef HG
    for (int t = 0; t < 6; t++)
        for (size_t i = 0; i < GN[t]; i++)
            if (!isfinite(GPTR[t][i])) {
                fprintf(stderr, "FATAL: non-finite grad tensor %d at step %zu\n",
                        t, g_step); exit(2);
            }
    lancius_clip_global_norm(GPTR, GN, 6, CLIP);
    return loss;
}

/* SGDM with decoupled weight decay applied directly to parameters. */
static void apply_step(model* m, double lr){
#define UPD(W,M,G,N,WDV) do{ for(size_t i=0;i<(size_t)(N);i++){ \
    (M)[i]=MOM*(M)[i]+(G)[i]; \
    (W)[i]=(1.0-lr*(WDV))*(W)[i]-lr*(M)[i]; } }while(0)
    UPD(m->W1,m->mW1,m->gW1,(size_t)FEAT*H1,WD);
    UPD(m->b1,m->mb1,m->gb1,H1,0.0);
    UPD(m->W2,m->mW2,m->gW2,(size_t)H1*H2,WD);
    UPD(m->b2,m->mb2,m->gb2,H2,0.0);
    UPD(m->W3,m->mW3,m->gW3,(size_t)H2*NCLASS,WD);
    UPD(m->b3,m->mb3,m->gb3,NCLASS,0.0);
#undef UPD
    for (size_t i = 0; i < (size_t)FEAT*H1; i++)
        if (!isfinite(m->W1[i])) { fprintf(stderr,"FATAL: W1[%zu] diverged at step %zu\n",i,g_step); exit(2); }
    for (size_t i = 0; i < (size_t)H1*H2; i++)
        if (!isfinite(m->W2[i])) { fprintf(stderr,"FATAL: W2[%zu] diverged at step %zu\n",i,g_step); exit(2); }
    for (size_t i = 0; i < (size_t)H2*NCLASS; i++)
        if (!isfinite(m->W3[i])) { fprintf(stderr,"FATAL: W3[%zu] diverged at step %zu\n",i,g_step); exit(2); }
}

/* ---- inference forward (OpenMP, deterministic per-row order) ---- */
static void fwd_logits(const model* m,const double* X,size_t n,double* z){
    #pragma omp parallel
    {
        double* hid1=(double*)malloc(H1*sizeof(double));
        double* hid2=(double*)malloc(H2*sizeof(double));
        if(!hid1||!hid2) die("fwd tmp");
        #pragma omp for schedule(static)
        for(size_t i=0;i<n;i++){
            for(size_t j=0;j<H1;j++){
                double s=m->b1[j];
                for(size_t k=0;k<FEAT;k++) s+=X[i*FEAT+k]*m->W1[k*H1+j];
                hid1[j]=tanh(s);
            }
            for(size_t j=0;j<H2;j++){
                double s=m->b2[j];
                for(size_t k=0;k<H1;k++) s+=hid1[k]*m->W2[k*H2+j];
                hid2[j]=tanh(s);
            }
            for(size_t c=0;c<NCLASS;c++){
                double s=m->b3[c];
                for(size_t j=0;j<H2;j++) s+=hid2[j]*m->W3[j*NCLASS+c];
                z[i*NCLASS+c]=s;
            }
        }
        free(hid1); free(hid2);
    }
}

/* ---- metrics (overall rows vs per-step rows split by S>0) ---- */
typedef struct {
    size_t n, exact, within1;
    double mae, mse, macro_f1, kappa;
    long cm[NCMAX][NCMAX];
} met;

static void eval_split(met* mo, met* ms, const model* md,
                       const double* X, const int* L, const int* S, size_t n,
                       double logit_scale){
    if (!(logit_scale > 0.0) || !isfinite(logit_scale))
        die("logit scale must be finite and positive");
    double* z=(double*)malloc(n*NCLASS*sizeof(double));
    if(!z) die("eval z");
    fwd_logits(md,X,n,z);
    memset(mo,0,sizeof *mo); memset(ms,0,sizeof *ms);
    for(size_t i=0;i<n;i++){
        met* m = S[i]>0 ? ms : mo;
        int p=0;
        for(size_t c=1;c<NCLASS;c++) if(z[i*NCLASS+c]>z[i*NCLASS+p]) p=(int)c;
        int t=L[i];
        m->cm[t][p]++;
        if(p==t) m->exact++;
        if((int)p==t || (p>t? p-t : t-p)<=1) m->within1++;
        /* Match serving: continuous verdict is E[level | logits], not the
         * argmax class level. Use max-subtracted softmax for stability. */
        double mx = z[i*NCLASS] * logit_scale;
        for(size_t c=1;c<NCLASS;c++) {
            double scaled = z[i*NCLASS+c] * logit_scale;
            if (!isfinite(scaled)) { free(z); die("non-finite scaled logit"); }
            if(scaled>mx) mx=scaled;
        }
        double denom = 0.0, expected = 0.0;
        for(size_t c=0;c<NCLASS;c++) {
            double scaled = z[i*NCLASS+c] * logit_scale;
            if (!isfinite(scaled)) { free(z); die("non-finite scaled logit"); }
            double e = exp(scaled - mx);
            denom += e;
            expected += e * LEVEL[c];
        }
        if (!(denom > 0.0) || !isfinite(denom) || !isfinite(expected)) {
            free(z);
            die("non-finite softmax during continuous metric evaluation");
        }
        expected /= denom;
        if (expected < -1.0 + 1e-6) expected = -1.0 + 1e-6;
        if (expected >  1.0 - 1e-6) expected =  1.0 - 1e-6;
        double err = expected - LEVEL[t];
        m->mae += fabs(err);
        m->mse += err * err;
        m->n++;
    }
    for (int which = 0; which < 2; which++){
        met* m = which ? ms : mo;
        if (!m->n) continue;
        m->mae /= (double)m->n;
        m->mse /= (double)m->n;
        double f1=0;
        for(size_t c=0;c<NCLASS;c++){
            double tp=m->cm[c][c], fp=0, fn=0;
            for(size_t r=0;r<NCLASS;r++){ if((int)r!=(int)c){ fp+=m->cm[r][c]; fn+=m->cm[c][r]; } }
            double pr=(tp+fp>0)?tp/(tp+fp):0.0, rc=(tp+fn>0)?tp/(tp+fn):0.0;
            f1 += (pr+rc>0)? 2*pr*rc/(pr+rc):0.0;
        }
        m->macro_f1=f1/(double)NCLASS;
        double po=(double)m->exact/(double)m->n, pe=0;
        for(size_t c=0;c<NCLASS;c++){
            double true_count=0, pred_count=0;
            for(size_t r=0;r<NCLASS;r++) {
                true_count += m->cm[c][r];
                pred_count += m->cm[r][c];
            }
            pe += (true_count/(double)m->n)*(pred_count/(double)m->n);
        }
        m->kappa = (pe<1.0)? (po-pe)/(1.0-pe) : 0.0;
    }
    free(z);
}

/* ---- checkpoint / resume ---- */
static int save_state(const char* path, const model* m, const double* mu,
                      const double* sd, long step, double best_score){
    FILE* f=fopen(path,"wb"); if(!f) return -1;
    uint32_t magic=CKPT_MAGIC;
    size_t dims[5]={FEAT,NCLASS,H1,H2,BATCH};
    fwrite(&magic,sizeof magic,1,f);
    fwrite(dims,sizeof dims,1,f);
    fwrite(&step,sizeof step,1,f);
    fwrite(&best_score,sizeof best_score,1,f);
    fwrite(&rng_state,sizeof rng_state,1,f);
    fwrite(mu,sizeof(double),FEAT,f);
    fwrite(sd,sizeof(double),FEAT,f);
    fwrite(m->W1,8,(size_t)FEAT*H1,f);   fwrite(m->b1,8,H1,f);
    fwrite(m->W2,8,(size_t)H1*H2,f);     fwrite(m->b2,8,H2,f);
    fwrite(m->W3,8,(size_t)H2*NCLASS,f); fwrite(m->b3,8,NCLASS,f);
    fwrite(m->mW1,8,(size_t)FEAT*H1,f);  fwrite(m->mb1,8,H1,f);
    fwrite(m->mW2,8,(size_t)H1*H2,f);    fwrite(m->mb2,8,H2,f);
    fwrite(m->mW3,8,(size_t)H2*NCLASS,f);fwrite(m->mb3,8,NCLASS,f);
    int ok = ferror(f)==0;
    fclose(f);
    return ok ? 0 : -1;
}

static int load_state(const char* path, model* m, double* mu, double* sd,
                      long* step, double* best_score){
    FILE* f=fopen(path,"rb"); if(!f) return -1;
    uint32_t magic=0; size_t dims[5];
    if (fread(&magic,sizeof magic,1,f)!=1 || magic!=CKPT_MAGIC) { fclose(f); return -1; }
    if (fread(dims,sizeof dims,1,f)!=1) { fclose(f); return -1; }
    if (dims[0]!=FEAT||dims[1]!=NCLASS||dims[2]!=H1||dims[3]!=H2||dims[4]!=BATCH)
        { fprintf(stderr,"FATAL: checkpoint dims mismatch\n"); exit(2); }
    long st; double bs; uint64_t rs;
    if (fread(&st,sizeof st,1,f)!=1 || fread(&bs,sizeof bs,1,f)!=1 ||
        fread(&rs,sizeof rs,1,f)!=1) { fclose(f); return -1; }
    if (fread(mu,8,FEAT,f)!=FEAT) { fclose(f); return -1; }
    if (fread(sd,8,FEAT,f)!=FEAT) { fclose(f); return -1; }
    if (fread(m->W1,8,(size_t)FEAT*H1,f)!=(size_t)FEAT*H1) { fclose(f); return -1; }
    if (fread(m->b1,8,H1,f)!=H1) { fclose(f); return -1; }
    if (fread(m->W2,8,(size_t)H1*H2,f)!=(size_t)H1*H2) { fclose(f); return -1; }
    if (fread(m->b2,8,H2,f)!=H2) { fclose(f); return -1; }
    if (fread(m->W3,8,(size_t)H2*NCLASS,f)!=(size_t)H2*NCLASS) { fclose(f); return -1; }
    if (fread(m->b3,8,NCLASS,f)!=NCLASS) { fclose(f); return -1; }
    if (fread(m->mW1,8,(size_t)FEAT*H1,f)!=(size_t)FEAT*H1) { fclose(f); return -1; }
    if (fread(m->mb1,8,H1,f)!=H1) { fclose(f); return -1; }
    if (fread(m->mW2,8,(size_t)H1*H2,f)!=(size_t)H1*H2) { fclose(f); return -1; }
    if (fread(m->mb2,8,H2,f)!=H2) { fclose(f); return -1; }
    if (fread(m->mW3,8,(size_t)H2*NCLASS,f)!=(size_t)H2*NCLASS) { fclose(f); return -1; }
    if (fread(m->mb3,8,NCLASS,f)!=NCLASS) { fclose(f); return -1; }
    fclose(f);
    g_step=(size_t)st; *step=st; *best_score=bs; rng_state=rs;
    return 0;
}

/* ---- .lancius export (default rows=1) ---- */
static int export_lancius(const model* m, const char* path, size_t rows) {
    if (rows < 1) rows = 1;
    lancius_graph* g = lancius_graph_create();
    if (!g) return -1;
    lancius_node* Xn  = lancius_input(g, rows, FEAT);
    lancius_node* W1n = lancius_input(g, FEAT, H1);
    lancius_node* b1n = lancius_input(g, 1, H1);
    lancius_node* W2n = lancius_input(g, H1, H2);
    lancius_node* b2n = lancius_input(g, 1, H2);
    lancius_node* W3n = lancius_input(g, H2, NCLASS);
    lancius_node* b3n = lancius_input(g, 1, NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n||!W3n||!b3n) {
        fprintf(stderr, "export: node builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    lancius_node_bind_external(W1n, m->W1);
    lancius_node_bind_external(b1n, m->b1);
    lancius_node_bind_external(W2n, m->W2);
    lancius_node_bind_external(b2n, m->b2);
    lancius_node_bind_external(W3n, m->W3);
    lancius_node_bind_external(b3n, m->b3);
    lancius_node* h1 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, Xn, W1n), b1n));
    lancius_node* h2 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, h1, W2n), b2n));
    lancius_node* z3 = lancius_add(g, lancius_matmul(g, h2, W3n), b3n);
    if (!h1 || !h2 || !z3) {
        fprintf(stderr, "export: op builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    int rc = lancius_graph_save(g, path);
    if (rc != 0) fprintf(stderr, "lancius_graph_save failed: %s\n",
                         lancius_error_string(lancius_get_error()));
    lancius_graph_destroy(g);
    return rc;
}

static int verify_lancius(const char* path, const model* m,
                          const double* X, size_t n) {
    lancius_graph* g = lancius_graph_load(path);
    if (!g) { fprintf(stderr, "load %s failed: %s\n", path,
                      lancius_error_string(lancius_get_error())); return -1; }
    lancius_node* Xn = NULL;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* nd = g->nodes[i];
        if (nd && nd->op == LANCIUS_OP_INPUT && nd->ndim == 2 &&
            nd->shape[0] == n && nd->shape[1] == FEAT) {
            Xn = nd; break;
        }
    }
    if (!Xn) { fprintf(stderr, "no [%zu,%zu] INPUT in %s\n", n, FEAT, path);
               lancius_graph_destroy(g); return -1; }
    lancius_arena* ar = lancius_arena_create(256u*1024u*1024u);
    if (!ar) { lancius_graph_destroy(g); return -1; }
    lancius_node_bind_external(Xn, (double*)X);
    lancius_schedule* sch = lancius_ir_schedule(g);
    if (!sch) { lancius_arena_destroy(ar); lancius_graph_destroy(g); return -1; }
    lancius_schedule_execute(sch, ar);
    int err = (int)lancius_get_error();
    if (err != 0) {
        fprintf(stderr, "reloaded execute err=%s\n",
                lancius_error_string(lancius_get_error()));
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); return -1;
    }
    lancius_node* out = NULL;
    for (uint32_t i = 0; i < g->node_count && !out; i++) {
        lancius_node* nd = g->nodes[i];
        if (!nd || !nd->runtime_data || nd->ndim != 2) continue;
        if (nd->op == LANCIUS_OP_INPUT || nd->op == LANCIUS_OP_CONST) continue;
        if (nd->shape[0] != n || nd->shape[1] != NCLASS) continue;
        int consumed = 0;
        for (uint32_t j = 0; j < g->node_count && !consumed; j++) {
            lancius_node* mm = g->nodes[j];
            if (!mm || !mm->inputs) continue;
            for (uint32_t c = 0; c < mm->input_count; c++)
                if (mm->inputs[c] == nd) { consumed = 1; break; }
        }
        if (!consumed) out = nd;
    }
    if (!out) { fprintf(stderr, "no unconsumed [n,%zu] node\n", NCLASS);
                lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
                lancius_graph_destroy(g); return -1; }
    double worst = 0.0;
    double* ref = (double*)malloc(n*NCLASS*sizeof(double));
    fwd_logits(m, X, n, ref);
    for (size_t i = 0; i < n*NCLASS; i++) {
        double d = fabs(ref[i] - out->runtime_data[i]);
        if (d > worst) worst = d;
    }
    free(ref);
    lancius_schedule_destroy(sch);
    lancius_arena_destroy(ar);
    lancius_graph_destroy(g);
    printf("verify-lancius: max|diff| = %.3e\n", worst);
    return (worst > 1e-12) ? -1 : 0;
}

/* ---- affine feature normalization (train-fit or pinned stats) ---- */
static void zscore_fit(const corpus* tr, double* mu, double* sd){
    for(size_t k=0;k<FEAT;k++) mu[k]=0.0;
    for(size_t i=0;i<tr->n;i++)
        for(size_t k=0;k<FEAT;k++) mu[k]+=tr->X[i*FEAT+k];
    for(size_t k=0;k<FEAT;k++) mu[k]/=(double)tr->n;
    for(size_t k=0;k<FEAT;k++) sd[k]=0.0;
    for(size_t i=0;i<tr->n;i++)
        for(size_t k=0;k<FEAT;k++){ double d=tr->X[i*FEAT+k]-mu[k]; sd[k]+=d*d; }
    for(size_t k=0;k<FEAT;k++){ sd[k]=sqrt(sd[k]/(double)tr->n);
        if(!(sd[k]>1e-12)) sd[k]=1.0; }
}
static void zscore_apply(corpus* c, const double* mu, const double* sd){
    for(size_t i=0;i<c->n;i++)
        for(size_t k=0;k<FEAT;k++){
            double v=(c->X[i*FEAT+k]-mu[k])/sd[k];
            if(v>8.0) v=8.0;
            if(v<-8.0) v=-8.0;
            c->X[i*FEAT+k]=v;
        }
}

/* ---- Fisher-Yates shuffle ---- */
static void shuffle_idx(size_t* idx, size_t n){
    for(size_t i=n;i>1;i--){
        size_t j=(size_t)u01()*(double)i;
        size_t t=idx[i-1]; idx[i-1]=idx[j]; idx[j]=t;
    }
}

/* ---- hard negative mining: build next-epoch index pool with oversampled
 * misclassified rows ---- */
static void build_epoch_index(const model* m, const corpus* tr,
                              size_t* idx, size_t* nidx, int epoch){
    /* Start with all train rows, then repeat the verdict rows (and class 1
     * if a caller asks for it). */
    size_t q=0;
    for(size_t i=0;i<tr->n;i++) idx[q++]=i;
    for(size_t r=1;r<POS_REPEAT;r++)
        for(size_t i=0;i<tr->n;i++) if(tr->L[i]==1) idx[q++]=i;
    for(size_t r=1;r<OVERALL_REPEAT;r++)
        for(size_t i=0;i<tr->n;i++) if(tr->S[i]==0) idx[q++]=i;
    *nidx = q;

    if (epoch < HARD_NEG_START || HARD_NEG_FRAC <= 0.0) return;

    /* Evaluate on train to find misclassified rows. */
    size_t eval_n = tr->n;
    double* z=(double*)malloc(eval_n*NCLASS*sizeof(double));
    if(!z) die("hardneg z");
    fwd_logits(m, tr->X, eval_n, z);
    size_t nwrong=0;
    size_t* wrong=(size_t*)xcalloc(eval_n,sizeof(size_t),"wrong");
    for(size_t i=0;i<eval_n;i++){
        int p=0;
        for(size_t c=1;c<NCLASS;c++) if(z[i*NCLASS+c]>z[i*NCLASS+p]) p=(int)c;
        if(p != tr->L[i]) wrong[nwrong++]=i;
    }
    free(z);

    if (nwrong == 0) { free(wrong); return; }

    /* The repeated verdict rows sit at the end of the base pool; shuffle so
     * the kept half is a random sample instead of the un-repeated prefix. */
    shuffle_idx(idx, q);

    /* Replace HARD_NEG_FRAC of the pool with misclassified rows (sampled with
     * replacement if needed). */
    size_t nreplace = (size_t)((double)q * HARD_NEG_FRAC);
    if (nreplace > q) nreplace = q;
    size_t keep = q - nreplace;
    /* The first `keep` shuffled entries stay; the tail becomes hard rows. */
    for(size_t i=0;i<nreplace;i++){
        size_t w = wrong[(size_t)(u01()*(double)nwrong)];
        idx[keep+i]=w;
    }
    free(wrong);
}

/* ---- main ---- */
int main(int argc, char** argv){
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffer stdout so log file shows progress */
    const char* dir = NULL;
    const char* export_path = NULL;
    const char* resume_path = NULL;
    const char* init_from = NULL;
    const char* mu_sd_path = NULL;
    const char* save_dir = ".";
    int do_verify_rt = 0;
    int strict_targets = 0;
    size_t lancius_rows = 1;
    size_t hdr_feat=0, hdr_nclass=0;
    size_t ntrain=0, nval=0, ntest=0;
    int feat_cli=0, nclass_cli=0;

    int nposarg = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i+1 < argc) SEED=(unsigned)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) LR=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--lr-min") && i+1 < argc) LR_MIN_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--wd") && i+1 < argc) WD=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--mom") && i+1 < argc) MOM=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--clip") && i+1 < argc) CLIP=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--warmup-frac") && i+1 < argc) WARMUP_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--eval-frac") && i+1 < argc) EVAL_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--pos-repeat") && i+1 < argc) POS_REPEAT=(size_t)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--overall-repeat") && i+1 < argc) OVERALL_REPEAT=(size_t)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--hard-negative-frac") && i+1 < argc) HARD_NEG_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--hard-negative-start") && i+1 < argc) HARD_NEG_START=(int)strtol(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--export-lancius") && i+1 < argc) export_path=argv[++i];
        else if (!strcmp(argv[i], "--lancius-rows") && i+1 < argc) lancius_rows=(size_t)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--verify-lancius")) do_verify_rt=1;
        else if (!strcmp(argv[i], "--strict-targets")) strict_targets=1;
        else if (!strcmp(argv[i], "--resume") && i+1 < argc) resume_path=argv[++i];
        else if (!strcmp(argv[i], "--init-from") && i+1 < argc) init_from=argv[++i];
        else if (!strcmp(argv[i], "--mu-sd") && i+1 < argc) mu_sd_path=argv[++i];
        else if (!strcmp(argv[i], "--save-dir") && i+1 < argc) save_dir=argv[++i];
        else if (!strcmp(argv[i], "--feat") && i+1 < argc) { FEAT=(size_t)strtoul(argv[++i],NULL,10); feat_cli=1; }
        else if (!strcmp(argv[i], "--nclass") && i+1 < argc) { NCLASS=(size_t)strtoul(argv[++i],NULL,10); nclass_cli=1; }
        else if (argv[i][0] != '-') {
            switch (nposarg++) {
                case 0: dir = argv[i]; break;
                case 1: H1=strtoul(argv[i],NULL,10); break;
                case 2: H2=strtoul(argv[i],NULL,10); break;
                case 3: EPOCHS=atoi(argv[i]); break;
                case 4: BATCH=strtoul(argv[i],NULL,10); break;
                default: fprintf(stderr,"too many positional args\n"); return 2;
            }
        }
    }
    if (POS_REPEAT < 1) POS_REPEAT = 1;
    if (OVERALL_REPEAT < 1) OVERALL_REPEAT = 1;
    if (!dir) {
        fprintf(stderr,
        "usage: train_i6 <dir> [H1 H2 EPOCHS BATCH]\n"
        "  [--seed N] [--lr L] [--lr-min frac] [--wd f] [--mom f] [--clip f]\n"
        "  [--warmup-frac f] [--eval-frac f]\n"
        "  [--mu-sd PATH] [--init-from PATH rejected for I6 schema safety]\n"
        "  [--export-lancius PATH] [--lancius-rows N] [--verify-lancius]\n"
        "  [--resume PATH] [--save-dir D] [--strict-targets]\n"
        "  [--hard-negative-frac f] [--hard-negative-start EPOCH]\n"
        "\n"
        "I6 trainer: 65 -> H1 -> H2 -> 5 verifier, CE loss, persistent graph,\n"
        "algorithm-held-out validation. Overall and per-step MAE/MSE target < 1e-3.\n");
        return 2;
    }
    if (init_from) {
        fprintf(stderr, "ERROR: --init-from is incompatible with I6's sparse "
                "feature schema and train-only normalizer; train from fresh "
                "initialization.\n");
        return 2;
    }
    if (NCLASS < 2 || NCLASS > NCMAX) die("train_i6 expects 2 <= NCLASS <= 8");
    if (FEAT == 0) die("FEAT must be > 0");
    if (BATCH == 0 || EPOCHS <= 0) die("BATCH and EPOCHS must be positive");

    read_hdr(dir, &hdr_feat, &hdr_nclass, &ntrain, &nval, &ntest);
    if (!feat_cli)  FEAT = hdr_feat;
    if (!nclass_cli) NCLASS = hdr_nclass;
    if (hdr_feat != 0 && feat_cli && hdr_feat != FEAT) {
        fprintf(stderr, "WARN: --feat %zu differs from corpus.hdr FEAT=%zu\n", FEAT, hdr_feat);
    }

    corpus tr = load(dir, "train", FEAT, ntrain, "train");
    corpus va; memset(&va,0,sizeof va);
    corpus te = load(dir, "test",  FEAT, ntest,  "test");
    if (te.n == 0) die("empty test split");

    /* If the header had no NVAL, split VALFRAC off the train rows for
     * validation. Otherwise load the separate val split. */
    int had_nval = 0;
    {
        char p[512]; snprintf(p, sizeof p, "%s/corpus.hdr", dir);
        FILE* f=fopen(p,"r");
        if (f) { char line[128]; char k[64]; long v;
            while (fgets(line,sizeof line,f)) {
                if (sscanf(line,"%63s %ld",k,&v)==2 && !strcmp(k,"NVAL")) { had_nval=1; break; }
            }
            fclose(f);
        }
    }
    if (had_nval) {
        va = load(dir, "val", FEAT, nval, "val");
        if (va.n == 0) die("empty val split");
    } else {
        /* carve VALFRAC off the end of train as validation (deterministic,
         * no shuffle -- the corpus is already shuffled by gen_data). */
        size_t nv = (size_t)((double)tr.n * VALFRAC);
        if (nv < 1) nv = 1;
        if (nv >= tr.n) nv = tr.n / 2;
        va.n = nv;
        va.X = (double*)xcalloc(nv*FEAT,8,"va.X");
        va.L = (int*)xcalloc(nv,8,"va.L");
        va.S = (int*)xcalloc(nv,8,"va.S");
        va.A = (int*)xcalloc(nv,8,"va.A");
        for(size_t i=0;i<nv;i++){
            size_t si = tr.n - nv + i;
            memcpy(va.X+i*FEAT, tr.X+si*FEAT, FEAT*sizeof(double));
            va.L[i]=tr.L[si]; va.S[i]=tr.S[si]; va.A[i]=tr.A[si];
        }
        tr.n -= nv;
    }

    /* z-score normalization: use pinned --mu-sd if provided (so training
     * and evaluation share the same feature space), otherwise fit from train. */
    double* mu=(double*)xcalloc(FEAT,8,"mu");
    double* sd=(double*)xcalloc(FEAT,8,"sd");
    int mu_sd_loaded = 0;
    if (mu_sd_path) {
        FILE* f=fopen(mu_sd_path,"r");
        if (f) {
            int nm=0, ns=0;
            char line[256];
            while (fgets(line,sizeof line,f)) {
                if (line[0]=='M'&&line[1]=='U'&&line[2]==' ' && nm<65)
                    mu[nm++]=strtod(line+3,NULL);
                else if (line[0]=='S'&&line[1]=='D'&&line[2]==' ' && ns<65)
                    sd[ns++]=strtod(line+3,NULL);
            }
            fclose(f);
            if (nm==65 && ns==65) {
                for(size_t k=0;k<65;k++) if(!(sd[k]>1e-12)) sd[k]=1.0;
                mu_sd_loaded = 1;
                printf("using pinned MU/SD from %s\n", mu_sd_path);
            } else {
                fprintf(stderr, "WARN: --mu-sd %s has %d MU / %d SD lines, "
                        "falling back to train fit\n", mu_sd_path, nm, ns);
            }
        } else {
            fprintf(stderr, "WARN: --mu-sd %s not found, falling back to train fit\n", mu_sd_path);
        }
    }
    if (!mu_sd_loaded) {
        zscore_fit(&tr, mu, sd);
        printf("fitted MU/SD from train rows\n");
    }
    zscore_apply(&tr, mu, sd);
    zscore_apply(&va, mu, sd);
    zscore_apply(&te, mu, sd);

    size_t npos=0, nover=0;
    for(size_t i=0;i<tr.n;i++) {
        if(tr.L[i]==1) npos++;
        if(tr.S[i]==0) nover++;
    }
    if(!npos) die("train split has no positive rows");
    size_t nfit = tr.n;
    if (npos && POS_REPEAT - 1 > (SIZE_MAX - nfit) / npos)
        die("positive-repeat index count overflows size_t");
    nfit += npos * (POS_REPEAT - 1);
    if (nover && OVERALL_REPEAT - 1 > (SIZE_MAX - nfit) / nover)
        die("overall-repeat index count overflows size_t");
    nfit += nover * (OVERALL_REPEAT - 1);
    size_t* idx=(size_t*)xcalloc(nfit,sizeof(size_t),"idx");
    size_t nidx=0;

    model m; minit(&m);

    /* Load checkpoint if resuming. */
    long ckpt_step=0; double ckpt_best=INFINITY;
    int has_ckpt = 0;
    char ckpt_last[1024]; snprintf(ckpt_last,sizeof ckpt_last,"%s/last.bin",save_dir);
    if (resume_path) {
        if (load_state(resume_path, &m, mu, sd, &ckpt_step, &ckpt_best) != 0)
            die("resume: cannot load checkpoint");
        has_ckpt = 1;
    } else if (fsize(ckpt_last) >= 0) {
        /* Auto-resume from save_dir/last.bin if present. */
        if (load_state(ckpt_last, &m, mu, sd, &ckpt_step, &ckpt_best) == 0)
            has_ckpt = 1;
        else
            fprintf(stderr, "WARN: ignoring incompatible or unreadable %s\n", ckpt_last);
    }

    ARENA = lancius_arena_create(256u*1024u*1024u);
    if (!ARENA) die("arena");
    tgraph_build(&m);
    grad_registry(&m);

    if (has_ckpt && (ckpt_step < 0 || !isfinite(ckpt_best)))
        die("resume checkpoint has invalid step or best score");
    double best_score = has_ckpt ? ckpt_best : INFINITY;
    if (nfit == 0) die("training split is empty");
    size_t steps_per_epoch = nfit / BATCH + (nfit % BATCH != 0);
    if (steps_per_epoch == 0 || steps_per_epoch > (size_t)INT_MAX / (size_t)EPOCHS)
        die("training schedule exceeds integer step range");
    int total_steps = EPOCHS * (int)steps_per_epoch;
    size_t start_epoch = has_ckpt ? (size_t)ckpt_step / steps_per_epoch : 0;
    if (start_epoch > (size_t)EPOCHS) start_epoch = (size_t)EPOCHS;

    printf("================================================================\n");
    printf("  Lancius I6 decision-surface trainer  %zu -> %zu -> %zu -> %zu\n"
           "  (%.3fM params)  CE objective, persistent graph\n"
           "  initialization: fresh Glorot weights\n"
           "  hard negative: frac=%.2f start_epoch=%d\n"
           "================================================================\n",
           FEAT, H1, H2, NCLASS, (double)nparams()/1e6,
           HARD_NEG_FRAC, HARD_NEG_START);

    for (int ep = (int)start_epoch; ep < EPOCHS; ep++) {
        /* Build this epoch's index pool (with hard negatives if enabled). */
        build_epoch_index(&m, &tr, idx, &nidx, ep+1);
        shuffle_idx(idx, nidx);

        double epoch_loss_sum = 0.0; size_t epoch_rows = 0, epoch_steps = 0;
        for (size_t s = 0; s < nidx; s += BATCH) {
            size_t rows = nidx - s;
            if (rows > BATCH) rows = BATCH;
            /* Fill batch buffers. */
            for (size_t i = 0; i < BATCH; i++) {
                /* Replay the tail to fill the fixed graph batch, assigning
                 * each copy 1/repeats of its row's weight. */
                size_t copy = i % rows;
                size_t repeats = BATCH / rows + (copy < BATCH % rows);
                double row_weight = (double)BATCH /
                                    ((double)rows * (double)repeats);
                size_t ri = idx[s + copy];
                memcpy(m.Xb + i*FEAT, tr.X + ri*FEAT, FEAT*sizeof(double));
                for(size_t c=0;c<NCLASS;c++)
                    m.Tb[i*NCLASS+c] = (tr.L[ri]==(int)c)?row_weight:0.0;
            }
            double loss = step_model(&m);
            epoch_loss_sum += loss * (double)rows;
            epoch_rows += rows;
            epoch_steps++;

            /* LR schedule: warmup + cosine (step-based, not frac-based). */
            int warmup_steps = (int)(WARMUP_FRAC * (double)total_steps);
            if (g_step > (size_t)INT_MAX) die("global step exceeds scheduler range");
            double lr = lancius_lr_warmup_cosine((int)g_step, warmup_steps,
                                                 total_steps, LR, LR * LR_MIN_FRAC);
            apply_step(&m, lr);
            g_step++;

            if (epoch_steps % 100 == 0) {
                printf("  epoch %d step %zu loss=%.6f lr=%.6g\n",
                       ep+1, epoch_steps, loss, lr);
            }
        }
        double epoch_loss = epoch_rows ? epoch_loss_sum / (double)epoch_rows : 0.0;

        /* Checkpoint selection uses validation only. The test split is read
         * once after training, so it cannot steer epoch-by-epoch decisions. */
        met mo, ms; eval_split(&mo, &ms, &m, va.X, va.L, va.S, va.n, 1.0);
        printf("epoch %2d/%d train_loss=%.6f  "
               "val overall exact=%.4f MAE=%.6g MSE=%.6g  "
               "per-step exact=%.4f MAE=%.6g MSE=%.6g\n",
               ep+1, EPOCHS, epoch_loss,
               mo.n?(double)mo.exact/(double)mo.n:0.0, mo.mae, mo.mse,
               ms.n?(double)ms.exact/(double)ms.n:0.0, ms.mae, ms.mse);

        /* Minimize the worst of the four required validation measurements.
         * Scale each metric by its 1e-3 target so no condition can hide behind
         * a better score in another one. */
        double score = 0.0;
        int score_terms = 0;
#define SCORE_METRIC(METRIC) do { \
    if ((METRIC).n) { \
        double q_ = (METRIC).mae / 1e-3; \
        if (q_ > score) score = q_; \
        q_ = (METRIC).mse / 1e-3; \
        if (q_ > score) score = q_; \
        score_terms++; \
    } \
} while (0)
        SCORE_METRIC(mo);
        SCORE_METRIC(ms);
#undef SCORE_METRIC
        if (!score_terms) die("validation split has no rows");

        char path_best[1024], path_last[1024];
        snprintf(path_best,sizeof path_best,"%s/best.bin",save_dir);
        snprintf(path_last,sizeof path_last,"%s/last.bin",save_dir);

        int improved = 0;
        if (score < best_score - 1e-7) {
            improved = 1;
            best_score = score;
        }
        if (improved) {
            if (save_state(path_best, &m, mu, sd, (long)g_step, best_score) != 0)
                fprintf(stderr, "WARN: failed to save best.bin\n");
            if (improved) printf("  -> new best (worst normalized val error=%.6g)\n",
                                 best_score);
        }
        if (save_state(path_last, &m, mu, sd, (long)g_step, best_score) != 0)
            fprintf(stderr, "WARN: failed to save last.bin\n");

        /* Warn if continuous validation MAE remains high after hard negatives
         * have had time to affect the fit. */
        if (HARD_NEG_FRAC > 0.0 && ep+1 >= HARD_NEG_START &&
            mo.mae > 0.1 && ep+1 >= HARD_NEG_START+3) {
            fprintf(stderr, "WARN: val MAE still %.6f after hard-negative mining; "
                    "consider more data or a wider trunk\n", mo.mae);
        }
    }

    /* Final test evaluation with the best state loaded (if we saved one). */
    char path_best[1024]; snprintf(path_best,sizeof path_best,"%s/best.bin",save_dir);
    if (fsize(path_best) >= 0) {
        long st; double bs;
        if (load_state(path_best, &m, mu, sd, &st, &bs) == 0) {
            printf("loaded best.bin (step=%ld score=%.6f) for final eval\n", st, bs);
        } else fprintf(stderr, "WARN: best.bin is incompatible or unreadable; evaluating current state\n");
    }
    /* Fit a single positive logit scale on validation only. This changes
     * confidence, not the argmax class, and is folded into W3/b3 before export
     * so the continuous readout measured here matches the deployed artifact. */
    static const double calibration_scales[] = {
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0
    };
    double calibration_scale = 1.0;
    double calibration_score = INFINITY;
    met cmo, cms;
    for(size_t i=0;i<sizeof calibration_scales/sizeof calibration_scales[0];i++) {
        double scale = calibration_scales[i];
        eval_split(&cmo, &cms, &m, va.X, va.L, va.S, va.n, scale);
        double score = 0.0;
#define CAL_SCORE(METRIC) do { \
    if ((METRIC).n) { \
        double q_ = (METRIC).mae; if (q_ > score) score = q_; \
        q_ = (METRIC).mse; if (q_ > score) score = q_; \
    } \
} while (0)
        CAL_SCORE(cmo);
        CAL_SCORE(cms);
#undef CAL_SCORE
        if (score < calibration_score) {
            calibration_score = score;
            calibration_scale = scale;
        }
    }
    for(size_t i=0;i<(size_t)H2*NCLASS;i++) m.W3[i] *= calibration_scale;
    for(size_t i=0;i<NCLASS;i++) m.b3[i] *= calibration_scale;
    eval_split(&cmo, &cms, &m, va.X, va.L, va.S, va.n, 1.0);
    printf("VALIDATION calibration: logit_scale=%.6g overall MAE=%.9g MSE=%.9g "
           "per-step MAE=%.9g MSE=%.9g\n",
           calibration_scale, cmo.mae, cmo.mse, cms.mae, cms.mse);
    met to, ts; eval_split(&to, &ts, &m, te.X, te.L, te.S, te.n, 1.0);
    printf("================================================================\n");
    printf("FINAL test overall exact=%.4f within1=%.4f macroF1=%.4f "
           "MAE=%.9g MSE=%.9g kappa=%.4f\n",
           to.n?(double)to.exact/(double)to.n:0.0,
           to.n?(double)to.within1/(double)to.n:0.0,
           to.macro_f1, to.mae, to.mse, to.kappa);
    printf("FINAL test per-step exact=%.4f within1=%.4f macroF1=%.4f "
           "MAE=%.9g MSE=%.9g kappa=%.4f\n",
           ts.n?(double)ts.exact/(double)ts.n:0.0,
           ts.n?(double)ts.within1/(double)ts.n:0.0,
           ts.macro_f1, ts.mae, ts.mse, ts.kappa);
    int targets_pass = to.mae < 1e-3 && to.mse < 1e-3 &&
                       ts.mae < 1e-3 && ts.mse < 1e-3;
    printf("Targets overall MAE/MSE < 1e-3: %s/%s; "
           "per-step MAE/MSE < 1e-3: %s/%s\n",
           to.mae < 1e-3 ? "PASS" : "FAIL",
           to.mse < 1e-3 ? "PASS" : "FAIL",
           ts.mae < 1e-3 ? "PASS" : "FAIL",
           ts.mse < 1e-3 ? "PASS" : "FAIL");
    printf("================================================================\n");

    if (export_path) {
        if (export_lancius(&m, export_path, lancius_rows) != 0)
            die("export failed");
        printf("exported %s (%zu rows)\n", export_path, lancius_rows);
        if (do_verify_rt) {
            /* Verify on the first `lancius_rows` test rows. */
            size_t vn = lancius_rows < te.n ? lancius_rows : te.n;
            if (verify_lancius(export_path, &m, te.X, vn) != 0)
                die("verify-lancius failed");
        }
    }

    tgraph_free();
    mfree(&m);
    free(idx);
    free(mu); free(sd);
    lancius_arena_destroy(ARENA);
    return strict_targets && !targets_pass ? 3 : 0;
}

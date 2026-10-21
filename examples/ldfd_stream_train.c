/* Lancius x LDFD stream plug (experimental, v12R3) -- one trainer, two data
 * sources, one result.
 *
 * The v12R3 requirement: the whole framework can train from a PREDOWNLOADED
 * corpus and from a LIVE LDFD stream interchangeably. This binary proves the
 * seam end to end on a small model:
 *
 *   phase 1  LANDED        > synthetic CSV corpus written to
 *                          temp/ldfd_stream/fixture.csv (the "downloaded
 *                          dataset" sitting on disk);
 *   phase 2  LIVE STREAM   > loopback HTTP server serves the very same
 *                          corpus in tiny slices; LDFD (snap_fetch_once +
 *                          snap_csv_parser) consumes the chunks and the row
 *                          callback fills the training buffers directly --
 *                          no fopen on the live data path;
 *   phase 3  TRAIN (LIVE)  > persistent-graph trainer (train_verifier
 *                          pattern) 2 -> 8 -> 2 tanh, fused softmax CE,
 *                          momentum SGD, Fisher-Yates batches;
 *   phase 4  PREDOWNLOADED > reread fixture.csv from disk, feed the same
 *                          parser from file chunks, train an identically
 *                          initialised model;
 *   phase 5  PARITY        > both weight sets compared tensor by tensor +
 *                          hold-out accuracy gated;
 *   phase 6  ROUND-TRIP    > live-streamed model saved as .lancius,
 *                          reloaded, executed and compared against the
 *                          in-process forward.
 *
 * Any failed gate exits nonzero: rows swallowed mid-chunk-split, accuracy at
 * chance, source disagreement, or a round-trip mismatch all turn red.
 */
#define _POSIX_C_SOURCE 200809L

#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <lancius/lancius_train.h>
#include <snapshot.h>
#include <parser_csv.h>

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define FEAT    2
#define NCLASS  2
#define H1      8
#define BATCH   32
#define NROWS   512
#define NFIT    432          /* rows 0..431 train, 432..511 hold-out */
#define NHOLD   (NROWS - NFIT)
#define EPOCHS  24
#define LR      0.08
#define MOM     0.9
#define CLIP    5.0
#define SEED    7u
#define MODEL_PATH "temp/ldfd_stream/ldfd_plug.lancius"

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(2); }
static void *xcalloc(size_t n, size_t sz, const char *what) {
    void *p = calloc(n ? n : 1, sz);
    if (!p) { fprintf(stderr, "FATAL: oom %s\n", what); exit(2); }
    return p;
}

/* xorshift64*: same generator train_verifier uses, so both sources see
 * identical initialisation and shuffle streams. */
static uint64_t rng_state;
static void rseed(uint64_t s) {
    rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL;
}
static double u01(void) {
    uint64_t z = rng_state;
    z ^= z >> 12; z ^= z << 25; z ^= z >> 27;
    rng_state = z * 0x2545F4914F6CDD1DULL;
    z = rng_state;
    return (double)(z >> 11) / 9007199254740992.0;
}
static double gauss_(void) {
    double u1 = u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = u01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* ---------------------------------------------------------- ground truth */

static double g_X[NROWS * FEAT];
static int    g_L[NROWS];

static void build_dataset(void) {
    rseed(SEED);
    for (size_t i = 0; i < NROWS; i++) {
        double x1 = u01() * 2.0 - 1.0;
        double x2 = u01() * 2.0 - 1.0;
        double score = x1 - 0.9 * x2 + 0.25 * sin(3.0 * x1);
        g_X[i * FEAT + 0] = x1;
        g_X[i * FEAT + 1] = x2;
        g_L[i] = score > 0.0 ? 1 : 0;
    }
}

/* CSV text is built from the same doubles with %.17g so the on-disk, on-wire
 * and in-memory forms round-trip exactly -- required for the parity gate. */
static char *fixture_text;
static size_t fixture_len;

static void build_fixture(const char *path) {
    size_t cap = NROWS * 80 + 64;
    fixture_text = xcalloc(cap, 1, "fixture");
    fixture_len = (size_t)snprintf(fixture_text, cap, "x1,x2,label\n");
    for (size_t i = 0; i < NROWS; i++) {
        fixture_len += (size_t)snprintf(fixture_text + fixture_len,
                                        cap - fixture_len, "%.17g,%.17g,%d\n",
                                        g_X[i * FEAT + 0], g_X[i * FEAT + 1],
                                        g_L[i]);
    }
    if (fixture_len >= cap) die("fixture overflow");
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write fixture.csv");
    if (fwrite(fixture_text, 1, fixture_len, f) != fixture_len) {
        fprintf(stderr, "FATAL: short fixture write\n"); exit(2);
    }
    if (fflush(f) != 0) die("fixture flush");
    fclose(f);
}

/* -------------------------------------------------------- row ingestion */

typedef struct {
    double *X;          /* [NROWS, FEAT] */
    int    *L;
    size_t  n, bad;
} rowsink;

static void on_row(const char **f, int nf, void *udata) {
    rowsink *s = udata;
    if (nf != 3) { s->bad++; return; }
    char *end1 = NULL, *end2 = NULL;
    double x1 = strtod(f[0], &end1);
    double x2 = strtod(f[1], &end2);
    int lab = atoi(f[2]);
    if (end1 == f[0] || end2 == f[1] || (lab != 0 && lab != 1)) { s->bad++; return; }
    if (!isfinite(x1) || !isfinite(x2)) { s->bad++; return; }
    if (s->n >= NROWS) { s->bad++; return; }
    s->X[s->n * FEAT + 0] = x1;
    s->X[s->n * FEAT + 1] = x2;
    s->L[s->n] = lab;
    s->n++;
}

/* Feed raw bytes through the LDFD csv parser (used by the PREDOWNLOADED
 * path; the LIVE path feeds via snap_fetch_once instead). 997-byte chunks
 * guarantee mid-row splits. */
static void rows_feed_bytes(rowsink *s, const uint8_t *data, size_t len) {
    snap_parser_t csvp;
    memset(&csvp, 0, sizeof csvp);
    csvp = snap_csv_parser;
    if (csvp.init(&csvp.ctx, "delimiter=,") != SNAP_OK) die("csv parser init");
    snap_csv_set_callback(csvp.ctx, on_row, s);
    while (len) {
        snap_chunk_t c;
        c.data = (uint8_t *)data;
        c.len = len > 997 ? 997 : len;
        c.is_final = 0;
        if (csvp.feed(csvp.ctx, &c) != SNAP_OK) die("csv feed error");
        data += c.len;
        len -= c.len;
    }
    if (csvp.flush(csvp.ctx) != SNAP_OK) die("csv flush error");
    csvp.free(csvp.ctx);
}

/* Live path: one streaming fetch through the LDFD parser pipeline, exactly
 * like context.c's snap_ctx_run_once arms a pipeline parser. */
static void live_fetch_rows(const char *url, rowsink *s) {
    snap_parser_t csvp;
    memset(&csvp, 0, sizeof csvp);
    csvp = snap_csv_parser;
    if (csvp.init(&csvp.ctx, "delimiter=,") != SNAP_OK) die("csv parser init");
    snap_csv_set_callback(csvp.ctx, on_row, s);
    snap_error_t rc = snap_fetch_once(url, NULL, &csvp, csvp.ctx);
    csvp.free(csvp.ctx);
    if (rc != SNAP_OK) die("snap_fetch_once failed");
}

/* ------------------------------------------------- loopback HTTP server */

typedef struct {
    int listen_fd;
    int port;
    pthread_t th;
} srv_t;

static uint64_t slice_rng;   /* server-thread-local; main LCG is untouched */

static void *srv_thread(void *arg) {
    srv_t *s = arg;
    for (;;) {
        int fd = accept(s->listen_fd, NULL, NULL);
        if (fd < 0) break;
        char req[1024];
        ssize_t got = recv(fd, req, sizeof req - 1, 0);
        (void)got;
        const char *head = "HTTP/1.1 200 OK\r\n"
                           "Content-Type: text/csv\r\n"
                           "Connection: close\r\n\r\n";
        if (send(fd, head, strlen(head), 0) < 0) { close(fd); continue; }
        /* Tiny slices so chunk boundaries land mid-row most of the time. */
        size_t off = 0;
        while (off < fixture_len) {
            slice_rng = slice_rng * 6364136223846793007ULL +
                        1442695040888963407ULL;
            size_t step = 13 + (size_t)((slice_rng >> 33) % 377u);
            if (step > fixture_len - off) step = fixture_len - off;
            size_t done = 0;
            while (done < step) {
                ssize_t w = send(fd, fixture_text + off + done, step - done, 0);
                if (w <= 0) break;
                done += (size_t)w;
            }
            off += step;
        }
        close(fd);
    }
    return NULL;
}

static void srv_start(srv_t *s) {
    s->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s->listen_fd < 0) die("socket");
    int one = 1;
    setsockopt(s->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bind(s->listen_fd, (struct sockaddr *)&a, sizeof a) != 0) die("bind");
    if (listen(s->listen_fd, 2) != 0) die("listen");
    socklen_t al = sizeof a;
    if (getsockname(s->listen_fd, (struct sockaddr *)&a, &al) != 0) die("getsockname");
    s->port = ntohs(a.sin_port);
    slice_rng = 0xC0FFEE123456789ULL;
    if (pthread_create(&s->th, NULL, srv_thread, s) != 0) die("server thread");
}

static void srv_stop(srv_t *s) {
    shutdown(s->listen_fd, SHUT_RDWR);
    close(s->listen_fd);
    pthread_join(s->th, NULL);
}

/* ------------------------------------------------------------------ model */

typedef struct {
    double *W1, *b1, *W2, *b2;
    double *mW1, *mb1, *mW2, *mb2;
    double *gW1, *gb1, *gW2, *gb2;
    double *Xb, *Tb;
} model_t;

#define SZW1 ((size_t)FEAT * H1)
#define SZB1 ((size_t)H1)
#define SZW2 ((size_t)H1 * NCLASS)
#define SZB2 ((size_t)NCLASS)

static void minit(model_t *m) {
    m->W1 = xcalloc(SZW1, 8, "W1"); m->b1 = xcalloc(SZB1, 8, "b1");
    m->W2 = xcalloc(SZW2, 8, "W2"); m->b2 = xcalloc(SZB2, 8, "b2");
    m->mW1 = xcalloc(SZW1, 8, "mW1"); m->mb1 = xcalloc(SZB1, 8, "mb1");
    m->mW2 = xcalloc(SZW2, 8, "mW2"); m->mb2 = xcalloc(SZB2, 8, "mb2");
    m->gW1 = xcalloc(SZW1, 8, "gW1"); m->gb1 = xcalloc(SZB1, 8, "gb1");
    m->gW2 = xcalloc(SZW2, 8, "gW2"); m->gb2 = xcalloc(SZB2, 8, "gb2");
    m->Xb = xcalloc(BATCH * FEAT, 8, "Xb");
    m->Tb = xcalloc(BATCH * NCLASS, 8, "Tb");
    double s1 = sqrt(2.0 / (double)FEAT), s2 = sqrt(2.0 / (double)H1);
    for (size_t i = 0; i < SZW1; i++) m->W1[i] = gauss_() * s1;
    for (size_t i = 0; i < SZW2; i++) m->W2[i] = gauss_() * s2;
}

static void mfree(model_t *m) {
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->mW1); free(m->mb1); free(m->mW2); free(m->mb2);
    free(m->gW1); free(m->gb1); free(m->gW2); free(m->gb2);
    free(m->Xb); free(m->Tb);
}

static lancius_arena *ARENA;
static lancius_graph *tg_fwd;
static lancius_training_graph *tg;
static lancius_schedule *sched;
static lancius_node *nW1, *nb1, *nW2, *nb2, *nloss;

static void tgraph_build(model_t *m) {
    tg_fwd = lancius_graph_create(); if (!tg_fwd) die("graph");
    lancius_node *nX = lancius_input(tg_fwd, BATCH, FEAT);
    nW1 = lancius_input(tg_fwd, FEAT, H1);
    nb1 = lancius_input(tg_fwd, 1, H1);
    nW2 = lancius_input(tg_fwd, H1, NCLASS);
    nb2 = lancius_input(tg_fwd, 1, NCLASS);
    lancius_node *nY = lancius_input(tg_fwd, BATCH, NCLASS);
    if (!nX || !nW1 || !nb1 || !nW2 || !nb2 || !nY) die("builders");
    lancius_node_bind_external(nX, m->Xb);
    lancius_node_bind_external(nW1, m->W1);
    lancius_node_bind_external(nb1, m->b1);
    lancius_node_bind_external(nW2, m->W2);
    lancius_node_bind_external(nb2, m->b2);
    lancius_node_bind_external(nY, m->Tb);
    lancius_node *h = lancius_tanh(tg_fwd, lancius_add(tg_fwd,
        lancius_matmul(tg_fwd, nX, nW1), nb1));
    lancius_node *z = lancius_add(tg_fwd, lancius_matmul(tg_fwd, h, nW2), nb2);
    nloss = lancius_cross_entropy(tg_fwd, z, nY);
    if (!h || !z || !nloss) die("loss builder");
    tg = lancius_ir_autodiff(tg_fwd, nloss);
    if (!tg) die("autodiff");
    sched = lancius_ir_schedule(tg->graph);
    if (!sched) die("sched");
}

static void tgraph_free(void) {
    if (sched) lancius_schedule_destroy(sched);
    if (tg) lancius_training_graph_destroy(tg);
    if (tg_fwd) lancius_graph_destroy(tg_fwd);
    sched = NULL; tg = NULL; tg_fwd = NULL;
}

static double step_model(model_t *m) {
    lancius_arena_reset(ARENA);
    lancius_schedule_execute(sched, ARENA);
    if (lancius_get_error() != LANCIUS_ERROR_OK) die("schedule execute");
    lancius_clear_error();
    if (!tg->loss_node || !tg->loss_node->runtime_data) die("no loss value");
    double loss = tg->loss_node->runtime_data[0];
    if (!isfinite(loss)) { fprintf(stderr, "FATAL: loss=%g\n", loss); exit(2); }
    #define HG(dst, node, cnt) do { lancius_node *g_ = tg->grad_nodes[(node)->id]; \
        if (!g_ || !g_->runtime_data) die("missing grad"); \
        memcpy((dst), g_->runtime_data, (cnt) * sizeof(double)); } while (0)
    HG(m->gW1, nW1, SZW1); HG(m->gb1, nb1, SZB1);
    HG(m->gW2, nW2, SZW2); HG(m->gb2, nb2, SZB2);
    #undef HG
    double *gp[4]; size_t gn[4];
    gp[0] = m->gW1; gn[0] = SZW1; gp[1] = m->gb1; gn[1] = SZB1;
    gp[2] = m->gW2; gn[2] = SZW2; gp[3] = m->gb2; gn[3] = SZB2;
    for (int t = 0; t < 4; t++)
        for (size_t i = 0; i < gn[t]; i++)
            if (!isfinite(gp[t][i])) {
                fprintf(stderr, "FATAL: grad %d NaN\n", t); exit(2);
            }
    lancius_clip_global_norm(gp, gn, 4, CLIP);
    return loss;
}

static void apply_step(model_t *m) {
    #define UPD(W, M, G, N) do { for (size_t i = 0; i < (N); i++) { \
        (M)[i] = MOM * (M)[i] + (G)[i]; (W)[i] -= LR * (M)[i]; } } while (0)
    UPD(m->W1, m->mW1, m->gW1, SZW1);
    UPD(m->b1, m->mb1, m->gb1, SZB1);
    UPD(m->W2, m->mW2, m->gW2, SZW2);
    UPD(m->b2, m->mb2, m->gb2, SZB2);
    #undef UPD
}

static void fwd_logits(const model_t *m, const double *X, size_t n, double *z) {
    for (size_t i = 0; i < n; i++) {
        double hid[H1];
        for (size_t j = 0; j < H1; j++) {
            double s = m->b1[j];
            for (size_t k = 0; k < FEAT; k++) s += X[i * FEAT + k] * m->W1[k * H1 + j];
            hid[j] = tanh(s);
        }
        for (size_t c = 0; c < NCLASS; c++) {
            double s = m->b2[c];
            for (size_t j = 0; j < H1; j++) s += hid[j] * m->W2[j * NCLASS + c];
            z[i * NCLASS + c] = s;
        }
    }
}

static double accuracy(const model_t *m, const double *X, const int *L, size_t n) {
    double *z = xcalloc(n * NCLASS, 8, "eval z");
    fwd_logits(m, X, n, z);
    size_t ok = 0;
    for (size_t i = 0; i < n; i++) {
        int pred = z[i * NCLASS + 1] > z[i * NCLASS + 0] ? 1 : 0;
        if (pred == L[i]) ok++;
    }
    free(z);
    return (double)ok / (double)n;
}

/* Train over rows[f0..f0+n): returns the mean CE of the final epoch, stores
 * the first step's CE through loss_first. */
static double train(model_t *m, const double *X, const int *L, size_t f0,
                    size_t n, const char *tag, double *loss_first) {
    size_t idx[NFIT];
    for (size_t i = 0; i < n; i++) idx[i] = i;
    rseed(SEED ^ 0xA777u);
    double last_avg = 0.0;
    for (int ep = 0; ep < EPOCHS; ep++) {
        for (size_t i = n - 1; i > 0; i--) { /* Fisher-Yates */
            size_t j = (size_t)(u01() * (double)(i + 1));
            size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        double sum = 0.0; size_t steps = 0;
        for (size_t s = 0; s < n; s += BATCH) {
            size_t rows = n - s; if (rows > BATCH) rows = BATCH;
            for (size_t r = 0; r < BATCH; r++) { /* final batch padded by replay */
                size_t src = f0 + idx[s + (r % rows)];
                memcpy(m->Xb + r * FEAT, X + src * FEAT, FEAT * sizeof(double));
                for (size_t c = 0; c < NCLASS; c++) m->Tb[r * NCLASS + c] = 0.0;
                m->Tb[r * NCLASS + L[src]] = 1.0;
            }
            double ce = step_model(m);
            if (steps == 0 && ep == 0) *loss_first = ce;
            apply_step(m);
            sum += ce; steps++;
        }
        last_avg = sum / (double)steps;
        if (ep == 0) printf("   %s   epoch 1: mean CE %.4f -> ", tag, last_avg);
        else if (ep == EPOCHS - 1)
            printf("epoch %d: mean CE %.6f\n", EPOCHS, last_avg);
        fflush(stdout);
    }
    return last_avg;
}

/* ------------------------------------------------ .lancius save + reload */

static int export_model(const model_t *m, const char *path, size_t rows) {
    lancius_graph *g = lancius_graph_create();
    if (!g) return -1;
    lancius_node *Xn  = lancius_input(g, rows, FEAT);
    lancius_node *W1n = lancius_input(g, FEAT, H1);
    lancius_node *b1n = lancius_input(g, 1, H1);
    lancius_node *W2n = lancius_input(g, H1, NCLASS);
    lancius_node *b2n = lancius_input(g, 1, NCLASS);
    if (!Xn || !W1n || !b1n || !W2n || !b2n) {
        fprintf(stderr, "export: node builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_clear_error();
        lancius_graph_destroy(g); return -1;
    }
    lancius_node_bind_external(W1n, m->W1);
    lancius_node_bind_external(b1n, m->b1);
    lancius_node_bind_external(W2n, m->W2);
    lancius_node_bind_external(b2n, m->b2);
    lancius_node *h = lancius_tanh(g, lancius_add(g,
        lancius_matmul(g, Xn, W1n), b1n));
    lancius_node *z = lancius_add(g, lancius_matmul(g, h, W2n), b2n);
    if (!h || !z) {
        fprintf(stderr, "export: op builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_clear_error();
        lancius_graph_destroy(g); return -1;
    }
    int rc = lancius_graph_save(g, path);
    if (rc != 0) {
        fprintf(stderr, "lancius_graph_save failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_clear_error();
    }
    lancius_graph_destroy(g);
    return rc;
}

static int verify_model(const char *path, const model_t *m,
                        const double *X, size_t n, int *labels) {
    lancius_graph *g = lancius_graph_load(path);
    if (!g) {
        fprintf(stderr, "load %s failed: %s\n", path,
                lancius_error_string(lancius_get_error()));
        lancius_clear_error();
        return -1;
    }
    lancius_node *Xn = NULL;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node *nd = g->nodes[i];
        if (nd && nd->op == LANCIUS_OP_INPUT && nd->ndim == 2 &&
            nd->shape[0] == n && nd->shape[1] == FEAT) { Xn = nd; break; }
    }
    if (!Xn) {
        fprintf(stderr, "ROUND-TRIP: no [%zu,%zu] INPUT in %s\n", n,
                (size_t)FEAT, path);
        lancius_graph_destroy(g); return -1;
    }
    lancius_arena *ar = lancius_arena_create(256u * 1024u * 1024u);
    if (!ar) { lancius_graph_destroy(g); return -1; }
    lancius_node_bind_external(Xn, (double *)X);
    lancius_schedule *sch = lancius_ir_schedule(g);
    if (!sch) { lancius_arena_destroy(ar); lancius_graph_destroy(g); return -1; }
    lancius_schedule_execute(sch, ar);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        fprintf(stderr, "reloaded execute err=%s\n",
                lancius_error_string(lancius_get_error()));
        lancius_clear_error();
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); return -1;
    }
    lancius_clear_error();
    /* unconsumed [n, NCLASS] node = logits */
    lancius_node *out = NULL;
    for (uint32_t i = 0; i < g->node_count && !out; i++) {
        lancius_node *nd = g->nodes[i];
        if (!nd || !nd->runtime_data || nd->ndim != 2) continue;
        if (nd->op == LANCIUS_OP_INPUT || nd->op == LANCIUS_OP_CONST) continue;
        if (nd->shape[0] != n || nd->shape[1] != NCLASS) continue;
        int consumed = 0;
        for (uint32_t j = 0; j < g->node_count && !consumed; j++) {
            lancius_node *mm = g->nodes[j];
            if (!mm || !mm->inputs) continue;
            for (uint32_t c = 0; c < mm->input_count; c++)
                if (mm->inputs[c] == nd) { consumed = 1; break; }
        }
        if (!consumed) out = nd;
    }
    if (!out) {
        fprintf(stderr, "ROUND-TRIP: no unconsumed [%zu,%zu] node\n", n,
                (size_t)NCLASS);
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); return -1;
    }
    double worst = 0.0;
    double *ref = xcalloc(n * NCLASS, 8, "verify ref");
    fwd_logits(m, X, n, ref);
    for (size_t i = 0; i < n * NCLASS; i++) {
        double d = fabs(ref[i] - out->runtime_data[i]);
        if (d > worst) worst = d;
    }
    /* and the accuracy must survive the round-trip */
    size_t ok = 0;
    for (size_t i = 0; i < n; i++) {
        int pred = out->runtime_data[i * NCLASS + 1] >
                   out->runtime_data[i * NCLASS + 0] ? 1 : 0;
        if (pred == labels[i]) ok++;
    }
    double acc = (double)ok / (double)n;
    free(ref);
    lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
    lancius_graph_destroy(g);
    printf("   .lancius round-trip (reloaded on rows %zu): "
           "max |diff| = %.3e, acc %.4f  %s\n",
           n, worst, acc, worst < 1e-9 ? "OK" : "MISMATCH");
    return worst < 1e-9 ? 0 : -1;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    printf("================================================================\n");
    printf("  Lancius x LDFD stream plug -- one trainer, two data sources\n");
    printf("================================================================\n\n");

    build_dataset();
    if (system("mkdir -p temp/ldfd_stream") != 0) die("mkdir temp/ldfd_stream");
    build_fixture("temp/ldfd_stream/fixture.csv");
    printf("[1/6] LANDED:        %zu rows, %zu bytes -> %s\n",
           (size_t)NROWS, fixture_len, "temp/ldfd_stream/fixture.csv");

    rowsink live = { xcalloc(NROWS * FEAT, 8, "liveX"), xcalloc(NROWS, 4, "liveL"), 0, 0 };
    rowsink disk = { xcalloc(NROWS * FEAT, 8, "diskX"), xcalloc(NROWS, 4, "diskL"), 0, 0 };

    /* ---- phase 2: LIVE stream through the LDFD parser (no file I/O) ---- */
    srv_t srv;
    srv_start(&srv);
    char url[96];
    snprintf(url, sizeof url, "http://127.0.0.1:%d/fixture.csv", srv.port);
    printf("[2/6] LIVE STREAM:  LDFD snap_fetch_once <- %s\n", url);
    live_fetch_rows(url, &live);
    srv_stop(&srv);
    printf("                    parsed %zu/%zu rows, %zu malformed "
           "(mid-row splits exercised)\n", live.n, (size_t)NROWS, live.bad);
    if (live.n != NROWS || live.bad != 0) die("live stream lost rows");

    /* ---- phase 3: train on the LIVE-streamed rows ---- */
    model_t a, b;
    ARENA = lancius_arena_create(64u * 1024u * 1024u);
    if (!ARENA) die("arena");
    rseed(SEED);
    minit(&a);
    tgraph_build(&a);
    double la_first = 0.0, la_last = 0.0;
    printf("[3/6] TRAIN(LIVE):  2 -> %d -> %d tanh, batch %d, %d epochs\n",
           H1, NCLASS, BATCH, EPOCHS);
    la_last = train(&a, live.X, live.L, 0, NFIT, "live", &la_first);
    tgraph_free();
    double acc_live_fit = accuracy(&a, live.X, live.L, NFIT);
    double acc_live = accuracy(&a, live.X + NFIT * FEAT, g_L + NFIT, NHOLD);
    printf("                    fit acc %.4f | hold-out acc %.4f\n",
           acc_live_fit, acc_live);

    /* ---- phase 4: PREDOWNLOADED fixture reread from disk ---- */
    printf("[4/6] PREDOWNLOAD:  rereading temp/ldfd_stream/fixture.csv...\n");
    FILE *f = fopen("temp/ldfd_stream/fixture.csv", "rb");
    if (!f) die("fixture.csv missing on disk");
    if (fseek(f, 0, SEEK_END) != 0) die("seek");
    long fl = ftell(f);
    if (fl <= 0 || (size_t)fl != fixture_len) die("fixture size changed");
    rewind(f);
    char *fb = xcalloc(fixture_len, 1, "fixture readback");
    if (fread(fb, 1, fixture_len, f) != fixture_len) die("fixture read");
    int fc = fclose(f);
    if (fc != 0) die("fixture close");
    rows_feed_bytes(&disk, (const uint8_t *)fb, fixture_len);
    printf("                    parsed %zu/%zu rows, %zu malformed\n",
           disk.n, (size_t)NROWS, disk.bad);
    if (disk.n != NROWS || disk.bad != 0) die("disk stream lost rows");
    if (memcmp(live.X, disk.X, NROWS * FEAT * sizeof(double)) != 0 ||
        memcmp(live.L, disk.L, NROWS * sizeof(int)) != 0)
        die("live stream and disk fixture disagree");

    /* ---- phase 5: PREDOWNLOADED model + source parity ---- */
    rseed(SEED);
    minit(&b);
    tgraph_build(&b);
    double lb_first = 0.0, lb_last = 0.0;
    printf("[5/6] TRAIN(DISK):  identical init, identical batches\n");
    lb_last = train(&b, disk.X, disk.L, 0, NFIT, "disk", &lb_first);
    tgraph_free();
    double acc_disk = accuracy(&b, disk.X + NFIT * FEAT, g_L + NFIT, NHOLD);
    printf("                    fit acc %.4f | hold-out acc %.4f\n",
           accuracy(&b, disk.X, disk.L, NFIT), acc_disk);

    double worst = 0.0;
    #define CMP(T, N) do { for (size_t i = 0; i < (N); i++) { \
        double d = fabs(a.T[i] - b.T[i]); if (d > worst) worst = d; } } while (0)
    CMP(W1, SZW1); CMP(b1, SZB1); CMP(W2, SZW2); CMP(b2, SZB2);
    #undef CMP
    printf("\n   PARITY:         max |live - predownloaded| over all "
           "weights = %.3e\n", worst);

    if (worst > 1e-9) {
        fprintf(stderr, "FATAL: sources disagree after training "
                "(max diff %.3e > 1e-9)\n", worst);
        return 1;
    }
    if (acc_live < 0.90 || acc_disk < 0.90) {
        fprintf(stderr, "FATAL: hold-out acc at or under 90%% "
                "(live %.4f disk %.4f)\n", acc_live, acc_disk);
        return 1;
    }
    if (!(la_last < la_first)) die("live loss did not fall");
    if (!(lb_last < lb_first)) die("disk loss did not fall");
    printf("   loss fell on both sources: %.4f -> %.6f / %.6f\n",
           la_first, la_last, lb_last);
    printf("   hold-out: live %.4f, disk %.4f\n", acc_live, acc_disk);

    /* ---- phase 6: export the live-streamed model, reload, verify ---- */
    printf("[6/6] ROUND-TRIP:   saving live-streamed model -> %s\n", MODEL_PATH);
    if (export_model(&a, MODEL_PATH, NHOLD) != 0) die("model export");
    if (verify_model(MODEL_PATH, &a, live.X + NFIT * FEAT, NHOLD,
                     g_L + NFIT) != 0) die("model round-trip");

    lancius_arena_destroy(ARENA);
    mfree(&a); mfree(&b);
    free(live.X); free(live.L); free(disk.X); free(disk.L);
    free(fb); free(fixture_text);
    printf("\nLDFD STREAM PLUG COMPLETE: live-streamed and predownloaded "
           "sources train one model, agree, and round-trip.\n");
    return 0;
}

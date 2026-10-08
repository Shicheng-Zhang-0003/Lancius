/* R3-2 gate: TRANSPOSE_BATCHED + GELU/LayerNorm/RMSNorm/MATMUL_BATCHED VJPs.
 * Proves exact values, v2 roundtrip, and analytic-vs-finite-difference
 * agreement for every new backward. Exits 1 on divergence (no false-green).
 */
// R3-2 proof: TRANSPOSE_BATCHED values + GELU/LN/RMSN/MATMUL_BATCHED VJPs
// vs finite differences + v2 roundtrip of TRANSPOSE_BATCHED.
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>
#include <lancius.h>

/* Scratch directory for this audit's save/load probes.
 *
 * This used to be the literal /tmp/opencode/, which is a path that exists only
 * on the machine the audits were written on. On a GitHub runner the directory
 * does not exist, lancius_graph_save returns an IO error, and `make check`
 * fails on "v2 save ..." -- so the gate could never pass anywhere but the
 * author's laptop, which is the exact failure a standing gate exists to
 * prevent. LANCIUS_SCRATCH overrides it; the default is repo-relative and
 * created on demand. */
static const char* lancius_scratch_dir(void) {
    const char* s = getenv("LANCIUS_SCRATCH");
    if (!s || !*s) s = "temp/scratch";
    /* mkdir(2) does NOT create parent directories, and a fresh checkout has no
     * temp/ at all, so creating only the leaf is not enough -- the first fix
     * here passed locally precisely because temp/ already existed here and
     * failed on the runner for exactly that reason. Walk the chain instead. */
    char buf[512];
    size_t n = strlen(s);
    if (n >= sizeof buf) n = sizeof buf - 1;
    memcpy(buf, s, n);
    buf[n] = '\0';
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\0') {
            char save = buf[i];
            buf[i] = '\0';
            mkdir(buf, 0777);          /* EEXIST is the normal case */
            buf[i] = save;
        }
    }
    return s;
}static int checks = 0;
static int failures = 0;
#define CHECK(cond, msg) \
    do { \
        checks++; \
        if (!(cond)) { \
            printf("  FAIL: %s (err=%d)\n", msg, (int)lancius_get_error()); \
            failures++; \
        } \
    } while (0)

static lancius_arena* sc = NULL;
// forward loss helper: zero non-inputs, execute, read scalar, reset
static double floss(lancius_schedule* fs, lancius_node* loss) {
    for (uint32_t w = 0; w < fs->wave_count; w++)
        for (uint32_t k = 0; k < fs->waves[w].node_count; k++) {
            lancius_node* n = fs->waves[w].nodes[k];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
        }
    lancius_schedule_execute(fs, sc);
    double l = loss->runtime_data ? loss->runtime_data[0] : 1e300;
    lancius_arena_reset(sc);
    return l;
}
// compare analytic grad of param P (n elems) vs central differences
static void gradcheck(lancius_graph* g, lancius_schedule* fs, lancius_training_graph* tg,
                      lancius_node* P, size_t n, const char* name) {
    lancius_schedule* bs = lancius_ir_schedule(tg->graph);
    lancius_schedule_execute(bs, sc);
    lancius_node* gP = tg->grad_nodes[P->id];
    double* ana = NULL;
    if (gP && gP->runtime_data) {
        ana = malloc(n * sizeof(double));
        memcpy(ana, gP->runtime_data, n * sizeof(double));
    }
    lancius_arena_reset(sc);
    CHECK(ana != NULL, name);
    if (!ana) { lancius_schedule_destroy(bs); return; }
    double maxrel = 0.0;
    for (size_t i = 0; i < n; i++) {
        lancius_node* L = tg->loss_node;
        (void)L;
        // find loss node in fwd graph: last node
        lancius_node* loss = g->nodes[g->node_count - 1];
        P->runtime_data[i] += 1e-5;
        double lp = floss(fs, loss);
        P->runtime_data[i] -= 2e-5;
        double lm = floss(fs, loss);
        P->runtime_data[i] += 1e-5;
        double fd = (lp - lm) / 2e-5;
        double rel = fabs(ana[i] - fd) / (fabs(ana[i]) + fabs(fd) + 1e-8);
        if (rel > maxrel) maxrel = rel;
    }
    printf("  maxrel %s: %e\n", name, maxrel);
    char msg[128];
    snprintf(msg, sizeof(msg), "%s finite-diff", name);
    CHECK(maxrel < 1e-4, msg);
    free(ana);
    lancius_schedule_destroy(bs);
}
static void t_transpose_batched(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_3d(g, 2, 2, 3);
    double xd[12]; for (int i = 0; i < 12; i++) xd[i] = (double)(i + 1);
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    lancius_node* T = lancius_transpose_batched(g, X);
    CHECK(T && T->shape[0] == 2 && T->shape[1] == 3 && T->shape[2] == 2, "tbatch shape");
    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_schedule_execute(s, sc);
    lancius_arena_reset(sc);
    // T[0,0,:] = X[0,:,0] = [1,4]; T[1,2,:] = X[1,:,2] = [9,12]
    int ok = T && T->runtime_data && fabs(T->runtime_data[0] - 1.0) < 1e-12
        && fabs(T->runtime_data[1] - 4.0) < 1e-12
        && fabs(T->runtime_data[10] - 9.0) < 1e-12 && fabs(T->runtime_data[11] - 12.0) < 1e-12;
    CHECK(ok, "tbatch values");
    lancius_clear_error();
    char sp[512]; snprintf(sp, sizeof sp, "%s/r3_tbatch.lancius", lancius_scratch_dir());
    CHECK(lancius_graph_save(g, sp) == 0, "tbatch v2 save");
    lancius_graph* g2 = lancius_graph_load(sp);
    CHECK(g2 && g2->node_count == 2, "tbatch v2 roundtrip");
    if (g2) lancius_graph_destroy(g2);
    lancius_schedule_destroy(s);
    free(X->runtime_data); lancius_graph_destroy(g);
}
static void t_gelu(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 4);
    lancius_node* Y = lancius_input(g, 2, 4);
    double xd[8] = {-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.0, 0.25};
    double yd[8] = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* G = lancius_gelu(g, X);
    lancius_node* L = lancius_mse(g, G, Y);
    lancius_schedule* fs = lancius_ir_schedule(g);
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "gelu trains");
    if (tg) {
        gradcheck(g, fs, tg, X, 8, "gelu");
        lancius_training_graph_destroy(tg);
    }
    lancius_schedule_destroy(fs);
    free(X->runtime_data); free(Y->runtime_data); lancius_graph_destroy(g);
}
static void t_layernorm(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 3);
    lancius_node* Gm = lancius_input(g, 1, 3);
    lancius_node* B = lancius_input(g, 1, 3);
    lancius_node* Y = lancius_input(g, 2, 3);
    double xd[6] = {0.5, -1.0, 2.0, 0.1, 0.2, 0.3};
    double gd[3] = {1.0, 0.5, 2.0};
    double bd[3] = {0.1, -0.1, 0.0};
    double yd[6] = {0.0, 0.5, -0.5, 0.2, -0.2, 0.1};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    Gm->runtime_data = malloc(sizeof(gd)); memcpy(Gm->runtime_data, gd, sizeof(gd));
    B->runtime_data = malloc(sizeof(bd)); memcpy(B->runtime_data, bd, sizeof(bd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* N = lancius_layernorm(g, X, Gm, B);
    lancius_node* L = lancius_mse(g, N, Y);
    lancius_schedule* fs = lancius_ir_schedule(g);
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "layernorm trains (x+gamma+beta)");
    if (tg) {
        gradcheck(g, fs, tg, X, 6, "layernorm-x");
        gradcheck(g, fs, tg, Gm, 3, "layernorm-gamma");
        gradcheck(g, fs, tg, B, 3, "layernorm-beta");
        lancius_training_graph_destroy(tg);
    }
    lancius_schedule_destroy(fs);
    free(X->runtime_data); free(Gm->runtime_data); free(B->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);
}
static void t_rmsnorm(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 4);
    lancius_node* Gm = lancius_input(g, 1, 4);
    lancius_node* Y = lancius_input(g, 2, 4);
    double xd[8] = {0.5, -1.0, 2.0, 0.1, -0.3, 0.7, 1.2, -0.9};
    double gd[4] = {1.0, 0.5, 2.0, 1.5};
    double yd[8] = {0.0, 0.5, -0.5, 0.2, -0.2, 0.1, 0.4, -0.4};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    Gm->runtime_data = malloc(sizeof(gd)); memcpy(Gm->runtime_data, gd, sizeof(gd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* N = lancius_rmsnorm(g, X, Gm);
    lancius_node* L = lancius_mse(g, N, Y);
    lancius_schedule* fs = lancius_ir_schedule(g);
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "rmsnorm trains (x+gamma)");
    if (tg) {
        gradcheck(g, fs, tg, X, 8, "rmsnorm-x");
        gradcheck(g, fs, tg, Gm, 4, "rmsnorm-gamma");
        lancius_training_graph_destroy(tg);
    }
    lancius_schedule_destroy(fs);
    free(X->runtime_data); free(Gm->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);
}
static void t_matmul_batched(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input_3d(g, 2, 2, 3);
    lancius_node* B = lancius_input_3d(g, 2, 3, 2);
    lancius_node* Y = lancius_input_3d(g, 2, 2, 2);
    double ad[12], bd[12], yd[8];
    for (int i = 0; i < 12; i++) { ad[i] = 0.1 * (i - 5); bd[i] = 0.1 * (i - 3); }
    for (int i = 0; i < 8; i++) yd[i] = (i % 2) ? 0.5 : -0.5;
    A->runtime_data = malloc(sizeof(ad)); memcpy(A->runtime_data, ad, sizeof(ad));
    B->runtime_data = malloc(sizeof(bd)); memcpy(B->runtime_data, bd, sizeof(bd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* M = lancius_matmul_batched(g, A, B);
    lancius_node* L = lancius_mse(g, M, Y);
    lancius_schedule* fs = lancius_ir_schedule(g);
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "matmul_batched trains (was fail-loud)");
    if (tg) {
        gradcheck(g, fs, tg, A, 12, "bmm-A");
        gradcheck(g, fs, tg, B, 12, "bmm-B");
        lancius_training_graph_destroy(tg);
    }
    lancius_schedule_destroy(fs);
    free(A->runtime_data); free(B->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);
}
int main(void) {
    printf("TRAIN BACKWARD AUDIT (R3-2)\n");
    sc = lancius_arena_create(64 * 1024 * 1024);
    t_transpose_batched();
    t_gelu();
    t_layernorm();
    t_rmsnorm();
    t_matmul_batched();
    lancius_arena_destroy(sc);
    printf("TRAIN BACKWARD AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

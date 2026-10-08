/* R3-1 gate: SUM_AXIS_ND per-axis N-dim reduction + N-dim broadcast backward.
 *
 * Proves: exact values on 1..4-D, equivalence with SUM_AXIS0/1 on 2D,
 * builder rejects bad axis, v2 roundtrip, 3D partial-broadcast autodiff
 * succeeds with exact grads, and analytic grads match finite differences.
 * Exits 1 on any divergence (no false-green).
 */
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>

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
}
static int checks = 0;
static int failures = 0;
static lancius_arena* scratch = NULL;

#define CHECK(cond, msg) \
    do { \
        checks++; \
        if (!(cond)) { \
            printf("  FAIL: %s (err=%d)\n", msg, (int)lancius_get_error()); \
            failures++; \
        } \
    } while (0)

static int close_d(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

static void test_values_1d(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 1, 4);
    double xd[4] = {1.0, 2.0, 3.0, 4.0};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    lancius_node* S = lancius_sum_axis_nd(g, X, 1);
    CHECK(S && S->ndim == 2 && S->shape[0] == 1 && S->shape[1] == 1, "1D(sum over only axis) shape");
    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_schedule_execute(s, scratch);
    lancius_arena_reset(scratch);
    CHECK(S && S->runtime_data && close_d(S->runtime_data[0], 10.0, 1e-12), "1D axis1 value 10");
    lancius_schedule_destroy(s);
    free(X->runtime_data); lancius_graph_destroy(g);
}

static void test_values_4d(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 2, 1, 2, 2);
    double xd[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    lancius_node* S = lancius_sum_axis_nd(g, X, 0);
    CHECK(S && S->ndim == 4 && S->shape[0] == 1 && S->shape[3] == 2, "4D axis0 shape");
    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_schedule_execute(s, scratch);
    lancius_arena_reset(scratch);
    /* out[0,c,h,w] = in[0,c,h,w] + in[1,c,h,w] */
    CHECK(S && S->runtime_data && close_d(S->runtime_data[0], 6.0, 1e-12), "4D axis0 value [0]=1+5");
    CHECK(S && S->runtime_data && close_d(S->runtime_data[3], 12.0, 1e-12), "4D axis0 value [3]=4+8");
    lancius_schedule_destroy(s);
    free(X->runtime_data); lancius_graph_destroy(g);
}

static void test_equiv_2d(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 3);
    double xd[6] = {1, 2, 3, 4, 5, 6};
    X->runtime_data = malloc(sizeof(xd)); memcpy(X->runtime_data, xd, sizeof(xd));
    lancius_node* A = lancius_sum_axis0(g, X);
    lancius_node* B = lancius_sum_axis_nd(g, X, 0);
    lancius_node* C = lancius_sum_axis1(g, X);
    lancius_node* D = lancius_sum_axis_nd(g, X, 1);
    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_schedule_execute(s, scratch);
    lancius_arena_reset(scratch);
    int ok = A && B && A->runtime_data && B->runtime_data;
    for (int i = 0; ok && i < 3; i++) ok = close_d(A->runtime_data[i], B->runtime_data[i], 1e-12);
    CHECK(ok, "axis_nd(0) == axis0 on 2D");
    ok = C && D && C->runtime_data && D->runtime_data;
    for (int i = 0; ok && i < 2; i++) ok = close_d(C->runtime_data[i], D->runtime_data[i], 1e-12);
    CHECK(ok, "axis_nd(1) == axis1 on 2D");
    lancius_schedule_destroy(s);
    free(X->runtime_data); lancius_graph_destroy(g);
}

static void test_bad_axis(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 2);
    lancius_clear_error();
    lancius_node* S = lancius_sum_axis_nd(g, X, 2);
    CHECK(!S && lancius_get_error() != LANCIUS_ERROR_OK, "axis == ndim rejected");
    lancius_clear_error();
    lancius_node* N = NULL;
    S = lancius_sum_axis_nd(g, N, 0);
    CHECK(!S && lancius_get_error() != LANCIUS_ERROR_OK, "NULL input rejected");
    lancius_clear_error();
    lancius_graph_destroy(g);
}

static void test_broadcast_3d_exact(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input_3d(g, 2, 1, 4);
    lancius_node* B = lancius_input_3d(g, 2, 3, 4);
    lancius_node* Y = lancius_input_3d(g, 2, 3, 4);
    double ad[8], bd[24], yd[24];
    for (int i = 0; i < 8; i++) ad[i] = 0.5;
    for (int i = 0; i < 24; i++) { bd[i] = 0.25; yd[i] = 1.0; }
    A->runtime_data = malloc(sizeof(ad)); memcpy(A->runtime_data, ad, sizeof(ad));
    B->runtime_data = malloc(sizeof(bd)); memcpy(B->runtime_data, bd, sizeof(bd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* Z = lancius_add(g, A, B);
    lancius_node* L = lancius_mse(g, Z, Y);
    CHECK(Z && L, "3D broadcast graph builds");
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "3D partial broadcast trains (no fail-loud)");
    if (tg) {
        lancius_schedule* bs = lancius_ir_schedule(tg->graph);
        CHECK(bs != NULL, "3D backward schedules");
        if (bs) {
            lancius_schedule_execute(bs, scratch);
            lancius_arena_reset(scratch);
            lancius_node* gA = tg->grad_nodes[A->id];
            lancius_node* gB = tg->grad_nodes[B->id];
            /* dL/dA[i] = 2*(0.75-1)/24 summed over 3 = -0.0625; dL/dB = -0.020833.. */
            CHECK(gA && gA->runtime_data && close_d(gA->runtime_data[0], -0.0625, 1e-9), "3D broadcast dA exact");
            CHECK(gB && gB->runtime_data && close_d(gB->runtime_data[0], 2.0 * (0.75 - 1.0) / 24.0, 1e-9), "3D broadcast dB exact");
            lancius_schedule_destroy(bs);
        }
        lancius_training_graph_destroy(tg);
    }
    free(A->runtime_data); free(B->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);
}

static void test_finite_diff(void) {
    /* d/dA MSE(add(A,B),Y) with A[1,1,3] vs B[1,2,3]: numeric vs analytic. */
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input_3d(g, 1, 1, 3);
    lancius_node* B = lancius_input_3d(g, 1, 2, 3);
    lancius_node* Y = lancius_input_3d(g, 1, 2, 3);
    double ad[3] = {0.3, -0.7, 1.1};
    double bd[6] = {0.1, 0.2, 0.3, -0.4, 0.5, -0.6};
    double yd[6] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    A->runtime_data = malloc(sizeof(ad)); memcpy(A->runtime_data, ad, sizeof(ad));
    B->runtime_data = malloc(sizeof(bd)); memcpy(B->runtime_data, bd, sizeof(bd));
    Y->runtime_data = malloc(sizeof(yd)); memcpy(Y->runtime_data, yd, sizeof(yd));
    lancius_node* Z = lancius_add(g, A, B);
    lancius_node* L = lancius_mse(g, Z, Y);
    lancius_schedule* fs = lancius_ir_schedule(g);
    lancius_training_graph* tg = lancius_ir_autodiff(g, L);
    CHECK(tg != NULL, "finitediff graph trains");
    double analytic[3] = {0, 0, 0};
    if (tg) {
        lancius_schedule* bs = lancius_ir_schedule(tg->graph);
        lancius_schedule_execute(bs, scratch);
        lancius_node* gA = tg->grad_nodes[A->id];
        CHECK(gA && gA->runtime_data, "analytic grad exists");
        if (gA && gA->runtime_data) memcpy(analytic, gA->runtime_data, sizeof(analytic));
        lancius_arena_reset(scratch);
        lancius_schedule_destroy(bs);
        lancius_training_graph_destroy(tg);
    }
    /* Numeric: perturb A, re-execute forward. */
    const double eps = 1e-5;
    double maxrel = 0.0;
    for (int i = 0; i < 3; i++) {
        double lp, lm;
        A->runtime_data[i] += eps;
        for (uint32_t w = 0; w < fs->wave_count; w++)
            for (uint32_t k = 0; k < fs->waves[w].node_count; k++) {
                lancius_node* n = fs->waves[w].nodes[k];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }
        lancius_schedule_execute(fs, scratch);
        lp = L->runtime_data[0];
        lancius_arena_reset(scratch);
        A->runtime_data[i] -= 2 * eps;
        for (uint32_t w = 0; w < fs->wave_count; w++)
            for (uint32_t k = 0; k < fs->waves[w].node_count; k++) {
                lancius_node* n = fs->waves[w].nodes[k];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }
        lancius_schedule_execute(fs, scratch);
        lm = L->runtime_data[0];
        lancius_arena_reset(scratch);
        A->runtime_data[i] += eps;
        double fd = (lp - lm) / (2 * eps);
        double rel = fabs(analytic[i] - fd) / (fabs(analytic[i]) + fabs(fd) + 1e-8);
        if (rel > maxrel) maxrel = rel;
    }
    printf("  max finite-diff rel err: %e\n", maxrel);
    CHECK(maxrel < 1e-4, "analytic matches finite-diff");
    lancius_schedule_destroy(fs);
    free(A->runtime_data); free(B->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);
}

static void test_roundtrip(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_3d(g, 2, 3, 4);
    lancius_node* S = lancius_sum_axis_nd(g, X, 2);
    CHECK(S != NULL, "roundtrip graph builds");
    lancius_clear_error();
    char sp[512]; snprintf(sp, sizeof sp, "%s/r3_audit_nd.lancius", lancius_scratch_dir());
    CHECK(lancius_graph_save(g, sp) == 0, "v2 save SUM_AXIS_ND");
    lancius_graph* g2 = lancius_graph_load(sp);
    CHECK(g2 && g2->node_count == g->node_count, "v2 load roundtrip node count");
    if (g2) {
        lancius_node* s2 = NULL;
        for (uint32_t i = 0; i < g2->node_count; i++)
            if (g2->nodes[i]->op == LANCIUS_OP_SUM_AXIS_ND) s2 = g2->nodes[i];
        CHECK(s2 && s2->axes[0] == 2 && s2->shape[2] == 1, "axis persists as 2");
        lancius_graph_destroy(g2);
    }
    lancius_graph_destroy(g);
}

int main(void) {
    printf("SUM_AXIS_ND AUDIT (R3-1)\n");
    scratch = lancius_arena_create(64 * 1024 * 1024);
    if (!scratch) { printf("FATAL: OOM scratch\n"); return 1; }
    test_values_1d();
    test_values_4d();
    test_equiv_2d();
    test_bad_axis();
    test_broadcast_3d_exact();
    test_finite_diff();
    test_roundtrip();
    lancius_arena_destroy(scratch);
    printf("SUM_AXIS_ND AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

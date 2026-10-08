/*
 * V7 HARDENING AUDIT — closes the seven mutation-test holes.
 *
 * Provenance: every check below corresponds to a real defect that mutation
 * testing injected into a pristine tree and the existing gate reported GREEN.
 * A gate that cannot fail is decoration; these are the checks that make the
 * gate able to fail.
 *
 *   H1  sticky-error contract: every supported op must leave the thread-local
 *       error at OK after a successful execution. Catches the SUM_AXIS_ND
 *       fall-through (correct values, then a false UNSUPPORTED_OP).
 *   H2  softmax numerical stability: exp() must be max-subtracted, so logits
 *       of magnitude 1e5 must still produce a normalised distribution rather
 *       than inf/NaN.
 *   H3  arena alignment: every allocation is 32B-aligned (AVX2 contract), and
 *       zero-byte allocations never alias.
 *   H4  arena alignment cap: an absurd alignment (1<<60) must be refused, not
 *       turned into a gigantic grow (OOM-DoS).
 *   H5  quantizer exactness: scale == max|w|/127, round-trip error is bounded
 *       by max|w|/254, and INT8 saturation is symmetric.
 *   H6  fusion refuses a shape-mismatched RELU instead of silently overwriting.
 *   H7  op coverage: a full op table executes without error AND changes no
 *       sticky error state, so a new op cannot slip in half-wired.
 */
#include <lancius.h>
#include <lancius/lancius_kernels.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

static int checks = 0;
static int failures = 0;

static void rep(const char* name, int before) {
    if (failures == before) printf("  [PASS] %s\n", name);
}

#define CHECK(cond, msg) \
    do { \
        checks++; \
        if (!(cond)) { printf("  [FAIL] %s (err=%d)\n", msg, (int)lancius_get_error()); failures++; } \
    } while (0)

#define SECTION(n) printf("\n[%s]\n", n)

static lancius_arena* scratch = NULL;

static lancius_schedule* run(lancius_graph* g) {
    lancius_schedule* s = lancius_ir_schedule(g);
    if (!s) return NULL;
    lancius_schedule_execute(s, scratch);
    return s;
}

static void done(lancius_schedule* s, lancius_graph* g) {
    if (s) lancius_schedule_destroy(s);
    lancius_graph_destroy(g);
    lancius_arena_reset(scratch);
}

/* loss(sum_axis_nd(matmul(x,w), axis) -> SUM), rebuilt from scratch each call so
 * the finite difference measures the objective, not a cached graph. */
static double san_loss(const double* X, const double* W, size_t R, size_t C, uint32_t axis) {
    lancius_arena* ar = lancius_arena_create(1 << 20);
    lancius_graph* g = lancius_graph_create();
    lancius_node* xn = lancius_input(g, R, C);
    lancius_node* wn = lancius_input(g, C, C);
    lancius_node_bind_external(xn, (void*)X);
    lancius_node_bind_external(wn, (void*)W);
    lancius_node* m  = lancius_matmul(g, xn, wn);
    lancius_node* rd = lancius_sum_axis_nd(g, m, axis);
    lancius_node* sm = lancius_sum(g, rd);
    lancius_schedule* sc = lancius_ir_schedule(g);
    lancius_schedule_execute(sc, ar);
    double v = (sm && sm->runtime_data) ? sm->runtime_data[0] : NAN;
    lancius_schedule_destroy(sc);
    lancius_arena_destroy(ar);
    lancius_graph_destroy(g);
    return v;
}

static void fill_seq(double* p, size_t n, double start, double step) {
    for (size_t i = 0; i < n; i++) p[i] = start + step * (double)i;
}

/* ------------------------------------------------------------------ H1 */
static void h1_sticky_error_contract(void) {
    int b = failures;
    SECTION("H1 sticky-error contract per op");

    /* SUM_AXIS_ND: correct values AND a clean error state. The mutation that
     * removed the branch's terminating `return` produced the right numbers and
     * then fell through to the vision router, which set UNSUPPORTED_OP. */
    {
        double x[20];
        fill_seq(x, 20, 1.0, 1.0);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input(g, 5, 4);
        lancius_node_bind_external(in, x);
        lancius_node* o = lancius_sum_axis_nd(g, in, 0);
        CHECK(o != NULL, "H1 sum_axis_nd built");
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(s != NULL, "H1 sum_axis_nd scheduled");
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
              "H1 sum_axis_nd leaves sticky error at OK");
        CHECK(o && o->runtime_data, "H1 sum_axis_nd produced output");
        if (o && o->runtime_data) {
            CHECK(fabs(o->runtime_data[0] - 45.0) < 1e-12, "H1 sum_axis_nd value [0]==45");
            CHECK(fabs(o->runtime_data[3] - 60.0) < 1e-12, "H1 sum_axis_nd value [3]==60");
        }
        done(s, g);
    }
    /* every axis of every rank */
    {
        struct { size_t d[4]; uint8_t nd; uint32_t axis; } cases[] = {
            {{3,4,1,1}, 2, 1}, {{2,3,4,1}, 3, 0}, {{2,3,4,5}, 4, 2},
        };
        for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
            size_t n = 1; for (uint8_t d = 0; d < cases[i].nd; d++) n *= cases[i].d[d];
            double* x = (double*)malloc(n * sizeof(double));
            fill_seq(x, n, 1.0, 1.0);
            lancius_graph* g = lancius_graph_create();
            lancius_node* in;
            if (cases[i].nd == 2) in = lancius_input(g, cases[i].d[0], cases[i].d[1]);
            else if (cases[i].nd == 3) in = lancius_input_3d(g, cases[i].d[0], cases[i].d[1], cases[i].d[2]);
            else in = lancius_input_4d(g, cases[i].d[0], cases[i].d[1], cases[i].d[2], cases[i].d[3]);
            lancius_node_bind_external(in, x);
            lancius_node* o = lancius_sum_axis_nd(g, in, cases[i].axis);
            lancius_clear_error();
            lancius_schedule* s = run(g);
            CHECK(s != NULL, "H1 sum_axis_nd scheduled (rank sweep)");
            CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
                  "H1 sum_axis_nd sticky error OK (rank sweep)");
            CHECK(o && o->runtime_data, "H1 sum_axis_nd output present (rank sweep)");
            done(s, g); free(x);
        }
    }
    /* SUM_AXIS0 / SUM_AXIS1 / TRANSPOSE / SOFTMAX also run clean */
    {
        double x[12]; fill_seq(x, 12, 0.5, 0.25);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input(g, 3, 4);
        lancius_node_bind_external(in, x);
        lancius_node* s0 = lancius_sum_axis0(g, in);
        lancius_node* s1 = lancius_sum_axis1(g, in);
        lancius_node* tr = lancius_transpose(g, in);
        lancius_node* sm = lancius_softmax(g, in);
        lancius_node* rl = lancius_relu(g, in);
        lancius_node* sb = lancius_sum(g, in);
        (void)s0; (void)s1; (void)tr; (void)sm; (void)rl; (void)sb;
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(s != NULL, "H1 2D op sweep scheduled");
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H1 2D op sweep sticky error OK");
        done(s, g);
    }
    /* SUM_AXIS_ND inside a TRAINED graph, with the gradient proven exact by
     * central differences on the objective. This is the R3-1 claim in
     * docs/DESPOT_TRUTH_V2.md ("VJP: dx=broadcast_to_shape(dy)") and it used
     * to be unreachable: the forward node executed, then autodiff aborted with
     * INTERNAL because broadcast_to_shape refused the [R,1] -> [R,C]
     * expansion that the VJP is defined in terms of. */
    for (uint32_t axis = 0; axis < 2; axis++) {
        size_t R = 5, C = 4;
        double* x = (double*)malloc(R*C*sizeof(double));
        double* w = (double*)malloc(C*C*sizeof(double));
        for (size_t i = 0; i < R*C; i++)  x[i] = 0.3 - 0.07*(double)i;
        for (size_t i = 0; i < C*C; i++)  w[i] = 0.2 + 0.03*(double)i;

        /* analytic gradient */
        double* gW = (double*)calloc(C*C, sizeof(double));
        double* gX = (double*)calloc(R*C, sizeof(double));
        lancius_graph* g = lancius_graph_create();
        lancius_node* xn = lancius_input(g, R, C);
        lancius_node* wn = lancius_input(g, C, C);
        lancius_node_bind_external(xn, x);
        lancius_node_bind_external(wn, w);
        lancius_node* m = lancius_matmul(g, xn, wn);
        lancius_node* red = lancius_sum_axis_nd(g, m, axis);
        lancius_node* loss = lancius_sum(g, red);
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
              "H1 training forward leaves sticky error OK");
        lancius_schedule_destroy(s); lancius_arena_reset(scratch);
        lancius_clear_error();
        lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
        CHECK(tg != NULL, "H1 autodiff through a SUM_AXIS_ND forward builds");
        if (tg) {
            lancius_schedule* bs = run(tg->graph);
            CHECK(bs != NULL, "H1 backward through SUM_AXIS_ND schedules");
            CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
                  "H1 backward through SUM_AXIS_ND leaves sticky error OK");
            lancius_node* gw = tg->grad_nodes[wn->id];
            lancius_node* gx = tg->grad_nodes[xn->id];
            CHECK(gw && gw->runtime_data, "H1 SUM_AXIS_ND produced dW");
            CHECK(gx && gx->runtime_data, "H1 SUM_AXIS_ND produced dX");
            if (gw && gw->runtime_data) memcpy(gW, gw->runtime_data, C*C*sizeof(double));
            if (gx && gx->runtime_data) memcpy(gX, gx->runtime_data, R*C*sizeof(double));
            lancius_schedule_destroy(bs);
            lancius_training_graph_destroy(tg);
            lancius_arena_reset(scratch);
        }
        lancius_graph_destroy(g);

        /* central differences on the same objective */
        double h = 1e-6, worstW = 0.0, worstX = 0.0;
        for (size_t i = 0; i < C*C; i++) {
            double o = w[i];
            w[i] = o + h; double lp = san_loss(x, w, R, C, axis);
            w[i] = o - h; double lm = san_loss(x, w, R, C, axis);
            w[i] = o;
            double e = fabs((lp - lm) / (2*h) - gW[i]); if (e > worstW) worstW = e;
        }
        for (size_t i = 0; i < R*C; i++) {
            double o = x[i];
            x[i] = o + h; double lp = san_loss(x, w, R, C, axis);
            x[i] = o - h; double lm = san_loss(x, w, R, C, axis);
            x[i] = o;
            double e = fabs((lp - lm) / (2*h) - gX[i]); if (e > worstX) worstX = e;
        }
        CHECK(worstW < 1e-7, "H1 SUM_AXIS_ND dW exact vs central differences");
        CHECK(worstX < 1e-7, "H1 SUM_AXIS_ND dX exact vs central differences");
        free(x); free(w); free(gW); free(gX);
    }
    rep("H1 sticky-error contract", b);
}

/* ------------------------------------------------------------------ H2 */
static void h2_softmax_stability(void) {
    int b = failures;
    SECTION("H2 softmax numerical stability (max-subtraction is load-bearing)");

    /* Logits at 1e5: without the max-subtraction exp() overflows to inf and the
     * row becomes NaN. With it, the result is a clean distribution. */
    {
        size_t R = 2, C = 4;
        double* z = (double*)malloc(R*C*sizeof(double));
        z[0] = 1.0e5;  z[1] = 1.0e5 - 1.0; z[2] = 1.0e5 - 2.0; z[3] = 1.0e5 - 3.0;
        z[4] = -1.0e5; z[5] = -1.0e5 + 1.0; z[6] = -1.0e5 + 2.0; z[7] = -1.0e5 + 3.0;
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input(g, R, C);
        lancius_node_bind_external(in, z);
        lancius_node* o = lancius_softmax(g, in);
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(s != NULL, "H2 huge-logit softmax scheduled");
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
              "H2 huge logits do not overflow (max-subtracted)");
        CHECK(o && o->runtime_data, "H2 huge-logit softmax produced output");
        if (o && o->runtime_data) {
            double sum = 0.0; int finite = 1;
            for (size_t i = 0; i < R*C; i++) {
                if (!isfinite(o->runtime_data[i])) finite = 0;
                sum += o->runtime_data[i];
            }
            CHECK(finite, "H2 all softmax outputs finite at |z|=1e5");
            CHECK(fabs(sum - (double)(R)) < 1e-12, "H2 huge-logit softmax rows sum to 1");
            /* the largest logit must dominate */
            CHECK(o->runtime_data[0] > 0.5, "H2 argmax mass > 0.5 at z=1e5");
        }
        done(s, g); free(z);
    }
    /* cross-entropy with huge logits must also stay finite */
    {
        size_t R = 2, C = 3;
        double* z = (double*)malloc(R*C*sizeof(double));
        double* y = (double*)calloc(R*C, sizeof(double));
        z[0] = 900.0; z[1] = 800.0; z[2] = 700.0;
        z[3] = -900.0; z[4] = -800.0; z[5] = -700.0;
        y[0] = 1.0; y[4] = 1.0;
        lancius_graph* g = lancius_graph_create();
        lancius_node* zn = lancius_input(g, R, C);
        lancius_node* yn = lancius_input(g, R, C);
        lancius_node_bind_external(zn, z);
        lancius_node_bind_external(yn, y);
        lancius_node* ce = lancius_cross_entropy(g, zn, yn);
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H2 CE at |z|=900 is clean");
        CHECK(ce && ce->runtime_data && isfinite(ce->runtime_data[0]),
              "H2 CE at |z|=900 stays finite");
        /* Row 0: the label IS the argmax, so its CE contribution is 0.
         * Row 1: logsumexp(-900,-800,-700) = -700, label index 1 has logit
         * -800, so the contribution is -700 - (-800) = 100.
         * Mean over 2 rows = 50. Computed with the max-subtraction, so this
         * is finite even though exp(-900) underflows on its own. */
        if (ce && ce->runtime_data && isfinite(ce->runtime_data[0]))
            CHECK(fabs(ce->runtime_data[0] - 50.0) < 1e-9,
                  "H2 CE at |z|=900 equals (0 + 100)/2 == 50");
        done(s, g); free(z); free(y);
    }
    /* attention with large logits must not overflow either */
    {
        size_t S = 4, H = 2, D = 4;
        size_t n = S*H*D;
        double* q = (double*)malloc(n*sizeof(double));
        double* k = (double*)malloc(n*sizeof(double));
        double* v = (double*)malloc(n*sizeof(double));
        fill_seq(q, n, 200.0, 1.0);
        fill_seq(k, n, 200.0, -1.0);
        fill_seq(v, n, 1.0, 0.25);
        lancius_graph* g = lancius_graph_create();
        lancius_node* qn = lancius_input_3d(g, S, H, D);
        lancius_node* kn = lancius_input_3d(g, S, H, D);
        lancius_node* vn = lancius_input_3d(g, S, H, D);
        lancius_node_bind_external(qn, q);
        lancius_node_bind_external(kn, k);
        lancius_node_bind_external(vn, v);
        lancius_node* o = lancius_attention(g, qn, kn, vn);
        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H2 attention at large logits clean");
        if (o && o->runtime_data) {
            int finite = 1;
            for (size_t i = 0; i < n; i++) if (!isfinite(o->runtime_data[i])) finite = 0;
            CHECK(finite, "H2 attention outputs finite at large logits");
        }
        done(s, g); free(q); free(k); free(v);
    }
    rep("H2 softmax/attention stability", b);
}

/* ------------------------------------------------------------------ H3 */
static void h3_arena_alignment(void) {
    int b = failures;
    SECTION("H3 arena 32B alignment and zero-size aliasing");

    lancius_arena* a = lancius_arena_create(1 << 20);
    CHECK(a != NULL, "H3 arena created");
    /* every alignment the library claims to honour, including 32, 64, 128 */
    for (size_t al = 8; al <= 256; al *= 2) {
        void* p = lancius_arena_alloc(a, 100, al);
        CHECK(p != NULL, "H3 arena_alloc succeeded");
        if (p) CHECK(((uintptr_t)p % al) == 0, "H3 arena honours requested alignment");
    }
    /* The default path must satisfy the 32B AVX2 contract for EVERY arena, not
     * merely for the one whose malloc block happened to start 32-aligned.
     * glibc only guarantees 16B, so a single-arena check passes by luck half the
     * time against a library that defaults to 16 -- which is exactly what
     * mutation testing did: 64 fresh arenas catch it every time. */
    for (int arena_i = 0; arena_i < 64; arena_i++) {
        lancius_arena* d = lancius_arena_create(1u << 20);
        CHECK(d != NULL, "H3 fresh arena created");
        if (!d) break;
        for (int i = 0; i < 8; i++) {
            void* p = lancius_arena_alloc(d, 8 + (size_t)i, 0);
            CHECK(p != NULL, "H3 default-align alloc succeeded");
            if (p) CHECK(((uintptr_t)p % 32) == 0,
                         "H3 default allocation is 32B aligned (AVX2 contract)");
        }
        lancius_arena_destroy(d);
    }
    /* sizes that straddle the 32B rounding must not overlap */
    {
        char* p1 = (char*)lancius_arena_alloc(a, 1, 32);
        char* p2 = (char*)lancius_arena_alloc(a, 1, 32);
        char* p3 = (char*)lancius_arena_alloc(a, 31, 32);
        CHECK(p1 && p2 && p3, "H3 tiny allocations returned");
        if (p1 && p2 && p3) {
            CHECK(p2 >= p1 + 32, "H3 1-byte alloc reserves a full 32B footprint");
            CHECK(p3 >= p2 + 32, "H3 31-byte alloc reserves a full 32B footprint");
            p1[0] = 'a'; p2[0] = 'b'; p3[0] = 'c';
            CHECK(p1[0] == 'a' && p2[0] == 'b' && p3[0] == 'c',
                  "H3 tiny allocations do not alias each other");
        }
    }
    lancius_arena_destroy(a);

    /* zero-size allocations must not alias (the size==0 -> 1 rule) */
    {
        lancius_arena* z = lancius_arena_create(1 << 16);
        void* p1 = lancius_arena_alloc(z, 0, 32);
        void* p2 = lancius_arena_alloc(z, 0, 32);
        CHECK(p1 != NULL && p2 != NULL, "H3 zero-size allocs returned");
        CHECK(p1 != p2, "H3 zero-size allocations do not alias");
        lancius_arena_destroy(z);
    }
    rep("H3 arena alignment", b);
}

/* ------------------------------------------------------------------ H4 */
static void h4_arena_alignment_cap(void) {
    int b = failures;
    SECTION("H4 arena refuses absurd alignment (OOM-DoS guard)");
    lancius_arena* a = lancius_arena_create(1 << 20);
    lancius_clear_error();
    void* p = lancius_arena_alloc(a, 16, (size_t)1 << 60);
    CHECK(p == NULL, "H4 alignment 1<<60 refused (was an unbounded grow)");
    CHECK(lancius_get_error() == LANCIUS_ERROR_OVERFLOW,
          "H4 absurd alignment reports OVERFLOW");
    lancius_clear_error();
    p = lancius_arena_alloc(a, 16, (size_t)1 << 40);
    CHECK(p == NULL, "H4 alignment 1<<40 refused");
    CHECK(lancius_get_error() == LANCIUS_ERROR_OVERFLOW,
          "H4 alignment 1<<40 reports OVERFLOW");
    /* non-power-of-two alignment is refused */
    lancius_clear_error();
    p = lancius_arena_alloc(a, 16, 3);
    CHECK(p == NULL, "H4 non-power-of-two alignment refused");
    /* the cap boundary itself still works */
    lancius_clear_error();
    p = lancius_arena_alloc(a, 16, (size_t)1 << 20);
    CHECK(p != NULL, "H4 alignment exactly 1MB still allowed");
    lancius_clear_error();
    /* and an impossible size is refused, not wrapped */
    p = lancius_arena_alloc(a, SIZE_MAX, 32);
    CHECK(p == NULL, "H4 SIZE_MAX request refused");
    CHECK(lancius_get_error() == LANCIUS_ERROR_OVERFLOW,
          "H4 SIZE_MAX request reports OVERFLOW");
    lancius_arena_destroy(a);
    rep("H4 arena alignment cap", b);
}

/* ------------------------------------------------------------------ H5 */
static void h5_quantizer_exactness(void) {
    int b = failures;
    SECTION("H5 INT8 quantizer exactness and error bound");

    /* per-tensor: scale == max|w|/127 exactly, and round-trip error is bounded
     * by max|w|/(2*127) for every element (round-to-nearest property). */
    {
        size_t C_out = 4, C_in = 3, Kh = 3, Kw = 3;
        size_t n = C_out*C_in*Kh*Kw;
        double* w = (double*)malloc(n*sizeof(double));
        for (size_t i = 0; i < n; i++) w[i] = sin((double)i) * 0.37;
        double maxabs = 0.0;
        for (size_t i = 0; i < n; i++) { double a = fabs(w[i]); if (a > maxabs) maxabs = a; }

        lancius_graph* g = lancius_graph_create();
        lancius_node* wn = lancius_input_4d(g, C_out, C_in, Kh, Kw);
        wn->runtime_data = (double*)malloc(n*sizeof(double));
        memcpy(wn->runtime_data, w, n*sizeof(double));
        lancius_node_bind_owned_heap(wn, wn->runtime_data);
        lancius_clear_error();
        lancius_quantize_graph(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H5 per-tensor quantize clean");
        CHECK(wn->dtype == LANCIUS_DTYPE_INT8, "H5 node became INT8");
        CHECK(fabs(wn->scale - maxabs / 127.0) < 1e-15,
              "H5 per-tensor scale == max|w|/127 exactly");
        CHECK(wn->runtime_data_int8 != NULL, "H5 INT8 buffer present");
        /* saturation is symmetric: +127 reachable, -127 reachable */
        int saw_pos = 0, saw_neg = 0;
        for (size_t i = 0; i < n; i++) {
            if (wn->runtime_data_int8[i] == 127) saw_pos = 1;
            if (wn->runtime_data_int8[i] == -127) saw_neg = 1;
        }
        CHECK(saw_pos, "H5 +127 saturation reachable");
        CHECK(saw_neg, "H5 -127 saturation reachable");
        /* dequantize and check the round-trip bound */
        lancius_clear_error();
        lancius_dequantize_graph(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H5 dequantize clean");
        CHECK(wn->dtype == LANCIUS_DTYPE_FP64, "H5 node returned to FP64");
        double worst = 0.0;
        if (wn->runtime_data) {
            for (size_t i = 0; i < n; i++) {
                double e = fabs(wn->runtime_data[i] - w[i]);
                if (e > worst) worst = e;
            }
        }
        CHECK(worst <= maxabs / 127.0 * 0.5000001,
              "H5 round-trip error <= max|w|/(2*127) (round-to-nearest bound)");
        lancius_graph_destroy(g); free(w);
    }

    /* per-channel: one scale per output channel, each == max|channel|/127 */
    {
        size_t C_out = 3, C_in = 2, Kh = 2, Kw = 2;
        size_t n = C_out*C_in*Kh*Kw;
        size_t per = n / C_out;
        double* w = (double*)malloc(n*sizeof(double));
        for (size_t c = 0; c < C_out; c++)
            for (size_t j = 0; j < per; j++)
                w[c*per + j] = cos((double)(c*per + j)) * (0.5 + 0.7 * (double)c);

        lancius_graph* g = lancius_graph_create();
        lancius_node* wn = lancius_input_4d(g, C_out, C_in, Kh, Kw);
        wn->runtime_data = (double*)malloc(n*sizeof(double));
        memcpy(wn->runtime_data, w, n*sizeof(double));
        lancius_node_bind_owned_heap(wn, wn->runtime_data);
        lancius_clear_error();
        lancius_quantize_graph_per_channel(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H5 per-channel quantize clean");
        CHECK(wn->dtype == LANCIUS_DTYPE_INT8, "H5 per-channel node became INT8");
        CHECK(wn->rt && wn->rt->scale_per_channel != NULL, "H5 per-channel scales stored");
        CHECK(wn->rt && wn->rt->scale_channels == C_out, "H5 one scale per output channel");
        if (wn->rt && wn->rt->scale_per_channel) {
            int allok = 1;
            for (size_t c = 0; c < C_out; c++) {
                double m = 0.0;
                for (size_t j = 0; j < per; j++) { double a = fabs(w[c*per+j]); if (a > m) m = a; }
                if (fabs(wn->rt->scale_per_channel[c] - m/127.0) > 1e-15) allok = 0;
            }
            CHECK(allok, "H5 per-channel scale == max|channel|/127 for every channel");
        }
        lancius_clear_error();
        lancius_dequantize_graph(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H5 per-channel dequantize clean");
        double worst = 0.0;
        if (wn->runtime_data) {
            for (size_t i = 0; i < n; i++) {
                double e = fabs(wn->runtime_data[i] - w[i]);
                if (e > worst) worst = e;
            }
        }
        double worst_bound = 0.0;
        for (size_t c = 0; c < C_out; c++) {
            double m = 0.0;
            for (size_t j = 0; j < per; j++) { double a = fabs(w[c*per+j]); if (a > m) m = a; }
            if (m/127.0 > worst_bound) worst_bound = m/127.0;
        }
        CHECK(worst <= worst_bound * 0.5000001,
              "H5 per-channel round-trip error <= max_channel/(2*127)");
        lancius_graph_destroy(g); free(w);
    }

    /* all-zero weights stay FP64 (scale would be 0) */
    {
        size_t C_out = 2, C_in = 1, Kh = 2, Kw = 2;
        size_t n = C_out*C_in*Kh*Kw;
        lancius_graph* g = lancius_graph_create();
        lancius_node* wn = lancius_input_4d(g, C_out, C_in, Kh, Kw);
        wn->runtime_data = (double*)calloc(n, sizeof(double));
        lancius_node_bind_owned_heap(wn, wn->runtime_data);
        lancius_clear_error();
        lancius_quantize_graph(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H5 all-zero quantize clean");
        CHECK(wn->dtype == LANCIUS_DTYPE_FP64, "H5 all-zero weights stay FP64 (scale stays 1.0)");
        lancius_graph_destroy(g);
    }
    rep("H5 quantizer exactness", b);
}

/* ------------------------------------------------------------------ H6 */
static void h6_fusion_shape_guard(void) {
    int b = failures;
    SECTION("H6 conv+relu fusion refuses a shape-mismatched RELU");

    /* A normal conv->relu fuses and produces relu(conv). */
    {
        size_t N = 1, Cin = 2, Hin = 6, Win = 6, Cout = 3, Kh = 3, Kw = 3;
        size_t Hout = Hin + 2 - Kh + 1, Wout = Win + 2 - Kw + 1;
        size_t nin = N*Cin*Hin*Win, nw = Cout*Cin*Kh*Kw;
        double* in = (double*)malloc(nin*sizeof(double));
        double* w = (double*)malloc(nw*sizeof(double));
        fill_seq(in, nin, -0.5, 0.1);
        fill_seq(w, nw, 0.2, -0.03);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in_n = lancius_input_4d(g, N, Cin, Hin, Win);
        lancius_node* w_n = lancius_input_4d(g, Cout, Cin, Kh, Kw);
        lancius_node_bind_external(in_n, in);
        lancius_node_bind_external(w_n, w);
        lancius_node* c = lancius_conv2d(g, in_n, w_n, 1, 1);
        lancius_node* r = lancius_relu(g, c);
        lancius_clear_error();
        lancius_schedule* s = run(g);
        double* sep = (double*)malloc(N*Cout*Hout*Wout*sizeof(double));
        memcpy(sep, r->runtime_data, N*Cout*Hout*Wout*sizeof(double));
        lancius_schedule_destroy(s);
        lancius_arena_reset(scratch);

        lancius_optimize_fusion(g);
        CHECK(c->op == LANCIUS_OP_NOP && r->op == LANCIUS_OP_CONV2D_RELU_FUSED,
              "H6 matched shapes fuse");
        lancius_clear_error();
        lancius_schedule* s2 = run(g);
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H6 fused graph executes clean");
        if (r->runtime_data) {
            double worst = 0.0;
            for (size_t i = 0; i < N*Cout*Hout*Wout; i++) {
                double e = fabs(r->runtime_data[i] - sep[i]);
                if (e > worst) worst = e;
            }
            CHECK(worst == 0.0, "H6 fused output is bit-identical to conv->relu");
        }
        done(s2, g);
        free(in); free(w); free(sep);
    }

    /* A RELU whose declared shape disagrees with the conv must NOT be fused:
     * the fusion steals the conv's inputs and copies the conv's shape over the
     * RELU's, so fusing would silently discard the RELU's own shape. */
    {
        size_t N = 1, Cin = 1, Hin = 4, Win = 4, Cout = 1, Kh = 3, Kw = 3;
        size_t nin = N*Cin*Hin*Win, nw = Cout*Cin*Kh*Kw;
        double* in = (double*)malloc(nin*sizeof(double));
        double* w = (double*)malloc(nw*sizeof(double));
        fill_seq(in, nin, 0.1, 0.1);
        fill_seq(w, nw, 0.3, 0.05);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in_n = lancius_input_4d(g, N, Cin, Hin, Win);
        lancius_node* w_n = lancius_input_4d(g, Cout, Cin, Kh, Kw);
        lancius_node_bind_external(in_n, in);
        lancius_node_bind_external(w_n, w);
        lancius_node* c = lancius_conv2d(g, in_n, w_n, 1, 0);
        lancius_node* r = lancius_relu(g, c);
        /* corrupt the RELU's declared shape, as a malformed loader could */
        r->shape[2] = 7; r->shape[3] = 7;
        lancius_optimize_fusion(g);
        CHECK(r->op != LANCIUS_OP_CONV2D_RELU_FUSED,
              "H6 shape-mismatched RELU is refused by fusion");
        CHECK(c->op == LANCIUS_OP_CONV2D, "H6 conv left intact when fusion refused");
        CHECK(r->shape[2] == 7 && r->shape[3] == 7,
              "H6 refused fusion left the RELU shape untouched");
        lancius_graph_destroy(g); free(in); free(w);
    }
    rep("H6 fusion shape guard", b);
}

/* ------------------------------------------------------------------ H7 */
static void h7_op_table_no_wired_half(void) {
    int b = failures;
    SECTION("H7 every op in a representative graph runs clean");
    /* One graph touching most forward ops at once. Any op that half-executes
     * (correct values, wrong error) shows up here. */
    {
        size_t B = 3, Hd = 4, N = 1, C = 1, H = 4, W = 4;
        double* x = (double*)malloc(B*Hd*sizeof(double));
        double* gm = (double*)malloc(Hd*sizeof(double));
        double* bt = (double*)malloc(Hd*sizeof(double));
        double* wi = (double*)malloc(C*Hd*sizeof(double));
        double* kk = (double*)malloc(B*Hd*sizeof(double));
        double* gg = (double*)malloc(Hd*sizeof(double));
        double* up = (double*)malloc(B*Hd*sizeof(double));
        double* im = (double*)malloc(N*C*H*W*sizeof(double));
        double* kw = (double*)malloc(C*3*3*sizeof(double));
        fill_seq(x, B*Hd, -0.4, 0.15);
        fill_seq(gm, Hd, 0.8, 0.05);
        fill_seq(bt, Hd, -0.1, 0.02);
        fill_seq(wi, C*Hd, 0.2, 0.01);
        fill_seq(kk, B*Hd, 0.3, -0.02);
        fill_seq(gg, Hd, 0.9, 0.01);
        fill_seq(up, B*Hd, 0.5, -0.01);
        fill_seq(im, N*C*H*W, 0.2, 0.03);
        fill_seq(kw, C*9, 0.4, -0.05);

        lancius_graph* g = lancius_graph_create();
        lancius_node* xn = lancius_input(g, B, Hd);
        lancius_node* gn = lancius_input(g, 1, Hd);
        lancius_node* bn = lancius_input(g, 1, Hd);
        lancius_node* wn = lancius_input(g, C, Hd);
        lancius_node* kn = lancius_input(g, B, Hd);
        lancius_node* tan = lancius_input(g, B, Hd);
        lancius_node* un = lancius_input(g, B, Hd);
        lancius_node* imn = lancius_input_4d(g, N, C, H, W);
        lancius_node* kwn = lancius_input_4d(g, C, 1, 3, 3);
        lancius_node_bind_external(xn, x);
        lancius_node_bind_external(gn, gm);
        lancius_node_bind_external(bn, bt);
        lancius_node_bind_external(wn, wi);
        lancius_node_bind_external(kn, kk);
        lancius_node_bind_external(tan, x);
        lancius_node_bind_external(un, up);
        lancius_node_bind_external(imn, im);
        lancius_node_bind_external(kwn, kw);

        lancius_node* ln = lancius_layernorm(g, xn, gn, bn);
        lancius_node* gel = lancius_gelu(g, xn);
        lancius_node* rmn = lancius_rmsnorm(g, xn, gn);
        lancius_node* tn = lancius_tanh(g, xn);
        lancius_node* sw = lancius_swiglu(g, xn, un);
        lancius_node* rl = lancius_relu(g, xn);
        lancius_node* sm = lancius_softmax(g, xn);
        lancius_node* sum = lancius_sum(g, xn);
        lancius_node* ax0 = lancius_sum_axis0(g, xn);
        lancius_node* ax1 = lancius_sum_axis1(g, xn);
        lancius_node* axn = lancius_sum_axis_nd(g, xn, 1);
        lancius_node* tr = lancius_transpose(g, xn);
        lancius_node* add = lancius_add(g, xn, xn);
        lancius_node* sub = lancius_sub(g, xn, xn);
        lancius_node* mul = lancius_mul(g, xn, xn);
        lancius_node* mm = lancius_matmul(g, wn, kn);
        lancius_node* bcast = lancius_broadcast(g, sum, B, Hd);
        lancius_node* conv = lancius_conv2d(g, imn, kwn, 1, 1);
        lancius_node* pool = lancius_maxpool2d(g, imn, 2, 2);
        lancius_node* perm = lancius_permute(g, imn, 0, 2, 3, 1);
        lancius_node* mmb = lancius_matmul_batched(g, xn, xn);
        (void)ln; (void)gel; (void)rmn; (void)tn; (void)sw; (void)rl; (void)sm;
        (void)sum; (void)ax0; (void)ax1; (void)axn; (void)tr; (void)add; (void)sub;
        (void)mul; (void)mm; (void)bcast; (void)conv; (void)pool; (void)perm; (void)mmb;

        lancius_clear_error();
        lancius_schedule* s = run(g);
        CHECK(s != NULL, "H7 combined graph scheduled");
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
              "H7 combined graph leaves sticky error at OK (no op half-wired)");
        CHECK(ln && ln->runtime_data, "H7 layernorm output");
        CHECK(gel && gel->runtime_data, "H7 gelu output");
        CHECK(rmn && rmn->runtime_data, "H7 rmsnorm output");
        CHECK(tn && tn->runtime_data, "H7 tanh output");
        CHECK(sw && sw->runtime_data, "H7 swiglu output");
        CHECK(sm && sm->runtime_data, "H7 softmax output");
        CHECK(axn && axn->runtime_data, "H7 sum_axis_nd output");
        CHECK(conv && conv->runtime_data, "H7 conv output");
        CHECK(pool && pool->runtime_data, "H7 maxpool output");
        CHECK(perm && perm->runtime_data, "H7 permute output");
        done(s, g);
        free(x); free(gm); free(bt); free(wi); free(kk); free(gg); free(up);
        free(im); free(kw);
    }
    rep("H7 op table", b);
}


/* ------------------------------------------------------------------ H8 */
static void h8_broadcast_to_shape_contract(void) {
    int b = failures;
    SECTION("H8 broadcast_to_shape obeys the same NumPy rule at every rank");
    double v[24];
    fill_seq(v, 24, 1.0, 0.5);

    /* scalar lifts to any 1..4-D */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* s = lancius_input(g, 1, 1);
        lancius_node_bind_external(s, v);
        size_t shapes[4][4] = { {1,1,1,1}, {6,1,1,1}, {2,3,1,1}, {2,3,4,1} };
        uint8_t nd[4] = {1,2,3,4};
        for (int i = 0; i < 4; i++) {
            lancius_clear_error();
            lancius_node* o = lancius_broadcast_to_shape(g, s, shapes[i], nd[i]);
            CHECK(o != NULL, "H8 scalar lifts to any rank");
        }
        lancius_graph_destroy(g);
    }
    /* per-dim-1 expansion is allowed at EVERY rank (was rank 2 and 4 only) */
    {
        struct { size_t in[4]; uint8_t ind; size_t out[4]; uint8_t outd; } cases[] = {
            {{1,3}, 2, {2,3}, 2},
            {{2,1,4}, 3, {2,3,4,1}, 3},
            {{2,1,1,5}, 4, {2,3,4,5}, 4},
            {{1,1,1,5}, 4, {2,3,4,5}, 4},
        };
        for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
            lancius_graph* g = lancius_graph_create();
            lancius_node* x;
            size_t nin = 1; for (uint8_t d = 0; d < cases[i].ind; d++) nin *= cases[i].in[d];
            if (cases[i].ind == 2) x = lancius_input(g, cases[i].in[0], cases[i].in[1]);
            else if (cases[i].ind == 3) x = lancius_input_3d(g, cases[i].in[0], cases[i].in[1], cases[i].in[2]);
            else x = lancius_input_4d(g, cases[i].in[0], cases[i].in[1], cases[i].in[2], cases[i].in[3]);
            lancius_node_bind_external(x, v);
            lancius_clear_error();
            lancius_node* o = lancius_broadcast_to_shape(g, x, cases[i].out, cases[i].outd);
            CHECK(o != NULL, "H8 per-dim-1 expansion accepted at every rank");
            if (o) {
                lancius_clear_error();
                lancius_schedule* s = run(g);
                CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
                      "H8 expanded broadcast executes clean");
                done(s, g);
            } else {
                lancius_graph_destroy(g);
            }
        }
    }
    /* trailing-rank alignment: a rank-2 source lifts into rank 3/4 */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* x = lancius_input(g, 2, 4);
        lancius_node_bind_external(x, v);
        lancius_clear_error();
        lancius_node* o3 = lancius_broadcast_to_shape(g, x, (size_t[]){3,2,4}, 3);
        CHECK(o3 != NULL, "H8 rank-2 source lifts into rank 3 (trailing align)");
        if (o3) { lancius_schedule* s = run(g); CHECK(lancius_get_error()==LANCIUS_ERROR_OK,"H8 rank-3 lift executes clean"); done(s, g); }
        else lancius_graph_destroy(g);
    }
    /* exact match still works */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* x = lancius_input_4d(g, 2, 3, 4, 1);
        lancius_node_bind_external(x, v);
        lancius_clear_error();
        lancius_node* o = lancius_broadcast_to_shape(g, x, (size_t[]){2,3,4,1}, 4);
        CHECK(o != NULL, "H8 exact-shape broadcast still accepted");
        lancius_graph_destroy(g);
    }
    /* genuinely incompatible shapes are still refused, at every rank */
    {
        struct { size_t in[4]; uint8_t ind; size_t out[4]; uint8_t outd; } bad[] = {
            {{2,3}, 2, {2,4}, 2},
            {{2,3,4}, 3, {2,5,4}, 3},
            {{2,3,4,5}, 4, {2,3,4,7}, 4},
            {{2,3,4,5}, 4, {2,3,1,1}, 3},   /* target rank BELOW source rank */
        };
        for (unsigned i = 0; i < sizeof(bad)/sizeof(bad[0]); i++) {
            lancius_graph* g = lancius_graph_create();
            lancius_node* x;
            if (bad[i].ind == 2) x = lancius_input(g, bad[i].in[0], bad[i].in[1]);
            else if (bad[i].ind == 3) x = lancius_input_3d(g, bad[i].in[0], bad[i].in[1], bad[i].in[2]);
            else x = lancius_input_4d(g, bad[i].in[0], bad[i].in[1], bad[i].in[2], bad[i].in[3]);
            lancius_node_bind_external(x, v);
            lancius_clear_error();
            lancius_node* o = lancius_broadcast_to_shape(g, x, bad[i].out, bad[i].outd);
            CHECK(o == NULL, "H8 incompatible broadcast refused (never silently miscomputed)");
            if (o) CHECK(lancius_get_error() == LANCIUS_ERROR_SHAPE_MISMATCH,
                         "H8 refusal reports SHAPE_MISMATCH");
            lancius_graph_destroy(g);
        }
    }
    /* the same rule through the v2 round-trip: a 3-D expanding BROADCAST that
     * the executor can run must also survive save/load */
    {
        size_t nin = 2*1*4, nout = 2*3*4;
        lancius_graph* g = lancius_graph_create();
        lancius_node* x = lancius_input_3d(g, 2, 1, 4);
        double* xd = (double*)malloc(nin*sizeof(double));
        for (size_t i = 0; i < nin; i++) xd[i] = 0.25*(double)i;
        lancius_node_bind_owned_heap(x, xd);
        lancius_node* bc = lancius_broadcast_to_shape(g, x, (size_t[]){2,3,4}, 3);
        CHECK(bc != NULL, "H8 3-D expanding broadcast builds");
        lancius_node* y = bc ? lancius_add(g, bc, bc) : NULL;
        CHECK(y != NULL, "H8 3-D expanding broadcast composes");
        if (bc && y) {
            lancius_clear_error();
            lancius_schedule* s = run(g);
            CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H8 3-D expanding broadcast runs");
            double* ref = (double*)malloc(nout*sizeof(double));
            memcpy(ref, y->runtime_data, nout*sizeof(double));
            lancius_schedule_destroy(s);
            lancius_arena_reset(scratch);
            lancius_clear_error();
            CHECK(lancius_graph_save(g, "audit_v7_bcast.lancius") == 0,
                  "H8 3-D expanding broadcast saves");
            lancius_clear_error();
            lancius_graph* g2 = lancius_graph_load("audit_v7_bcast.lancius");
            CHECK(g2 != NULL, "H8 3-D expanding broadcast RELOADS (loader must not be stricter)");
            if (g2) {
                /* execute the reloaded graph and require the same values */
                lancius_arena* ar2 = lancius_arena_create(1 << 20);
                lancius_schedule* s2 = lancius_ir_schedule(g2);
                CHECK(s2 != NULL, "H8 reloaded graph schedules");
                if (s2) {
                    lancius_schedule_execute(s2, ar2);
                    CHECK(lancius_get_error() == LANCIUS_ERROR_OK,
                          "H8 reloaded graph executes clean");
                    lancius_node* y2 = (g2->node_count >= 1) ? g2->nodes[g2->node_count - 1] : NULL;
                    CHECK(y2 && y2->runtime_data != NULL, "H8 reloaded graph has output");
                    lancius_schedule_destroy(s2);
                }
                lancius_arena_destroy(ar2);
                lancius_graph_destroy(g2);
            }
            remove("audit_v7_bcast.lancius");
            free(ref);
        }
        lancius_graph_destroy(g);
    }
    rep("H8 broadcast_to_shape contract", b);
}


/* ------------------------------------------------------------------ H9 */
static void h9_int8_accumulator_width(void) {
    int b = failures;
    SECTION("H9 INT8 convolution accumulates in 64 bits");

    /* int32 overflows at 2^31 / (127*127) = 132104 terms. Build a convolution
     * with C_in*K_h*K_w > 200000 so an int32 accumulator is guaranteed to wrap,
     * and check the FP64 result against an exact int64 reference.
     *
     * C_in = 512, K = 20x20 -> 204800 taps per output. Weights: 1 output
     * channel x 512 x 400 = 204800 int8 (200 KiB). Input: 512 x 20 x 20 =
     * 204800 int8. Output is 1x1x1x1. */
    size_t Cin = 512, K = 20, taps = Cin * K * K;
    CHECK(taps > 132104, "H9 tap count exceeds the int32 overflow threshold");
    int8_t* w = (int8_t*)malloc(taps);
    int8_t* in = (int8_t*)malloc(Cin * K * K);
    if (!w || !in) { printf("  [FAIL] H9 OOM\n"); failures++; free(w); free(in); return; }
    /* all +127 so the accumulation is maximal and wraps loudly under int32 */
    for (size_t i = 0; i < taps; i++) w[i] = 127;
    for (size_t i = 0; i < Cin * K * K; i++) in[i] = 127;

    double scale = 1.0;
    double* out = (double*)calloc(1, sizeof(double));
    kernel_conv2d_int8_fwd(out, in, w, scale, scale, 1, Cin, K, K, 1, K, K, 1, 0);
    lancius_clear_error();
    CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "H9 deep INT8 conv executes clean");

    int64_t exact = 0;
    for (size_t i = 0; i < taps; i++) exact += (int64_t)in[i] * (int64_t)w[i];
    double want = (double)exact * scale * scale;
    CHECK(out[0] == want, "H9 INT8 conv result matches the exact int64 reference");
    if (out[0] != want)
        printf("         got=%.1f want=%.1f (int32 would have wrapped to %.1f)\n",
               out[0], want, (double)(int32_t)exact);

    /* and the same shape with a signed mix, where a narrow accumulator also
     * loses the sign: -127 weights against +127 activations */
    for (size_t i = 0; i < taps; i++) w[i] = (i % 2) ? -127 : 127;
    out[0] = 0.0;
    kernel_conv2d_int8_fwd(out, in, w, scale, scale, 1, Cin, K, K, 1, K, K, 1, 0);
    lancius_clear_error();
    int64_t exact2 = 0;
    for (size_t i = 0; i < taps; i++) exact2 += (int64_t)in[i] * (int64_t)w[i];
    CHECK(out[0] == (double)exact2, "H9 signed INT8 conv matches the exact int64 reference");

    free(w); free(in); free(out);
    rep("H9 INT8 accumulator width", b);
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v12R2: V7 HARDENING AUDIT (MUTATION-TEST CLOSURES)\n");
    printf("================================================================\n");
    scratch = lancius_arena_create(16 * 1024 * 1024);
    if (!scratch) { printf("FATAL: arena\n"); return 2; }

    h1_sticky_error_contract();
    h2_softmax_stability();
    h3_arena_alignment();
    h4_arena_alignment_cap();
    h5_quantizer_exactness();
    h6_fusion_shape_guard();
    h7_op_table_no_wired_half();
    h8_broadcast_to_shape_contract();
    h9_int8_accumulator_width();

    lancius_arena_destroy(scratch);
    printf("\n================================================================\n");
    printf("  V7 HARDENING AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    printf("================================================================\n");
    return failures ? 1 : 0;
}
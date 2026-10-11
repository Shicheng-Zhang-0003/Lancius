/* bench_kernels.c -- the throughput harness.
 *
 * WHY THIS EXISTS. The oracles in tools/audit prove correctness and never
 * measure speed, so every performance sentence in this repository was
 * unfalsifiable: KNOWN_LIMITATIONS recorded that nothing measured throughput,
 * and this file is the response. It is a measurement, not a claim: run it on
 * the machine you care about and read the numbers it prints. Nothing here is
 * asserted as a target, and nothing here is compared against another
 * framework, because a benchmark that beats a strawman is a lie in the
 * making. It prints the thread environment beside the numbers so a result
 * cannot be quoted without its conditions.
 *
 * METHOD. Each case warms up once (first-touch, page faults, libgomp thread
 * pool creation and any lazy CPUID all happen here, never inside a timed
 * run), then times `reps` calls and reports the MEDIAN, not the mean: a
 * median is not moved by the one descheduled run. FLOP counts are printed in
 * the table header formula so a reader can re-derive every rate.
 *
 *   matmul  M x K x N        2*M*K*N          (one mul + one add per MAC)
 *   conv    NCHW forward     2*N*Co*Ho*Wo*Cin*Kh*Kw
 *   element (gelu/norm)      --               reported as Melem/s, because
 *                                             counting a tanh as "N flops"
 *                                             is a fudge, and saying so is
 *                                             more honest than inventing N
 *   attention (per head)     4*S*S*D          scores + out projection; the
 *                                             online softmax pass is counted
 *                                             as S*S*(D+1) and is folded in
 *
 * THREADS. kernel_matmul and friends carry `#pragma omp parallel for`, so the
 * rate scales with OMP_NUM_THREADS. The runtime's pthread pool pins each
 * worker's OpenMP team to 1 (see src/runtime/lancius_threadpool.c), so a
 * pooled scheduler run is single-threaded per worker by design. Both paths
 * are measured below so the trade is visible rather than assumed.
 *
 * Build:  make bench_kernels      Run:  ./bench_kernels
 * Not a gate member: timing is machine-dependent and must never be able to
 * turn the suite red. Its absence from `make check` is deliberate.
 */
#include <lancius.h>
#include <lancius/lancius_kernels.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Median of `reps` timed calls of fn(ctx). Warms up once outside the clock. */
static double median_seconds(void (*fn)(void*), void* ctx, int reps) {
    fn(ctx); /* warm-up: pages, libgomp team, caches */
    double* t = (double*)malloc((size_t)reps * sizeof(double));
    if (!t) { fprintf(stderr, "bench: oom\n"); exit(2); }
    for (int i = 0; i < reps; i++) {
        double s = now_s();
        fn(ctx);
        t[i] = now_s() - s;
    }
    /* insertion sort: reps is small and this keeps the binary free of qsort's
     * function-pointer indirection in the hot-ish path */
    for (int i = 1; i < reps; i++) {
        double k = t[i];
        int j = i - 1;
        while (j >= 0 && t[j] > k) { t[j + 1] = t[j]; j--; }
        t[j + 1] = k;
    }
    double m = t[reps / 2];
    free(t);
    return m;
}

/* --------------------------------------------------------------- matmul */
typedef struct { double *o, *a, *b; size_t M, K, N; } mm_ctx;
static void run_mm64(void* p) {
    mm_ctx* c = (mm_ctx*)p;
    kernel_matmul(c->o, c->a, c->b, c->M, c->K, c->N);
}
static void run_mm32(void* p) {
    mm_ctx* c = (mm_ctx*)p;
    kernel_matmul_f32((float*)c->o, (const float*)c->a, (const float*)c->b, c->M, c->K, c->N);
}

/* ------------------------------------------------------------- elementwise */
typedef struct { double *o, *in, *g, *b; size_t n, h; } el_ctx;
static void run_gelu(void* p) { el_ctx* c = (el_ctx*)p; kernel_gelu(c->o, c->in, c->n); }
static void run_norm(void* p) {
    el_ctx* c = (el_ctx*)p;
    kernel_layernorm(c->o, c->in, c->g, c->b, c->n / c->h, c->h, LANCIUS_NORM_EPS);
}

/* ------------------------------------------------------------- attention */
typedef struct { double *o, *q, *k, *v; size_t S, H, D; } at_ctx;
static void run_attn(void* p) {
    at_ctx* c = (at_ctx*)p;
    kernel_attention(c->o, c->q, c->k, c->v, c->S, c->H, c->D);
}

/* --------------------------------------------------------- scheduler e2e
 * One FP64 matmul node, executed serially and then on the 4-thread pool.
 * This is the number a caller actually experiences: IR + validate + schedule
 * + plan + execute, not just the kernel. */
typedef struct {
    lancius_graph* g; lancius_node *x, *w, *m;
    lancius_arena* arena; double *xv, *wv;
    size_t M, K, N; int pooled;
} e2e_ctx;
static void run_e2e(void* p) {
    e2e_ctx* c = (e2e_ctx*)p;
    lancius_clear_error();
    lancius_schedule* s = lancius_ir_schedule(c->g);
    if (!s) { fprintf(stderr, "bench: schedule failed err=%d\n", (int)lancius_get_error()); exit(2); }
    if (c->pooled) {
        lancius_pool* pool = lancius_pool_create(4);
        lancius_schedule_execute_parallel(s, c->arena, pool);
        lancius_pool_destroy(pool);
    } else {
        lancius_schedule_execute(s, c->arena);
    }
    int err = (int)lancius_get_error();
    lancius_schedule_destroy(s);
    if (err) { fprintf(stderr, "bench: execute failed err=%d\n", err); exit(2); }
}

static void table_row(const char* name, const char* shape, double secs,
                      double flops, double elems) {
    if (flops > 0.0)
        printf("  %-22s %-18s %9.3f ms %10.1f GFLOP/s\n", name, shape, secs * 1e3,
               flops / secs / 1e9);
    else
        printf("  %-22s %-18s %9.3f ms %10.2f Melem/s\n", name, shape, secs * 1e3,
               elems / secs / 1e6);
}

int main(void) {
    printf("================================================================\n");
    printf("  Lancius kernel throughput bench (median of 7, 1 warm-up)\n");
    printf("  FLOP formulas are printed in bench_kernels.c; rates are\n");
    printf("  measurements on THIS machine, not targets and not comparisons.\n");
    printf("================================================================\n");
    int nproc = 1;
#ifdef _OPENMP
    nproc = omp_get_num_procs();
    printf("  OpenMP: enabled, max_threads=%d, num_procs=%d\n",
           omp_get_max_threads(), nproc);
#else
    printf("  OpenMP: disabled (serial build)\n");
#endif
#ifdef __AVX2__
    printf("  SIMD:   AVX2+FMA code path compiled in\n");
#else
    printf("  SIMD:   scalar (no -mavx2)\n");
#endif
    printf("  Build:  %s %s\n", __DATE__, __TIME__);
    printf("\n");

    const int REPS = 7;

    /* ---- FP64 / FP32 matmul, the two hot paths ---------------------- */
    {
        size_t dims[3] = {256, 512, 1024};
        for (int i = 0; i < 3; i++) {
            size_t n = dims[i];
            mm_ctx c;
            c.M = c.K = c.N = n;
            c.a = (double*)malloc(n * n * sizeof(double));
            c.b = (double*)malloc(n * n * sizeof(double));
            c.o = (double*)malloc(n * n * sizeof(double));
            if (!c.a || !c.b || !c.o) { fprintf(stderr, "bench: oom\n"); return 2; }
            for (size_t k = 0; k < n * n; k++) {
                c.a[k] = (double)((k * 1103515245u + 12345u) % 1000) / 1000.0 - 0.5;
                c.b[k] = (double)((k * 22695477u + 1u) % 1000) / 1000.0 - 0.5;
            }
            char shape[32];
            snprintf(shape, sizeof(shape), "%zux%zux%zu", n, n, n);
            double s = median_seconds(run_mm64, &c, REPS);
            table_row("matmul fp64", shape, s, 2.0 * (double)n * n * n, 0.0);
            free(c.a); free(c.b); free(c.o);
        }
    }
    {
        size_t n = 512;
        mm_ctx c;
        c.M = c.K = c.N = n;
        c.a = (double*)malloc(n * n * sizeof(float));
        c.b = (double*)malloc(n * n * sizeof(float));
        c.o = (double*)malloc(n * n * sizeof(float));
        if (!c.a || !c.b || !c.o) { fprintf(stderr, "bench: oom\n"); return 2; }
        float* af = (float*)c.a; float* bf = (float*)c.b;
        for (size_t k = 0; k < n * n; k++) {
            af[k] = (float)((k * 1103515245u + 12345u) % 1000) / 1000.0f - 0.5f;
            bf[k] = (float)((k * 22695477u + 1u) % 1000) / 1000.0f - 0.5f;
        }
        double s = median_seconds(run_mm32, &c, REPS);
        table_row("matmul fp32", "512x512x512", s, 2.0 * (double)n * n * n, 0.0);
        free(c.a); free(c.b); free(c.o);
    }

    /* ---- elementwise: rate in elements, not fudged FLOPs ------------- */
    {
        size_t n = 4u * 1024u * 1024u;
        el_ctx c;
        c.n = n; c.h = 1024;
        c.o = (double*)malloc(n * sizeof(double));
        c.in = (double*)malloc(n * sizeof(double));
        c.g = (double*)malloc(c.h * sizeof(double));
        c.b = (double*)malloc(c.h * sizeof(double));
        if (!c.o || !c.in || !c.g || !c.b) { fprintf(stderr, "bench: oom\n"); return 2; }
        for (size_t k = 0; k < n; k++)
            c.in[k] = (double)((k * 1103515245u + 12345u) % 2000) / 1000.0 - 1.0;
        for (size_t k = 0; k < c.h; k++) { c.g[k] = 1.0; c.b[k] = 0.0; }
        double s = median_seconds(run_gelu, &c, REPS);
        char shape[32];
        snprintf(shape, sizeof(shape), "%zu el", n);
        table_row("gelu (tanh approx)", shape, s, 0.0, (double)n);
        s = median_seconds(run_norm, &c, REPS);
        table_row("layernorm", shape, s, 0.0, (double)n);
        free(c.o); free(c.in); free(c.g); free(c.b);
    }

    /* ---- causal attention ------------------------------------------- */
    {
        size_t S = 256, H = 8, D = 64;
        at_ctx c;
        c.S = S; c.H = H; c.D = D;
        size_t qkv = S * H * D;
        c.o = (double*)malloc(qkv * sizeof(double));
        c.q = (double*)malloc(qkv * sizeof(double));
        c.k = (double*)malloc(qkv * sizeof(double));
        c.v = (double*)malloc(qkv * sizeof(double));
        if (!c.o || !c.q || !c.k || !c.v) { fprintf(stderr, "bench: oom\n"); return 2; }
        for (size_t i = 0; i < qkv; i++) {
            c.q[i] = (double)((i * 1103515245u + 12345u) % 1000) / 1000.0 - 0.5;
            c.k[i] = (double)((i * 22695477u + 1u) % 1000) / 1000.0 - 0.5;
            c.v[i] = (double)((i * 69069u + 1u) % 1000) / 1000.0 - 0.5;
        }
        double s = median_seconds(run_attn, &c, REPS);
        char shape[32];
        snprintf(shape, sizeof(shape), "S=%zu H=%zu D=%zu", S, H, D);
        /* scores + out per head = 4*S*S*D; the online softmax pass is S*S*(D+1) */
        double flops = (double)H * (4.0 * (double)S * S * D + (double)S * S * (D + 1));
        table_row("attention (causal)", shape, s, flops, 0.0);
        free(c.o); free(c.q); free(c.k); free(c.v);
    }

    /* ---- end-to-end: IR -> validate -> schedule -> plan -> execute ---- */
    {
        size_t M = 256, K = 256, N = 256;
        e2e_ctx c;
        c.M = M; c.K = K; c.N = N; c.pooled = 0;
        c.xv = (double*)malloc(M * K * sizeof(double));
        c.wv = (double*)malloc(K * N * sizeof(double));
        if (!c.xv || !c.wv) { fprintf(stderr, "bench: oom\n"); return 2; }
        for (size_t i = 0; i < M * K; i++)
            c.xv[i] = (double)((i * 1103515245u + 12345u) % 1000) / 1000.0 - 0.5;
        for (size_t i = 0; i < K * N; i++)
            c.wv[i] = (double)((i * 22695477u + 1u) % 1000) / 1000.0 - 0.5;
        c.g = lancius_graph_create();
        c.x = lancius_input(c.g, M, K);
        c.w = lancius_input(c.g, K, N);
        c.m = lancius_matmul(c.g, c.x, c.w);
        lancius_node_bind_external(c.x, c.xv);
        lancius_node_bind_external(c.w, c.wv);
        c.arena = lancius_arena_create(64u * 1024u * 1024u);
        if (!c.arena) { fprintf(stderr, "bench: arena oom\n"); return 2; }
        char shape[32];
        snprintf(shape, sizeof(shape), "%zux%zux%zu", M, K, N);
        double s = median_seconds(run_e2e, &c, REPS);
        table_row("graph e2e serial", shape, s, 2.0 * (double)M * K * N, 0.0);
        c.pooled = 1;
        s = median_seconds(run_e2e, &c, REPS);
        table_row("graph e2e pool=4", shape, s, 2.0 * (double)M * K * N, 0.0);
        lancius_arena_destroy(c.arena);
        lancius_graph_destroy(c.g);
        free(c.xv); free(c.wv);
    }

    printf("\nNote: the pooled scheduler path pins each worker's OpenMP team to 1\n"
           "(src/runtime/lancius_threadpool.c), so it trades kernel-level\n"
           "parallelism for wave-level parallelism and determinism. Compare the\n"
           "two e2e rows before choosing a mode for your shape.\n");
    return 0;
}

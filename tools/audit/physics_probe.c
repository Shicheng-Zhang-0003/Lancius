/*
 * physics_probe.c -- the MACHINE-TRUTH arm of `make check-physics`.
 *
 * Everything this file prints is something a CPU actually did under the
 * project's own toolchain (-O3 -mavx2 -mfma -fopenmp -std=c11), not a claim.
 * tools/audit/physics_oracle.py then re-derives each number from an
 * independent reference (ISO/IEC 60929 decimal arithmetic for the
 * transcendental truth, IEEE 754-2019 for the format constants, zlib for
 * CRC-32, and a second execution path inside Lancius for determinism) and
 * fails on any divergence.
 *
 * Why this gate exists: the standing oracle gate proves that Lancius computes
 * the right FUNCTION. It does not pin the physical substrate the function is
 * evaluated on -- whether GCC contracts a*b+c into an FMA (it would move every
 * result by 1 ulp and break every "bit-identical" claim in this repo), whether
 * the OpenMP-parallel kernel is bit-identical to the serial one, whether a
 * 4-thread pool quietly becomes 72 OS threads because every worker opens its
 * own OMP team, and whether the CRC-32 in the model format is the CRC-32 of
 * IEEE 802.3. Those are facts about this machine, they change with the
 * compiler, and until now nothing measured them.
 */
#include "lancius/lancius.h"
#include "lancius/lancius_error.h"
#include "lancius/lancius_kernels.h"
#include "lancius/lancius_threadpool.h"
#include "lancius/lancius_model_format.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---------------------------------------------------------------- helpers */

/* Same repo-relative scratch rule the audits use: create every prefix of
 * temp/scratch on demand so the gate runs from a fresh checkout. */
static int make_scratch(void) {
    if (mkdir("temp", 0755) != 0 && errno != EEXIST) return 0;
    if (mkdir("temp/scratch", 0755) != 0 && errno != EEXIST) return 0;
    return 1;
}

static uint64_t fnv_bits(const double* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        uint64_t u;
        memcpy(&u, &p[i], 8);
        h ^= u;
        h *= 1099511628211ULL;
    }
    return h;
}

static double rnd01(void) { return (double)rand() / (double)RAND_MAX - 0.5; }

static int os_threads(void) {
    DIR* d = opendir("/proc/self/task");
    if (!d) return -1;
    struct dirent* e;
    int n = 0;
    while ((e = readdir(d))) if (e->d_name[0] != '.') n++;
    closedir(d);
    return n;
}

/* ---------------------------------------------- 1. IEEE 754-2019 anchors */

static void probe_ieee(void) {
    double eps = 0x1.0p-52;
    double one_plus_half = 1.0 + 0x1p-53;      /* ties-to-even: must round to 1 */
    double one_plus_eps_half_ok = 1.0 + 0x1p-52; /* one ulp above 1: must NOT */
    uint64_t b;
    memcpy(&b, &eps, 8);
    printf("IEEE epsilon_bits %016llx\n", (unsigned long long)b);
    printf("IEEE one_plus_half_eps %d\n", one_plus_half == 1.0);
    printf("IEEE one_plus_eps_is_next %d\n", one_plus_eps_half_ok > 1.0);
    memcpy(&b, &one_plus_eps_half_ok, 8);
    printf("IEEE next_after_1_bits %016llx\n", (unsigned long long)b);
    printf("IEEE sqrt2 %.17g\n", sqrt(2.0));
    printf("IEEE exp1 %.17g\n", exp(1.0));
    printf("IEEE log2 %.17g\n", log(2.0));
    /* subnormals are live on this machine (no FTZ/DAZ): a physics outcome of
     * the MXCSR control word, which gcc leaves at the x86-64 default. */
    double dn = 0x1p-1070;
    printf("IEEE denorm_flushed %d\n", (dn * 2.0 == 0.0));
    printf("IEEE denorm_min %.17g\n", 0x1p-1074);
}

/* --------------------------------- 2. libm values as this toolchain gives */

static const double kX[] = {
    -12.0, -8.0, -5.0, -2.7, -1.0, -0.5, -0.1, 0.0,
     0.1,  0.5,  1.0,  2.6989, 5.0,  8.0, 10.0, 12.0
};

static void probe_libm(void) {
    for (size_t i = 0; i < sizeof kX / sizeof kX[0]; i++) {
        double x = kX[i];
        printf("EXP %.17g %.17g\n", x, exp(x));
        printf("TANH %.17g %.17g\n", x, tanh(x));
        printf("SIN %.17g %.17g\n", x, sin(x));
        printf("COS %.17g %.17g\n", x, cos(x));
        printf("SQRT %.17g %.17g\n", x, sqrt(x));
        if (x > 0.0) printf("LOG %.17g %.17g\n", x, log(x));
    }
    static const double kE[] = {-12.0, -10.0, -5.0, -2.6989, -1.0, 0.0,
                                 1.0,   2.6989, 5.0, 10.0, 12.0};
    for (size_t i = 0; i < sizeof kE / sizeof kE[0]; i++)
        printf("ERF %.17g %.17g\n", kE[i], erf(kE[i]));
    /* the constant the GELU kernel hard-codes */
    printf("CONST sqrt2_over_pi %.17g\n", sqrt(2.0 / 3.14159265358979311599796346854));
}

/* ------------------ 3. FMA contraction: the reproducibility landmine */

__attribute__((noinline)) static double fp_prod(double a, double b) { return a * b; }
__attribute__((noinline)) static double fp_sum(double p, double c) { return p + c; }
__attribute__((noinline)) static double fp_madd(double a, double b, double c) { return a * b + c; }

static void probe_fma(void) {
    /* a*b = 1 - 2^-54, which rounds to 1 when materialised, so an unfused
     * a*b+c gives 0 and a fused one gives -2^-54. If this prints CONTRACTION 1
     * the whole repository's bit-identity story is false under these flags. */
    volatile double a = 1.0 + 0x1p-27, b = 1.0 - 0x1p-27, c = -1.0;
    double direct = fp_madd(a, b, c);
    double split = fp_sum(fp_prod(a, b), c);
    printf("FMA contraction %d\n", direct != split);
    uint64_t bd, bs;
    memcpy(&bd, &direct, 8);
    memcpy(&bs, &split, 8);
    printf("FMA direct_bits %016llx\n", (unsigned long long)bd);
    printf("FMA split_bits %016llx\n", (unsigned long long)bs);
}

/* ------------- 4. determinism: parallel kernel vs serial kernel, bits */

static void probe_kernel_determinism(void) {
    const size_t M = 97, K = 200, N = 113;
    double* a = (double*)malloc(M * K * sizeof(double));
    double* b = (double*)malloc(K * N * sizeof(double));
    double* o1 = (double*)malloc(M * N * sizeof(double));
    double* o2 = (double*)malloc(M * N * sizeof(double));
    if (!a || !b || !o1 || !o2) { printf("DETMAT OOM\n"); free(a); free(b); free(o1); free(o2); return; }
    srand(20261010);
    for (size_t i = 0; i < M * K; i++) a[i] = rnd01() * 4.0;
    for (size_t i = 0; i < K * N; i++) b[i] = rnd01() * 4.0;

    int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    lancius_clear_error();
    kernel_matmul(o1, a, b, M, K, N);
    int e1 = lancius_get_error();
    omp_set_num_threads(8);
    lancius_clear_error();
    kernel_matmul(o2, a, b, M, K, N);
    int e2 = lancius_get_error();
    omp_set_num_threads(saved_threads); /* leave the default for later probes */

    uint64_t h1 = fnv_bits(o1, M * N), h2 = fnv_bits(o2, M * N);
    size_t diff = 0;
    for (size_t i = 0; i < M * N; i++) if (o1[i] != o2[i]) diff++;
    printf("DETMAT err1 %d err8 %d\n", e1, e2);
    printf("DETMAT hash1 %016llx hash8 %016llx\n",
           (unsigned long long)h1, (unsigned long long)h2);
    printf("DETMAT bitdiffs %zu\n", diff);
    free(a); free(b); free(o1); free(o2);
}

/* ------------- 5. scheduler determinism + pool thread budget */

static volatile int g_stop_sampling = 0;
static int g_max_os_threads = 0;

static void* sampler(void* arg) {
    (void)arg;
    while (!g_stop_sampling) {
        int n = os_threads();
        if (n > g_max_os_threads) g_max_os_threads = n;
        struct timespec ts = {0, 1000000L}; /* 1 ms */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static uint64_t run_schedule(lancius_schedule* sched, lancius_arena* arena,
                             lancius_pool* pool, lancius_node** outs, int nout) {
    lancius_arena_reset(arena);
    lancius_clear_error();
    if (pool) lancius_schedule_execute_parallel(sched, arena, pool);
    else lancius_schedule_execute(sched, arena);
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < nout; i++) {
        uint64_t o = fnv_bits((const double*)outs[i]->runtime_data, 64 * 64);
        h ^= o;
        h *= 1099511628211ULL;
    }
    return h;
}

static void probe_scheduler_determinism(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* x1 = lancius_input(g, 64, 128);
    lancius_node* w1 = lancius_input(g, 128, 64);
    lancius_node* x2 = lancius_input(g, 64, 128);
    lancius_node* w2 = lancius_input(g, 128, 64);
    lancius_node* m1 = lancius_matmul(g, x1, w1);
    lancius_node* m2 = lancius_matmul(g, x2, w2);
    lancius_node* s = lancius_add(g, m1, m2);
    /* A second sink that consumes m1, so m1 stays live past its producing
     * wave and the pooled/liveness path cannot recycle its buffer under the
     * hash. */
    lancius_node* r1 = lancius_relu(g, m1);

    double* xv1 = (double*)calloc(64 * 128, sizeof(double));
    double* wv1 = (double*)calloc(128 * 64, sizeof(double));
    double* xv2 = (double*)calloc(64 * 128, sizeof(double));
    double* wv2 = (double*)calloc(128 * 64, sizeof(double));
    if (!xv1 || !wv1 || !xv2 || !wv2) { printf("DETSCHED OOM\n"); return; }
    srand(20261011);
    for (int i = 0; i < 64 * 128; i++) xv1[i] = rnd01();
    for (int i = 0; i < 128 * 64; i++) wv1[i] = rnd01();
    for (int i = 0; i < 64 * 128; i++) xv2[i] = rnd01();
    for (int i = 0; i < 128 * 64; i++) wv2[i] = rnd01();
    lancius_node_bind_external(x1, xv1);
    lancius_node_bind_external(w1, wv1);
    lancius_node_bind_external(x2, xv2);
    lancius_node_bind_external(w2, wv2);

    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* arena = lancius_arena_create(64u << 20);
    if (!sched || !arena) { printf("DETSCHED setup failed err=%d\n", lancius_get_error()); return; }

    lancius_node* outs[2] = {r1, s};
    uint64_t hs = run_schedule(sched, arena, NULL, outs, 2);
    int err_serial = lancius_get_error();

    lancius_pool* pool = lancius_pool_create(4);
    pthread_t th;
    g_stop_sampling = 0;
    /* Baseline AFTER the pool exists (its worker threads are already live) and
     * after the serial run above (which has already warmed the OMP team), so
     * the delta we report is exactly the thread traffic the pool's execution
     * adds on top. Before the V10 pinning fix that delta was ~4xOMP_THREADS;
     * after it, it is the pool plus this sampler. */
    int baseline_threads = os_threads();
    g_max_os_threads = baseline_threads;
    pthread_create(&th, NULL, sampler, NULL);
    uint64_t hp = run_schedule(sched, arena, pool, outs, 2);
    int err_pool = lancius_get_error();
    g_stop_sampling = 1;
    pthread_join(th, NULL);
    int pool_threads_observed = g_max_os_threads;
    lancius_pool_destroy(pool);

    printf("DETSCHED serial_hash %016llx pool_hash %016llx\n",
           (unsigned long long)hs, (unsigned long long)hp);
    printf("DETSCHED err_serial %d err_pool %d\n", err_serial, err_pool);
    printf("POOLTHREADS max_os %d baseline %d pool_size %d delta %d\n",
           pool_threads_observed, baseline_threads, 4,
           pool_threads_observed - baseline_threads);

    lancius_arena_destroy(arena);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    free(xv1); free(wv1); free(xv2); free(wv2); /* bind_external: caller owns */
}

/* ------------- 6. a real v2 model on disk for the CRC-32 cross-check */

static void probe_crc(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* x = lancius_input(g, 4, 4);
    lancius_node* w = lancius_input(g, 4, 4);
    lancius_node* y = lancius_matmul(g, x, w);
    (void)y;
    double* xv = (double*)calloc(16, sizeof(double));
    double* wv = (double*)calloc(16, sizeof(double));
    if (!xv || !wv) { printf("CRC file OOM\n"); return; }
    for (int i = 0; i < 16; i++) { xv[i] = i * 0.25; wv[i] = (i % 5) * 0.5; }
    lancius_node_bind_external(x, xv);
    lancius_node_bind_external(w, wv);
    if (lancius_graph_save(g, "temp/scratch/physics_crc.lancius") != 0) {
        printf("CRCFILE save_failed err=%d\n", lancius_get_error());
    } else {
        printf("CRCFILE path temp/scratch/physics_crc.lancius\n");
    }
    lancius_graph_destroy(g);
}

int main(void) {
    if (!make_scratch()) { fprintf(stderr, "cannot create temp/scratch\n"); return 2; }
    probe_ieee();
    probe_libm();
    probe_fma();
    probe_kernel_determinism();
    probe_scheduler_determinism();
    probe_crc();
    return 0;
}

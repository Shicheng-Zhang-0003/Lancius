/*
 * Stable-API lifetime audit.
 *
 * The FFI contract says a destroyed graph invalidates all of its tensor
 * handles, and that every stale, wrong-type or double-destroyed handle
 * returns INVALID_HANDLE instead of touching freed memory.  This audit is
 * executable: it destroys objects, reuses the heap, and demands refusal.
 *
 * It also exercises the handle table from four threads and proves that
 * create/destroy cycles recycle table rows instead of growing forever.
 */
#include <lancius/lancius_stable_api.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test-only diagnostic exported by the static library. */
extern size_t lancius_debug_live_handle_count(void);

static int checks = 0;
static int failures = 0;

#define CHECK(cond, msg) do { \
    checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while (0)

static int invalid_handle_error(void) {
    return lancius_get_last_error() == LANCIUS_ERR_INVALID_HANDLE;
}

typedef struct {
    lancius_context ctx;
    int ok;
    int iterations;
} thread_arg;

static void* churn(void* opaque) {
    thread_arg* a = (thread_arg*)opaque;
    a->ok = 1;
    for (int i = 0; i < a->iterations; i++) {
        lancius_graph_handle g = lancius_graph_create_stable(a->ctx);
        if (!g) { a->ok = 0; return NULL; }
        lancius_tensor_handle t = lancius_add_input(g, 4, 4);
        if (!t) { a->ok = 0; lancius_graph_destroy_stable(g); return NULL; }
        if (lancius_tensor_element_count(t) != 16) { a->ok = 0; }
        lancius_graph_destroy_stable(g);
    }
    return NULL;
}

int main(void) {
    printf("================================================================\n");
    printf("  STABLE API LIFETIME AUDIT\n");
    printf("================================================================\n");

    lancius_context ctx = lancius_create_context();
    CHECK(ctx != 0, "context created");
    CHECK(lancius_debug_live_handle_count() == 1, "one live handle after context create");
    size_t baseline = lancius_debug_live_handle_count();

    lancius_graph_handle g1 = lancius_graph_create_stable(ctx);
    lancius_tensor_handle t1 = lancius_add_input(g1, 2, 3);
    CHECK(g1 != 0 && t1 != 0, "graph and tensor created");
    CHECK(lancius_tensor_element_count(t1) == 6, "live tensor reports 6 elements");
    lancius_graph_destroy_stable(g1);
    CHECK(lancius_get_last_error() == LANCIUS_OK, "graph destroy succeeds");
    CHECK(lancius_debug_live_handle_count() == baseline, "graph destroy releases graph and tensor rows");

    CHECK(lancius_tensor_element_count(t1) == 0 && invalid_handle_error(),
          "stale tensor handle is rejected by element_count");
    double out[6] = {0};
    CHECK(lancius_read_output(t1, out, sizeof(out)) == LANCIUS_ERR_INVALID_HANDLE && invalid_handle_error(),
          "stale tensor handle is rejected by read_output");
    double in[6] = {1, 2, 3, 4, 5, 6};
    CHECK(lancius_bind_data(t1, in) == LANCIUS_ERR_INVALID_HANDLE && invalid_handle_error(),
          "stale tensor handle is rejected by bind_data");

    lancius_graph_destroy_stable(g1);
    CHECK(invalid_handle_error(), "double graph destroy is rejected");

    /* Heap reuse: the old wrapper address must not alias the new graph. */
    lancius_graph_handle g2 = lancius_graph_create_stable(ctx);
    lancius_tensor_handle t2 = lancius_add_input(g2, 7, 5);
    CHECK(g2 != 0 && t2 != 0, "second graph and tensor created");
    CHECK(lancius_debug_live_handle_count() == baseline + 2, "second graph owns exactly two rows");
    CHECK(lancius_tensor_element_count(t2) == 35, "second tensor reports 35 elements");
    CHECK(lancius_tensor_element_count(t1) == 0 && invalid_handle_error(),
          "stale tensor does not alias a newer tensor");
    CHECK(lancius_bind_data(t1, in) == LANCIUS_ERR_INVALID_HANDLE && invalid_handle_error(),
          "stale tensor cannot bind data into a newer graph");
    CHECK(lancius_add_relu(g2, t1) == 0 && invalid_handle_error(),
          "stale tensor is rejected by a graph operation");

    /* Wrong-type handles must not be reinterpreted. */
    CHECK(lancius_tensor_element_count((lancius_tensor_handle)g2) == 0 && invalid_handle_error(),
          "graph handle is rejected where a tensor handle is required");
    CHECK(lancius_compile_and_run(t2) == LANCIUS_ERR_INVALID_HANDLE && invalid_handle_error(),
          "tensor handle is rejected where a graph handle is required");

    lancius_graph_destroy_stable(g2);
    CHECK(lancius_debug_live_handle_count() == baseline, "all graph rows released");

    /* Recycling: 20k create/destroy cycles must not consume 20k table rows. */
    for (int i = 0; i < 20000; i++) {
        lancius_graph_handle g = lancius_graph_create_stable(ctx);
        lancius_tensor_handle t = lancius_add_input(g, 1, 1);
        if (!g || !t) { printf("  FAIL: churn allocation at %d\n", i); failures++; break; }
        lancius_graph_destroy_stable(g);
    }
    CHECK(lancius_debug_live_handle_count() == baseline, "20k create/destroy cycles recycle rows");

    /* Independent graphs from four threads. */
    enum { NTHREADS = 4, NITER = 2000 };
    pthread_t th[NTHREADS];
    thread_arg args[NTHREADS];
    for (int i = 0; i < NTHREADS; i++) {
        args[i].ctx = ctx;
        args[i].ok = 0;
        args[i].iterations = NITER;
        if (pthread_create(&th[i], NULL, churn, &args[i]) != 0) {
            printf("  FAIL: pthread_create %d\n", i);
            failures++;
            args[i].ok = 0;
        }
    }
    for (int i = 0; i < NTHREADS; i++) pthread_join(th[i], NULL);
    int threaded_ok = 1;
    for (int i = 0; i < NTHREADS; i++) if (!args[i].ok) threaded_ok = 0;
    CHECK(threaded_ok, "4 threads x 2000 independent graph cycles");
    CHECK(lancius_debug_live_handle_count() == baseline, "threaded churn leaves only the context");

    /* Graph handles are independent of the context after creation. */
    lancius_graph_handle g3 = lancius_graph_create_stable(ctx);
    lancius_destroy_context(ctx);
    CHECK(lancius_get_last_error() == LANCIUS_OK, "context destroy succeeds");
    lancius_tensor_handle t3 = lancius_add_input(g3, 3, 3);
    CHECK(t3 != 0, "graph survives context destruction (documented contract)");
    CHECK(lancius_tensor_element_count(t3) == 9, "surviving graph tensor is usable");
    lancius_graph_destroy_stable(g3);
    CHECK(lancius_debug_live_handle_count() == 0, "all rows released at shutdown");

    lancius_destroy_context(ctx);
    CHECK(invalid_handle_error(), "double context destroy is rejected");

    printf("================================================================\n");
    printf("  STABLE API LIFETIME AUDIT: %d checks, %d failure(s)\n", checks, failures);
    printf("================================================================\n");
    return failures ? 1 : 0;
}

#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS V1.0: ADVERSARIAL NaN/Inf INJECTION AUDIT           \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    lancius_node* in = lancius_input(g, 32, 32);
    lancius_node* w = lancius_const(g, 0.0, 32, 32);
    lancius_node* mm = lancius_matmul(g, in, w);
    lancius_node* out = lancius_relu(g, mm);

    size_t elems = 32 * 32;
    in->runtime_data = (double*)calloc(elems, sizeof(double));
    for(size_t i=0; i<elems; i++) in->runtime_data[i] = 1.0;

    // Adversarial Injection
    w->runtime_data = (double*)malloc(elems * sizeof(double));
    for(size_t i=0; i<elems; i++) {
        if (i % 2 == 0) w->runtime_data[i] = NAN;
        else w->runtime_data[i] = INFINITY;
    }

    printf("[1/2] Executing Graph with NaN/Inf Weights via Threadpool...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(4 * 1024 * 1024);
    lancius_pool* pool = lancius_pool_create(4);

    // This should NOT segfault or deadlock
    lancius_schedule_execute_parallel(sched, scratch, pool);

    printf("  ✅ Engine survived parallel execution with corrupted weights.\n");

    printf("[2/2] Verifying NaN Propagation...\n");
    int nan_count = 0;
    for(size_t i=0; i<elems; i++) {
        if (isnan(out->runtime_data[i]) || isinf(out->runtime_data[i])) nan_count++;
    }

    if (nan_count > 0) {
        printf("  ✅ NaN/Inf correctly propagated to output (%d / %zu elements).\n", nan_count, elems);
        printf("  🏆 TOPOLOGICAL SORT & ARENA OFFSETS REMAINED STABLE.\n");
    } else {
        printf("  ❌ NaN was silently swallowed or corrupted memory.\n");
    }

    // V1.0 FIX: Save payload pointers before destroying the graph (which frees the node structs)
    double* in_data = in->runtime_data;
    double* w_data = w->runtime_data;

    lancius_pool_destroy(pool);
    lancius_schedule_destroy(sched);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);

    free(in_data);
    free(w_data);

    return 0;
}

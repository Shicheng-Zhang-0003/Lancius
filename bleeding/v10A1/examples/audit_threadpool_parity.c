#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS V1.0: THREADPOOL PARITY & RACE CONDITION AUDIT      \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    // Build a wide graph with independent nodes in the same wave
    lancius_node* in1 = lancius_input(g, 64, 64);
    lancius_node* in2 = lancius_input(g, 64, 64);
    lancius_node* in3 = lancius_input(g, 64, 64);

    lancius_node* r1 = lancius_relu(g, in1);
    lancius_node* r2 = lancius_relu(g, in2);
    lancius_node* r3 = lancius_relu(g, in3);

    lancius_node* out1 = lancius_add(g, r1, r2);
    lancius_node* out2 = lancius_mul(g, r2, r3);

    size_t elems = 64 * 64;
    in1->runtime_data = (double*)malloc(elems * sizeof(double));
    in2->runtime_data = (double*)malloc(elems * sizeof(double));
    in3->runtime_data = (double*)malloc(elems * sizeof(double));

    for(size_t i=0; i<elems; i++) {
        in1->runtime_data[i] = 1.5;
        in2->runtime_data[i] = -2.0;
        in3->runtime_data[i] = 3.0;
    }

    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(16 * 1024 * 1024);

    // 1. Sequential Execution
    printf("[1/3] Executing Sequentially...\n");
    lancius_schedule_execute(sched, scratch);

    double* seq_out1 = (double*)malloc(elems * sizeof(double));
    double* seq_out2 = (double*)malloc(elems * sizeof(double));
    memcpy(seq_out1, out1->runtime_data, elems * sizeof(double));
    memcpy(seq_out2, out2->runtime_data, elems * sizeof(double));

    lancius_arena_reset(scratch);

    // 2. Parallel Execution via Threadpool
    printf("[2/3] Executing via Threadpool (4 Threads)...\n");
    lancius_pool* pool = lancius_pool_create(4);
    lancius_schedule_execute_parallel(sched, scratch, pool);
    lancius_pool_destroy(pool);

    // 3. Differential Parity Check
    printf("[3/3] Comparing Sequential vs Parallel Outputs...\n");
    double max_diff1 = 0.0, max_diff2 = 0.0;
    for(size_t i=0; i<elems; i++) {
        double d1 = fabs(seq_out1[i] - out1->runtime_data[i]);
        double d2 = fabs(seq_out2[i] - out2->runtime_data[i]);
        if(d1 > max_diff1) max_diff1 = d1;
        if(d2 > max_diff2) max_diff2 = d2;
    }

    printf("  Max Diff (Add Node): %e\n", max_diff1);
    printf("  Max Diff (Mul Node): %e\n", max_diff2);

    if (max_diff1 == 0.0 && max_diff2 == 0.0) {
        printf("\n  ✅ PARITY VERIFIED: Threadpool execution matches sequential perfectly.\n");
        printf("  🏆 ZERO RACE CONDITIONS DETECTED IN WAVE SCHEDULER.\n");
    } else {
        printf("\n  ❌ DIVERGENCE DETECTED: Threadpool caused race conditions!\n");
    }

    free(seq_out1); free(seq_out2);
    free(in1->runtime_data); free(in2->runtime_data); free(in3->runtime_data);
    lancius_schedule_destroy(sched);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);

    return 0;
}

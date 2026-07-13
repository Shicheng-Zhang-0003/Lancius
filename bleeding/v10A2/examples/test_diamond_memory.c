#include <lancius.h>
#include "lancius/lancius_memory_planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS v10A2: DIAMOND GRAPH MEMORY CORRUPTOR (RESNET TEST)\n");
    printf("================================================================\n");

    lancius_graph* g = lancius_graph_create();

    // 1. Build Diamond Graph
    // Input(1, 3, 8, 8) -> Conv(3, 3, 3, 3, pad=1) -> ReLU -> Add(ReLU, Input)
    lancius_node* X = lancius_input_4d(g, 1, 3, 8, 8);
    lancius_node* W = lancius_input_4d(g, 3, 3, 3, 3);
    lancius_node* C1 = lancius_conv2d(g, X, W, 1, 1);
    lancius_node* R1 = lancius_relu(g, C1);
    lancius_node* Out = lancius_add(g, R1, X);

    // 2. Schedule & Plan Memory
    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_memory_plan* plan = lancius_build_memory_plan(sched, g);

    if (!plan) {
        printf("  ❌ FAILED: Could not build memory plan.\n");
        return 1;
    }

    printf("  🧠 Pooled Peak Memory: %zu bytes\n", plan->peak_memory);

    void* flat_buffer = malloc(plan->peak_memory);
    lancius_schedule_attach_pool(sched, flat_buffer, plan);

    // 3. Allocate & Initialize Inputs on the Heap
    size_t x_elems = 1 * 3 * 8 * 8;
    size_t w_elems = 3 * 3 * 3 * 3;

    double* x_data = (double*)malloc(x_elems * sizeof(double));
    double* w_data = (double*)malloc(w_elems * sizeof(double));
    double* x_pristine = (double*)malloc(x_elems * sizeof(double));

    for(size_t i=0; i<x_elems; i++) {
        x_data[i] = (i % 10) * 0.1 - 0.5; // Mix of positive and negative
        x_pristine[i] = x_data[i];
    }
    for(size_t i=0; i<w_elems; i++) {
        w_data[i] = (i % 5) * 0.05;
    }

    X->runtime_data = x_data;
    W->runtime_data = w_data;

    // 4. Execute via Pool (Passing NULL scratch arena forces strict pool usage)
    printf("[1/2] Executing Diamond Graph via Pooled Flat Buffer...\n");
    lancius_schedule_execute(sched, NULL);

    // 5. Verify Math
    printf("[2/2] Verifying Skip-Connection Math Integrity...\n");

    // Because R1 is the output of a ReLU, R1[i] >= 0.
    // Therefore, Out[i] = R1[i] + X_pristine[i] MUST be >= X_pristine[i].
    // If the memory planner overwrote X's buffer with C1 or R1, this invariant breaks.

    int pass = 1;
    double* out_data = Out->runtime_data;
    for(size_t i=0; i<x_elems; i++) {
        if (out_data[i] < x_pristine[i] - 1e-9) {
            printf("  ❌ MATH CORRUPTION at index %zu: Out=%f < X=%f\n", i, out_data[i], x_pristine[i]);
            printf("     (Memory planner overwrote skip-connection input!)\n");
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("  ✅ DIAMOND GRAPH VERIFIED: Skip-connection memory remained pristine.\n");
        printf("  🏆 LINEAR SCAN PLANNER SURVIVED THE RESNET TEST.\n");
    }

    free(flat_buffer);
    free(x_data); free(w_data); free(x_pristine);
    lancius_memory_plan_destroy(plan);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);

    printf("================================================================\n");
    return pass ? 0 : 1;
}

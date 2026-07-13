
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS INTERNAL X-RAY AUDIT                                 \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    // Mini CNN: Input(1, 3, 8, 8) -> Conv(4, 3, 3, 3) -> ReLU -> MaxPool(2,2) -> Flatten -> MatMul -> Add -> Softmax
    lancius_node* X = lancius_input_4d(g, 1, 3, 8, 8);
    lancius_node* W1 = lancius_input_4d(g, 4, 3, 3, 3);
    lancius_node* C1 = lancius_conv2d(g, X, W1, 1, 1); // stride 1, pad 1 -> (1, 4, 8, 8)
    lancius_node* R1 = lancius_relu(g, C1);
    lancius_node* P1 = lancius_maxpool2d(g, R1, 2, 2); // -> (1, 4, 4, 4)
    lancius_node* F1 = lancius_flatten(g, P1); // -> (1, 64)

    lancius_node* W2 = lancius_input(g, 64, 10);
    lancius_node* Z1 = lancius_matmul(g, F1, W2); // -> (1, 10)

    lancius_node* b1 = lancius_input(g, 1, 10);
    lancius_node* B1 = lancius_broadcast(g, b1, 1, 10);
    lancius_node* A1 = lancius_add(g, Z1, B1); // -> (1, 10)

    lancius_node* S1 = lancius_softmax(g, A1); // -> (1, 10)

    // Allocate dummy data
    size_t x_sz = 1*3*8*8; X->runtime_data = (double*)calloc(x_sz, sizeof(double));
    size_t w1_sz = 4*3*3*3; W1->runtime_data = (double*)calloc(w1_sz, sizeof(double));
    size_t w2_sz = 64*10; W2->runtime_data = (double*)calloc(w2_sz, sizeof(double));
    size_t b1_sz = 1*10; b1->runtime_data = (double*)calloc(b1_sz, sizeof(double));

    for(size_t i=0; i<x_sz; i++) X->runtime_data[i] = 0.1;
    for(size_t i=0; i<w1_sz; i++) W1->runtime_data[i] = 0.05;
    for(size_t i=0; i<w2_sz; i++) W2->runtime_data[i] = 0.02;
    for(size_t i=0; i<b1_sz; i++) b1->runtime_data[i] = 0.0;

    printf("[1/3] Compiling Schedule & Profiling Topology...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);

    size_t total_mem = 0;
    for(uint32_t w=0; w<sched->wave_count; w++) {
        size_t wave_mem = 0;
        for(uint32_t i=0; i<sched->waves[w].node_count; i++) {
            lancius_node* n = sched->waves[w].nodes[i];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) {
                size_t elems = lancius_node_elements(n);
                wave_mem += elems * sizeof(double);
            }
        }
        printf("  Wave %u: %u nodes | %zu bytes (%.2f KB)\n", w, sched->waves[w].node_count, wave_mem, wave_mem / 1024.0);
        if (wave_mem > total_mem) total_mem = wave_mem;
    }
    printf("  [PEAK MEMORY]: %zu bytes (%.2f KB)\n\n", total_mem, total_mem / 1024.0);

    printf("[2/3] Executing with High-Resolution Timers...\n");
    lancius_arena* scratch = lancius_arena_create(total_mem + 1024*1024);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    lancius_schedule_execute(sched, scratch);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) / 1e6; // ms

    printf("  Total Execution Time: %.4f ms\n\n", time_taken);

    printf("[3/3] Inspecting Output Logits & Precision...\n");
    printf("  Softmax Probabilities: [");
    for(int i=0; i<10; i++) {
        printf("%.6f ", S1->runtime_data[i]);
    }
    printf("]\n");
    printf("  Sum of Probabilities: ");
    double sum = 0;
    for(int i=0; i<10; i++) sum += S1->runtime_data[i];
    printf("%.15f (Should be exactly 1.0)\n", sum);

    printf("\n================================================================\n");
    printf("  INTERNAL X-RAY COMPLETE. INSPECT THE NUMBERS.                \n");
    printf("================================================================\n");

    free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(b1->runtime_data);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);
    return 0;
}

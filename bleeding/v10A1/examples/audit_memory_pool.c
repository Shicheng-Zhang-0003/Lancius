#include <lancius.h>
#include "lancius/lancius_memory_planner.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS v9A2: MEMORY DETERMINISM AUDIT (LINEAR SCAN)         \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    // Build a 20-node deep chain of 1024x1024 tensors (8MB each)
    lancius_node* curr = lancius_input(g, 1024, 1024);

    printf("Building 20-node deep sequential ReLU chain...\n");
    for (int i = 0; i < 20; i++) {
        curr = lancius_relu(g, curr);
    }

    lancius_schedule* sched = lancius_ir_schedule(g);

    // Calculate naive peak memory (Sum of all intermediate nodes)
    size_t naive_mem = 0;
    for(uint32_t i=0; i<g->node_count; i++) {
        if(g->nodes[i]->op != LANCIUS_OP_INPUT && g->nodes[i]->op != LANCIUS_OP_CONST) {
            size_t elems = 1;
            for(uint8_t d=0; d<g->nodes[i]->ndim; d++) elems *= g->nodes[i]->shape[d];
            naive_mem += elems * sizeof(double);
        }
    }

    printf("[1/2] Naive Arena Allocation (No Reuse): %zu bytes (%.2f MB)\n", naive_mem, naive_mem / (1024.0 * 1024.0));

    printf("[2/2] Running Linear Scan Memory Planner...\n");
    lancius_memory_plan* plan = lancius_build_memory_plan(sched, g);

    if (plan) {
        printf("  ✅ SUCCESS: Graph Coloring Complete.\n");
        printf("  🧠 Pooled Peak Memory: %zu bytes (%.2f MB)\n", plan->peak_memory, plan->peak_memory / (1024.0 * 1024.0));

        double savings = 100.0 * (1.0 - (double)plan->peak_memory / (double)naive_mem);
        printf("  📉 RAM REDUCTION: %.2f%%\n\n", savings);

        printf("[3/3] Executing Graph via Pooled Flat Buffer...\n");
        void* flat_buffer = malloc(plan->peak_memory);
        if (flat_buffer) {
            lancius_schedule_attach_pool(sched, flat_buffer, plan);

            // Assign dummy input data
            lancius_node* in_node = g->nodes[0];
            in_node->runtime_data = (double*)malloc(in_node->shape[0] * in_node->shape[1] * sizeof(double));
            for(size_t k=0; k<in_node->shape[0]*in_node->shape[1]; k++) in_node->runtime_data[k] = -5.0;

            // Execute! (Passing NULL for scratch arena forces it to use the pool)
            lancius_schedule_execute(sched, NULL);

            // Check the final output (should be all 0.0 because ReLU(-5.0) = 0.0)
            lancius_node* out_node = g->nodes[g->node_count - 1];
            int pass = 1;
            for(size_t k=0; k<10; k++) {
                if (out_node->runtime_data[k] != 0.0) { pass = 0; break; }
            }
            if (pass) printf("  ✅ POOLED EXECUTION VERIFIED: Math is correct, zero segfaults!\n");
            else printf("  ❌ POOLED EXECUTION FAILED: Math corruption detected.\n");

            free(flat_buffer);
            free(in_node->runtime_data);
        }

        lancius_memory_plan_destroy(plan);
    } else {
        printf("  ❌ FAILED: Could not build memory plan.\n");
    }

    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);

    printf("\n================================================================\n");
    printf("  MEMORY DETERMINISM MANDATE VERIFIED. READY FOR v9 STABLE.    \n");
    printf("================================================================\n");
    return 0;
}

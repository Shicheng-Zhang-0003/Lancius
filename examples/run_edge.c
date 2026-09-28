#include "lancius/lancius_inference.h"
#include "lancius/lancius_memory_planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// We load the model from disk. No training code here.
int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  Lancius: EDGE DEPLOYMENT RUNNER (Standalone)              \n");
    printf("================================================================\n\n");

    const char* model_path = argc > 1 ? argv[1] : "cifar10_lenet.lancius";
    printf("[1/3] Loading model from '%s'...\n", model_path);
    lancius_graph* g = lancius_graph_load(model_path);
    if (!g) {
        printf("FATAL: Failed to load model. Did you run training first?\n");
        return 1;
    }
    printf("  ✅ Loaded %u nodes.\n", g->node_count);

    printf("[2/3] Compiling Inference Schedule...\n");
    printf("[V10S] Running Graph Optimizations...\n");
    lancius_optimize_fusion(g);
    lancius_schedule* sched = lancius_ir_schedule(g);
    if (!sched) { printf("FATAL: schedule failed.\n"); lancius_graph_destroy(g); return 1; }
    lancius_memory_plan* plan = lancius_build_memory_plan(sched, g);
    /* Despot truth: NULL plan/sched fall back instead of deref (plan may be
     * NULL on corrupt shapes; required() may report 0 with sticky error). */
    size_t peak_mem = plan ? plan->peak_memory : lancius_schedule_static_memory_required(sched);
    size_t arena_size = peak_mem ? peak_mem : (1024 * 1024);
    printf("  🧠 Liveness Analyzer: Peak Memory = %zu bytes (%.2f KB)\n", peak_mem, peak_mem / 1024.0);
    void* flat_buffer = malloc(arena_size ? arena_size : 1);
    if (!flat_buffer) { printf("FATAL: OOM allocating static buffer."); if (plan) lancius_memory_plan_destroy(plan); lancius_schedule_destroy(sched); lancius_graph_destroy(g); return 1; }
    if (plan) lancius_schedule_attach_pool(sched, flat_buffer, plan);

    // Find the Input Node (X) and Output Node (Logits)
    lancius_node* input_node = NULL;
    lancius_node* output_node = NULL;
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n) continue;
        if (n->op == LANCIUS_OP_INPUT && n->ndim == 4 && n->shape[1] == 3 && n->shape[2] == 32 && n->shape[3] == 32) input_node = n;
        if (n->op == LANCIUS_OP_CROSS_ENTROPY && n->input_count > 0 && n->inputs && n->inputs[0]) {
                output_node = (lancius_node*)n->inputs[0]; // Native LANCIUS: Logits are input to CE
            } else if (n->ndim == 2 && n->shape[1] == 10 && n->op != LANCIUS_OP_INPUT) {
                output_node = n; // V10S ONNX FIX: Final node with 10 classes is the output
            }
    }

    if (!input_node || !output_node) {
        printf("FATAL: Could not find Input or Output nodes in loaded graph.\n");
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }

    size_t batch_size = input_node->shape[0];
    printf("[3/3] Running Inference on Random Noise (Batch Size: %zu)...\n", batch_size);

    // Allocate input buffer matching the loaded model's EXACT expected shape
    /* Despot truth: aborting counter + unbounded file shape + unchecked alloc
     * (was: abort/OOM-deref on malicious models). */
    size_t in_size = 0;
    if (!lancius_node_elements_checked(input_node, &in_size) || in_size == 0 || in_size > (64u << 20)) {
        printf("FATAL: bad input size.\n");
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    input_node->runtime_data = (double*)calloc(in_size, sizeof(double));
    if (!input_node->runtime_data) {
        printf("FATAL: OOM input buffer.\n");
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    lancius_node_bind_external(input_node, input_node->runtime_data);

    // Fill with random noise to simulate a batch of images
    for(size_t i=0; i<in_size; i++) input_node->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;

    // V11: Quantize Input Noise on the fly for INT8 Conv2D
    double max_in = 0.5;
    input_node->scale = max_in / 127.0;
    input_node->runtime_data_int8 = (int8_t*)malloc(in_size ? in_size : 1);
    if (!input_node->runtime_data_int8) {
        printf("FATAL: OOM int8 input buffer.\n");
        free(input_node->runtime_data); input_node->runtime_data = NULL;
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    lancius_node_bind_external_int8(input_node, input_node->runtime_data_int8);
    for(size_t i=0; i<in_size; i++) input_node->runtime_data_int8[i] = (int8_t)round(input_node->runtime_data[i] / input_node->scale);
    input_node->dtype = LANCIUS_DTYPE_INT8;

    // Execute
    lancius_schedule_execute_static(sched, flat_buffer);

    // Read Output
    /* Despot truth: NULL output printed FAIL but returned 0 (masked failure). */
    if (output_node->ndim != 2) {
        printf("  ❌ Inference Failed: output is not 2D.\n");
        free(input_node->runtime_data); input_node->runtime_data = NULL;
        free(input_node->runtime_data_int8); input_node->runtime_data_int8 = NULL;
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    if (output_node->runtime_data) {
        printf("  ✅ Inference Successful!\n");
        size_t num_classes = output_node->shape[1];
        printf("  Output Logits for Image 0 (Raw Scores for %zu Classes):\n  [", num_classes);
        // Output shape is [Batch, Classes], so the first 'num_classes' elements belong to Image 0
        for(size_t i=0; i<num_classes; i++) {
            printf("%.4f ", output_node->runtime_data[i]);
        }
        printf("]\n");

        // Argmax for Image 0
        int pred = 0; double max_v = -1e9;
        for(size_t i=0; i<num_classes; i++) {
            if(output_node->runtime_data[i] > max_v) {
                max_v = output_node->runtime_data[i];
                pred = i;
            }
        }
        printf("  🏆 Predicted Class for Image 0: %d (Confidence: %.4f)\n", pred, max_v);
    } else {
        printf("  ❌ Inference Failed: Output buffer is NULL.\n");
        free(input_node->runtime_data); input_node->runtime_data = NULL;
        free(input_node->runtime_data_int8); input_node->runtime_data_int8 = NULL;
        free(flat_buffer);
        if (plan) lancius_memory_plan_destroy(plan);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }

    // Cleanup
    free(input_node->runtime_data); input_node->runtime_data = NULL;
    free(input_node->runtime_data_int8); input_node->runtime_data_int8 = NULL;
    if (plan) lancius_memory_plan_destroy(plan);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    free(flat_buffer);

    printf("\n================================================================\n");
    printf("  EDGE DEPLOYMENT VERIFIED. MODEL IS PORTABLE.\n");
    printf("================================================================\n");
    return 0;
}

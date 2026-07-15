#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("  [C] Loading pytorch_lenet.lancius...\n");
    lancius_graph* g = lancius_graph_load("pytorch_lenet.lancius");
    if(!g) { printf("  [C] FATAL: Failed to load model.\n"); return 1; }

    // Find Input (4D) and Output (2D, 10 classes)
    lancius_node* in_node = NULL;
    lancius_node* out_node = NULL;
    for(uint32_t i=0; i<g->node_count; i++) {
    lancius_node* n = g->nodes[i];
    // Input node is an INPUT with no weights and shape starting with 1,3,...
    if (n->op == LANCIUS_OP_INPUT && n->ndim == 4 && n->shape[1] == 3 && n->runtime_data == NULL) {
        in_node = n;
    }
    // Output: non-input node, 2D, 10 classes
    if (n->ndim == 2 && n->shape[1] == 10 && n->op != LANCIUS_OP_INPUT) {
        out_node = n;
    }
}

        printf("  [DEBUG C] Graph Topology Dump:\n");
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        printf("    Node %u: Op=%d Shape=[%zux%zux%zux%zu] Inputs=[", n->id, n->op, n->shape[0], n->shape[1], n->shape[2], n->shape[3]);
        for(uint32_t k=0; k<n->input_count; k++) printf(" %u", n->inputs[k] ? n->inputs[k]->id : 9999);
        printf(" ]\n");
    }

if(!in_node || !out_node) {
        printf("  [C] FATAL: Could not find Input/Output nodes.\n");
        return 1;
    }

    // Load FP32 input from Python and cast to Lancius FP64
    size_t in_elems = in_node->shape[0] * in_node->shape[1] * in_node->shape[2] * in_node->shape[3];
    float* temp_in = (float*)malloc(in_elems * sizeof(float));
    FILE* f_in = fopen("parity_input.bin", "rb");
    if(!f_in) { printf("  [C] FATAL: Missing parity_input.bin\n"); return 1; }
    size_t _r = fread(temp_in, sizeof(float), in_elems, f_in); (void)_r;
    fclose(f_in);

    // V1.0 FIX: Explicitly allocate memory for the input node before writing
    in_node->runtime_data = (double*)malloc(in_elems * sizeof(double));
    for(size_t i=0; i<in_elems; i++) in_node->runtime_data[i] = (double)temp_in[i];
    free(temp_in);

    // Execute
    printf("  [C] Compiling Schedule & Executing...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(16 * 1024 * 1024);
    lancius_schedule_execute(sched, scratch);

    // Cast FP64 output back to FP32 for Python comparison
    size_t out_elems = out_node->shape[0] * out_node->shape[1];
    float* temp_out = (float*)malloc(out_elems * sizeof(float));
    for(size_t i=0; i<out_elems; i++) temp_out[i] = (float)out_node->runtime_data[i];

    FILE* f_out = fopen("lancius_out.bin", "wb");
    fwrite(temp_out, sizeof(float), out_elems, f_out);
    fclose(f_out);
    free(temp_out);

    // Cleanup
    // V1.0 ASAN FIX: Free heap-allocated INPUT data (Weights + Input Image)
    for(uint32_t i=0; i<g->node_count; i++) {
        if (g->nodes[i]->op == LANCIUS_OP_INPUT) {
            if (g->nodes[i]->runtime_data) free(g->nodes[i]->runtime_data);
            if (g->nodes[i]->runtime_data_int8) free(g->nodes[i]->runtime_data_int8);
        }
    }

    lancius_schedule_destroy(sched);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);

    printf("  [C] Execution Complete. Output saved to lancius_out.bin\n");
    return 0;
}

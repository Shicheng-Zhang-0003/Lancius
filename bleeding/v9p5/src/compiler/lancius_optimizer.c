#include "lancius/lancius_ir.h"
#include <stdio.h>

static int count_consumers(lancius_graph* g, lancius_node* target) {
    int count = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        for (uint32_t j = 0; j < n->input_count; j++) {
            if (n->inputs[j] == target) count++;
        }
    }
    return count;
}

void lancius_optimize_fusion(lancius_graph* g) {
    if (!g) return;
    int fused_count = 0;
    for (int i = g->node_count - 1; i >= 0; i--) {
        lancius_node* n = g->nodes[i];
        if (n->op == LANCIUS_OP_RELU && n->input_count > 0 && n->inputs[0]->op == LANCIUS_OP_CONV2D) {
            lancius_node* conv = (lancius_node*)n->inputs[0];
            if (count_consumers(g, conv) == 1) {
                n->op = LANCIUS_OP_CONV2D_RELU_FUSED;
                // V9 FIX: Steal the Conv2D's inputs array directly to avoid arena OOB write!
                n->inputs = conv->inputs;
                n->input_count = 2;
                n->kernel_h = conv->kernel_h;
                n->kernel_w = conv->kernel_w;
                n->stride = conv->stride;
                n->pad = conv->pad;
                for(int d=0; d<4; d++) n->shape[d] = conv->shape[d];
                n->ndim = 4;
                conv->op = LANCIUS_OP_NOP;
                conv->input_count = 0;
                fused_count++;
            }
        }
    }
    printf("[V9 OPTIMIZER] Fused %d Conv2D+ReLU patterns. Arena memory saved.\n", fused_count);
}

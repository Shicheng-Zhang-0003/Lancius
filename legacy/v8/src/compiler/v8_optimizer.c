#include "basis/v8_ir.h"
#include <stdio.h>

static int count_consumers(v8_graph* g, v8_node* target) {
    int count = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        v8_node* n = g->nodes[i];
        for (uint32_t j = 0; j < n->input_count; j++) {
            if (n->inputs[j] == target) count++;
        }
    }
    return count;
}

void v8_optimize_fusion(v8_graph* g) {
    if (!g) return;
    int fused_count = 0;
    for (int i = g->node_count - 1; i >= 0; i--) {
        v8_node* n = g->nodes[i];
        if (n->op == V8_OP_RELU && n->input_count > 0 && n->inputs[0]->op == V8_OP_CONV2D) {
            v8_node* conv = (v8_node*)n->inputs[0];
            if (count_consumers(g, conv) == 1) {
                n->op = V8_OP_CONV2D_RELU_FUSED;
                // V9 FIX: Steal the Conv2D's inputs array directly to avoid arena OOB write!
                n->inputs = conv->inputs;
                n->input_count = 2;
                n->kernel_h = conv->kernel_h;
                n->kernel_w = conv->kernel_w;
                n->stride = conv->stride;
                n->pad = conv->pad;
                for(int d=0; d<4; d++) n->shape[d] = conv->shape[d];
                n->ndim = 4;
                conv->op = V8_OP_NOP;
                conv->input_count = 0;
                fused_count++;
            }
        }
    }
    printf("[V9 OPTIMIZER] Fused %d Conv2D+ReLU patterns. Arena memory saved.\n", fused_count);
}

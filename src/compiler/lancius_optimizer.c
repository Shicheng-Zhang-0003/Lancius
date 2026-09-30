#include "lancius/lancius_ir.h"
#include "lancius/lancius_error.h"

static int count_consumers(lancius_graph* g, const lancius_node* target) {
    int count = 0;
    uint32_t i, j;
    if (!g || !g->nodes || !target) return 0;
    for (i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n || !n->inputs) continue;
        for (j = 0; j < n->input_count; j++) {
            if (n->inputs[j] == target) count++;
        }
    }
    return count;
}

void lancius_optimize_fusion(lancius_graph* g) {
    if (!g || !g->nodes) return;
    int fused_count = 0;
    for (int i = (int)g->node_count - 1; i >= 0; i--) {
        lancius_node* n = g->nodes[i];
        if (!n) continue;
        if (n->op == LANCIUS_OP_RELU && n->input_count > 0 && n->inputs && n->inputs[0] && n->inputs[0]->op == LANCIUS_OP_CONV2D) {
            /* v12R1 fix: n->inputs is const-qualified — do not cast away const.
             * Fusion mutates the conv node, so resolve the non-const handle
             * through the graph's node array. */
            lancius_node* conv = NULL;
            for (uint32_t k = 0; k < g->node_count; k++) {
                if (g->nodes[k] == n->inputs[0]) { conv = g->nodes[k]; break; }
            }
            if (!conv) continue;
            /* Despot truth: stolen inputs array must hold 2 entries (was: OOB
             * read when conv had <2 inputs). */
            if (!conv->inputs || conv->input_count != 2 || !conv->inputs[0] || !conv->inputs[1]) continue;
            if (conv->ndim != 4 || n->ndim != 4) continue;
            if (count_consumers(g, conv) == 1) {
                n->op = LANCIUS_OP_CONV2D_RELU_FUSED;
                // V10S FIX: Steal the Conv2D's inputs array directly to avoid arena OOB write!
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
                conv->inputs = NULL;
                fused_count++;
            }
        }
    }
    /* Despot truth: do NOT clear errors on success — a prior sticky error
     * from a failed build must not be masked by a successful optimize pass. */
}

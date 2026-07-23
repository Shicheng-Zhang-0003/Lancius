#include "lancius/lancius_autodiff.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

static void accum_grad(lancius_graph* g, lancius_node** grad_map, uint32_t fwd_input_id, lancius_node* new_grad, lancius_node** fwd_to_full) {
    if (!new_grad) return;
    lancius_node* full_input = fwd_to_full[fwd_input_id];
    if (!full_input) return;

    bool exact_match = (new_grad->ndim == full_input->ndim);
    if (exact_match) {
        for(uint8_t i=0; i<new_grad->ndim; i++) {
            if(new_grad->shape[i] != full_input->shape[i]) { exact_match = false; break; }
        }
    }

    if (exact_match) {
        if (grad_map[fwd_input_id] == NULL) grad_map[fwd_input_id] = new_grad;
        else grad_map[fwd_input_id] = lancius_add(g, grad_map[fwd_input_id], new_grad);
        return;
    }

    if (new_grad->ndim == 2 && full_input->ndim == 2) {
        if (new_grad->shape[0] == 1 && new_grad->shape[1] == 1 && (full_input->shape[0] > 1 || full_input->shape[1] > 1)) {
            new_grad = lancius_broadcast(g, new_grad, full_input->shape[0], full_input->shape[1]);
        }
        else if ((new_grad->shape[0] > 1 || new_grad->shape[1] > 1) && full_input->shape[0] == 1 && full_input->shape[1] == 1) {
            new_grad = lancius_sum(g, new_grad);
        }
        else if (new_grad->shape[0] > 1 && new_grad->shape[1] == full_input->shape[1] && full_input->shape[0] == 1) {
            new_grad = lancius_sum_axis0(g, new_grad);
        }
        else if (new_grad->shape[1] > 1 && new_grad->shape[0] == full_input->shape[0] && full_input->shape[1] == 1) {
            new_grad = lancius_sum_axis1(g, new_grad);
        }
    }

    if (grad_map[fwd_input_id] == NULL) grad_map[fwd_input_id] = new_grad;
    else grad_map[fwd_input_id] = lancius_add(g, grad_map[fwd_input_id], new_grad);
}

lancius_training_graph* lancius_ir_autodiff(lancius_graph* fwd_g, lancius_node* loss_node) {
    if (!loss_node) return NULL; // Prevent segfault on malformed graphs
    lancius_training_graph* tg = (lancius_training_graph*)calloc(1, sizeof(lancius_training_graph));
    tg->graph = lancius_graph_create();
    tg->max_id = fwd_g->next_id;
    tg->grad_nodes = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));

    lancius_node** fwd_to_full = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));
    for(uint32_t i=0; i<fwd_g->node_count; i++) {
        lancius_node* old = fwd_g->nodes[i];
        lancius_node* n = NULL;
        const lancius_node* in0 = old->input_count > 0 ? fwd_to_full[old->inputs[0]->id] : NULL;
        const lancius_node* in1 = old->input_count > 1 ? fwd_to_full[old->inputs[1]->id] : NULL;

        switch(old->op) {
            case LANCIUS_OP_INPUT:
                if (old->ndim == 4) n = lancius_input_4d(tg->graph, old->shape[0], old->shape[1], old->shape[2], old->shape[3]);
                else n = lancius_input(tg->graph, old->shape[0], old->shape[1]);
                if(n) { n->runtime_data = old->runtime_data; lancius_runtime_sync_from_legacy(n); }
                break;
            case LANCIUS_OP_CONST: n = lancius_const(tg->graph, old->attr_val, old->shape[0], old->shape[1]); break;
            case LANCIUS_OP_ADD: n = lancius_add(tg->graph, in0, in1); break;
            case LANCIUS_OP_SUB: n = lancius_sub(tg->graph, in0, in1); break;
            case LANCIUS_OP_MUL: n = lancius_mul(tg->graph, in0, in1); break;
            case LANCIUS_OP_MATMUL: n = lancius_matmul(tg->graph, in0, in1); break;
            case LANCIUS_OP_RELU: n = lancius_relu(tg->graph, in0); break;
            case LANCIUS_OP_TRANSPOSE: n = lancius_transpose(tg->graph, in0); break;
            case LANCIUS_OP_SUM: n = lancius_sum(tg->graph, in0); break;
            case LANCIUS_OP_BROADCAST: n = lancius_broadcast(tg->graph, in0, old->shape[0], old->shape[1]); break;
            case LANCIUS_OP_SOFTMAX: n = lancius_softmax(tg->graph, in0); break;
            case LANCIUS_OP_CROSS_ENTROPY: n = lancius_cross_entropy(tg->graph, in0, in1); break;
            case LANCIUS_OP_PERMUTE: n = lancius_permute(tg->graph, in0, old->axes[0], old->axes[1], old->axes[2], old->axes[3]); break;
            case LANCIUS_OP_MATMUL_BATCHED: n = lancius_matmul_batched(tg->graph, in0, in1); break;
            case LANCIUS_OP_CONV2D: n = lancius_conv2d(tg->graph, in0, in1, old->stride, old->pad); break;
            case LANCIUS_OP_MAXPOOL2D: n = lancius_maxpool2d(tg->graph, in0, old->kernel_h, old->stride); break;
            case LANCIUS_OP_FLATTEN: n = lancius_flatten(tg->graph, in0); break;
            case LANCIUS_OP_RESHAPE: n = lancius_reshape(tg->graph, in0, old->ndim, old->shape[0], old->shape[1], old->shape[2], old->shape[3]); break;
            case LANCIUS_OP_NOP: n = NULL; break; // V9 Fix: Skip neutralized nodes
            case LANCIUS_OP_CONV2D_RELU_FUSED:
                // V10S FIX: Manually allocate to preserve the FUSED opcode!
                n = (lancius_node*)lancius_arena_alloc(tg->graph->arena, sizeof(lancius_node), 8);
                if (n) {
                    memset(n, 0, sizeof(lancius_node));
                    n->id = tg->graph->next_id++;
                    n->op = LANCIUS_OP_CONV2D_RELU_FUSED; // Crucial: Keep the fused opcode
                    n->ndim = 4;
                    n->input_count = 2;
                    n->inputs = (const lancius_node**)lancius_arena_alloc(tg->graph->arena, sizeof(lancius_node*) * 2, 8);
                    n->inputs[0] = in0; n->inputs[1] = in1;
                    n->shape[0] = old->shape[0]; n->shape[1] = old->shape[1]; n->shape[2] = old->shape[2]; n->shape[3] = old->shape[3];
                    n->kernel_h = old->kernel_h; n->kernel_w = old->kernel_w; n->stride = old->stride; n->pad = old->pad;
                    n->runtime_data = old->runtime_data;
                    lancius_runtime_sync_from_legacy(n);
                    // Track node in training graph
                    if (tg->graph->node_count >= tg->graph->node_cap) {
                        tg->graph->node_cap = tg->graph->node_cap == 0 ? 1024 : tg->graph->node_cap * 2;
                        tg->graph->nodes = (lancius_node**)realloc(tg->graph->nodes, sizeof(lancius_node*) * tg->graph->node_cap);
                    }
                    tg->graph->nodes[tg->graph->node_count++] = n;
                }
                break;
            default: fprintf(stderr, "[AUTODIFF FATAL] Unhandled op %d\n", old->op); break;
        }
        fwd_to_full[old->id] = n;
    }

    lancius_node** grad_map = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));
    grad_map[loss_node->id] = lancius_const(tg->graph, 1.0, 1, 1);

    for (int i = fwd_g->node_count - 1; i >= 0; i--) {
        lancius_node* fwd_n = fwd_g->nodes[i];
        lancius_node* grad_out = grad_map[fwd_n->id];
        if (!grad_out) continue;
        if (fwd_n->op == LANCIUS_OP_INPUT || fwd_n->op == LANCIUS_OP_CONST) continue;
        if (fwd_n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes

        if (fwd_n->op == LANCIUS_OP_ADD) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUB) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            lancius_node* neg = lancius_const(tg->graph, -1.0, grad_out->shape[0], grad_out->shape[1]);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_mul(tg->graph, grad_out, neg), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MUL) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            lancius_node* gA = grad_out;
            lancius_node* gB = grad_out;

            if (gA && gA->ndim == 2 && gA->shape[0] == 1 && gA->shape[1] == 1 && (A->shape[0] > 1 || A->shape[1] > 1)) {
                gA = lancius_broadcast(tg->graph, gA, A->shape[0], A->shape[1]);
            }
            if (gB && gB->ndim == 2 && gB->shape[0] == 1 && gB->shape[1] == 1 && (B->shape[0] > 1 || B->shape[1] > 1)) {
                gB = lancius_broadcast(tg->graph, gB, B->shape[0], B->shape[1]);
            }

            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_mul(tg->graph, gA, B), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_mul(tg->graph, gB, A), fwd_to_full);
        
        } else if (fwd_n->op == LANCIUS_OP_RMSNORM || fwd_n->op == LANCIUS_OP_SWIGLU || fwd_n->op == LANCIUS_OP_GQA) {
            // v10S HONESTY: Fail loudly instead of passing mathematically incorrect gradients.
            fprintf(stderr, "[AUTODIFF FATAL] Transformer backward pass not implemented in v10S.\n");
            free(grad_map); free(fwd_to_full); 
            lancius_graph_destroy(tg->graph); free(tg); 
            return NULL;
} else if (fwd_n->op == LANCIUS_OP_MATMUL) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_matmul(tg->graph, grad_out, lancius_transpose(tg->graph, B)), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_matmul(tg->graph, lancius_transpose(tg->graph, A), grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_RELU) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_relu_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUM) {
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            if (full_input && full_input->ndim == 4 && grad_out->ndim == 2 && grad_out->shape[0] == 1 && grad_out->shape[1] == 1) {
                lancius_node* bcast = lancius_broadcast_4d(tg->graph, grad_out, full_input->shape[0], full_input->shape[1], full_input->shape[2], full_input->shape[3]);
                accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, bcast, fwd_to_full);
            } else {
                accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            }
        } else if (fwd_n->op == LANCIUS_OP_BROADCAST) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_TRANSPOSE) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_transpose(tg->graph, grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MATMUL_BATCHED) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_PERMUTE) {
            uint32_t inv_axes[4] = {0,0,0,0};
            for(int i=0; i<4; i++) inv_axes[fwd_n->axes[i]] = i;
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_permute(tg->graph, grad_out, inv_axes[0], inv_axes[1], inv_axes[2], inv_axes[3]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_CROSS_ENTROPY) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* Y = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_cross_entropy_bwd(tg->graph, A, Y, grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SOFTMAX) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_softmax_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->id]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_FLATTEN) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_reshape(tg->graph, grad_out, A->ndim, A->shape[0], A->shape[1], A->shape[2], A->shape[3]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MAXPOOL2D) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_maxpool2d_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id], fwd_n->kernel_h, fwd_n->stride), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_CONV2D) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* W = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_conv2d_bwd(tg->graph, grad_out, A, W, fwd_n->stride, fwd_n->pad), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_conv2d_bwd_w(tg->graph, grad_out, A, fwd_n->kernel_h, fwd_n->kernel_w, fwd_n->stride, fwd_n->pad), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            // V9 Fix: Backward pass for fused Conv2D + ReLU
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* W = fwd_to_full[fwd_n->inputs[1]->id];
            lancius_node* fused_out = fwd_to_full[fwd_n->id];
            lancius_node* masked_grad = lancius_relu_bwd(tg->graph, grad_out, fused_out);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_conv2d_bwd(tg->graph, masked_grad, A, W, fwd_n->stride, fwd_n->pad), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_conv2d_bwd_w(tg->graph, masked_grad, A, fwd_n->kernel_h, fwd_n->kernel_w, fwd_n->stride, fwd_n->pad), fwd_to_full);
        }
    }

    for(uint32_t i=0; i<fwd_g->next_id; i++) tg->grad_nodes[i] = grad_map[i];
    tg->loss_node = fwd_to_full[loss_node->id];

    free(grad_map); free(fwd_to_full);
    return tg;
}

void lancius_training_graph_destroy(lancius_training_graph* tg) {
    if (!tg) return;
    free(tg->grad_nodes);
    lancius_graph_destroy(tg->graph);
    free(tg);
}

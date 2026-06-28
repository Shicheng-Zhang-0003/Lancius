#include "basis/v8_autodiff.h"
#include <stdlib.h>
#include <string.h>

static void accum_grad(v8_graph* g, v8_node** grad_map, uint32_t fwd_input_id, v8_node* new_grad, v8_node** fwd_to_full) {
    if (!new_grad) return;
    v8_node* full_input = fwd_to_full[fwd_input_id];
    if (new_grad->rows == 1 && new_grad->cols == 1 && (full_input->rows > 1 || full_input->cols > 1)) {
        new_grad = v8_broadcast(g, new_grad, full_input->rows, full_input->cols);
    } else if ((new_grad->rows > 1 || new_grad->cols > 1) && full_input->rows == 1 && full_input->cols == 1) {
        new_grad = v8_sum(g, new_grad);
    } else if (new_grad->rows > 1 && new_grad->cols == full_input->cols && full_input->rows == 1) {
        new_grad = v8_sum_axis0(g, new_grad);
    } else if (new_grad->cols > 1 && new_grad->rows == full_input->rows && full_input->cols == 1) {
        new_grad = v8_sum_axis1(g, new_grad);
    }

    if (grad_map[fwd_input_id] == NULL) grad_map[fwd_input_id] = new_grad;
    else grad_map[fwd_input_id] = v8_add(g, grad_map[fwd_input_id], new_grad);
}

v8_training_graph* v8_ir_autodiff(v8_graph* fwd_g, v8_node* loss_node) {
    v8_training_graph* tg = (v8_training_graph*)calloc(1, sizeof(v8_training_graph));
    tg->graph = v8_graph_create();
    tg->max_id = fwd_g->next_id;
    tg->grad_nodes = (v8_node**)calloc(fwd_g->next_id, sizeof(v8_node*));

    v8_node** fwd_to_full = (v8_node**)calloc(fwd_g->next_id, sizeof(v8_node*));
    for(uint32_t i=0; i<fwd_g->node_count; i++) {
        v8_node* old = fwd_g->nodes[i];
        v8_node* n = NULL;
        const v8_node* in0 = old->input_count > 0 ? fwd_to_full[old->inputs[0]->id] : NULL;
        const v8_node* in1 = old->input_count > 1 ? fwd_to_full[old->inputs[1]->id] : NULL;
        switch(old->op) {
            case V8_OP_INPUT: n = v8_input(tg->graph, old->rows, old->cols); n->runtime_data = old->runtime_data; break;
            case V8_OP_CONST: n = v8_const(tg->graph, old->attr_val, old->rows, old->cols); break;
            case V8_OP_ADD: n = v8_add(tg->graph, in0, in1); break;
            case V8_OP_SUB: n = v8_sub(tg->graph, in0, in1); break;
            case V8_OP_MUL: n = v8_mul(tg->graph, in0, in1); break;
            case V8_OP_MATMUL: n = v8_matmul(tg->graph, in0, in1); break;
            case V8_OP_RELU: n = v8_relu(tg->graph, in0); break;
            case V8_OP_TRANSPOSE: n = v8_transpose(tg->graph, in0); break;
            case V8_OP_SUM: n = v8_sum(tg->graph, in0); break;
            case V8_OP_BROADCAST: n = v8_broadcast(tg->graph, in0, old->rows, old->cols); break;
            case V8_OP_SOFTMAX: n = v8_softmax(tg->graph, in0); break;
            default: break;
        }
        fwd_to_full[old->id] = n;
    }

    v8_node** grad_map = (v8_node**)calloc(fwd_g->next_id, sizeof(v8_node*));
    grad_map[loss_node->id] = v8_const(tg->graph, 1.0, 1, 1);

    for (int i = fwd_g->node_count - 1; i >= 0; i--) {
        v8_node* fwd_n = fwd_g->nodes[i];
        v8_node* grad_out = grad_map[fwd_n->id];
        if (!grad_out) continue;
        if (fwd_n->op == V8_OP_INPUT || fwd_n->op == V8_OP_CONST) continue;

        if (fwd_n->op == V8_OP_ADD) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == V8_OP_SUB) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            v8_node* neg = v8_const(tg->graph, -1.0, grad_out->rows, grad_out->cols);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, v8_mul(tg->graph, grad_out, neg), fwd_to_full);
        } else if (fwd_n->op == V8_OP_MUL) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, v8_mul(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[1]->id]), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, v8_mul(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id]), fwd_to_full);
        } else if (fwd_n->op == V8_OP_MATMUL) {
            v8_node* A = fwd_to_full[fwd_n->inputs[0]->id]; v8_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, v8_matmul(tg->graph, grad_out, v8_transpose(tg->graph, B)), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, v8_matmul(tg->graph, v8_transpose(tg->graph, A), grad_out), fwd_to_full);
        } else if (fwd_n->op == V8_OP_RELU) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, v8_relu_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id]), fwd_to_full);
        } else if (fwd_n->op == V8_OP_SUM) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == V8_OP_BROADCAST) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == V8_OP_TRANSPOSE) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, v8_transpose(tg->graph, grad_out), fwd_to_full);
        } else if (fwd_n->op == V8_OP_SOFTMAX) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, v8_softmax_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->id]), fwd_to_full);
        }
    }

    for(uint32_t i=0; i<fwd_g->next_id; i++) tg->grad_nodes[i] = grad_map[i];
    tg->loss_node = fwd_to_full[loss_node->id];
    free(grad_map); free(fwd_to_full);
    return tg;
}

void v8_training_graph_destroy(v8_training_graph* tg) {
    if (!tg) return;
    free(tg->grad_nodes);
    v8_graph_destroy(tg->graph);
    free(tg);
}

#include "basis/l6_learning/autodiff.h"
#include <stdlib.h>
#include <string.h>

static void accum_grad(basis_ir_graph* g, basis_ir_node** grad_map, uint32_t fwd_input_id, basis_ir_node* new_grad, basis_ir_node** fwd_to_full) {
    if (!new_grad) return;
    basis_ir_node* full_input = fwd_to_full[fwd_input_id];

    // Forward Broadcast: 1x1 grad -> MxN input
    if (new_grad->rows == 1 && new_grad->cols == 1 && (full_input->rows > 1 || full_input->cols > 1)) {
        new_grad = basis_ir_broadcast(g, new_grad, full_input->rows, full_input->cols);
    }
    // CRITICAL FIX: Reverse Reduction: MxN grad -> 1x1 input (e.g. from broadcasting a scalar bias)
    else if ((new_grad->rows > 1 || new_grad->cols > 1) && full_input->rows == 1 && full_input->cols == 1) {
        new_grad = basis_ir_sum(g, new_grad);
    }
    if (grad_map[fwd_input_id] == NULL) {
        grad_map[fwd_input_id] = new_grad;
    } else {
        grad_map[fwd_input_id] = basis_ir_add(g, grad_map[fwd_input_id], new_grad);
    }
}

basis_training_graph* basis_ir_autodiff(basis_ir_graph* fwd_g, basis_ir_node* loss_node) {
    basis_training_graph* tg = (basis_training_graph*)calloc(1, sizeof(basis_training_graph));
    tg->graph = basis_ir_graph_create();
    tg->max_id = fwd_g->next_id;
    tg->grad_nodes = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));

    basis_ir_node** fwd_to_full = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));
    for(uint32_t i=0; i<fwd_g->node_count; i++) {
        basis_ir_node* old = fwd_g->nodes[i];
        basis_ir_node* n = NULL;
        const basis_ir_node* in0 = old->input_count > 0 ? fwd_to_full[old->inputs[0]->id] : NULL;
        const basis_ir_node* in1 = old->input_count > 1 ? fwd_to_full[old->inputs[1]->id] : NULL;
        switch(old->op) {
            case BASIS_IR_OP_INPUT: n = basis_ir_input(tg->graph, old->rows, old->cols); break;
            case BASIS_IR_OP_CONST: n = basis_ir_const(tg->graph, old->attr_val, old->rows, old->cols); break;
            case BASIS_IR_OP_ADD: n = basis_ir_add(tg->graph, in0, in1); break;
            case BASIS_IR_OP_SUB: n = basis_ir_sub(tg->graph, in0, in1); break;
            case BASIS_IR_OP_MUL: n = basis_ir_mul(tg->graph, in0, in1); break;
            case BASIS_IR_OP_MATMUL: n = basis_ir_matmul(tg->graph, in0, in1); break;
            case BASIS_IR_OP_RELU: n = basis_ir_relu(tg->graph, in0); break;
            case BASIS_IR_OP_TRANSPOSE: n = basis_ir_transpose(tg->graph, in0); break;
            case BASIS_IR_OP_SUM: n = basis_ir_sum(tg->graph, in0); break;
            case BASIS_IR_OP_BROADCAST: n = basis_ir_broadcast(tg->graph, in0, old->rows, old->cols); break;
            default: break;
        }
        fwd_to_full[old->id] = n;
    }

    basis_ir_node** grad_map = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));
    grad_map[loss_node->id] = basis_ir_const(tg->graph, 1.0, 1, 1);

    for (int i = fwd_g->node_count - 1; i >= 0; i--) {
        basis_ir_node* fwd_n = fwd_g->nodes[i];
        basis_ir_node* grad_out = grad_map[fwd_n->id];
        if (!grad_out) continue;
        if (fwd_n->op == BASIS_IR_OP_INPUT || fwd_n->op == BASIS_IR_OP_CONST) continue;

        if (fwd_n->op == BASIS_IR_OP_ADD) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_SUB) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            basis_ir_node* neg = basis_ir_const(tg->graph, -1.0, grad_out->rows, grad_out->cols);
            basis_ir_node* gB = basis_ir_mul(tg->graph, grad_out, neg);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, gB, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_MUL) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            basis_ir_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_mul(tg->graph, grad_out, B), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, basis_ir_mul(tg->graph, grad_out, A), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_MATMUL) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            basis_ir_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            basis_ir_node* B_T = basis_ir_transpose(tg->graph, B);
            basis_ir_node* A_T = basis_ir_transpose(tg->graph, A);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_matmul(tg->graph, grad_out, B_T), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, basis_ir_matmul(tg->graph, A_T, grad_out), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_RELU) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_relu_bwd(tg->graph, grad_out, A), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_SUM) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_BROADCAST) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_sum(tg->graph, grad_out), fwd_to_full);
        }
    }

    for(uint32_t i=0; i<fwd_g->next_id; i++) tg->grad_nodes[i] = grad_map[i];
    tg->loss_node = fwd_to_full[loss_node->id];

    free(grad_map); free(fwd_to_full);
    return tg;
}

void basis_training_graph_destroy(basis_training_graph* tg) {
    if (!tg) return;
    free(tg->grad_nodes);
    basis_ir_graph_destroy(tg->graph);
    free(tg);
}

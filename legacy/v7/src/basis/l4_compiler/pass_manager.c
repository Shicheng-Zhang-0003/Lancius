#include "basis/l4_compiler/pass_manager.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

static void basis_ir_dce(basis_ir_graph* g) {
    if (!g || g->node_count == 0) return;
    basis_ir_node* out = g->nodes[g->node_count - 1];
    bool* reachable = (bool*)calloc(g->next_id, sizeof(bool));
    if (!reachable) return;
    reachable[out->id] = true;
    for (int i = (int)g->node_count - 1; i >= 0; i--) {
        basis_ir_node* n = g->nodes[i];
        if (reachable[n->id]) {
            for (uint32_t j = 0; j < n->input_count; j++) {
                if (n->inputs[j]) reachable[n->inputs[j]->id] = true;
            }
        }
    }
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (reachable[g->nodes[i]->id]) g->nodes[write_idx++] = g->nodes[i];
    }
    g->node_count = write_idx;
    free(reachable);
}

basis_ir_graph* basis_ir_run_passes(basis_ir_graph* old_g) {
    if (!old_g) return NULL;
    basis_ir_graph* new_g = basis_ir_graph_create();
    basis_ir_node** map = (basis_ir_node**)calloc(old_g->next_id, sizeof(basis_ir_node*));
    if (!map) { basis_ir_graph_destroy(new_g); return NULL; }

    for (uint32_t i = 0; i < old_g->node_count; i++) {
        basis_ir_node* old_n = old_g->nodes[i];
        basis_ir_node* new_n = NULL;

        const basis_ir_node* in0 = (old_n->input_count > 0 && old_n->inputs[0]) ? map[old_n->inputs[0]->id] : NULL;
        const basis_ir_node* in1 = (old_n->input_count > 1 && old_n->inputs[1]) ? map[old_n->inputs[1]->id] : NULL;

        // Pass 1: Constant Folding
        if (old_n->op == BASIS_IR_OP_ADD && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val + in1->attr_val, old_n->rows, old_n->cols);
        } else if (old_n->op == BASIS_IR_OP_MUL && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val * in1->attr_val, old_n->rows, old_n->cols);
        } else if (old_n->op == BASIS_IR_OP_SUB && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val - in1->attr_val, old_n->rows, old_n->cols);
        }
        // Pass 2: Identity Elimination
        else if (old_n->op == BASIS_IR_OP_ADD && in0 && in1) {
            if (in0->op == BASIS_IR_OP_CONST && in0->attr_val == 0.0) new_n = (basis_ir_node*)in1;
            else if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 0.0) new_n = (basis_ir_node*)in0;
        } else if (old_n->op == BASIS_IR_OP_MUL && in0 && in1) {
            if (in0->op == BASIS_IR_OP_CONST && in0->attr_val == 1.0) new_n = (basis_ir_node*)in1;
            else if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 1.0) new_n = (basis_ir_node*)in0;
        } else if (old_n->op == BASIS_IR_OP_SUB && in0 && in1) {
            if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 0.0) new_n = (basis_ir_node*)in0;
        }

        // Fallback: Rebuild Node
        if (!new_n) {
            switch (old_n->op) {
                case BASIS_IR_OP_INPUT:
                    new_n = basis_ir_input(new_g, old_n->rows, old_n->cols);
                    // CRITICAL FIX: Share pointer so external training loops can update weights
                    new_n->runtime_data = old_n->runtime_data;
                    break;
                case BASIS_IR_OP_CONST:
                    // CRITICAL FIX: basis_ir_const already allocates and fills persistent memory
                    new_n = basis_ir_const(new_g, old_n->attr_val, old_n->rows, old_n->cols);
                    break;
                case BASIS_IR_OP_ADD: new_n = basis_ir_add(new_g, in0, in1); break;
                case BASIS_IR_OP_SUB: new_n = basis_ir_sub(new_g, in0, in1); break;
                case BASIS_IR_OP_MUL: new_n = basis_ir_mul(new_g, in0, in1); break;
                case BASIS_IR_OP_MATMUL: new_n = basis_ir_matmul(new_g, in0, in1); break;
                case BASIS_IR_OP_RELU: new_n = basis_ir_relu(new_g, in0); break;
                case BASIS_IR_OP_TRANSPOSE: new_n = basis_ir_transpose(new_g, in0); break;
                case BASIS_IR_OP_SOFTMAX: new_n = basis_ir_softmax(new_g, in0); break;
                default: break;
            }
        }
        if (new_n) map[old_n->id] = new_n;
    }
    free(map);
    basis_ir_dce(new_g);
    return new_g;
}

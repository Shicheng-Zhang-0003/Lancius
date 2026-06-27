#include "basis/core/ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void graph_track_node(basis_ir_graph* g, basis_ir_node* n) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 1024 : g->node_cap * 2;
        g->nodes = (basis_ir_node**)realloc(g->nodes, sizeof(basis_ir_node*) * g->node_cap);
    }
    g->nodes[g->node_count++] = n;
}

basis_ir_graph* basis_ir_graph_create(void) {
    basis_ir_graph* g = (basis_ir_graph*)calloc(1, sizeof(basis_ir_graph));
    if (!g) return NULL;
    g->arena = basis_arena_create(1024 * 1024 * 16); // 16MB blocks for massive graphs
    g->node_cap = 1024;
    g->nodes = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * g->node_cap);
    return g;
}

void basis_ir_graph_destroy(basis_ir_graph* g) {
    if (!g) return;
    free(g->nodes);             // Free the tracking metadata
    basis_arena_destroy(g->arena); // O(1) destruction of ALL nodes!
    free(g);
}

basis_ir_node* basis_ir_input(basis_ir_graph* g, size_t r, size_t c) {
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_INPUT;
    n->rows = r; n->cols = c;
    n->input_count = 0; n->inputs = NULL;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_const(basis_ir_graph* g, double val, size_t r, size_t c) {
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_CONST;
    n->rows = r; n->cols = c;
    n->input_count = 0; n->inputs = NULL;
    n->attr_val = val;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_add(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL; // Shape Inference
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_ADD;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_matmul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->cols != b->rows) return NULL; // Shape Inference
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_MATMUL;
    n->rows = a->rows; n->cols = b->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_relu(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_RELU;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}


basis_ir_node* basis_ir_mul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_MUL;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_sub(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SUB;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_transpose(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_TRANSPOSE;
    n->rows = a->cols; n->cols = a->rows;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_softmax(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SOFTMAX;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}


basis_ir_node* basis_ir_sum(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SUM;
    n->rows = 1; n->cols = 1;
    n->input_count = 1; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}
basis_ir_node* basis_ir_broadcast(basis_ir_graph* g, const basis_ir_node* a, size_t r, size_t c) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_BROADCAST;
    n->rows = r; n->cols = c;
    n->input_count = 1; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}
basis_ir_node* basis_ir_relu_bwd(basis_ir_graph* g, const basis_ir_node* grad, const basis_ir_node* fwd_a) {
    if (!grad || !fwd_a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_RELU_BWD;
    n->rows = grad->rows; n->cols = grad->cols;
    n->input_count = 2; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = grad; n->inputs[1] = fwd_a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}

static const char* ir_op_str(basis_ir_opcode op) {
    switch(op) {
        case BASIS_IR_OP_INPUT: return "INPUT";
        case BASIS_IR_OP_CONST: return "CONST";
        case BASIS_IR_OP_ADD: return "ADD";
        case BASIS_IR_OP_MATMUL: return "MATMUL";
        case BASIS_IR_OP_RELU: return "RELU";
        case BASIS_IR_OP_MUL: return "MUL";
        case BASIS_IR_OP_SUB: return "SUB";
        case BASIS_IR_OP_TRANSPOSE: return "TRANSPOSE";
        case BASIS_IR_OP_SOFTMAX: return "SOFTMAX";
        case BASIS_IR_OP_SUM: return "SUM";
        case BASIS_IR_OP_BROADCAST: return "BROADCAST";
        case BASIS_IR_OP_RELU_BWD: return "RELU_BWD";
        default: return "UNKNOWN";
    }
}

void basis_ir_print(const basis_ir_graph* g) {
    printf("=== V7 Immutable Graph (%u nodes) ===\n", g->node_count);
    for (uint32_t i = 0; i < g->node_count; i++) {
        const basis_ir_node* n = g->nodes[i];
        printf("%%%u = %-8s [%3zu x %3zu] | Inputs: ", n->id, ir_op_str(n->op), n->rows, n->cols);
        if (n->input_count == 0) {
            if (n->op == BASIS_IR_OP_CONST) printf("val=%.2f", n->attr_val);
            else printf("none");
        } else {
            for(uint32_t j=0; j<n->input_count; j++) printf("%%%u ", n->inputs[j]->id);
        }
        printf("\n");
    }
}

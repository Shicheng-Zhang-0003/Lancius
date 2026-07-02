#include "basis/stage6_ir/graph.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static char* ir_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* c = (char*)malloc(len);
    if (c) memcpy(c, s, len);
    return c;
}

static basis_node* graph_alloc_node(basis_graph* g, basis_op_type op, size_t r, size_t c, const char* name) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 16 : g->node_cap * 2;
        g->nodes = (basis_node*)realloc(g->nodes, sizeof(basis_node) * g->node_cap);
    }
    basis_node* n = &g->nodes[g->node_count++];
    n->id = (int)g->node_count - 1;
    n->op = op;
    n->input_ids[0] = -1;
    n->input_ids[1] = -1;
    n->rows = r;
    n->cols = c;
    n->name = ir_strdup(name);
    return n;
}

basis_graph* basis_graph_new(void) {
    basis_graph* g = (basis_graph*)calloc(1, sizeof(basis_graph));
    return g;
}

basis_node* basis_graph_input(basis_graph* g, const char* name, size_t r, size_t c) {
    return graph_alloc_node(g, BASIS_OP_INPUT, r, c, name);
}

basis_node* basis_graph_matmul(basis_graph* g, basis_node* a, basis_node* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "[IR ERROR] MatMul shape mismatch: [%zux%zu] x [%zux%zu]\n", a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_MATMUL, a->rows, b->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = b->id;
    return n;
}

basis_node* basis_graph_add(basis_graph* g, basis_node* a, basis_node* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "[IR ERROR] Add shape mismatch\n");
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_ADD, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = b->id;
    return n;
}

basis_node* basis_graph_broadcast_add(basis_graph* g, basis_node* a, basis_node* bias) {
    if (bias->rows != 1 || bias->cols != a->cols) {
        fprintf(stderr, "[IR ERROR] Broadcast Add shape mismatch\n");
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_BROADCAST_ADD, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = bias->id;
    return n;
}

basis_node* basis_graph_relu(basis_graph* g, basis_node* a) {
    basis_node* n = graph_alloc_node(g, BASIS_OP_RELU, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    return n;
}

basis_node* basis_graph_tanh(basis_graph* g, basis_node* a) {
    basis_node* n = graph_alloc_node(g, BASIS_OP_TANH, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    return n;
}

static const char* op_to_string(basis_op_type op) {
    switch(op) {
        case BASIS_OP_INPUT: return "INPUT";
        case BASIS_OP_MATMUL: return "MATMUL";
        case BASIS_OP_ADD: return "ADD";
        case BASIS_OP_BROADCAST_ADD: return "BROADCAST_ADD";
        case BASIS_OP_RELU: return "RELU";
        case BASIS_OP_TANH: return "TANH";
        default: return "UNKNOWN";
    }
}

void basis_graph_print(basis_graph* g) {
    printf("=== BASIS Static Graph IR ===\n");
    for (size_t i = 0; i < g->node_count; i++) {
        basis_node* n = &g->nodes[i];
        printf("Node %2d | %-15s | Shape: [%3zu x %3zu] | Inputs: ",
               n->id, op_to_string(n->op), n->rows, n->cols);
        if (n->op == BASIS_OP_INPUT) {
            printf("None (Name: \"%s\")", n->name);
        } else {
            printf("[%d, %d]", n->input_ids[0], n->input_ids[1]);
        }
        printf("\n");
    }
    printf("=============================\n");
}

void basis_graph_free(basis_graph* g) {
    if (!g) return;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].name) free(g->nodes[i].name);
    }
    if (g->nodes) free(g->nodes);
    free(g);
}

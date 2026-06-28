#include "basis/v8_ir.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void track(v8_graph* g, v8_node* n) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 1024 : g->node_cap * 2;
        g->nodes = (v8_node**)realloc(g->nodes, sizeof(v8_node*) * g->node_cap);
    }
    g->nodes[g->node_count++] = n;
}

v8_graph* v8_graph_create(void) {
    v8_graph* g = (v8_graph*)calloc(1, sizeof(v8_graph));
    if (!g) return NULL;
    g->arena = v8_arena_create(16 * 1024 * 1024);
    g->node_cap = 1024;
    g->nodes = (v8_node**)malloc(sizeof(v8_node*) * g->node_cap);
    return g;
}

void v8_graph_destroy(v8_graph* g) {
    if (!g) return;
    free(g->nodes);
    v8_arena_destroy(g->arena);
    free(g);
}

static v8_node* alloc_node(v8_graph* g, v8_opcode op, size_t r, size_t c, uint32_t in_count) {
    v8_node* n = (v8_node*)v8_arena_alloc(g->arena, sizeof(v8_node), 8);
    if (!n) return NULL;
    memset(n, 0, sizeof(v8_node)); // Zero-state contract
    n->id = g->next_id++;
    n->op = op;
    n->rows = r;
    n->cols = c;
    n->input_count = in_count;
    if (in_count > 0) {
        n->inputs = (const v8_node**)v8_arena_alloc(g->arena, sizeof(v8_node*) * in_count, 8);
    }
    track(g, n);
    return n;
}

v8_node* v8_input(v8_graph* g, size_t r, size_t c) {
    return alloc_node(g, V8_OP_INPUT, r, c, 0);
}

v8_node* v8_const(v8_graph* g, double val, size_t r, size_t c) {
    v8_node* n = alloc_node(g, V8_OP_CONST, r, c, 0);
    if (n) {
        n->attr_val = val;
    }
    return n;
}

v8_node* v8_add(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    v8_node* n = alloc_node(g, V8_OP_ADD, a->rows, a->cols, 2);
    if (n) {
        n->inputs[0] = a;
        n->inputs[1] = b;
    }
    return n;
}

v8_node* v8_sub(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUB, a->rows, a->cols, 2);
    if (n) {
        n->inputs[0] = a;
        n->inputs[1] = b;
    }
    return n;
}

v8_node* v8_mul(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    v8_node* n = alloc_node(g, V8_OP_MUL, a->rows, a->cols, 2);
    if (n) {
        n->inputs[0] = a;
        n->inputs[1] = b;
    }
    return n;
}

v8_node* v8_matmul(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    v8_node* n = alloc_node(g, V8_OP_MATMUL, a->rows, b->cols, 2);
    if (n) {
        n->inputs[0] = a;
        n->inputs[1] = b;
    }
    return n;
}

v8_node* v8_relu(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_RELU, a->rows, a->cols, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_softmax(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SOFTMAX, a->rows, a->cols, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_sum(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM, 1, 1, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_broadcast(v8_graph* g, const v8_node* a, size_t r, size_t c) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_BROADCAST, r, c, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_transpose(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_TRANSPOSE, a->cols, a->rows, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_relu_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_a) {
    if (!grad || !fwd_a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_RELU_BWD, grad->rows, grad->cols, 2);
    if (n) {
        n->inputs[0] = grad;
        n->inputs[1] = fwd_a;
    }
    return n;
}

v8_node* v8_softmax_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_y) {
    if (!grad || !fwd_y) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SOFTMAX_BWD, grad->rows, grad->cols, 2);
    if (n) {
        n->inputs[0] = grad;
        n->inputs[1] = fwd_y;
    }
    return n;
}

v8_node* v8_sum_axis0(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM_AXIS0, 1, a->cols, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

v8_node* v8_sum_axis1(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM_AXIS1, a->rows, 1, 1);
    if (n) {
        n->inputs[0] = a;
    }
    return n;
}

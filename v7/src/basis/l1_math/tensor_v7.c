#include "basis/l1_math/tensor_v7.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static basis_v7_tensor* create_v7_tensor(basis_ir_graph* g, basis_ir_node* n) {
    // Allocate the metadata struct on the arena so it's freed in O(1)
    basis_v7_tensor* t = (basis_v7_tensor*)basis_arena_alloc(g->arena, sizeof(basis_v7_tensor), 8);
    t->rows = n->rows;
    t->cols = n->cols;
    t->node = n;
    t->graph = g;
    return t;
}

basis_v7_tensor* basis_v7_input(basis_ir_graph* g, size_t r, size_t c, const double* initial_data) {
    basis_ir_node* n = basis_ir_input(g, r, c);
    size_t elements = r * c;
    // Allocate data buffer on the arena!
    n->runtime_data = (double*)basis_arena_alloc(g->arena, elements * sizeof(double), 32);
    if (initial_data) {
        memcpy(n->runtime_data, initial_data, elements * sizeof(double));
    } else {
        memset(n->runtime_data, 0, elements * sizeof(double));
    }
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_const(basis_ir_graph* g, double val, size_t r, size_t c) {
    basis_ir_node* n = basis_ir_const(g, val, r, c);
    if (!n) return NULL;
    // runtime_data is now allocated and populated by basis_ir_const
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_add(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_add(g, a->node, b->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_matmul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_matmul(g, a->node, b->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_relu(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_relu(g, a->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}


basis_v7_tensor* basis_v7_mul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_mul(g, a->node, b->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_sub(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_sub(g, a->node, b->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_transpose(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_transpose(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_softmax(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_softmax(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}


basis_v7_tensor* basis_v7_sum(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_sum(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_broadcast(basis_ir_graph* g, basis_v7_tensor* a, size_t r, size_t c) {
    basis_ir_node* n = basis_ir_broadcast(g, a->node, r, c); if (!n) return NULL; return create_v7_tensor(g, n);
}

void basis_v7_execute(basis_ir_graph* g) {
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        size_t elements = n->rows * n->cols;

        if (n->op == BASIS_IR_OP_INPUT || n->op == BASIS_IR_OP_CONST) continue; // Already populated

        // Allocate intermediate activation buffer on the Arena (32-byte aligned for AVX)
        n->runtime_data = (double*)basis_arena_alloc(g->arena, elements * sizeof(double), 32);

        if (n->op == BASIS_IR_OP_ADD) {
            double* a = n->inputs[0]->runtime_data;
            double* b = n->inputs[1]->runtime_data;
            double* out = n->runtime_data;
            for(size_t k=0; k<elements; k++) out[k] = a[k] + b[k];
        }
        else if (n->op == BASIS_IR_OP_MATMUL) {
            double* a = n->inputs[0]->runtime_data;
            double* b = n->inputs[1]->runtime_data;
            double* out = n->runtime_data;
            size_t M = n->inputs[0]->rows;
            size_t K = n->inputs[0]->cols;
            size_t N = n->inputs[1]->cols;

            for(size_t i=0; i<M; i++) {
                for(size_t j=0; j<N; j++) {
                    double sum = 0.0;
                    for(size_t k=0; k<K; k++) {
                        sum += a[i*K + k] * b[k*N + j];
                    }
                    out[i*N + j] = sum;
                }
            }
        }
        else if (n->op == BASIS_IR_OP_RELU) {
            double* a = n->inputs[0]->runtime_data;
            double* out = n->runtime_data;
            for(size_t k=0; k<elements; k++) out[k] = a[k] > 0.0 ? a[k] : 0.0;
        }
    }
}

void basis_v7_tensor_print(basis_v7_tensor* t, const char* name) {
    if (!t || !t->node || !t->node->runtime_data) {
        printf("Tensor %s: Not yet executed.\n", name);
        return;
    }
    printf("Tensor %s (%zu x %zu):\n", name, t->rows, t->cols);
    for (size_t i = 0; i < t->rows; i++) {
        printf("  [");
        for (size_t j = 0; j < t->cols; j++) {
            printf(" %8.4f ", t->node->runtime_data[i * t->cols + j]);
        }
        printf("]\n");
    }
}

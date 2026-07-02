#ifndef BASIS_V7_TENSOR_H
#define BASIS_V7_TENSOR_H

#include "basis/core/ir.h"
#include <stddef.h>

// Lightweight V7 Tensor: Just metadata and a link to the IR Graph
typedef struct {
    size_t rows;
    size_t cols;
    basis_ir_node* node;
    basis_ir_graph* graph;
} basis_v7_tensor;

// Graph Building (Deferred Execution)
basis_v7_tensor* basis_v7_input(basis_ir_graph* g, size_t r, size_t c, const double* initial_data);
basis_v7_tensor* basis_v7_const(basis_ir_graph* g, double val, size_t r, size_t c);
basis_v7_tensor* basis_v7_add(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_matmul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_mul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_sub(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_transpose(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_sum(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_broadcast(basis_ir_graph* g, basis_v7_tensor* a, size_t r, size_t c);
basis_v7_tensor* basis_v7_softmax(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_relu(basis_ir_graph* g, basis_v7_tensor* a);

// Execution & Teardown
void basis_v7_execute(basis_ir_graph* g);
void basis_v7_tensor_print(basis_v7_tensor* t, const char* name);

#endif // BASIS_V7_TENSOR_H

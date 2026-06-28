#ifndef BASIS_V7_IR_H
#define BASIS_V7_IR_H

#include "basis/core/memory.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    BASIS_IR_OP_INPUT,
    BASIS_IR_OP_CONST,
    BASIS_IR_OP_ADD,
    BASIS_IR_OP_SUB,
    BASIS_IR_OP_MUL,
    BASIS_IR_OP_MATMUL,
    BASIS_IR_OP_RELU,
    BASIS_IR_OP_TANH,
    BASIS_IR_OP_TRANSPOSE,
    BASIS_IR_OP_SOFTMAX,
    BASIS_IR_OP_SUM,
    BASIS_IR_OP_BROADCAST,
    BASIS_IR_OP_RELU_BWD,
    BASIS_IR_OP_SUM_AXIS0,
    BASIS_IR_OP_SUM_AXIS1,
    BASIS_IR_OP_TRANSPOSE_BWD,
    BASIS_IR_OP_SOFTMAX_BWD
} basis_ir_opcode;

// Immutable Node: Inputs are const pointers. Once created, a node cannot be mutated.
typedef struct basis_ir_node {
    uint32_t id;
    basis_ir_opcode op;
    size_t rows;
    size_t cols;

    const struct basis_ir_node** inputs;
    uint32_t input_count;

    double attr_val; // For CONST nodes or operation attributes
    double* runtime_data; // Populated by the V7 Executor
} basis_ir_node;

typedef struct basis_ir_graph {
    basis_arena* arena;          // All nodes and input arrays live here

    basis_ir_node** nodes;       // Tracking array for topological traversal
    uint32_t node_count;
    uint32_t node_cap;

    uint32_t next_id;
} basis_ir_graph;

basis_ir_graph* basis_ir_graph_create(void);
void basis_ir_graph_destroy(basis_ir_graph* g); // O(1) Teardown!

// Graph Builders (Enforce SSA & Shape Inference)
basis_ir_node* basis_ir_input(basis_ir_graph* g, size_t r, size_t c);
basis_ir_node* basis_ir_const(basis_ir_graph* g, double val, size_t r, size_t c);
basis_ir_node* basis_ir_add(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_matmul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_relu(basis_ir_graph* g, const basis_ir_node* a);


basis_ir_node* basis_ir_mul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_sub(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_transpose(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_sum(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_broadcast(basis_ir_graph* g, const basis_ir_node* a, size_t r, size_t c);
basis_ir_node* basis_ir_sum_axis0(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_sum_axis1(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_softmax_bwd(basis_ir_graph* g, const basis_ir_node* grad, const basis_ir_node* fwd_y);
basis_ir_node* basis_ir_relu_bwd(basis_ir_graph* g, const basis_ir_node* grad, const basis_ir_node* fwd_a);
basis_ir_node* basis_ir_softmax(basis_ir_graph* g, const basis_ir_node* a);

void basis_ir_print(const basis_ir_graph* g);

#endif // BASIS_V7_IR_H

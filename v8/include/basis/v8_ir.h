#ifndef BASIS_V8_IR_H
#define BASIS_V8_IR_H
#include "basis/v8_arena.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    V8_OP_INPUT, V8_OP_CONST, V8_OP_ADD, V8_OP_SUB, V8_OP_MUL,
    V8_OP_MATMUL, V8_OP_RELU, V8_OP_SOFTMAX, V8_OP_SUM, V8_OP_BROADCAST,
    V8_OP_TRANSPOSE, V8_OP_RELU_BWD, V8_OP_SOFTMAX_BWD, V8_OP_SUM_AXIS0, V8_OP_SUM_AXIS1
} v8_opcode;

typedef struct v8_node {
    uint32_t id;
    v8_opcode op;
    size_t rows, cols;
    const struct v8_node** inputs;
    uint32_t input_count;
    double attr_val;
    double* runtime_data;
} v8_node;

typedef struct v8_graph {
    v8_arena* arena;
    v8_node** nodes;
    uint32_t node_count, node_cap, next_id;
} v8_graph;

v8_graph* v8_graph_create(void);
void v8_graph_destroy(v8_graph* g);

v8_node* v8_input(v8_graph* g, size_t r, size_t c);
v8_node* v8_const(v8_graph* g, double val, size_t r, size_t c);
v8_node* v8_add(v8_graph* g, const v8_node* a, const v8_node* b);
v8_node* v8_sub(v8_graph* g, const v8_node* a, const v8_node* b);
v8_node* v8_mul(v8_graph* g, const v8_node* a, const v8_node* b);
v8_node* v8_matmul(v8_graph* g, const v8_node* a, const v8_node* b);
v8_node* v8_relu(v8_graph* g, const v8_node* a);
v8_node* v8_softmax(v8_graph* g, const v8_node* a);
v8_node* v8_sum(v8_graph* g, const v8_node* a);
v8_node* v8_broadcast(v8_graph* g, const v8_node* a, size_t r, size_t c);
v8_node* v8_transpose(v8_graph* g, const v8_node* a);

// Backward Ops
v8_node* v8_relu_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_a);
v8_node* v8_softmax_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_y);
v8_node* v8_sum_axis0(v8_graph* g, const v8_node* a);
v8_node* v8_sum_axis1(v8_graph* g, const v8_node* a);

#endif

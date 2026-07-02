#ifndef BASIS_V8_IR_H
#define BASIS_V8_IR_H
#include "basis/v8_arena.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    V8_OP_NOP = 0, V8_OP_INPUT, V8_OP_CONST, V8_OP_ADD, V8_OP_SUB, V8_OP_MUL,
    V8_OP_MATMUL, V8_OP_RELU, V8_OP_SOFTMAX, V8_OP_SUM, V8_OP_BROADCAST,
    V8_OP_TRANSPOSE, V8_OP_RELU_BWD, V8_OP_SOFTMAX_BWD, V8_OP_SUM_AXIS0, V8_OP_SUM_AXIS1,
    V8_OP_CONV2D, V8_OP_MAXPOOL2D, V8_OP_FLATTEN, V8_OP_CONV2D_RELU_FUSED,
    V8_OP_CROSS_ENTROPY, V8_OP_CROSS_ENTROPY_BWD, V8_OP_PERMUTE, V8_OP_MATMUL_BATCHED,
    V8_OP_CONV2D_BWD, V8_OP_CONV2D_BWD_W, V8_OP_MAXPOOL2D_BWD, V8_OP_RESHAPE
} v8_opcode;

typedef struct v8_node {
    uint32_t id;
    v8_opcode op;
    size_t shape[4];
    uint8_t ndim;
    uint32_t kernel_h, kernel_w, stride, pad;
    uint32_t axes[4];
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

static inline size_t v8_node_elements(const v8_node* n) {
    size_t e = 1;
    for(uint8_t i=0; i<n->ndim; i++) e *= n->shape[i];
    return e;
}

v8_graph* v8_graph_create(void);
void v8_graph_destroy(v8_graph* g);

v8_node* v8_input(v8_graph* g, size_t r, size_t c);
v8_node* v8_input_4d(v8_graph* g, size_t n, size_t c, size_t h, size_t w);
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

v8_node* v8_conv2d(v8_graph* g, const v8_node* in, const v8_node* w, uint32_t stride, uint32_t pad);
v8_node* v8_maxpool2d(v8_graph* g, const v8_node* in, uint32_t kernel, uint32_t stride);
v8_node* v8_flatten(v8_graph* g, const v8_node* in);

v8_node* v8_relu_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_a);
v8_node* v8_softmax_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_y);
v8_node* v8_sum_axis0(v8_graph* g, const v8_node* a);
v8_node* v8_sum_axis1(v8_graph* g, const v8_node* a);
v8_node* v8_cross_entropy(v8_graph* g, const v8_node* logits, const v8_node* targets);
v8_node* v8_permute(v8_graph* g, const v8_node* in, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3);
v8_node* v8_matmul_batched(v8_graph* g, const v8_node* a, const v8_node* b);
v8_node* v8_cross_entropy_bwd(v8_graph* g, const v8_node* logits, const v8_node* targets, const v8_node* grad_out);
v8_node* v8_conv2d_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, const v8_node* fwd_w, uint32_t stride, uint32_t pad);
v8_node* v8_reshape(v8_graph* g, const v8_node* in, uint8_t ndim, size_t s0, size_t s1, size_t s2, size_t s3);
v8_node* v8_conv2d_bwd_w(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, uint32_t k_h, uint32_t k_w, uint32_t stride, uint32_t pad);
v8_node* v8_maxpool2d_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, uint32_t kernel, uint32_t stride);

void v8_graph_save(v8_graph* g, const char* path);
v8_graph* v8_graph_load(const char* path);

void v8_optimize_fusion(v8_graph* g);

#endif

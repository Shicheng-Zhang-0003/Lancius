#ifndef Lancius_LANCIUS_IR_H
#define Lancius_LANCIUS_IR_H
#include "lancius/lancius_arena.h"
#include <stddef.h>
#include <stdint.h>

typedef enum { LANCIUS_DTYPE_FP64 = 0, LANCIUS_DTYPE_INT8 } lancius_dtype;

typedef enum {
    LANCIUS_OP_NOP = 0, LANCIUS_OP_INPUT, LANCIUS_OP_CONST, LANCIUS_OP_ADD, LANCIUS_OP_SUB, LANCIUS_OP_MUL,
    LANCIUS_OP_MATMUL, LANCIUS_OP_RELU, LANCIUS_OP_SOFTMAX, LANCIUS_OP_SUM, LANCIUS_OP_BROADCAST,
    LANCIUS_OP_TRANSPOSE, LANCIUS_OP_RELU_BWD, LANCIUS_OP_SOFTMAX_BWD, LANCIUS_OP_SUM_AXIS0, LANCIUS_OP_SUM_AXIS1,
    LANCIUS_OP_CONV2D, LANCIUS_OP_MAXPOOL2D, LANCIUS_OP_FLATTEN, LANCIUS_OP_CONV2D_RELU_FUSED,
    LANCIUS_OP_CROSS_ENTROPY, LANCIUS_OP_CROSS_ENTROPY_BWD, LANCIUS_OP_PERMUTE, LANCIUS_OP_MATMUL_BATCHED,
    LANCIUS_OP_CONV2D_BWD, LANCIUS_OP_CONV2D_BWD_W, LANCIUS_OP_MAXPOOL2D_BWD, LANCIUS_OP_RESHAPE,
    // V13 TRANSFORMER MANDATE (LLM Inference)
    LANCIUS_OP_EMBEDDING, LANCIUS_OP_LAYERNORM, LANCIUS_OP_GELU, LANCIUS_OP_ROPE,
    LANCIUS_OP_ATTENTION, LANCIUS_OP_KV_CACHE_READ, LANCIUS_OP_KV_CACHE_WRITE
} lancius_opcode;

typedef struct lancius_node {
    uint32_t id;
    lancius_opcode op;
    size_t shape[4];
    uint8_t ndim;
    uint32_t kernel_h, kernel_w, stride, pad;
    uint32_t axes[4];
    const struct lancius_node** inputs;
    uint32_t input_count;
    double attr_val;
    double* runtime_data;
    lancius_dtype dtype;
    double scale;
    int8_t* runtime_data_int8;
} lancius_node;

typedef struct lancius_graph {
    lancius_arena* arena;
    lancius_node** nodes;
    uint32_t node_count, node_cap, next_id;
} lancius_graph;

static inline size_t lancius_node_elements(const lancius_node* n) {
    size_t e = 1;
    for(uint8_t i=0; i<n->ndim; i++) e *= n->shape[i];
    return e;
}

lancius_graph* lancius_graph_create(void);
void lancius_graph_destroy(lancius_graph* g);

lancius_node* lancius_input(lancius_graph* g, size_t r, size_t c);
lancius_node* lancius_input_4d(lancius_graph* g, size_t n, size_t c, size_t h, size_t w);
lancius_node* lancius_const(lancius_graph* g, double val, size_t r, size_t c);
lancius_node* lancius_add(lancius_graph* g, const lancius_node* a, const lancius_node* b);
lancius_node* lancius_sub(lancius_graph* g, const lancius_node* a, const lancius_node* b);
lancius_node* lancius_mul(lancius_graph* g, const lancius_node* a, const lancius_node* b);
lancius_node* lancius_matmul(lancius_graph* g, const lancius_node* a, const lancius_node* b);
lancius_node* lancius_relu(lancius_graph* g, const lancius_node* a);
lancius_node* lancius_softmax(lancius_graph* g, const lancius_node* a);
lancius_node* lancius_sum(lancius_graph* g, const lancius_node* a);
lancius_node* lancius_broadcast_4d(lancius_graph* g, const lancius_node* a, size_t n, size_t c, size_t h, size_t w);
lancius_node* lancius_broadcast(lancius_graph* g, const lancius_node* a, size_t r, size_t c);
lancius_node* lancius_transpose(lancius_graph* g, const lancius_node* a);

lancius_node* lancius_conv2d(lancius_graph* g, const lancius_node* in, const lancius_node* w, uint32_t stride, uint32_t pad);
lancius_node* lancius_maxpool2d(lancius_graph* g, const lancius_node* in, uint32_t kernel, uint32_t stride);
lancius_node* lancius_flatten(lancius_graph* g, const lancius_node* in);

lancius_node* lancius_relu_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_a);
lancius_node* lancius_softmax_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_y);
lancius_node* lancius_sum_axis0(lancius_graph* g, const lancius_node* a);
lancius_node* lancius_sum_axis1(lancius_graph* g, const lancius_node* a);
lancius_node* lancius_cross_entropy(lancius_graph* g, const lancius_node* logits, const lancius_node* targets);
lancius_node* lancius_permute(lancius_graph* g, const lancius_node* in, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3);
lancius_node* lancius_matmul_batched(lancius_graph* g, const lancius_node* a, const lancius_node* b);
lancius_node* lancius_cross_entropy_bwd(lancius_graph* g, const lancius_node* logits, const lancius_node* targets, const lancius_node* grad_out);
lancius_node* lancius_conv2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, const lancius_node* fwd_w, uint32_t stride, uint32_t pad);
lancius_node* lancius_reshape(lancius_graph* g, const lancius_node* in, uint8_t ndim, size_t s0, size_t s1, size_t s2, size_t s3);
lancius_node* lancius_conv2d_bwd_w(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t k_h, uint32_t k_w, uint32_t stride, uint32_t pad);
lancius_node* lancius_maxpool2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t kernel, uint32_t stride);

void lancius_graph_save(lancius_graph* g, const char* path);
lancius_graph* lancius_graph_load(const char* path);

void lancius_optimize_fusion(lancius_graph* g);
void lancius_quantize_graph(lancius_graph* g);


lancius_node* lancius_attention(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v);
lancius_node* lancius_layernorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma, const lancius_node* beta);
lancius_node* lancius_gelu(lancius_graph* g, const lancius_node* in);

#endif

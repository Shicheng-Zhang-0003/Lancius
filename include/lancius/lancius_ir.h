/**
 * @file lancius_ir.h
 * @brief Immutable SSA IR Graph: The core directed acyclic graph (DAG) representation of all tensor operations.
 */
#ifndef Lancius_LANCIUS_IR_H
#define Lancius_LANCIUS_IR_H
#include "lancius/lancius_arena.h"
#include "lancius/lancius_types.h"
#include "lancius/lancius_error.h"
#include "lancius/lancius_model_format.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* A1: lancius_dtype now lives in lancius_types.h */

typedef enum {
    LANCIUS_OP_NOP = 0, LANCIUS_OP_INPUT, LANCIUS_OP_CONST, LANCIUS_OP_ADD, LANCIUS_OP_SUB, LANCIUS_OP_MUL,
    LANCIUS_OP_MATMUL, LANCIUS_OP_RELU, LANCIUS_OP_SOFTMAX, LANCIUS_OP_SUM, LANCIUS_OP_BROADCAST,
    LANCIUS_OP_TRANSPOSE, LANCIUS_OP_RELU_BWD, LANCIUS_OP_SOFTMAX_BWD, LANCIUS_OP_SUM_AXIS0, LANCIUS_OP_SUM_AXIS1,
    LANCIUS_OP_CONV2D, LANCIUS_OP_MAXPOOL2D, LANCIUS_OP_FLATTEN, LANCIUS_OP_CONV2D_RELU_FUSED,
    LANCIUS_OP_CROSS_ENTROPY, LANCIUS_OP_CROSS_ENTROPY_BWD, LANCIUS_OP_PERMUTE, LANCIUS_OP_MATMUL_BATCHED,
    LANCIUS_OP_CONV2D_BWD, LANCIUS_OP_CONV2D_BWD_W, LANCIUS_OP_MAXPOOL2D_BWD, LANCIUS_OP_RESHAPE,
    // V10S TRANSFORMER MANDATE (LLM Inference)
    LANCIUS_OP_EMBEDDING, LANCIUS_OP_LAYERNORM, LANCIUS_OP_GELU, LANCIUS_OP_ROPE,
    LANCIUS_OP_ATTENTION, LANCIUS_OP_KV_CACHE_READ, LANCIUS_OP_KV_CACHE_WRITE,
    LANCIUS_OP_RMSNORM,
    LANCIUS_OP_SWIGLU,
    LANCIUS_OP_GQA
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
    float* runtime_data_f32;

    /* A1: separated mutable runtime state. Legacy fields above remain temporary compatibility mirrors. */
    lancius_runtime_state* rt;
    // V9.5 TENSOR VIEW FIELDS (Zero-Copy Reshape/Permute/Flatten)
    uint8_t is_view;                    // 1 if this node is a view (no owned memory)
    size_t strides[4];                  // Byte strides for each dimension
    const struct lancius_node* view_source; // Points to the underlying data owner
} lancius_node;

typedef struct lancius_graph {
    lancius_arena* arena;
    lancius_node** nodes;
    uint32_t node_count, node_cap, next_id;

    /* A1: runtime state table indexed by node id. */
    lancius_runtime_state* rt_states;
    size_t rt_cap;
} lancius_graph;

#ifndef LANCIUS_MAX_TENSOR_ELEMS
#define LANCIUS_MAX_TENSOR_ELEMS ((size_t)100000000)
#endif

#ifndef LANCIUS_MAX_TENSOR_BYTES
#define LANCIUS_MAX_TENSOR_BYTES ((size_t)800000000)
#endif

/*
 * v11S FATAL INVARIANT (not user-reachable via stable API):
 * lancius_node_elements() calls abort() if ndim > 4 or element count
 * overflows LANCIUS_MAX_TENSOR_ELEMS. These conditions indicate memory
 * corruption or a violated internal invariant. Continuing execution
 * would be unsafe. The stable FFI API validates all inputs before
 * calling this function, so FFI consumers will never trigger these.
 *
 * For checked (non-aborting) element counting, use:
 *   lancius_node_elements_checked(n, &out)
 */
static inline size_t lancius_node_elements(const lancius_node* n) {
    if (!n) return 0;

    if (n->ndim > 4) {
        fprintf(stderr,
            "[LANCIUS SAFETY FATAL] tensor ndim %u exceeds max rank 4\n",
            (unsigned)n->ndim);
        abort();
    }

    size_t e = 1;

    for (uint8_t i = 0; i < n->ndim; i++) {
        size_t dim = n->shape[i];

        if (dim == 0) {
            e = 0;
            continue;
        }

        if (dim > LANCIUS_MAX_TENSOR_ELEMS || e > LANCIUS_MAX_TENSOR_ELEMS / dim) {
            fprintf(stderr,
                "[LANCIUS SAFETY FATAL] tensor element count overflow or exceeds sanity limit\n");
            abort();
        }

        e *= dim;
    }

    return e;
}

static inline int lancius_node_elements_checked(const lancius_node* n, size_t* out) {
    if (!n || !out) return 0;
    if (n->ndim > 4) return 0;

    size_t e = 1;

    for (uint8_t i = 0; i < n->ndim; i++) {
        size_t dim = n->shape[i];

        if (dim == 0) {
            e = 0;
            continue;
        }

        if (dim > LANCIUS_MAX_TENSOR_ELEMS || e > LANCIUS_MAX_TENSOR_ELEMS / dim) {
            return 0;
        }

        e *= dim;
    }

    *out = e;
    return 1;
}

static inline size_t lancius_node_element_bytes(const lancius_node* n);

static inline int lancius_node_bytes_checked(const lancius_node* n, size_t* out) {
    size_t elems = 0;
    if (!lancius_node_elements_checked(n, &elems)) return 0;

    size_t elem_size = lancius_node_element_bytes(n);
    if (elem_size != 0 && elems > SIZE_MAX / elem_size) return 0;

    size_t bytes = elems * elem_size;
    if (bytes > LANCIUS_MAX_TENSOR_BYTES) return 0;

    *out = bytes;
    return 1;
}

/* A3: dtype-aware tensor byte sizing */
static inline size_t lancius_node_element_bytes(const lancius_node* n) {
    return lancius_dtype_size(n ? n->dtype : LANCIUS_DTYPE_FP64);
}

static inline size_t lancius_node_bytes(const lancius_node* n) {
    if (!n) return 0;
    return lancius_node_elements(n) * lancius_node_element_bytes(n);
}

lancius_graph* lancius_graph_create(void);
void lancius_graph_destroy(lancius_graph* g);

lancius_node* lancius_input(lancius_graph* g, size_t r, size_t c);
lancius_node* lancius_input_4d(lancius_graph* g, size_t n, size_t c, size_t h, size_t w);
lancius_node* lancius_input_3d(lancius_graph* g, size_t d0, size_t d1, size_t d2);
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

int lancius_graph_save(lancius_graph* g, const char* path); /* returns 0 on success, -1 on failure */
lancius_graph* lancius_graph_load(const char* path);

void lancius_optimize_fusion(lancius_graph* g);
void lancius_quantize_graph(lancius_graph* g);


lancius_node* lancius_attention(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v);
lancius_node* lancius_layernorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma, const lancius_node* beta);
lancius_node* lancius_gelu(lancius_graph* g, const lancius_node* in);


lancius_node* lancius_rmsnorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma);
lancius_node* lancius_swiglu(lancius_graph* g, const lancius_node* gate, const lancius_node* up);
lancius_node* lancius_gqa(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v, uint32_t n_heads_q, uint32_t n_heads_kv);

/* A1: runtime state API */
lancius_runtime_state* lancius_graph_runtime(lancius_graph* g, uint32_t node_id);
lancius_runtime_state* lancius_node_rt(const lancius_node* n);
void lancius_runtime_sync_from_legacy(lancius_node* n);
void lancius_runtime_sync_to_legacy(lancius_node* n);
void lancius_graph_sync_runtime_to_legacy(lancius_graph* g);

/* A2: tensor ownership API */
void lancius_node_set_owner(lancius_node* n, lancius_memory_owner owner);
lancius_memory_owner lancius_node_get_owner(const lancius_node* n);

void lancius_node_bind_external(lancius_node* n, void* data);
void lancius_node_bind_external_int8(lancius_node* n, int8_t* data);

void lancius_node_bind_owned_heap(lancius_node* n, void* data);
void lancius_node_bind_owned_heap_int8(lancius_node* n, int8_t* data);
void lancius_node_bind_external_f32(lancius_node* n, float* data);
void lancius_node_bind_owned_heap_f32(lancius_node* n, float* data);

void lancius_node_release_owned(lancius_node* n);
void lancius_graph_release_owned(lancius_graph* g);

/* v11A1: ownership query helpers */
bool lancius_node_buffer_is_external(const lancius_node* n);
bool lancius_node_int8_buffer_is_external(const lancius_node* n);
#endif

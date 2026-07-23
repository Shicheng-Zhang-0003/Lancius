#include "lancius/lancius_ir.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void track(lancius_graph* g, lancius_node* n) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 1024 : g->node_cap * 2;
        g->nodes = (lancius_node**)realloc(g->nodes, sizeof(lancius_node*) * g->node_cap);
    }
    g->nodes[g->node_count++] = n;
}

static void lancius_ensure_runtime_capacity(lancius_graph* g, uint32_t id) {
    if (!g || id < g->rt_cap) return;

    size_t new_cap = g->rt_cap ? g->rt_cap : 1024;
    while (new_cap <= id) new_cap *= 2;

    lancius_runtime_state* nr = (lancius_runtime_state*)realloc(
        g->rt_states,
        new_cap * sizeof(lancius_runtime_state)
    );
    if (!nr) return;

    memset(nr + g->rt_cap, 0, (new_cap - g->rt_cap) * sizeof(lancius_runtime_state));
    g->rt_states = nr;
    g->rt_cap = new_cap;
}

static lancius_node* alloc_node(lancius_graph* g, lancius_opcode op, uint8_t ndim, uint32_t in_count);

lancius_graph* lancius_graph_create(void) {
    /* A4: graph_create error instrumentation */
    lancius_graph* g = (lancius_graph*)calloc(1, sizeof(lancius_graph));
    if (!g) {
        lancius_set_error(LANCIUS_ERROR_OOM);
        return NULL;
    }

    g->arena = lancius_arena_create(16 * 1024 * 1024);
    if (!g->arena) {
        lancius_set_error(LANCIUS_ERROR_OOM);
        free(g);
        return NULL;
    }

    g->node_cap = 1024;
    g->nodes = (lancius_node**)malloc(sizeof(lancius_node*) * g->node_cap);
    if (!g->nodes) {
        lancius_set_error(LANCIUS_ERROR_OOM);
        lancius_arena_destroy(g->arena);
        free(g);
        return NULL;
    }

    /* A1: runtime state table */
    g->rt_cap = 1024;
    g->rt_states = (lancius_runtime_state*)calloc(g->rt_cap, sizeof(lancius_runtime_state));
    if (!g->rt_states) {
        lancius_set_error(LANCIUS_ERROR_OOM);
        free(g->nodes);
        lancius_arena_destroy(g->arena);
        free(g);
        return NULL;
    }

    return g;
}


lancius_node* lancius_attention(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v) {
    if (!q || !k || !v) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_ATTENTION, 3, 3);
    if (n) { n->shape[0] = q->shape[0]; n->shape[1] = q->shape[1]; n->shape[2] = q->shape[2]; n->inputs[0] = q; n->inputs[1] = k; n->inputs[2] = v; }
    return n;
}
lancius_node* lancius_layernorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma, const lancius_node* beta) {
    if (!in || !gamma || !beta) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_LAYERNORM, in->ndim, 3);
    if (n) { memcpy(n->shape, in->shape, sizeof(size_t)*in->ndim); n->inputs[0] = in; n->inputs[1] = gamma; n->inputs[2] = beta; }
    return n;
}
lancius_node* lancius_gelu(lancius_graph* g, const lancius_node* in) {
    if (!in) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_GELU, in->ndim, 1);
    if (n) { memcpy(n->shape, in->shape, sizeof(size_t)*in->ndim); n->inputs[0] = in; }
    return n;
}


lancius_node* lancius_broadcast_4d(lancius_graph* g, const lancius_node* a, size_t n, size_t c, size_t h, size_t w) {
    if (!a) return NULL;
    lancius_node* n_node = alloc_node(g, LANCIUS_OP_BROADCAST, 4, 1);
    if (n_node) { n_node->shape[0] = n; n_node->shape[1] = c; n_node->shape[2] = h; n_node->shape[3] = w; n_node->inputs[0] = a; }
    return n_node;
}

void lancius_graph_destroy(lancius_graph* g) {
    if (!g) return;
    free(g->nodes);
    free(g->rt_states); /* A1 */
    lancius_arena_destroy(g->arena);
    free(g);
}

static lancius_node* alloc_node(lancius_graph* g, lancius_opcode op, uint8_t ndim, uint32_t in_count) {
    lancius_node* n = (lancius_node*)lancius_arena_alloc(g->arena, sizeof(lancius_node), 8);
    if (!n) { lancius_set_error(LANCIUS_ERROR_OOM); /* A4 alloc_node OOM */ return NULL; }

    memset(n, 0, sizeof(lancius_node));
    n->id = g->next_id++;

    /* A1: attach separated runtime state */
    lancius_ensure_runtime_capacity(g, n->id);
    if (g->rt_states && n->id < g->rt_cap) {
        n->rt = &g->rt_states[n->id];
        n->rt->buffer = NULL;
        n->rt->buffer_int8 = NULL;
        n->rt->dtype = LANCIUS_DTYPE_FP64;
        n->rt->scale = 1.0;
        n->rt->owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL; /* A2 */
        n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;   /* A2 */
        n->rt->offset = 0;
        n->rt->flags = 0;
    }

    n->op = op;
    n->ndim = ndim;
    n->input_count = in_count;

    if (in_count > 0) {
        n->inputs = (const lancius_node**)lancius_arena_alloc(
            g->arena,
            sizeof(lancius_node*) * in_count,
            8
        );
    }

    track(g, n);
    return n;
}

lancius_node* lancius_input(lancius_graph* g, size_t r, size_t c) {
    lancius_node* n = alloc_node(g, LANCIUS_OP_INPUT, 2, 0);
    if(n) { n->shape[0] = r; n->shape[1] = c; } return n;
}
lancius_node* lancius_input_4d(lancius_graph* g, size_t n_dim, size_t c, size_t h, size_t w) {
    lancius_node* n = alloc_node(g, LANCIUS_OP_INPUT, 4, 0);
    if(n) { n->shape[0] = n_dim; n->shape[1] = c; n->shape[2] = h; n->shape[3] = w; } return n;
}
lancius_node* lancius_const(lancius_graph* g, double val, size_t r, size_t c) {
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONST, 2, 0);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->attr_val = val; } return n;
}
lancius_node* lancius_add(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
        if (!a || !b) { fprintf(stderr, "[LANCIUS IR FATAL] ADD NULL input"); return NULL; }
    // V10S FIX: Allow broadcast

    lancius_node* n = alloc_node(g, LANCIUS_OP_ADD, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_sub(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
        if (!a || !b) { fprintf(stderr, "[LANCIUS IR FATAL] SUB NULL input"); return NULL; }
    // V10S FIX: Allow broadcast

    lancius_node* n = alloc_node(g, LANCIUS_OP_SUB, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_mul(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
        if (!a || !b) { fprintf(stderr, "[LANCIUS IR FATAL] MUL NULL input"); return NULL; }
    // V10S FIX: Allow broadcast

    lancius_node* n = alloc_node(g, LANCIUS_OP_MUL, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_matmul(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b || a->ndim < 2 || b->ndim < 2) {
        fprintf(stderr, "[LANCIUS IR FATAL] MATMUL ndim < 2: a=%u b=%u", a?a->ndim:0, b?b->ndim:0);
        return NULL;
    }
    // V10S FIX: NDim-aware dimension extraction (handles 4D Reshape outputs)
    size_t a_rows = a->shape[a->ndim - 2];
    size_t a_cols = a->shape[a->ndim - 1];
    size_t b_rows = b->shape[b->ndim - 2];
    size_t b_cols = b->shape[b->ndim - 1];

    if (a_cols != b_rows) {
        fprintf(stderr, "[LANCIUS IR FATAL] MATMUL mismatch: a=[%zux%zu] b=[%zux%zu]", a_rows, a_cols, b_rows, b_cols);
        return NULL;
    }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MATMUL, 2, 2);
    if (n) { n->shape[0] = a_rows; n->shape[1] = b_cols; n->inputs[0] = a; n->inputs[1] = b; }
    return n;
}
lancius_node* lancius_relu(lancius_graph* g, const lancius_node* a) {
    if (!a) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_RELU, a->ndim, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; } return n;
}
lancius_node* lancius_softmax(lancius_graph* g, const lancius_node* a) {
    if (!a || a->ndim != 2) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SOFTMAX, 2, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*2); n->inputs[0] = a; } return n;
}
lancius_node* lancius_sum(lancius_graph* g, const lancius_node* a) {
    if (!a) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = a; } return n;
}
lancius_node* lancius_broadcast(lancius_graph* g, const lancius_node* a, size_t r, size_t c) {
    if (!a) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_BROADCAST, 2, 1);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->inputs[0] = a; } return n;
}
lancius_node* lancius_transpose(lancius_graph* g, const lancius_node* a) {
    if (!a || a->ndim != 2) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_TRANSPOSE, 2, 1);
    if (n) { n->shape[0] = a->shape[1]; n->shape[1] = a->shape[0]; n->inputs[0] = a; } return n;
}
lancius_node* lancius_relu_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_a) {
    if (!grad || !fwd_a) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_RELU_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_a; } return n;
}
lancius_node* lancius_softmax_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_y) {
    if (!grad || !fwd_y) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SOFTMAX_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_y; } return n;
}
lancius_node* lancius_sum_axis0(lancius_graph* g, const lancius_node* a) {
    if (!a || a->ndim != 2) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM_AXIS0, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = a->shape[1]; n->inputs[0] = a; } return n;
}
lancius_node* lancius_sum_axis1(lancius_graph* g, const lancius_node* a) {
    if (!a || a->ndim != 2) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM_AXIS1, 2, 1);
    if (n) { n->shape[0] = a->shape[0]; n->shape[1] = 1; n->inputs[0] = a; } return n;
}

lancius_node* lancius_conv2d(lancius_graph* g, const lancius_node* in, const lancius_node* w, uint32_t stride, uint32_t pad) {
    if (!in || !w || in->ndim != 4 || w->ndim != 4) { fprintf(stderr, "[LANCIUS IR FATAL] CONV2D ndim mismatch\n"); return NULL; }
    size_t N = in->shape[0], C_in = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    size_t C_out = w->shape[0], K_h = w->shape[2], K_w = w->shape[3];
    if (C_in != w->shape[1]) { fprintf(stderr, "[LANCIUS IR FATAL] CONV2D channels mismatch: in=%zu w=%zu\n", C_in, w->shape[1]); return NULL; }
    if (H_in + 2*pad < K_h || W_in + 2*pad < K_w) {
        fprintf(stderr, "[IR FATAL] Conv2D spatial dimensions underflow.\n");
        return NULL;
    }
    size_t H_out = (H_in + 2*pad - K_h) / stride + 1;
    size_t W_out = (W_in + 2*pad - K_w) / stride + 1;
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONV2D, 4, 2);
    if (n) {
        n->shape[0] = N; n->shape[1] = C_out; n->shape[2] = H_out; n->shape[3] = W_out;
        n->kernel_h = K_h; n->kernel_w = K_w; n->stride = stride; n->pad = pad;
        n->inputs[0] = in; n->inputs[1] = w;
    }
    return n;
}

lancius_node* lancius_maxpool2d(lancius_graph* g, const lancius_node* in, uint32_t kernel, uint32_t stride) {
    if (!in || in->ndim != 4) return NULL;
    size_t N = in->shape[0], C = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    if (H_in < kernel || W_in < kernel) {
        fprintf(stderr, "[IR FATAL] MaxPool2D spatial dimensions underflow.\n");
        return NULL;
    }
    size_t H_out = (H_in - kernel) / stride + 1;
    size_t W_out = (W_in - kernel) / stride + 1;
    lancius_node* n = alloc_node(g, LANCIUS_OP_MAXPOOL2D, 4, 1);
    if (n) {
        n->shape[0] = N; n->shape[1] = C; n->shape[2] = H_out; n->shape[3] = W_out;
        n->kernel_h = kernel; n->kernel_w = kernel; n->stride = stride; n->pad = 0;
        n->inputs[0] = in;
    }
    return n;
}

lancius_node* lancius_flatten(lancius_graph* g, const lancius_node* in) {
    if (!in || in->ndim != 4) return NULL;
    size_t N = in->shape[0];
    size_t flat = in->shape[1] * in->shape[2] * in->shape[3];
    lancius_node* n = alloc_node(g, LANCIUS_OP_FLATTEN, 2, 1);
    if (n) { n->shape[0] = N; n->shape[1] = flat; n->inputs[0] = in; }
    return n;
}


lancius_node* lancius_reshape(lancius_graph* g, const lancius_node* in, uint8_t ndim, size_t s0, size_t s1, size_t s2, size_t s3) {
    if (!in) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_RESHAPE, ndim, 1);
    if (n) {
        n->shape[0] = s0; if(ndim>1) n->shape[1] = s1; if(ndim>2) n->shape[2] = s2; if(ndim>3) n->shape[3] = s3;
        n->inputs[0] = in;
    }
    return n;
}
lancius_node* lancius_conv2d_bwd_w(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t k_h, uint32_t k_w, uint32_t stride, uint32_t pad) {
    if (!grad || !fwd_in) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONV2D_BWD_W, 4, 2);
    if (n) {
        n->shape[0] = grad->shape[1]; n->shape[1] = fwd_in->shape[1];
        n->shape[2] = k_h; n->shape[3] = k_w;
        n->kernel_h = k_h; n->kernel_w = k_w;
        n->stride = stride; n->pad = pad;
        n->inputs[0] = grad; n->inputs[1] = fwd_in;
    }
    return n;
}



lancius_node* lancius_permute(lancius_graph* g, const lancius_node* in, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    if (!in || in->ndim != 4) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_PERMUTE, 4, 1);
    if (n) {
        n->axes[0] = a0; n->axes[1] = a1; n->axes[2] = a2; n->axes[3] = a3;
        size_t dims[4] = {in->shape[0], in->shape[1], in->shape[2], in->shape[3]};
        n->shape[0] = dims[a0]; n->shape[1] = dims[a1]; n->shape[2] = dims[a2]; n->shape[3] = dims[a3];
        n->inputs[0] = in;
    }
    return n;
}
lancius_node* lancius_matmul_batched(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b || a->ndim != 3 || b->ndim != 3) return NULL;
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_MATMUL_BATCHED, 3, 2);
    if (n) {
        n->shape[0] = a->shape[0]; n->shape[1] = a->shape[1]; n->shape[2] = b->shape[2];
        n->inputs[0] = a; n->inputs[1] = b;
    }
    return n;
}

lancius_node* lancius_cross_entropy(lancius_graph* g, const lancius_node* logits, const lancius_node* targets) {
    if (!logits || !targets || logits->ndim != 2 || targets->ndim != 2) return NULL;
    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_CROSS_ENTROPY, 2, 2);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = logits; n->inputs[1] = targets; }
    return n;
}
lancius_node* lancius_cross_entropy_bwd(lancius_graph* g, const lancius_node* logits, const lancius_node* targets, const lancius_node* grad_out) {
    if (!logits || !targets || !grad_out) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_CROSS_ENTROPY_BWD, logits->ndim, 3);
    if (n) { memcpy(n->shape, logits->shape, sizeof(size_t)*logits->ndim); n->inputs[0] = logits; n->inputs[1] = targets; n->inputs[2] = grad_out; }
    return n;
}

lancius_node* lancius_conv2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, const lancius_node* fwd_w, uint32_t stride, uint32_t pad) {
    if (!grad || !fwd_in || !fwd_w) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONV2D_BWD, 4, 3);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->stride = stride; n->pad = pad;
        n->inputs[0] = grad; n->inputs[1] = fwd_in; n->inputs[2] = fwd_w;
    } return n;
}
lancius_node* lancius_maxpool2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t kernel, uint32_t stride) {
    if (!grad || !fwd_in) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_MAXPOOL2D_BWD, 4, 2);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->kernel_h = kernel; n->kernel_w = kernel; n->stride = stride;
        n->inputs[0] = grad; n->inputs[1] = fwd_in;
    } return n;
}

lancius_node* lancius_rmsnorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma) {
    if (!in || !gamma) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_RMSNORM, in->ndim, 2);
    if (n) {
        memcpy(n->shape, in->shape, sizeof(size_t) * in->ndim);
        n->inputs[0] = in; n->inputs[1] = gamma;
    }
    return n;
}

lancius_node* lancius_swiglu(lancius_graph* g, const lancius_node* gate, const lancius_node* up) {
    if (!gate || !up) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_SWIGLU, gate->ndim, 2);
    if (n) {
        memcpy(n->shape, gate->shape, sizeof(size_t) * gate->ndim);
        n->inputs[0] = gate; n->inputs[1] = up;
    }
    return n;
}

lancius_node* lancius_gqa(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v, uint32_t n_heads_q, uint32_t n_heads_kv) {
    if (!q || !k || !v) return NULL;
    lancius_node* n = alloc_node(g, LANCIUS_OP_GQA, q->ndim, 3);
    if (n) {
        memcpy(n->shape, q->shape, sizeof(size_t) * q->ndim);
        n->kernel_h = n_heads_q;
        n->kernel_w = n_heads_kv;
        n->inputs[0] = q; n->inputs[1] = k; n->inputs[2] = v;
    }
    return n;
}

/* A1: runtime state helpers */

lancius_runtime_state* lancius_graph_runtime(lancius_graph* g, uint32_t node_id) {
    if (!g || !g->rt_states || node_id >= g->rt_cap) return NULL;
    return &g->rt_states[node_id];
}

lancius_runtime_state* lancius_node_rt(const lancius_node* n) {
    return n ? n->rt : NULL;
}

void lancius_runtime_sync_from_legacy(lancius_node* n) {
    if (!n || !n->rt) return;

    n->rt->buffer = n->runtime_data;
    n->rt->buffer_int8 = n->runtime_data_int8;
    n->rt->dtype = n->dtype;
    n->rt->scale = n->scale;
}

void lancius_runtime_sync_to_legacy(lancius_node* n) {
    if (!n || !n->rt) return;

    n->runtime_data = (double*)n->rt->buffer;
    n->runtime_data_int8 = n->rt->buffer_int8;
    n->dtype = n->rt->dtype;
    n->scale = n->rt->scale;
}

void lancius_graph_sync_runtime_to_legacy(lancius_graph* g) {
    if (!g) return;

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_runtime_sync_to_legacy(g->nodes[i]);
    }
}

/* A2: tensor ownership helpers */

void lancius_node_set_owner(lancius_node* n, lancius_memory_owner owner) {
    if (!n || !n->rt) return;

    n->rt->owner = owner;
    n->rt->buffer_owner = owner;
    n->rt->int8_owner = owner;
}

lancius_memory_owner lancius_node_get_owner(const lancius_node* n) {
    if (!n || !n->rt) return LANCIUS_MEMORY_EXTERNAL;
    return n->rt->buffer_owner;
}

void lancius_node_bind_external(lancius_node* n, void* data) {
    if (!n) return;

    n->runtime_data = (double*)data;

    if (n->rt) {
        n->rt->buffer = data;
        n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->dtype = n->dtype; /* A3 external */
    }
}

void lancius_node_bind_external_int8(lancius_node* n, int8_t* data) {
    if (!n) return;

    n->runtime_data_int8 = data;

    if (n->rt) {
        n->rt->buffer_int8 = data;
        n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
    }
}

void lancius_node_bind_owned_heap(lancius_node* n, void* data) {
    if (!n) return;

    if (!data) {
        lancius_node_bind_external(n, NULL);
        return;
    }

    n->runtime_data = (double*)data;

    if (n->rt) {
        n->rt->buffer = data;
        n->rt->buffer_owner = LANCIUS_MEMORY_OWNED_HEAP;
        n->rt->owner = LANCIUS_MEMORY_OWNED_HEAP;
        n->rt->dtype = n->dtype; /* A3 owned */
    }
}

void lancius_node_bind_owned_heap_int8(lancius_node* n, int8_t* data) {
    if (!n) return;

    if (!data) {
        lancius_node_bind_external_int8(n, NULL);
        return;
    }

    n->runtime_data_int8 = data;

    if (n->rt) {
        n->rt->buffer_int8 = data;
        n->rt->int8_owner = LANCIUS_MEMORY_OWNED_HEAP;
        if (n->dtype == LANCIUS_DTYPE_INT8) n->rt->dtype = LANCIUS_DTYPE_INT8; /* A3 int8 */
    }
}

void lancius_node_release_owned(lancius_node* n) {
    if (!n) return;

    if (n->rt) {
        if (n->rt->buffer_owner == LANCIUS_MEMORY_OWNED_HEAP && n->runtime_data) {
            free(n->runtime_data);
            n->runtime_data = NULL;
            n->rt->buffer = NULL;
            n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL;
        }

        if (n->rt->int8_owner == LANCIUS_MEMORY_OWNED_HEAP && n->runtime_data_int8) {
            free(n->runtime_data_int8);
            n->runtime_data_int8 = NULL;
            n->rt->buffer_int8 = NULL;
            n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
        }

        if (
            n->rt->buffer_owner == LANCIUS_MEMORY_EXTERNAL &&
            n->rt->int8_owner == LANCIUS_MEMORY_EXTERNAL
        ) {
            n->rt->owner = LANCIUS_MEMORY_EXTERNAL;
        }
    }
}

void lancius_graph_release_owned(lancius_graph* g) {
    if (!g) return;

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node_release_owned(g->nodes[i]);
    }
}
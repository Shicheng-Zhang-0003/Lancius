#include "lancius/lancius_ir.h"
#include "lancius/lancius_validate.h"
#include "lancius/lancius_checked.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

static void track(lancius_graph* g, lancius_node* n) {
    if (!g || !n) return;
    if (g->node_count >= g->node_cap) {
        size_t new_cap = g->node_cap == 0 ? 1024 : (size_t)g->node_cap * 2;
        if (new_cap > (size_t)UINT32_MAX + 1) new_cap = (size_t)UINT32_MAX + 1;
        lancius_node** nn = (lancius_node**)realloc(g->nodes, sizeof(lancius_node*) * new_cap);
        if (!nn) { lancius_set_error(LANCIUS_ERROR_OOM); return; }
        g->nodes = nn;
        g->node_cap = (uint32_t)new_cap;
    }
    g->nodes[g->node_count++] = n;
}

static void lancius_ensure_runtime_capacity(lancius_graph* g, uint32_t id) {
    if (!g || id < g->rt_cap) return;

    size_t new_cap = g->rt_cap ? g->rt_cap : 1024;
    while (new_cap <= id) {
        if (new_cap > SIZE_MAX / 2) { lancius_set_error(LANCIUS_ERROR_OOM); return; }
        new_cap *= 2;
    }

    lancius_runtime_state* nr = (lancius_runtime_state*)realloc(
        g->rt_states,
        new_cap * sizeof(lancius_runtime_state)
    );
    if (!nr) { lancius_set_error(LANCIUS_ERROR_OOM); return; }

    memset(nr + g->rt_cap, 0, (new_cap - g->rt_cap) * sizeof(lancius_runtime_state));
    g->rt_states = nr;
    g->rt_cap = new_cap;
    
    /* v11A1 repair: realloc invalidates cached rt pointers; refresh existing nodes. */
    if (g->nodes) {
        for (uint32_t i = 0; i < g->node_count; i++) {
            lancius_node* existing = g->nodes[i];
            if (existing && existing->id < g->rt_cap) {
                existing->rt = &g->rt_states[existing->id];
            }
        }
    }
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
    if (!q || !k || !v) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (q->shape[1] == 0 || q->shape[2] == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    if (k->shape[1] != q->shape[1] || k->shape[2] != q->shape[2]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (v->shape[1] != q->shape[1] || v->shape[2] != q->shape[2]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    size_t shape[3] = {q->shape[0], q->shape[1], q->shape[2]};
    if (lancius_validate_shape(shape, 3) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_ATTENTION, 3, 3);
    if (n) { n->shape[0] = q->shape[0]; n->shape[1] = q->shape[1]; n->shape[2] = q->shape[2]; n->inputs[0] = q; n->inputs[1] = k; n->inputs[2] = v; }
    return n;
}
lancius_node* lancius_layernorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma, const lancius_node* beta) {
    if (!in || !gamma || !beta) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (in->ndim == 0 || in->ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (lancius_validate_shape(in->shape, in->ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    size_t total = 0;
    if (!lancius_checked_product_shape(in->shape, in->ndim, &total) || total == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    size_t ge = 0, be = 0;
    if (!lancius_node_elements_checked(gamma, &ge) || ge == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (!lancius_node_elements_checked(beta, &be) || be != ge) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (total % ge != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_LAYERNORM, in->ndim, 3);
    if (n) { memcpy(n->shape, in->shape, sizeof(size_t)*in->ndim); n->inputs[0] = in; n->inputs[1] = gamma; n->inputs[2] = beta; }
    return n;
}
lancius_node* lancius_gelu(lancius_graph* g, const lancius_node* in) {
    if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (lancius_validate_shape(in->shape, in->ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_GELU, in->ndim, 1);
    if (n) { memcpy(n->shape, in->shape, sizeof(size_t)*in->ndim); n->inputs[0] = in; }
    return n;
}


lancius_node* lancius_broadcast_4d(lancius_graph* g, const lancius_node* a, size_t n, size_t c, size_t h, size_t w) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t out[4] = {n, c, h, w};
    if (lancius_validate_shape(out, 4) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    size_t ae = 0, oe = 0;
    if (!lancius_node_elements_checked(a, &ae)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    if (!lancius_checked_product_shape(out, 4, &oe)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    /* Despot truth: oe%ae==0 is not broadcast (e.g. 12 into 24 can still be
     * incompatible); only scalar, exact, or per-dim 1-or-equal is executable. */
    if (ae != 1 && ae != oe) {
        bool compat = true;
        if (a->ndim == 0 || a->ndim > 4) compat = false;
        else {
            for (uint8_t i = 0; i < 4; i++) {
                int ai = (int)i - (4 - (int)a->ndim);
                size_t da = (ai < 0) ? 1 : a->shape[ai];
                if (!(da == 1 || da == out[i])) { compat = false; break; }
            }
        }
        if (!compat) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    }
    lancius_node* n_node = alloc_node(g, LANCIUS_OP_BROADCAST, 4, 1);
    if (n_node) { n_node->shape[0] = n; n_node->shape[1] = c; n_node->shape[2] = h; n_node->shape[3] = w; n_node->inputs[0] = a; }
    return n_node;
}

/* Despot truth: broadcast scalar (or broadcast-compatible) to any 1..4-D shape.
 * Math: out[I] = a[bcast(I)], trailing-rank NumPy semantics. Scheduler already
 * executes N-dim strided broadcast; this constructor makes it reachable for
 * autodiff SUM grads and scalar lifts without forcing [1,1]-wrong shapes. */
lancius_node* lancius_broadcast_to_shape(lancius_graph* g, const lancius_node* a, const size_t* shape, uint8_t ndim) {
    if (!g || !a || !shape) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (ndim == 0 || ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (lancius_validate_shape(shape, ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    size_t ae = 0, oe = 0;
    if (!lancius_node_elements_checked(a, &ae)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    if (!lancius_checked_product_shape(shape, ndim, &oe)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    if (ae != 1) {
        /* Non-scalar must already match exactly; partial N-dim reductions are
         * expressed via SUM_AXIS ops in accum_grad, not here. */
        if (a->ndim != ndim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
        for (uint8_t i = 0; i < ndim; i++) {
            if (a->shape[i] != shape[i]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
        }
    }
    lancius_node* n = alloc_node(g, LANCIUS_OP_BROADCAST, ndim, 1);
    if (n) {
        for (uint8_t i = 0; i < ndim; i++) n->shape[i] = shape[i];
        n->inputs[0] = a;
    }
    return n;
}

void lancius_graph_destroy(lancius_graph* g) {
    if (!g) return;
    lancius_graph_release_owned(g);
    free(g->nodes);
    free(g->rt_states); /* A1 */
    lancius_arena_destroy(g->arena);
    free(g);
}

static lancius_node* alloc_node(lancius_graph* g, lancius_opcode op, uint8_t ndim, uint32_t in_count) {
    if (!g || !g->arena) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (ndim == 0 || ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (g->next_id == UINT32_MAX) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }

    const lancius_node** inputs = NULL;
    if (in_count > 0) {
        inputs = (const lancius_node**)lancius_arena_alloc(
            g->arena,
            sizeof(lancius_node*) * in_count,
            8
        );
        if (!inputs) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    }

    lancius_node* n = (lancius_node*)lancius_arena_alloc(g->arena, sizeof(lancius_node), 8);
    if (!n) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }

    memset(n, 0, sizeof(lancius_node));
    n->id = g->next_id++;

    lancius_ensure_runtime_capacity(g, n->id);
    if (!g->rt_states || n->id >= g->rt_cap) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    if (g->rt_states && n->id < g->rt_cap) {
        n->rt = &g->rt_states[n->id];
        n->rt->buffer = NULL;
        n->rt->buffer_int8 = NULL;
        n->rt->buffer_f32 = NULL;
        n->rt->dtype = LANCIUS_DTYPE_FP64;
        n->rt->scale = 1.0;
        n->rt->scale_per_channel = NULL;
        n->rt->scale_channels = 0;
        n->rt->owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->f32_owner = LANCIUS_MEMORY_EXTERNAL;
        n->rt->offset = 0;
        n->rt->flags = 0;
    }

    n->op = op;
    n->ndim = ndim;
    n->input_count = in_count;
    n->inputs = inputs;

    size_t before = g->node_count;
    track(g, n);
    if (g->node_count == before) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    return n;
}

lancius_node* lancius_input(lancius_graph* g, size_t r, size_t c) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t s[2] = {r, c};
    if (lancius_validate_shape(s, 2) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_INPUT, 2, 0);
    if(n) { n->shape[0] = r; n->shape[1] = c; } return n;
}
lancius_node* lancius_input_4d(lancius_graph* g, size_t n_dim, size_t c, size_t h, size_t w) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t s[4] = {n_dim, c, h, w};
    if (lancius_validate_shape(s, 4) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_INPUT, 4, 0);
    if(n) { n->shape[0] = n_dim; n->shape[1] = c; n->shape[2] = h; n->shape[3] = w; } return n;
}

lancius_node* lancius_input_3d(lancius_graph* g, size_t d0, size_t d1, size_t d2) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t s[3] = {d0, d1, d2};
    if (lancius_validate_shape(s, 3) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_INPUT, 3, 0);
    if (n) {
        n->shape[0] = d0;
        n->shape[1] = d1;
        n->shape[2] = d2;
    }
    return n;
}
lancius_node* lancius_const(lancius_graph* g, double val, size_t r, size_t c) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t s[2] = {r, c};
    if (lancius_validate_shape(s, 2) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONST, 2, 0);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->attr_val = val; } return n;
}
static void broadcast_out_shape(const lancius_node* a, const lancius_node* b, size_t* out) {
    uint8_t nd = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    for (uint8_t i = 0; i < nd; i++) {
        int ai = (int)i - ((int)nd - (int)a->ndim);
        int bi = (int)i - ((int)nd - (int)b->ndim);
        size_t da = (ai < 0) ? 1 : a->shape[ai];
        size_t db = (bi < 0) ? 1 : b->shape[bi];
        out[i] = (da > db) ? da : db;
    }
}
static uint8_t broadcast_out_ndim(const lancius_node* a, const lancius_node* b) {
    return (a->ndim > b->ndim) ? a->ndim : b->ndim;
}
lancius_node* lancius_add(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    // V10S FIX: Allow broadcast

    /* Despot truth: broadcast incompatibility sets sticky error, never silent NULL. */
    if (lancius_validate_binary_broadcast(a, b) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    uint8_t ond = broadcast_out_ndim(a, b);
    lancius_node* n = alloc_node(g, LANCIUS_OP_ADD, ond, 2);
    if (n) { broadcast_out_shape(a, b, n->shape); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_sub(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    // V10S FIX: Allow broadcast

    if (lancius_validate_binary_broadcast(a, b) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    uint8_t ond = broadcast_out_ndim(a, b);
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUB, ond, 2);
    if (n) { broadcast_out_shape(a, b, n->shape); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_mul(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    // V10S FIX: Allow broadcast

    if (lancius_validate_binary_broadcast(a, b) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    uint8_t ond = broadcast_out_ndim(a, b);
    lancius_node* n = alloc_node(g, LANCIUS_OP_MUL, ond, 2);
    if (n) { broadcast_out_shape(a, b, n->shape); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
lancius_node* lancius_matmul(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim < 2 || b->ndim < 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    /* Despot truth: executor is 2D-only (scheduler reads shape[0..1]); an N-D
     * matmul node would silently drop batch dims at execution. Fail loud. */
    if (a->ndim != 2 || b->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    // V10S FIX: NDim-aware dimension extraction (handles 4D Reshape outputs)
    if (lancius_validate_matmul(a->shape, a->ndim, b->shape, b->ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    size_t a_rows = a->shape[a->ndim - 2];
    size_t a_cols = a->shape[a->ndim - 1];
    size_t b_rows = b->shape[b->ndim - 2];
    size_t b_cols = b->shape[b->ndim - 1];

    if (a_cols != b_rows) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MATMUL, 2, 2);
    if (n) { n->shape[0] = a_rows; n->shape[1] = b_cols; n->inputs[0] = a; n->inputs[1] = b; }
    return n;
}
lancius_node* lancius_relu(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_RELU, a->ndim, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; } return n;
}
lancius_node* lancius_softmax(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SOFTMAX, 2, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*2); n->inputs[0] = a; } return n;
}
lancius_node* lancius_sum(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = a; } return n;
}
lancius_node* lancius_broadcast(lancius_graph* g, const lancius_node* a, size_t r, size_t c) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t out[2] = {r, c};
    if (lancius_validate_shape(out, 2) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    size_t ae = 0;
    if (!lancius_node_elements_checked(a, &ae)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    if (a->ndim != 2 && ae != 1) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (ae != 1) {
        size_t in_R = a->shape[0], in_C = a->shape[1];
        int ok = (in_R == 1 && in_C == c) || (in_C == 1 && in_R == r) || (in_R == r && in_C == c);
        if (!ok) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    }
    lancius_node* n = alloc_node(g, LANCIUS_OP_BROADCAST, 2, 1);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->inputs[0] = a; } return n;
}
lancius_node* lancius_transpose(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_TRANSPOSE, 2, 1);
    if (n) { n->shape[0] = a->shape[1]; n->shape[1] = a->shape[0]; n->inputs[0] = a; } return n;
}
lancius_node* lancius_relu_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_a) {
    if (!grad || !fwd_a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_RELU_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_a; } return n;
}
lancius_node* lancius_softmax_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_y) {
    if (!grad || !fwd_y) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SOFTMAX_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_y; } return n;
}
/* v12R2 generic trainable primitives. Same-shape activation; MSE reduces to
 * scalar with exact-shape pred/target. Framework only: no truth semantics. */
lancius_node* lancius_tanh(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_TANH, a->ndim, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; } return n;
}
lancius_node* lancius_tanh_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_y) {
    if (!grad || !fwd_y) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (grad->ndim != fwd_y->ndim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    for (uint8_t i = 0; i < grad->ndim; i++)
        if (grad->shape[i] != fwd_y->shape[i]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_TANH_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_y; } return n;
}
lancius_node* lancius_mse(lancius_graph* g, const lancius_node* pred, const lancius_node* target) {
    if (!pred || !target) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (pred->ndim != target->ndim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    for (uint8_t i = 0; i < pred->ndim; i++)
        if (pred->shape[i] != target->shape[i]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MSE, 2, 2);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = pred; n->inputs[1] = target; } return n;
}
lancius_node* lancius_mse_bwd(lancius_graph* g, const lancius_node* pred, const lancius_node* target, const lancius_node* grad) {
    if (!pred || !target || !grad) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (pred->ndim != target->ndim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    for (uint8_t i = 0; i < pred->ndim; i++)
        if (pred->shape[i] != target->shape[i]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    size_t ge = 0;
    if (!lancius_node_elements_checked(grad, &ge) || ge != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MSE_BWD, pred->ndim, 3);
    if (n) { memcpy(n->shape, pred->shape, sizeof(size_t)*pred->ndim); n->inputs[0] = pred; n->inputs[1] = target; n->inputs[2] = grad; } return n;
}
lancius_node* lancius_sum_axis0(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM_AXIS0, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = a->shape[1]; n->inputs[0] = a; } return n;
}
lancius_node* lancius_sum_axis1(lancius_graph* g, const lancius_node* a) {
    if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SUM_AXIS1, 2, 1);
    if (n) { n->shape[0] = a->shape[0]; n->shape[1] = 1; n->inputs[0] = a; } return n;
}

lancius_node* lancius_conv2d(lancius_graph* g, const lancius_node* in, const lancius_node* w, uint32_t stride, uint32_t pad) {
    if (!in || !w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (in->ndim != 4 || w->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t N = in->shape[0], C_in = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    size_t C_out = w->shape[0], K_h = w->shape[2], K_w = w->shape[3];
    if (C_in != w->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (lancius_validate_conv2d(H_in, W_in, K_h, K_w, stride, pad) != LANCIUS_ERROR_OK) {
        lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE);
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
    if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (in->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    size_t N = in->shape[0], C = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    if (H_in < kernel || W_in < kernel) {
        lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE);
        return NULL;
    }
    if (stride == 0 || kernel == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
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
    if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (in->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    size_t N = in->shape[0];
    size_t flat_dims[3] = {in->shape[1], in->shape[2], in->shape[3]};
    size_t flat = 0;
    if (!lancius_checked_product_shape(flat_dims, 3, &flat)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_FLATTEN, 2, 1);
    if (n) { n->shape[0] = N; n->shape[1] = flat; n->inputs[0] = in; }
    return n;
}


lancius_node* lancius_reshape(lancius_graph* g, const lancius_node* in, uint8_t ndim, size_t s0, size_t s1, size_t s2, size_t s3) {
    if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    size_t resh_out_shape[4] = {s0, s1, s2, s3};
    size_t resh_in_elems = 0;
    if (!lancius_node_elements_checked(in, &resh_in_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    if (lancius_validate_reshape(resh_in_elems, resh_out_shape, ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_RESHAPE, ndim, 1);
    if (n) {
        n->shape[0] = s0; if(ndim>1) n->shape[1] = s1; if(ndim>2) n->shape[2] = s2; if(ndim>3) n->shape[3] = s3;
        n->inputs[0] = in;
    }
    return n;
}
lancius_node* lancius_conv2d_bwd_w(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t k_h, uint32_t k_w, uint32_t stride, uint32_t pad) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!grad || !fwd_in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    /* Despot truth: corrupt 2D inputs yielded wrong-shape grads silently. */
    if (grad->ndim != 4 || fwd_in->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (k_h == 0 || k_w == 0 || stride == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    if (grad->shape[1] == 0 || fwd_in->shape[1] == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
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
    if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (in->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    uint32_t perm_axes[4] = {a0, a1, a2, a3}; /* v12R1-202 */
    if (lancius_validate_permutation(perm_axes, 4) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_INVALID_PERMUTATION); return NULL; }
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
    if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (a->ndim != 3 || b->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MATMUL_BATCHED, 3, 2);
    if (n) {
        n->shape[0] = a->shape[0]; n->shape[1] = a->shape[1]; n->shape[2] = b->shape[2];
        n->inputs[0] = a; n->inputs[1] = b;
    }
    return n;
}

lancius_node* lancius_cross_entropy(lancius_graph* g, const lancius_node* logits, const lancius_node* targets) {
    if (!logits || !targets) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (logits->ndim != 2 || targets->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_CROSS_ENTROPY, 2, 2);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = logits; n->inputs[1] = targets; }
    return n;
}
lancius_node* lancius_cross_entropy_bwd(lancius_graph* g, const lancius_node* logits, const lancius_node* targets, const lancius_node* grad_out) {
    if (!logits || !targets || !grad_out) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (logits->ndim != 2 || targets->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    size_t ge = 0;
    if (!lancius_node_elements_checked(grad_out, &ge) || ge != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_CROSS_ENTROPY_BWD, logits->ndim, 3);
    if (n) { memcpy(n->shape, logits->shape, sizeof(size_t)*logits->ndim); n->inputs[0] = logits; n->inputs[1] = targets; n->inputs[2] = grad_out; }
    return n;
}

lancius_node* lancius_conv2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, const lancius_node* fwd_w, uint32_t stride, uint32_t pad) {
    if (!grad || !fwd_in || !fwd_w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    /* Despot V6 truth: mirror fwd rank/stride/pad checks (was unchecked). */
    if (grad->ndim != 4 || fwd_in->ndim != 4 || fwd_w->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (stride == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    if (fwd_in->shape[1] != fwd_w->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (lancius_validate_conv2d(fwd_in->shape[2], fwd_in->shape[3], fwd_w->shape[2], fwd_w->shape[3], stride, pad) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONV2D_BWD, 4, 3);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->stride = stride; n->pad = pad;
        n->inputs[0] = grad; n->inputs[1] = fwd_in; n->inputs[2] = fwd_w;
    } return n;
}
lancius_node* lancius_maxpool2d_bwd(lancius_graph* g, const lancius_node* grad, const lancius_node* fwd_in, uint32_t kernel, uint32_t stride) {
    if (!grad || !fwd_in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    /* Despot V6 truth: mirror fwd rank/stride checks (was unchecked). */
    if (grad->ndim != 4 || fwd_in->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (kernel == 0 || stride == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    if (fwd_in->shape[2] < kernel || fwd_in->shape[3] < kernel) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MAXPOOL2D_BWD, 4, 2);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->kernel_h = kernel; n->kernel_w = kernel; n->stride = stride;
        n->inputs[0] = grad; n->inputs[1] = fwd_in;
    } return n;
}

lancius_node* lancius_rmsnorm(lancius_graph* g, const lancius_node* in, const lancius_node* gamma) {
    if (!in || !gamma) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (lancius_validate_shape(in->shape, in->ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    size_t total = 0;
    if (!lancius_checked_product_shape(in->shape, in->ndim, &total) || total == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    size_t ge = 0;
    if (!lancius_node_elements_checked(gamma, &ge) || ge == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (total % ge != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_RMSNORM, in->ndim, 2);
    if (n) {
        memcpy(n->shape, in->shape, sizeof(size_t) * in->ndim);
        n->inputs[0] = in; n->inputs[1] = gamma;
    }
    return n;
}

lancius_node* lancius_swiglu(lancius_graph* g, const lancius_node* gate, const lancius_node* up) {
    if (!gate || !up) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (gate->ndim != up->ndim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    for (uint8_t i = 0; i < gate->ndim; i++) {
        if (gate->shape[i] != up->shape[i]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    }
    if (lancius_validate_shape(gate->shape, gate->ndim) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_SWIGLU, gate->ndim, 2);
    if (n) {
        memcpy(n->shape, gate->shape, sizeof(size_t) * gate->ndim);
        n->inputs[0] = gate; n->inputs[1] = up;
    }
    return n;
}

lancius_node* lancius_gqa(lancius_graph* g, const lancius_node* q, const lancius_node* k, const lancius_node* v, uint32_t n_heads_q, uint32_t n_heads_kv) {
    if (!q || !k || !v) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (lancius_validate_gqa(n_heads_q, n_heads_kv) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return NULL; }
    /* Despot V6 truth: Q/K/V must be 3D with matching seq/dim (was unchecked). */
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (q->shape[0] != k->shape[0] || q->shape[0] != v->shape[0]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (q->shape[1] != n_heads_q || k->shape[1] != n_heads_kv || v->shape[1] != n_heads_kv) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    if (q->shape[2] != k->shape[2] || q->shape[2] != v->shape[2]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_GQA, q->ndim, 3);
    if (n) {
        memcpy(n->shape, q->shape, sizeof(size_t) * q->ndim);
        n->kernel_h = n_heads_q;
        n->kernel_w = n_heads_kv;
        n->inputs[0] = q; n->inputs[1] = k; n->inputs[2] = v;
    }
    return n;
}

lancius_node* lancius_rope(lancius_graph* g, const lancius_node* qk, size_t seq_len, size_t n_heads, size_t head_dim) {
    if (!g || !qk) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (seq_len == 0 || n_heads == 0 || head_dim == 0 || (head_dim % 2) != 0) {
        lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE);
        return NULL;
    }
    if (qk->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    if (qk->shape[0] != seq_len || qk->shape[1] != n_heads || qk->shape[2] != 2 * head_dim) {
        lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
        return NULL;
    }
    lancius_node* n = alloc_node(g, LANCIUS_OP_ROPE, 3, 1);
    if (n) {
        n->shape[0] = seq_len;
        n->shape[1] = n_heads;
        n->shape[2] = 2 * head_dim;
        n->inputs[0] = qk;
    }
    return n;
}

lancius_node* lancius_const_scalar(lancius_graph* g, double val, uint8_t ndim) {
    if (!g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (ndim == 0 || ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return NULL; }
    lancius_node* n = alloc_node(g, LANCIUS_OP_CONST, ndim, 0);
    if (n) {
        for (uint8_t i = 0; i < ndim; i++) n->shape[i] = 1;
        n->attr_val = val;
    }
    return n;
}

/* A1: runtime state helpers */

/* Despot truth: hand-built nodes (autodiff fused-clone) need the same rt
 * invariant alloc_node provides; capacity growth can OOM (was ignored). */
int lancius_node_attach_runtime(lancius_graph* g, lancius_node* n) {
    if (!g || !n) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return 0; }
    lancius_ensure_runtime_capacity(g, n->id);
    if (!g->rt_states || n->id >= g->rt_cap) { lancius_set_error(LANCIUS_ERROR_OOM); return 0; }
    n->rt = &g->rt_states[n->id];
    n->rt->buffer = NULL;
    n->rt->buffer_int8 = NULL;
    n->rt->buffer_f32 = NULL;
    n->rt->dtype = LANCIUS_DTYPE_FP64;
    n->rt->scale = 1.0;
    n->rt->owner = LANCIUS_MEMORY_EXTERNAL;
    n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL;
    n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
    n->rt->f32_owner = LANCIUS_MEMORY_EXTERNAL;
    n->rt->offset = 0;
    n->rt->flags = 0;
    n->rt->transformer_state = NULL;
    return 1;
}

lancius_runtime_state* lancius_graph_runtime(lancius_graph* g, uint32_t node_id) {
    if (!g || !g->rt_states || node_id >= g->rt_cap) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    return &g->rt_states[node_id];
}

lancius_runtime_state* lancius_node_rt(const lancius_node* n) {
    if (!n || !n->rt) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    return n->rt;
}

void lancius_runtime_sync_from_legacy(lancius_node* n) {
    if (!n || !n->rt) return;

    n->rt->buffer = n->runtime_data;
    n->rt->buffer_int8 = n->runtime_data_int8;
    n->rt->buffer_f32 = n->runtime_data_f32;
    n->rt->dtype = n->dtype;
    n->rt->scale = n->scale;
}

void lancius_runtime_sync_to_legacy(lancius_node* n) {
    if (!n || !n->rt) return;

    n->runtime_data = (double*)n->rt->buffer;
    n->runtime_data_int8 = n->rt->buffer_int8;
    n->runtime_data_f32 = n->rt->buffer_f32;
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
    if (n->dtype == LANCIUS_DTYPE_FP32) n->rt->f32_owner = owner;
    if (n->dtype == LANCIUS_DTYPE_INT8) n->rt->int8_owner = owner;
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

void lancius_node_bind_external_f32(lancius_node* n, float* data) {
    if (!n) return;

    n->runtime_data_f32 = data;

    if (n->rt) {
        n->rt->buffer_f32 = data;
        n->rt->f32_owner = LANCIUS_MEMORY_EXTERNAL;
        if (data) n->rt->dtype = LANCIUS_DTYPE_FP32;
    }

    if (data) n->dtype = LANCIUS_DTYPE_FP32;
}

void lancius_node_bind_owned_heap_f32(lancius_node* n, float* data) {
    if (!n) return;

    if (!data) {
        lancius_node_bind_external_f32(n, NULL);
        return;
    }

    n->runtime_data_f32 = data;

    if (n->rt) {
        n->rt->buffer_f32 = data;
        n->rt->f32_owner = LANCIUS_MEMORY_OWNED_HEAP;
        n->rt->dtype = LANCIUS_DTYPE_FP32;
    }

    n->dtype = LANCIUS_DTYPE_FP32;
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

        /* Despot truth: f32 has its own owner (was aliased to buffer_owner,
         * leaking owned f32 or freeing external fp64). */
        if (n->rt->f32_owner == LANCIUS_MEMORY_OWNED_HEAP && n->runtime_data_f32) {
            free(n->runtime_data_f32);
            n->runtime_data_f32 = NULL;
            n->rt->buffer_f32 = NULL;
            n->rt->f32_owner = LANCIUS_MEMORY_EXTERNAL;
        }

        /* v12R1: per-channel quantization scales are a plain heap allocation. */
        if (n->rt->scale_per_channel) {
            free(n->rt->scale_per_channel);
            n->rt->scale_per_channel = NULL;
            n->rt->scale_channels = 0;
        }

        if (
            n->rt->buffer_owner == LANCIUS_MEMORY_EXTERNAL &&
            n->rt->int8_owner == LANCIUS_MEMORY_EXTERNAL &&
            n->rt->f32_owner == LANCIUS_MEMORY_EXTERNAL
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
/* v11A1: ownership query helpers */
bool lancius_node_buffer_is_external(const lancius_node* n) {
    return n && n->rt && n->rt->buffer_owner == LANCIUS_MEMORY_EXTERNAL;
}

bool lancius_node_int8_buffer_is_external(const lancius_node* n) {
    return n && n->rt && n->rt->int8_owner == LANCIUS_MEMORY_EXTERNAL;
}

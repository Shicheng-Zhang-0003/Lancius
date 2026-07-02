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

static v8_node* alloc_node(v8_graph* g, v8_opcode op, uint8_t ndim, uint32_t in_count) {
    v8_node* n = (v8_node*)v8_arena_alloc(g->arena, sizeof(v8_node), 8);
    if (!n) return NULL;
    memset(n, 0, sizeof(v8_node));
    n->id = g->next_id++;
    n->op = op;
    n->ndim = ndim;
    n->input_count = in_count;
    if (in_count > 0) {
        n->inputs = (const v8_node**)v8_arena_alloc(g->arena, sizeof(v8_node*) * in_count, 8);
    }
    track(g, n);
    return n;
}

v8_node* v8_input(v8_graph* g, size_t r, size_t c) {
    v8_node* n = alloc_node(g, V8_OP_INPUT, 2, 0);
    if(n) { n->shape[0] = r; n->shape[1] = c; } return n;
}
v8_node* v8_input_4d(v8_graph* g, size_t n_dim, size_t c, size_t h, size_t w) {
    v8_node* n = alloc_node(g, V8_OP_INPUT, 4, 0);
    if(n) { n->shape[0] = n_dim; n->shape[1] = c; n->shape[2] = h; n->shape[3] = w; } return n;
}
v8_node* v8_const(v8_graph* g, double val, size_t r, size_t c) {
    v8_node* n = alloc_node(g, V8_OP_CONST, 2, 0);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->attr_val = val; } return n;
}
v8_node* v8_add(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->ndim != b->ndim) { fprintf(stderr, "[V8 IR FATAL] ADD ndim mismatch: a=%p b=%p\n", a, b); return NULL; }
    for(int i=0; i<a->ndim; i++) if(a->shape[i] != b->shape[i]) { fprintf(stderr, "[V8 IR FATAL] ADD shape mismatch at dim %d: %zu != %zu\n", i, a->shape[i], b->shape[i]); return NULL; }
    v8_node* n = alloc_node(g, V8_OP_ADD, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
v8_node* v8_sub(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->ndim != b->ndim) { fprintf(stderr, "[V8 IR FATAL] ADD ndim mismatch: a=%p b=%p\n", a, b); return NULL; }
    for(int i=0; i<a->ndim; i++) if(a->shape[i] != b->shape[i]) { fprintf(stderr, "[V8 IR FATAL] ADD shape mismatch at dim %d: %zu != %zu\n", i, a->shape[i], b->shape[i]); return NULL; }
    v8_node* n = alloc_node(g, V8_OP_SUB, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
v8_node* v8_mul(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->ndim != b->ndim) { fprintf(stderr, "[V8 IR FATAL] ADD ndim mismatch: a=%p b=%p\n", a, b); return NULL; }
    for(int i=0; i<a->ndim; i++) if(a->shape[i] != b->shape[i]) { fprintf(stderr, "[V8 IR FATAL] ADD shape mismatch at dim %d: %zu != %zu\n", i, a->shape[i], b->shape[i]); return NULL; }
    v8_node* n = alloc_node(g, V8_OP_MUL, a->ndim, 2);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; n->inputs[1] = b; } return n;
}
v8_node* v8_matmul(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) { fprintf(stderr, "[V8 IR FATAL] MATMUL mismatch: a=[%zux%zu] b=[%zux%zu]\n", a?a->shape[0]:0, a?a->shape[1]:0, b?b->shape[0]:0, b?b->shape[1]:0); return NULL; }
    v8_node* n = alloc_node(g, V8_OP_MATMUL, 2, 2);
    if (n) { n->shape[0] = a->shape[0]; n->shape[1] = b->shape[1]; n->inputs[0] = a; n->inputs[1] = b; } return n;
}
v8_node* v8_relu(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_RELU, a->ndim, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*a->ndim); n->inputs[0] = a; } return n;
}
v8_node* v8_softmax(v8_graph* g, const v8_node* a) {
    if (!a || a->ndim != 2) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SOFTMAX, 2, 1);
    if (n) { memcpy(n->shape, a->shape, sizeof(size_t)*2); n->inputs[0] = a; } return n;
}
v8_node* v8_sum(v8_graph* g, const v8_node* a) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = a; } return n;
}
v8_node* v8_broadcast(v8_graph* g, const v8_node* a, size_t r, size_t c) {
    if (!a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_BROADCAST, 2, 1);
    if (n) { n->shape[0] = r; n->shape[1] = c; n->inputs[0] = a; } return n;
}
v8_node* v8_transpose(v8_graph* g, const v8_node* a) {
    if (!a || a->ndim != 2) return NULL;
    v8_node* n = alloc_node(g, V8_OP_TRANSPOSE, 2, 1);
    if (n) { n->shape[0] = a->shape[1]; n->shape[1] = a->shape[0]; n->inputs[0] = a; } return n;
}
v8_node* v8_relu_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_a) {
    if (!grad || !fwd_a) return NULL;
    v8_node* n = alloc_node(g, V8_OP_RELU_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_a; } return n;
}
v8_node* v8_softmax_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_y) {
    if (!grad || !fwd_y) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SOFTMAX_BWD, grad->ndim, 2);
    if (n) { memcpy(n->shape, grad->shape, sizeof(size_t)*grad->ndim); n->inputs[0] = grad; n->inputs[1] = fwd_y; } return n;
}
v8_node* v8_sum_axis0(v8_graph* g, const v8_node* a) {
    if (!a || a->ndim != 2) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM_AXIS0, 2, 1);
    if (n) { n->shape[0] = 1; n->shape[1] = a->shape[1]; n->inputs[0] = a; } return n;
}
v8_node* v8_sum_axis1(v8_graph* g, const v8_node* a) {
    if (!a || a->ndim != 2) return NULL;
    v8_node* n = alloc_node(g, V8_OP_SUM_AXIS1, 2, 1);
    if (n) { n->shape[0] = a->shape[0]; n->shape[1] = 1; n->inputs[0] = a; } return n;
}

v8_node* v8_conv2d(v8_graph* g, const v8_node* in, const v8_node* w, uint32_t stride, uint32_t pad) {
    if (!in || !w || in->ndim != 4 || w->ndim != 4) { fprintf(stderr, "[V8 IR FATAL] CONV2D ndim mismatch\n"); return NULL; }
    size_t N = in->shape[0], C_in = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    size_t C_out = w->shape[0], K_h = w->shape[2], K_w = w->shape[3];
    if (C_in != w->shape[1]) { fprintf(stderr, "[V8 IR FATAL] CONV2D channels mismatch: in=%zu w=%zu\n", C_in, w->shape[1]); return NULL; }
    size_t H_out = (H_in + 2*pad - K_h) / stride + 1;
    size_t W_out = (W_in + 2*pad - K_w) / stride + 1;
    v8_node* n = alloc_node(g, V8_OP_CONV2D, 4, 2);
    if (n) {
        n->shape[0] = N; n->shape[1] = C_out; n->shape[2] = H_out; n->shape[3] = W_out;
        n->kernel_h = K_h; n->kernel_w = K_w; n->stride = stride; n->pad = pad;
        n->inputs[0] = in; n->inputs[1] = w;
    }
    return n;
}

v8_node* v8_maxpool2d(v8_graph* g, const v8_node* in, uint32_t kernel, uint32_t stride) {
    if (!in || in->ndim != 4) return NULL;
    size_t N = in->shape[0], C = in->shape[1], H_in = in->shape[2], W_in = in->shape[3];
    size_t H_out = (H_in - kernel) / stride + 1;
    size_t W_out = (W_in - kernel) / stride + 1;
    v8_node* n = alloc_node(g, V8_OP_MAXPOOL2D, 4, 1);
    if (n) {
        n->shape[0] = N; n->shape[1] = C; n->shape[2] = H_out; n->shape[3] = W_out;
        n->kernel_h = kernel; n->kernel_w = kernel; n->stride = stride; n->pad = 0;
        n->inputs[0] = in;
    }
    return n;
}

v8_node* v8_flatten(v8_graph* g, const v8_node* in) {
    if (!in || in->ndim != 4) return NULL;
    size_t N = in->shape[0];
    size_t flat = in->shape[1] * in->shape[2] * in->shape[3];
    v8_node* n = alloc_node(g, V8_OP_FLATTEN, 2, 1);
    if (n) { n->shape[0] = N; n->shape[1] = flat; n->inputs[0] = in; }
    return n;
}


v8_node* v8_reshape(v8_graph* g, const v8_node* in, uint8_t ndim, size_t s0, size_t s1, size_t s2, size_t s3) {
    if (!in) return NULL;
    v8_node* n = alloc_node(g, V8_OP_RESHAPE, ndim, 1);
    if (n) {
        n->shape[0] = s0; if(ndim>1) n->shape[1] = s1; if(ndim>2) n->shape[2] = s2; if(ndim>3) n->shape[3] = s3;
        n->inputs[0] = in;
    }
    return n;
}
v8_node* v8_conv2d_bwd_w(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, uint32_t k_h, uint32_t k_w, uint32_t stride, uint32_t pad) {
    if (!grad || !fwd_in) return NULL;
    v8_node* n = alloc_node(g, V8_OP_CONV2D_BWD_W, 4, 2);
    if (n) {
        n->shape[0] = grad->shape[1]; n->shape[1] = fwd_in->shape[1];
        n->shape[2] = k_h; n->shape[3] = k_w;
        n->kernel_h = k_h; n->kernel_w = k_w;
        n->stride = stride; n->pad = pad;
        n->inputs[0] = grad; n->inputs[1] = fwd_in;
    }
    return n;
}



v8_node* v8_permute(v8_graph* g, const v8_node* in, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    if (!in || in->ndim != 4) return NULL;
    v8_node* n = alloc_node(g, V8_OP_PERMUTE, 4, 1);
    if (n) {
        n->axes[0] = a0; n->axes[1] = a1; n->axes[2] = a2; n->axes[3] = a3;
        size_t dims[4] = {in->shape[0], in->shape[1], in->shape[2], in->shape[3]};
        n->shape[0] = dims[a0]; n->shape[1] = dims[a1]; n->shape[2] = dims[a2]; n->shape[3] = dims[a3];
        n->inputs[0] = in;
    }
    return n;
}
v8_node* v8_matmul_batched(v8_graph* g, const v8_node* a, const v8_node* b) {
    if (!a || !b || a->ndim != 3 || b->ndim != 3) return NULL;
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) return NULL;
    v8_node* n = alloc_node(g, V8_OP_MATMUL_BATCHED, 3, 2);
    if (n) {
        n->shape[0] = a->shape[0]; n->shape[1] = a->shape[1]; n->shape[2] = b->shape[2];
        n->inputs[0] = a; n->inputs[1] = b;
    }
    return n;
}

v8_node* v8_cross_entropy(v8_graph* g, const v8_node* logits, const v8_node* targets) {
    if (!logits || !targets || logits->ndim != 2 || targets->ndim != 2) return NULL;
    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) return NULL;
    v8_node* n = alloc_node(g, V8_OP_CROSS_ENTROPY, 2, 2);
    if (n) { n->shape[0] = 1; n->shape[1] = 1; n->inputs[0] = logits; n->inputs[1] = targets; }
    return n;
}
v8_node* v8_cross_entropy_bwd(v8_graph* g, const v8_node* logits, const v8_node* targets, const v8_node* grad_out) {
    if (!logits || !targets || !grad_out) return NULL;
    v8_node* n = alloc_node(g, V8_OP_CROSS_ENTROPY_BWD, logits->ndim, 3);
    if (n) { memcpy(n->shape, logits->shape, sizeof(size_t)*logits->ndim); n->inputs[0] = logits; n->inputs[1] = targets; n->inputs[2] = grad_out; }
    return n;
}

v8_node* v8_conv2d_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, const v8_node* fwd_w, uint32_t stride, uint32_t pad) {
    if (!grad || !fwd_in || !fwd_w) return NULL;
    v8_node* n = alloc_node(g, V8_OP_CONV2D_BWD, 4, 3);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->stride = stride; n->pad = pad;
        n->inputs[0] = grad; n->inputs[1] = fwd_in; n->inputs[2] = fwd_w;
    } return n;
}
v8_node* v8_maxpool2d_bwd(v8_graph* g, const v8_node* grad, const v8_node* fwd_in, uint32_t kernel, uint32_t stride) {
    if (!grad || !fwd_in) return NULL;
    v8_node* n = alloc_node(g, V8_OP_MAXPOOL2D_BWD, 4, 2);
    if (n) {
        memcpy(n->shape, fwd_in->shape, sizeof(size_t)*4);
        n->kernel_h = kernel; n->kernel_w = kernel; n->stride = stride;
        n->inputs[0] = grad; n->inputs[1] = fwd_in;
    } return n;
}

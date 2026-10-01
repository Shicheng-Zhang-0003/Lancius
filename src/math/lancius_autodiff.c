#include "lancius/lancius_autodiff.h"
#include "lancius/lancius_validate.h"
#include "lancius/lancius_checked.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

/* Despot truth: returns 1 on success/neutral-skip, 0 on hard shape failure.
 * Hard failure must abort the whole autodiff (return NULL), never leave a
 * NULL grad that trains as zero. */
static int accum_grad(lancius_graph* g, lancius_node** grad_map, uint32_t fwd_input_id, lancius_node* new_grad, lancius_node** fwd_to_full) {
    /* Despot truth: NULL grad is never neutral (was: builder OOM silently
     * skipped, training that param as zero). Fail loud. */
    if (!new_grad) {
        if (lancius_get_error() == LANCIUS_ERROR_OK) lancius_set_error(LANCIUS_ERROR_INTERNAL);
        return 0;
    }
    lancius_node* full_input = fwd_to_full[fwd_input_id];
    if (!full_input) {
        if (lancius_get_error() == LANCIUS_ERROR_OK) lancius_set_error(LANCIUS_ERROR_INTERNAL);
        return 0;
    }

    bool exact_match = (new_grad->ndim == full_input->ndim);
    if (exact_match) {
        for(uint8_t i=0; i<new_grad->ndim; i++) {
            if(new_grad->shape[i] != full_input->shape[i]) { exact_match = false; break; }
        }
    }

    if (exact_match) {
        if (grad_map[fwd_input_id] == NULL) grad_map[fwd_input_id] = new_grad;
        else {
            lancius_node* acc = lancius_add(g, grad_map[fwd_input_id], new_grad);
            if (!acc) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
            grad_map[fwd_input_id] = acc;
        }
        return 1;
    }

    size_t in_elems = 0, grad_elems = 0;
    bool has_in_elems = lancius_node_elements_checked(full_input, &in_elems);
    bool has_grad_elems = lancius_node_elements_checked(new_grad, &grad_elems);

    if (has_in_elems && has_grad_elems && in_elems == 1 && grad_elems > 1) {
        lancius_node* sum_node = lancius_sum(g, new_grad);
        if (!sum_node) { lancius_set_error(LANCIUS_ERROR_INTERNAL); return 0; }
        {
            if (full_input->ndim == 2) {
                new_grad = sum_node;
            } else if (full_input->ndim >= 1 && full_input->ndim <= 4) {
                /* Safe: only read valid dims, pad remainder with 1. */
                size_t s[4] = {1,1,1,1};
                for (uint8_t i = 0; i < full_input->ndim; i++) s[i] = full_input->shape[i];
                new_grad = lancius_reshape(g, sum_node, full_input->ndim,
                                           s[0], s[1], s[2], s[3]);
                if (!new_grad) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
            } else {
                lancius_set_error(LANCIUS_ERROR_INVALID_RANK);
                return 0;
            }
        }
    } else if (new_grad->ndim == 2 && full_input->ndim == 2) {
        if (new_grad->shape[0] == 1 && new_grad->shape[1] == 1 && (full_input->shape[0] > 1 || full_input->shape[1] > 1)) {
            new_grad = lancius_broadcast(g, new_grad, full_input->shape[0], full_input->shape[1]);
            if (!new_grad) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
        }
        else if ((new_grad->shape[0] > 1 || new_grad->shape[1] > 1) && full_input->shape[0] == 1 && full_input->shape[1] == 1) {
            new_grad = lancius_sum(g, new_grad);
            if (!new_grad) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
        }
        else if (new_grad->shape[0] > 1 && new_grad->shape[1] == full_input->shape[1] && full_input->shape[0] == 1) {
            new_grad = lancius_sum_axis0(g, new_grad);
            if (!new_grad) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
        }
        else if (new_grad->shape[1] > 1 && new_grad->shape[0] == full_input->shape[0] && full_input->shape[1] == 1) {
            new_grad = lancius_sum_axis1(g, new_grad);
            if (!new_grad) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
        }
        else {
            /* Despot truth: 2D shapes differ but match no reduction rule.
             * Verify broadcast compat; if incompatible or unreducible, fail loud
             * instead of inserting a wrong-shaped gradient. */
            bool compat = true;
            {
                uint8_t nd = 2;
                for (uint8_t i = 0; i < nd; i++) {
                    size_t da = full_input->shape[i], db = new_grad->shape[i];
                    /* grad_out is output-shaped (>= input under broadcast), so
                     * input dim must be 1 or equal grad dim. */
                    if (!(da == 1 || da == db)) { compat = false; break; }
                }
            }
            if (!compat || new_grad->shape[0] != full_input->shape[0] || new_grad->shape[1] != full_input->shape[1]) {
                lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
                return 0;
            }
        }
    } else {
        /* Despot truth: N-dim (non-2D) gradient path. Exact match handled above;
         * scalar-input reduction handled above. Remaining cases: */
        if (has_grad_elems && grad_elems == 1) {
            /* Scalar grad_out (SUM output [1,1]) broadcast to N-dim input.
             * Exact math: d_input[i] = grad_out[0] for all i. */
            lancius_node* b = lancius_broadcast_to_shape(g, new_grad, full_input->shape, full_input->ndim);
            if (!b) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
            new_grad = b;
        } else {
            /* Broadcast-compatible partial reduction (e.g. [1,2,1,4] vs
             * [3,2,5,4]) needs per-axis N-dim sums the IR cannot yet express.
             * Fail loud instead of inserting a wrong-shaped gradient. */
            uint8_t nd_g = new_grad->ndim, nd_in = full_input->ndim;
            uint8_t nd = (nd_g > nd_in) ? nd_g : nd_in;
            bool compat = true;
            for (uint8_t i = 0; i < nd; i++) {
                int gi = (int)i - ((int)nd - (int)nd_g);
                int ii = (int)i - ((int)nd - (int)nd_in);
                size_t dg = (gi < 0) ? 1 : new_grad->shape[gi];
                size_t di = (ii < 0) ? 1 : full_input->shape[ii];
                if (!(di == 1 || di == dg)) { compat = false; break; }
                if (dg == 0 || di == 0) { compat = false; break; }
            }
            if (!compat) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
            /* Compatible but shapes differ => needs N-dim axis sums. No silent drop. */
            bool same = (nd_g == nd_in);
            if (same) {
                for (uint8_t i = 0; i < nd_g; i++) {
                    if (new_grad->shape[i] != full_input->shape[i]) { same = false; break; }
                }
            }
            if (!same) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return 0; }
        }
    }

    if (!new_grad) { lancius_set_error(LANCIUS_ERROR_INTERNAL); return 0; }
    if (grad_map[fwd_input_id] == NULL) grad_map[fwd_input_id] = new_grad;
    else {
        lancius_node* acc = lancius_add(g, grad_map[fwd_input_id], new_grad);
        if (!acc) {
            /* Broadcast-incompatible accumulation is a real shape error:
               do not silently drop the gradient. */
            lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
            return 0;
        }
        grad_map[fwd_input_id] = acc;
    }
    return 1;
}

lancius_training_graph* lancius_ir_autodiff(lancius_graph* fwd_g, lancius_node* loss_node) {
    if (!loss_node) return NULL; // Prevent segfault on malformed graphs
    if (!fwd_g || fwd_g->next_id == 0) return NULL;
    /* Despot truth: sticky errors from prior graphs must not poison this build,
     * and any shape failure inside backward aborts the whole training graph. */
    lancius_clear_error();
    lancius_training_graph* tg = (lancius_training_graph*)calloc(1, sizeof(lancius_training_graph));
    if (!tg) return NULL;
    tg->graph = lancius_graph_create();
    if (!tg->graph) { free(tg); return NULL; }
    tg->max_id = fwd_g->next_id;
    tg->grad_nodes = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));
    if (!tg->grad_nodes) { lancius_graph_destroy(tg->graph); free(tg); return NULL; }

    lancius_node** fwd_to_full = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));
    if (!fwd_to_full) { free(tg->grad_nodes); lancius_graph_destroy(tg->graph); free(tg); return NULL; }
    for(uint32_t i=0; i<fwd_g->node_count; i++) {
        lancius_node* old = fwd_g->nodes[i];
        if (!old) continue;
        lancius_node* n = NULL;
        const lancius_node* in0 = (old->input_count > 0 && old->inputs && old->inputs[0]) ? fwd_to_full[old->inputs[0]->id] : NULL;
        const lancius_node* in1 = (old->input_count > 1 && old->inputs && old->inputs[1]) ? fwd_to_full[old->inputs[1]->id] : NULL;
        /* Despot truth: inputs[2] was derefed unguarded (LAYERNORM/GQA/ATTENTION). */
        const lancius_node* in2 = (old->input_count > 2 && old->inputs && old->inputs[2]) ? fwd_to_full[old->inputs[2]->id] : NULL;

        switch(old->op) {
            case LANCIUS_OP_INPUT: {
/* v12R1-204: handle 3D inputs so matmul_batched reconstructs correctly */
/* Pad shape to 4D to avoid OOB reads when ndim < 4 */
size_t s[4] = {1,1,1,1};
for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
if (old->ndim == 4) n = lancius_input_4d(tg->graph, s[0], s[1], s[2], s[3]);
else if (old->ndim == 3) n = lancius_input_3d(tg->graph, s[0], s[1], s[2]);
else n = lancius_input(tg->graph, s[0], s[1]);
if(n) {
    n->runtime_data = old->runtime_data;
    n->runtime_data_int8 = old->runtime_data_int8;
    n->runtime_data_f32 = old->runtime_data_f32;
    n->dtype = old->dtype; n->scale = old->scale;
    lancius_runtime_sync_from_legacy(n);
    if (n->rt) { n->rt->int8_owner = old->rt ? old->rt->int8_owner : LANCIUS_MEMORY_EXTERNAL; n->rt->f32_owner = old->rt ? old->rt->f32_owner : LANCIUS_MEMORY_EXTERNAL; }
}
break;
            }
            case LANCIUS_OP_CONST: {
                size_t s[4] = {1,1,1,1};
                for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
                if (old->ndim == 2) n = lancius_const(tg->graph, old->attr_val, s[0], s[1]);
                else if (old->ndim >= 1 && old->ndim <= 4) {
                    n = lancius_const_scalar(tg->graph, old->attr_val, old->ndim);
                    if (n) {
                        for (uint8_t _i = 0; _i < old->ndim; _i++) n->shape[_i] = old->shape[_i];
                        n->dtype = old->dtype; n->scale = old->scale;
                    }
                } else n = NULL;
                break;
            }
            case LANCIUS_OP_ADD: n = lancius_add(tg->graph, in0, in1); break;
            case LANCIUS_OP_SUB: n = lancius_sub(tg->graph, in0, in1); break;
            case LANCIUS_OP_MUL: n = lancius_mul(tg->graph, in0, in1); break;
            case LANCIUS_OP_MATMUL: n = lancius_matmul(tg->graph, in0, in1); break;
            case LANCIUS_OP_RELU: n = lancius_relu(tg->graph, in0); break;
            case LANCIUS_OP_TRANSPOSE: n = lancius_transpose(tg->graph, in0); break;
            case LANCIUS_OP_SUM: n = lancius_sum(tg->graph, in0); break;
            case LANCIUS_OP_SUM_AXIS0: n = lancius_sum_axis0(tg->graph, in0); break;
            case LANCIUS_OP_SUM_AXIS1: n = lancius_sum_axis1(tg->graph, in0); break;
            case LANCIUS_OP_BROADCAST: {
                size_t s[4] = {1,1,1,1};
                for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
                if (old->ndim == 4) n = lancius_broadcast_4d(tg->graph, in0, s[0], s[1], s[2], s[3]);
                else if (old->ndim == 2) n = lancius_broadcast(tg->graph, in0, s[0], s[1]);
                else n = lancius_broadcast_to_shape(tg->graph, in0, s, old->ndim);
                break;
            }
            case LANCIUS_OP_SOFTMAX: n = lancius_softmax(tg->graph, in0); break;
            case LANCIUS_OP_CROSS_ENTROPY: n = lancius_cross_entropy(tg->graph, in0, in1); break;
            case LANCIUS_OP_TANH: n = lancius_tanh(tg->graph, in0); break;
            case LANCIUS_OP_MSE: n = lancius_mse(tg->graph, in0, in1); break;
            case LANCIUS_OP_PERMUTE: {
                uint32_t axes[4] = {0,0,0,0};
                for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) axes[_i] = old->axes[_i];
                n = lancius_permute(tg->graph, in0, axes[0], axes[1], axes[2], axes[3]);
                break;
            }
            case LANCIUS_OP_MATMUL_BATCHED: n = lancius_matmul_batched(tg->graph, in0, in1); break;
            case LANCIUS_OP_CONV2D: n = lancius_conv2d(tg->graph, in0, in1, old->stride, old->pad); break;
            case LANCIUS_OP_MAXPOOL2D: n = lancius_maxpool2d(tg->graph, in0, old->kernel_h, old->stride); break;
            case LANCIUS_OP_FLATTEN: n = lancius_flatten(tg->graph, in0); break;
            case LANCIUS_OP_RESHAPE: {
                size_t s[4] = {1,1,1,1};
                for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
                n = lancius_reshape(tg->graph, in0, old->ndim, s[0], s[1], s[2], s[3]);
                break;
            }
            /* Despot truth: transformer forward ops clone exactly so fwd_to_full
             * stays complete; backward still fails loud (no wrong grads). */
            case LANCIUS_OP_LAYERNORM: n = lancius_layernorm(tg->graph, in0, in1, in2); break;
            case LANCIUS_OP_RMSNORM: n = lancius_rmsnorm(tg->graph, in0, in1); break;
            case LANCIUS_OP_GELU: n = lancius_gelu(tg->graph, in0); break;
            case LANCIUS_OP_SWIGLU: n = lancius_swiglu(tg->graph, in0, in1); break;
            case LANCIUS_OP_GQA: n = lancius_gqa(tg->graph, in0, in1, in2, old->kernel_h, old->kernel_w); break;
            case LANCIUS_OP_ROPE: {
                size_t s[4] = {1,1,1,1};
                for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
                n = lancius_rope(tg->graph, in0, s[0], s[1], s[2] / 2);
                break;
            }
            case LANCIUS_OP_ATTENTION: n = lancius_attention(tg->graph, in0, in1, in2); break;
            case LANCIUS_OP_NOP: n = NULL; break; /* NOP: no forward clone, no backward */
            /* Despot truth: _BWD nodes must never appear in the forward graph.
             * Fail loud instead of silently cloning them. */
            case LANCIUS_OP_RELU_BWD:
            case LANCIUS_OP_SOFTMAX_BWD:
            case LANCIUS_OP_CROSS_ENTROPY_BWD:
            case LANCIUS_OP_TANH_BWD:
            case LANCIUS_OP_MSE_BWD:
            case LANCIUS_OP_CONV2D_BWD:
            case LANCIUS_OP_CONV2D_BWD_W:
            case LANCIUS_OP_MAXPOOL2D_BWD:
                lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
                free(fwd_to_full);
                free(tg->grad_nodes);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            case LANCIUS_OP_CONV2D_RELU_FUSED:
                // V10S FIX: Manually allocate to preserve the FUSED opcode!
                // Despot truth: rt attach checked (was NULL), output NOT aliased
                // (was shared with fwd graph: UAF + cross-execution overwrite).
                if (tg->graph->next_id == UINT32_MAX) {
                    lancius_set_error(LANCIUS_ERROR_INTERNAL);
                    free(fwd_to_full);
                    free(tg->grad_nodes);
                    lancius_graph_destroy(tg->graph); free(tg);
                    return NULL;
                }
                n = (lancius_node*)lancius_arena_alloc(tg->graph->arena, sizeof(lancius_node), 8);
                if (n) {
                    memset(n, 0, sizeof(lancius_node));
                    n->id = tg->graph->next_id++;
                    if (!lancius_node_attach_runtime(tg->graph, n)) {
                        lancius_set_error(LANCIUS_ERROR_INTERNAL);
                        free(fwd_to_full);
                        free(tg->grad_nodes);
                        lancius_graph_destroy(tg->graph); free(tg);
                        return NULL;
                    }
                    n->op = LANCIUS_OP_CONV2D_RELU_FUSED; // Crucial: Keep the fused opcode
                    n->ndim = 4;
                    n->input_count = 2;
                    n->inputs = (const lancius_node**)lancius_arena_alloc(tg->graph->arena, sizeof(lancius_node*) * 2, 8);
                    if (!n->inputs) {
                        lancius_set_error(LANCIUS_ERROR_INTERNAL);
                        free(fwd_to_full);
                        free(tg->grad_nodes);
                        lancius_graph_destroy(tg->graph); free(tg);
                        return NULL;
                    }
                    n->inputs[0] = in0; n->inputs[1] = in1;
                    /* Pad shape to 4D to avoid OOB reads when ndim < 4 */
                    size_t s[4] = {1,1,1,1};
                    for (uint8_t _i = 0; _i < old->ndim && _i < 4; _i++) s[_i] = old->shape[_i];
                    n->shape[0] = s[0]; n->shape[1] = s[1]; n->shape[2] = s[2]; n->shape[3] = s[3];
                    n->kernel_h = old->kernel_h; n->kernel_w = old->kernel_w; n->stride = old->stride; n->pad = old->pad;
                    n->dtype = LANCIUS_DTYPE_FP64;
                    n->scale = 1.0;
                    n->runtime_data = NULL;
                    lancius_runtime_sync_from_legacy(n);
                    // Track node in training graph
                    if (tg->graph->node_count >= tg->graph->node_cap) {
                        size_t new_cap = tg->graph->node_cap == 0 ? 1024 : (size_t)tg->graph->node_cap * 2;
                        if (new_cap >= (size_t)UINT32_MAX + 1) {
                            lancius_set_error(LANCIUS_ERROR_INTERNAL);
                            free(fwd_to_full);
                            free(tg->grad_nodes);
                            lancius_graph_destroy(tg->graph); free(tg);
                            return NULL;
                        }
                        lancius_node** nn = (lancius_node**)realloc(tg->graph->nodes, sizeof(lancius_node*) * new_cap);
                        if (!nn) {
                            lancius_set_error(LANCIUS_ERROR_OOM);
                            free(fwd_to_full);
                            free(tg->grad_nodes);
                            lancius_graph_destroy(tg->graph); free(tg);
                            return NULL;
                        }
                        tg->graph->nodes = nn;
                        tg->graph->node_cap = (uint32_t)new_cap;
                    }
                    tg->graph->nodes[tg->graph->node_count++] = n;
                }
                break;
            default:
                /* Despot truth: storing NULL and continuing builds a broken
                 * graph (was: silent). Fail loud. */
                lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
                free(fwd_to_full);
                free(tg->grad_nodes);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
        }
        fwd_to_full[old->id] = n;
    }

    /* Despot truth: foreign loss_node id OOB-read/wrote grad_map (was unchecked). */
    if (loss_node->id >= fwd_g->next_id) {
        lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
        free(fwd_to_full); free(tg->grad_nodes);
        lancius_graph_destroy(tg->graph); free(tg);
        return NULL;
    }
    {
        bool member = false;
        for (uint32_t _i = 0; _i < fwd_g->node_count; _i++) {
            if (fwd_g->nodes[_i] == loss_node) { member = true; break; }
        }
        if (!member) {
            lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
            free(fwd_to_full); free(tg->grad_nodes);
            lancius_graph_destroy(tg->graph); free(tg);
            return NULL;
        }
    }

    lancius_node** grad_map = (lancius_node**)calloc(fwd_g->next_id, sizeof(lancius_node*));
    if (!grad_map) { free(fwd_to_full); free(tg->grad_nodes); lancius_graph_destroy(tg->graph); free(tg); return NULL; }
    grad_map[loss_node->id] = lancius_const(tg->graph, 1.0, 1, 1);
    if (!grad_map[loss_node->id]) {
        /* Despot truth: NULL seed (OOM) made the whole loop skip and return
         * an empty grad graph as success (was unchecked). */
        lancius_set_error(LANCIUS_ERROR_OOM);
        free(grad_map); free(fwd_to_full); free(tg->grad_nodes);
        lancius_graph_destroy(tg->graph); free(tg);
        return NULL;
    }

    for (int i = fwd_g->node_count - 1; i >= 0; i--) {
        lancius_node* fwd_n = fwd_g->nodes[i];
        lancius_node* grad_out = grad_map[fwd_n->id];
        if (!grad_out) continue;
        if (fwd_n->op == LANCIUS_OP_INPUT || fwd_n->op == LANCIUS_OP_CONST) continue;
        if (fwd_n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes

        /* Despot truth: all backward ops below access fwd_n->inputs[N]->id.
         * Validate inputs exist before dereferencing. */
        if (!fwd_n->inputs) {
            lancius_set_error(LANCIUS_ERROR_INTERNAL);
            free(grad_map); free(fwd_to_full);
            lancius_graph_destroy(tg->graph); free(tg);
            return NULL;
        }

        if (fwd_n->op == LANCIUS_OP_ADD) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUB) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            lancius_node* neg = lancius_const_scalar(tg->graph, -1.0, grad_out ? grad_out->ndim : 2);
            lancius_node* prod = (neg && grad_out) ? lancius_mul(tg->graph, grad_out, neg) : NULL;
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, prod, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MUL) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            lancius_node* gA = grad_out;
            lancius_node* gB = grad_out;

            /* Despot truth: scalar grad_out ([1,1], 1 elem) lifted to operand
             * shape via exact N-dim broadcast, not 2D-only. */
            {
                size_t ge = 0;
                if (gA && lancius_node_elements_checked(gA, &ge) && ge == 1 && A && A->ndim >= 1 && A->ndim <= 4) {
                    size_t ae = 0;
                    if (lancius_node_elements_checked(A, &ae) && ae > 1) {
                        if (A->ndim == 2) gA = lancius_broadcast(tg->graph, gA, A->shape[0], A->shape[1]);
                        else gA = lancius_broadcast_to_shape(tg->graph, gA, A->shape, A->ndim);
                    }
                }
                if (gB && lancius_node_elements_checked(gB, &ge) && ge == 1 && B && B->ndim >= 1 && B->ndim <= 4) {
                    size_t be = 0;
                    if (lancius_node_elements_checked(B, &be) && be > 1) {
                        if (B->ndim == 2) gB = lancius_broadcast(tg->graph, gB, B->shape[0], B->shape[1]);
                        else gB = lancius_broadcast_to_shape(tg->graph, gB, B->shape, B->ndim);
                    }
                }
            }

            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_mul(tg->graph, gA, B), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_mul(tg->graph, gB, A), fwd_to_full);
        
        } else if (
        fwd_n->op == LANCIUS_OP_RMSNORM ||
        fwd_n->op == LANCIUS_OP_SWIGLU ||
        fwd_n->op == LANCIUS_OP_GQA ||
        fwd_n->op == LANCIUS_OP_LAYERNORM ||
        fwd_n->op == LANCIUS_OP_GELU ||
        fwd_n->op == LANCIUS_OP_ROPE ||
        fwd_n->op == LANCIUS_OP_ATTENTION ||
        fwd_n->op == LANCIUS_OP_KV_CACHE_READ ||
        fwd_n->op == LANCIUS_OP_KV_CACHE_WRITE ||
        fwd_n->op == LANCIUS_OP_EMBEDDING
    ) {
            // v10S HONESTY: Fail loudly instead of passing mathematically incorrect gradients.
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            free(grad_map); free(fwd_to_full); 
            lancius_graph_destroy(tg->graph); free(tg); 
            return NULL;
} else if (fwd_n->op == LANCIUS_OP_MATMUL) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_matmul(tg->graph, grad_out, lancius_transpose(tg->graph, B)), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_matmul(tg->graph, lancius_transpose(tg->graph, A), grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_RELU) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_relu_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUM) {
            /* Despot truth: SUM always reduces to scalar [1,1]; grad is scalar
             * broadcast to input shape for any 1..4-D input. Exact math:
             * d_input[i] = grad_out[0]. No [1,1]-for-3D lie. */
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            size_t ge = 0;
            bool is_scalar_grad = (lancius_node_elements_checked(grad_out, &ge) && ge == 1);
            if (full_input && is_scalar_grad && full_input->ndim >= 1 && full_input->ndim <= 4) {
                /* Despot truth: choose exact constructor directly; no trial that
                 * leaves sticky errors to poison the end-of-iteration gate. */
                lancius_clear_error();
                lancius_node* bcast = NULL;
                if (full_input->ndim == 4) {
                    bcast = lancius_broadcast_4d(tg->graph, grad_out, full_input->shape[0], full_input->shape[1], full_input->shape[2], full_input->shape[3]);
                } else if (full_input->ndim == 2) {
                    bcast = lancius_broadcast(tg->graph, grad_out, full_input->shape[0], full_input->shape[1]);
                } else {
                    bcast = lancius_broadcast_to_shape(tg->graph, grad_out, full_input->shape, full_input->ndim);
                }
                if (!bcast) {
                    lancius_set_error(LANCIUS_ERROR_INTERNAL);
                    free(grad_map); free(fwd_to_full);
                    lancius_graph_destroy(tg->graph); free(tg);
                    return NULL;
                }
                accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, bcast, fwd_to_full);
            } else {
                accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            }
        } else if (fwd_n->op == LANCIUS_OP_BROADCAST) {
            /* Despot truth: BROADCAST backward must reduce grad_out over the
             * broadcast dimensions. For y = broadcast(x, shape), dx = sum over
             * dimensions where input_dim == 1 and output_dim > 1. */
            if (!fwd_n->inputs || !fwd_n->inputs[0]) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            if (!full_input) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            /* Pad input shape to 4D */
            size_t in_shape[4] = {1,1,1,1};
            for (uint8_t _i = 0; _i < full_input->ndim && _i < 4; _i++) in_shape[_i] = full_input->shape[_i];
            /* Pad output shape to 4D */
            size_t out_shape[4] = {1,1,1,1};
            for (uint8_t _i = 0; _i < fwd_n->ndim && _i < 4; _i++) out_shape[_i] = fwd_n->shape[_i];
            /* Reduce over dimensions where input_dim == 1 and output_dim > 1.
             * Despot V6 truth: 4D partial reduction is exact via
             * permute+reshape+sum_axis0/1. Math: for y=broadcast(x),
             * dx[I] = sum_{J: bcast(J)=I} grad_out[J].
             * d==0: [D0,R]->sum_axis0; d==3: [P,D3]->sum_axis1;
             * d==1,2: permute axis to front, reduce, permute back.
             * ndim!=2,4 with needed reduction fails loud (no SUM_AXIS_ND). */
            lancius_node* grad = grad_out;
            size_t cur_shape[4] = {out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
            for (int d = 3; d >= 0; d--) {
                if (d < (int)fwd_n->ndim && in_shape[d] == 1 && out_shape[d] > 1) {
                    if (fwd_n->ndim == 2) {
                        if (!grad || grad->ndim != 2) {
                            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
                            free(grad_map); free(fwd_to_full);
                            lancius_graph_destroy(tg->graph); free(tg);
                            return NULL;
                        }
                        if (d == 0) {
                            grad = lancius_sum_axis0(tg->graph, grad);
                            cur_shape[0] = 1;
                        } else {
                            grad = lancius_sum_axis1(tg->graph, grad);
                            cur_shape[1] = 1;
                        }
                    } else if (fwd_n->ndim == 4) {
                        if (!grad || grad->ndim != 4) {
                            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
                            free(grad_map); free(fwd_to_full);
                            lancius_graph_destroy(tg->graph); free(tg);
                            return NULL;
                        }
                        if (grad->shape[0] != cur_shape[0] || grad->shape[1] != cur_shape[1] ||
                            grad->shape[2] != cur_shape[2] || grad->shape[3] != cur_shape[3]) {
                            lancius_set_error(LANCIUS_ERROR_INTERNAL);
                            free(grad_map); free(fwd_to_full);
                            lancius_graph_destroy(tg->graph); free(tg);
                            return NULL;
                        }
                        if (d == 0) {
                            size_t rest = 0;
                            if (!lancius_checked_product_shape(&cur_shape[1], 3, &rest) || rest == 0) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* reshaped = lancius_reshape(tg->graph, grad, 2, cur_shape[0], rest, 1, 1);
                            if (!reshaped) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* summed = lancius_sum_axis0(tg->graph, reshaped);
                            if (!summed) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            grad = lancius_reshape(tg->graph, summed, 4, 1, cur_shape[1], cur_shape[2], cur_shape[3]);
                            cur_shape[0] = 1;
                        } else if (d == 3) {
                            size_t pre = 0;
                            if (!lancius_checked_product_shape(cur_shape, 3, &pre) || pre == 0) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* reshaped = lancius_reshape(tg->graph, grad, 2, pre, cur_shape[3], 1, 1);
                            if (!reshaped) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* summed = lancius_sum_axis1(tg->graph, reshaped);
                            if (!summed) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            grad = lancius_reshape(tg->graph, summed, 4, cur_shape[0], cur_shape[1], cur_shape[2], 1);
                            cur_shape[3] = 1;
                        } else if (d == 1) {
                            lancius_node* perm = lancius_permute(tg->graph, grad, 1, 0, 2, 3);
                            if (!perm) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            size_t rest = cur_shape[0] * cur_shape[2] * cur_shape[3];
                            if (cur_shape[0] != 0 && cur_shape[2] != 0 && cur_shape[3] != 0) {
                                if (cur_shape[0] > SIZE_MAX / cur_shape[2] ||
                                    cur_shape[0] * cur_shape[2] > SIZE_MAX / cur_shape[3]) rest = 0;
                            }
                            if (rest == 0) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* reshaped = lancius_reshape(tg->graph, perm, 2, cur_shape[1], rest, 1, 1);
                            if (!reshaped) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* summed = lancius_sum_axis0(tg->graph, reshaped);
                            if (!summed) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* tmp = lancius_reshape(tg->graph, summed, 4, 1, cur_shape[0], cur_shape[2], cur_shape[3]);
                            if (!tmp) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            grad = lancius_permute(tg->graph, tmp, 1, 0, 2, 3);
                            cur_shape[1] = 1;
                        } else {
                            lancius_node* perm = lancius_permute(tg->graph, grad, 2, 0, 1, 3);
                            if (!perm) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            size_t rest = cur_shape[0] * cur_shape[1] * cur_shape[3];
                            if (cur_shape[0] != 0 && cur_shape[1] != 0 && cur_shape[3] != 0) {
                                if (cur_shape[0] > SIZE_MAX / cur_shape[1] ||
                                    cur_shape[0] * cur_shape[1] > SIZE_MAX / cur_shape[3]) rest = 0;
                            }
                            if (rest == 0) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* reshaped = lancius_reshape(tg->graph, perm, 2, cur_shape[2], rest, 1, 1);
                            if (!reshaped) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* summed = lancius_sum_axis0(tg->graph, reshaped);
                            if (!summed) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            lancius_node* tmp = lancius_reshape(tg->graph, summed, 4, 1, cur_shape[0], cur_shape[1], cur_shape[3]);
                            if (!tmp) {
                                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                                free(grad_map); free(fwd_to_full);
                                lancius_graph_destroy(tg->graph); free(tg);
                                return NULL;
                            }
                            grad = lancius_permute(tg->graph, tmp, 1, 2, 0, 3);
                            cur_shape[2] = 1;
                        }
                    } else {
                        lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
                        free(grad_map); free(fwd_to_full);
                        lancius_graph_destroy(tg->graph); free(tg);
                        return NULL;
                    }
                    if (!grad) {
                        lancius_set_error(LANCIUS_ERROR_INTERNAL);
                        free(grad_map); free(fwd_to_full);
                        lancius_graph_destroy(tg->graph); free(tg);
                        return NULL;
                    }
                }
            }
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_TRANSPOSE) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_transpose(tg->graph, grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MATMUL_BATCHED) {
            /* v12R1-203: correct batched-matmul backward needs a batched
               transpose that the IR does not provide. Fail loudly rather
               than emit mathematically wrong gradients. */
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            free(grad_map); free(fwd_to_full);
            lancius_graph_destroy(tg->graph); free(tg);
            return NULL;
        } else if (fwd_n->op == LANCIUS_OP_PERMUTE) {
            /* Despot truth: corrupt axes[i]>=4 wrote past inv_axes (stack OOB). */
            uint32_t inv_axes[4] = {0,0,0,0};
            if (lancius_validate_permutation(fwd_n->axes, 4) != LANCIUS_ERROR_OK) {
                lancius_set_error(LANCIUS_ERROR_INVALID_PERMUTATION);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            for(int i=0; i<4; i++) inv_axes[fwd_n->axes[i]] = i;
            {
                lancius_node* pg = lancius_permute(tg->graph, grad_out, inv_axes[0], inv_axes[1], inv_axes[2], inv_axes[3]);
                if (!pg || !accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, pg, fwd_to_full)) {
                    lancius_set_error(LANCIUS_ERROR_INTERNAL);
                    free(grad_map); free(fwd_to_full);
                    lancius_graph_destroy(tg->graph); free(tg);
                    return NULL;
                }
            }
        } else if (fwd_n->op == LANCIUS_OP_CROSS_ENTROPY) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* Y = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_cross_entropy_bwd(tg->graph, A, Y, grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SOFTMAX) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_softmax_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->id]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_TANH) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_tanh_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->id]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MSE) {
            lancius_node* P = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* T = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_mse_bwd(tg->graph, P, T, grad_out), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_FLATTEN) {
            if (!fwd_n->inputs || !fwd_n->inputs[0]) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            if (!A) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            /* Pad shape to 4D to avoid OOB reads */
            size_t s[4] = {1,1,1,1};
            for (uint8_t _i = 0; _i < A->ndim && _i < 4; _i++) s[_i] = A->shape[_i];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_reshape(tg->graph, grad_out, A->ndim, s[0], s[1], s[2], s[3]), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_MAXPOOL2D) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_maxpool2d_bwd(tg->graph, grad_out, fwd_to_full[fwd_n->inputs[0]->id], fwd_n->kernel_h, fwd_n->stride), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_CONV2D) {
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* W = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_conv2d_bwd(tg->graph, grad_out, A, W, fwd_n->stride, fwd_n->pad), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_conv2d_bwd_w(tg->graph, grad_out, A, fwd_n->kernel_h, fwd_n->kernel_w, fwd_n->stride, fwd_n->pad), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            // V9 Fix: Backward pass for fused Conv2D + ReLU
            lancius_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            lancius_node* W = fwd_to_full[fwd_n->inputs[1]->id];
            lancius_node* fused_out = fwd_to_full[fwd_n->id];
            lancius_node* masked_grad = lancius_relu_bwd(tg->graph, grad_out, fused_out);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, lancius_conv2d_bwd(tg->graph, masked_grad, A, W, fwd_n->stride, fwd_n->pad), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, lancius_conv2d_bwd_w(tg->graph, masked_grad, A, fwd_n->kernel_h, fwd_n->kernel_w, fwd_n->stride, fwd_n->pad), fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUM_AXIS0) {
            /* Despot truth: SUM_AXIS0 [R,C]->[1,C]; grad [1,C] broadcast to [R,C].
             * Exact: d_input[r,c] = grad_out[0,c]. */
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            if (!full_input || full_input->ndim != 2) {
                lancius_set_error(LANCIUS_ERROR_INVALID_RANK);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            lancius_node* b = lancius_broadcast(tg->graph, grad_out, full_input->shape[0], full_input->shape[1]);
            if (!b) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, b, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_SUM_AXIS1) {
            /* Despot truth: SUM_AXIS1 [R,C]->[R,1]; grad [R,1] broadcast to [R,C].
             * Exact: d_input[r,c] = grad_out[r,0]. */
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            if (!full_input || full_input->ndim != 2) {
                lancius_set_error(LANCIUS_ERROR_INVALID_RANK);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            lancius_node* b = lancius_broadcast(tg->graph, grad_out, full_input->shape[0], full_input->shape[1]);
            if (!b) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, b, fwd_to_full);
        } else if (fwd_n->op == LANCIUS_OP_RESHAPE) {
            /* Despot truth: RESHAPE is a pure reorder; grad reshapes back to
             * input shape. Element counts already validated at build. */
            if (!fwd_n->inputs || !fwd_n->inputs[0]) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            lancius_node* full_input = fwd_to_full[fwd_n->inputs[0]->id];
            if (!full_input) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            /* Pad shape to 4D to avoid OOB reads */
            size_t s[4] = {1,1,1,1};
            for (uint8_t _i = 0; _i < full_input->ndim && _i < 4; _i++) s[_i] = full_input->shape[_i];
            lancius_node* rg = lancius_reshape(tg->graph, grad_out, full_input->ndim,
                s[0], s[1], s[2], s[3]);
            if (!rg) {
                lancius_set_error(LANCIUS_ERROR_INTERNAL);
                free(grad_map); free(fwd_to_full);
                lancius_graph_destroy(tg->graph); free(tg);
                return NULL;
            }
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, rg, fwd_to_full);
        } else {
            /* Despot truth: every forward op reaching backward must have an
             * explicit VJP above. Silent drop is a mathematical lie. */
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            free(grad_map); free(fwd_to_full);
            lancius_graph_destroy(tg->graph); free(tg);
            return NULL;
        }
        /* Despot truth: any sticky shape error from accum_grad or builders
         * aborts the whole training graph. A NULL grad must never train as zero. */
        if (lancius_get_error() != LANCIUS_ERROR_OK) {
            lancius_set_error(LANCIUS_ERROR_INTERNAL);
            free(grad_map); free(fwd_to_full);
            lancius_graph_destroy(tg->graph); free(tg);
            return NULL;
        }
    }

    for(uint32_t i=0; i<fwd_g->next_id; i++) tg->grad_nodes[i] = grad_map[i];
    tg->loss_node = fwd_to_full[loss_node->id];

    free(grad_map); free(fwd_to_full);
    return tg;
}

void lancius_training_graph_destroy(lancius_training_graph* tg) {
    if (!tg) return;
    free(tg->grad_nodes);
    lancius_graph_destroy(tg->graph);
    free(tg);
}

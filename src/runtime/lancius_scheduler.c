#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_memory_planner.h"
#include "lancius/lancius_transformer.h"
#include "lancius/lancius_vision_ops.h"
#include "lancius/lancius_kernels.h"
#include "lancius/lancius_validate.h"
#include "lancius/lancius_checked.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

lancius_schedule* lancius_ir_schedule(lancius_graph* g) {
    if (!g || g->node_count == 0) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return NULL; }
    lancius_schedule* sched = (lancius_schedule*)calloc(1, sizeof(lancius_schedule));
    if (!sched) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    if (g->next_id == 0 || !g->nodes) { free(sched); lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return NULL; }
    if (g->next_id > LANCIUS_MAX_TENSOR_ELEMS) { free(sched); lancius_set_error(LANCIUS_ERROR_LIMIT); return NULL; }
    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    lancius_node** queue = (lancius_node**)malloc(sizeof(lancius_node*) * (size_t)g->node_count);
    if (!in_degree || !queue) { free(in_degree); free(queue); free(sched); lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }

    for (uint32_t i = 0; i < g->node_count; i++) {
        if (!g->nodes[i] || g->nodes[i]->id >= g->next_id) { free(in_degree); free(queue); free(sched->waves); free(sched); lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return NULL; }
        in_degree[g->nodes[i]->id] = g->nodes[i]->input_count;
    }

    uint32_t wave_cap = 16;
    sched->waves = (lancius_wave*)malloc(sizeof(lancius_wave) * wave_cap);
    if (!sched->waves) { free(in_degree); free(queue); free(sched); lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    uint32_t processed = 0;

    while (processed < g->node_count) {
        uint32_t q_tail = 0;
        for (uint32_t i = 0; i < g->node_count; i++) {
            lancius_node* n = g->nodes[i];
            if (in_degree[n->id] == 0) {
                queue[q_tail++] = n;
                in_degree[n->id] = UINT32_MAX;
            }
        }
        if (q_tail == 0) {
            lancius_set_error(LANCIUS_ERROR_GRAPH_CYCLE);
            for (uint32_t w = 0; w < sched->wave_count; w++) free(sched->waves[w].nodes);
            free(sched->waves); free(sched);
            free(in_degree); free(queue);
            return NULL;
        }
        if (sched->wave_count >= wave_cap) {
            if (wave_cap > UINT32_MAX / 2) {
                lancius_set_error(LANCIUS_ERROR_OOM);
                for (uint32_t w = 0; w < sched->wave_count; w++) free(sched->waves[w].nodes);
                free(sched->waves); free(sched);
                free(in_degree); free(queue);
                return NULL;
            }
            wave_cap *= 2;
            lancius_wave* nw = (lancius_wave*)realloc(sched->waves, sizeof(lancius_wave) * (size_t)wave_cap);
            if (!nw) {
                lancius_set_error(LANCIUS_ERROR_OOM);
                for (uint32_t w = 0; w < sched->wave_count; w++) free(sched->waves[w].nodes);
                free(sched->waves); free(sched);
                free(in_degree); free(queue);
                return NULL;
            }
            sched->waves = nw;
        }
        lancius_wave* w = &sched->waves[sched->wave_count++];
        w->node_count = q_tail;
        w->nodes = (lancius_node**)malloc(sizeof(lancius_node*) * (size_t)q_tail);
        if (!w->nodes) {
            lancius_set_error(LANCIUS_ERROR_OOM);
            sched->wave_count--;
            for (uint32_t ww = 0; ww < sched->wave_count; ww++) free(sched->waves[ww].nodes);
            free(sched->waves); free(sched);
            free(in_degree); free(queue);
            return NULL;
        }
        memcpy(w->nodes, queue, sizeof(lancius_node*) * q_tail);

        for (uint32_t i = 0; i < g->node_count; i++) {
            lancius_node* n = g->nodes[i];
            if (in_degree[n->id] == UINT32_MAX) continue;
            bool ready = true;
            for (uint32_t j = 0; j < n->input_count; j++) {
                if (!n->inputs || !n->inputs[j] || n->inputs[j]->id >= g->next_id) { ready = false; break; }
                if (in_degree[n->inputs[j]->id] != UINT32_MAX) { ready = false; break; }
            }
            if (ready) in_degree[n->id] = 0;
        }
        processed += q_tail;
    }
    free(in_degree); free(queue);
    return sched;
}

static void execute_permute(lancius_node* n);
static void execute_matmul_batched(lancius_node* n);

/* v12R1 hostile fix: N-dim broadcast-aware binary elementwise.
 * Previous code did flat a[k]+b[k], which is wrong + OOB whenever
 * shapes differ with a 1-dim (e.g. [2,2] vs [1,2]). Correct math:
 * out[I] = A[bcast(I)] OP B[bcast(I)], where bcast maps dim->0 if input dim==1. */
static void execute_broadcast_binary(lancius_node* n, int op) {
    if (!n || n->input_count < 2 || !n->inputs) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
    const lancius_node* A = n->inputs[0];
    const lancius_node* B = n->inputs[1];
    if (!A || !B) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    double* a = A->runtime_data; double* b = B->runtime_data; double* o = n->runtime_data;
    if (!a || !b || !o) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    uint8_t nd = n->ndim;
    if (nd == 0 || nd > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
    if (op < 0 || op > 2) { lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP); return; }
    if (A->ndim > nd || B->ndim > nd) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
    if (A->ndim == 0 || A->ndim > 4 || B->ndim == 0 || B->ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
    /* Trailing-rank alignment (NumPy semantics): leading missing dims are 1. */
    size_t out_shape[4] = {1,1,1,1}, a_shape[4] = {1,1,1,1}, b_shape[4] = {1,1,1,1};
    for (uint8_t i = 0; i < nd && i < 4; i++) {
        out_shape[i] = n->shape[i];
    }
    {
        uint8_t a_off = nd - A->ndim, b_off = nd - B->ndim;
        for (uint8_t i = 0; i < A->ndim; i++) a_shape[a_off + i] = A->shape[i];
        for (uint8_t i = 0; i < B->ndim; i++) b_shape[b_off + i] = B->shape[i];
    }
    /* Compatibility: each dim must satisfy a==1 || b==1 || a==b, and out==max(a,b). */
    for (uint8_t i = 0; i < nd; i++) {
        size_t da = a_shape[i], db = b_shape[i], dout = out_shape[i];
        if (da != 1 && db != 1 && da != db) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        size_t expect = (da > db) ? da : db;
        if (dout != expect) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
    }
    size_t a_stride[4] = {0,0,0,0}, b_stride[4] = {0,0,0,0};
    a_stride[nd-1] = 1; b_stride[nd-1] = 1;
    for (int i = (int)nd-2; i >= 0; i--) {
        size_t an = a_shape[i+1] ? a_shape[i+1] : 1;
        size_t bn = b_shape[i+1] ? b_shape[i+1] : 1;
        if (an && a_stride[i+1] > SIZE_MAX / an) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        if (bn && b_stride[i+1] > SIZE_MAX / bn) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        a_stride[i] = a_stride[i+1] * an;
        b_stride[i] = b_stride[i+1] * bn;
    }
    size_t total = 1;
    for (uint8_t i = 0; i < nd; i++) {
        if (out_shape[i] == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (out_shape[i] && total > SIZE_MAX / out_shape[i]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        total *= out_shape[i];
    }
    size_t a_elems = 0, b_elems = 0;
    if (!lancius_node_elements_checked(A, &a_elems) || !lancius_node_elements_checked(B, &b_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
    if (a_elems == 0 || b_elems == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
    size_t idx[4] = {0,0,0,0};
    for (size_t lin = 0; lin < total; lin++) {
        size_t rem = lin;
        for (int d = (int)nd-1; d >= 0; d--) {
            idx[d] = out_shape[d] ? (rem % out_shape[d]) : 0;
            rem /= (out_shape[d] ? out_shape[d] : 1);
        }
        size_t ai = 0, bi = 0;
        for (uint8_t d = 0; d < nd; d++) {
            size_t a_c = (a_shape[d] == 1) ? 0 : idx[d];
            size_t b_c = (b_shape[d] == 1) ? 0 : idx[d];
            if (a_c && a_stride[d] && a_c > (SIZE_MAX - ai) / a_stride[d]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            if (b_c && b_stride[d] && b_c > (SIZE_MAX - bi) / b_stride[d]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            ai += a_c * a_stride[d];
            bi += b_c * b_stride[d];
        }
        if (ai >= a_elems || bi >= b_elems) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        double av = a[ai], bv = b[bi];
        if (op == 0) o[lin] = av + bv;
        else if (op == 1) o[lin] = av - bv;
        else o[lin] = av * bv;
    }
}

static void execute_node_math(lancius_node* n) {
    if (!n) return;
    lancius_runtime_sync_from_legacy(n); /* A1 */

    /*
     * v11A2 Section 10/11:
     * FP32 tensors use runtime_data_f32.
     */
    if (n->dtype == LANCIUS_DTYPE_FP32) {
        if (!n->runtime_data_f32) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

        if (n->op != LANCIUS_OP_MATMUL &&
            n->op != LANCIUS_OP_INPUT &&
            n->op != LANCIUS_OP_CONST) {
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_DTYPE);
            return;
        }
    } else if (!n->runtime_data) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return;
    }

    size_t elements = 0;
    if (!lancius_node_elements_checked(n, &elements)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }

    // CRITICAL FIX: Execute Cross-Entropy BEFORE the Vision Ops router hijacks it
    if (n->op == LANCIUS_OP_CROSS_ENTROPY) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* x = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!x || !y) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        /* Despot truth: R==0/C==0 guards (bwd had them, fwd divided by R). */
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        /* Despot truth: forward and backward must use the same R/C source.
         * Forward uses input shape; backward uses output shape. They must match. */
        if (n->shape[0] != 1 || n->shape[1] != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        {
            size_t xe = 0, ye = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &xe) || !lancius_node_elements_checked(n->inputs[1], &ye) || xe != ye || xe != R * C) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        double total_loss = 0.0;
        for(size_t r=0; r<R; r++) {
            double max_val = x[r*C];
            for(size_t c=1; c<C; c++) if(x[r*C+c] > max_val) max_val = x[r*C+c];
            double sum_exp = 0.0;
            for(size_t c=0; c<C; c++) sum_exp += exp(x[r*C+c] - max_val);
            /* Despot truth: degenerate softmax denominator is NUMERICAL, not 1e30.
               Must match BWD which fails loud — no silent masking. */
            if (sum_exp <= 0.0 || sum_exp != sum_exp) { lancius_set_error(LANCIUS_ERROR_NUMERICAL); return; }
            double log_sum_exp = log(sum_exp) + max_val;
            for(size_t c=0; c<C; c++) {
                double yc = y[r*C+c];
                if (yc > 0.0) total_loss -= yc * (x[r*C+c] - log_sum_exp);
            }
        }
        n->runtime_data[0] = total_loss / R;
        if (n->runtime_data[0] < 0.0 && n->runtime_data[0] > -1e-12) n->runtime_data[0] = 0.0;
        return;
    

    }
    else if (n->op == LANCIUS_OP_CROSS_ENTROPY_BWD) {
        if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* x = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data; double* g = n->inputs[2]->runtime_data;
        if (!x || !y || !g) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t R = n->shape[0]; size_t C = n->shape[1];
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        /* Despot V6 truth: forward uses input shape R/C; backward output
         * shape must match inputs elementwise, grad must be scalar. */
        {
            size_t xe = 0, ye = 0, ge = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &xe) || !lancius_node_elements_checked(n->inputs[1], &ye) || xe != ye || xe != R * C) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            if (!lancius_node_elements_checked(n->inputs[2], &ge) || ge != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        double scale = g[0] / R;
        for(size_t r=0; r<R; r++) {
            double max_val = x[r*C];
            for(size_t c=1; c<C; c++) if(x[r*C+c] > max_val) max_val = x[r*C+c];
            double sum_exp = 0.0;
            for(size_t c=0; c<C; c++) sum_exp += exp(x[r*C+c] - max_val);
            if (sum_exp <= 0.0 || sum_exp != sum_exp) { lancius_set_error(LANCIUS_ERROR_NUMERICAL); return; }
            for(size_t c=0; c<C; c++) {
                double sm = exp(x[r*C+c] - max_val) / sum_exp;
                n->runtime_data[r*C+c] = (sm - y[r*C+c]) * scale;
            }
        }
        return;
    }

    
    else if (n->op == LANCIUS_OP_LAYERNORM) {
        if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        double* gamma = n->inputs[1]->runtime_data;
        double* beta = n->inputs[2]->runtime_data;
        if(!in || !gamma || !beta) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

        size_t hidden = 0;
        if (!lancius_node_elements_checked(n->inputs[1], &hidden) || hidden == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        size_t be = 0;
        if (!lancius_node_elements_checked(n->inputs[2], &be) || be != hidden) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        size_t total = 0;
        if (!lancius_node_elements_checked(n, &total) || total % hidden != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        /* Despot V6 truth: input elems must equal output elems (was OOB). */
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != total) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        size_t num_instances = total / hidden;

        kernel_layernorm(n->runtime_data, in, gamma, beta, num_instances, hidden, LANCIUS_NORM_EPS);
    }
    else if (n->op == LANCIUS_OP_GELU) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        if(!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        /* Despot truth: elementwise loops read in[k] for k<elements(output). */
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        kernel_gelu(n->runtime_data, in, elements);
    }

    
    else if (n->op == LANCIUS_OP_ATTENTION) {
        /* Despot V6 truth: validate inputs before deref (was NULL-deref). */
        if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        const lancius_node* q_node = n->inputs[0];
        const lancius_node* k_node = n->inputs[1];
        const lancius_node* v_node = n->inputs[2];

        double* q = q_node->runtime_data;
        if (!q) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

        size_t q_seq = q_node->shape[0];
        size_t n_heads = q_node->shape[1];
        size_t head_dim = q_node->shape[2];

        /*
         * v11A2 cache-aware attention contract:
         *
         * If this attention node has a bound KV-cache and Q is a single
         * token, execute generation against the cache's active length.
         *
         * This removes the A1 demo requirement to mutate K/V IR shapes
         * on every generated token.
         */
        lancius_kv_cache* cache = NULL;
        if (n->rt && n->rt->transformer_state) {
            cache = (lancius_kv_cache*)n->rt->transformer_state;
        }

        if (cache && q_seq == 1) {
            size_t active_seq = lancius_kv_cache_seq_len(cache);
            const double* k_cache = (const double*)lancius_kv_cache_k_buffer(cache, NULL);
            const double* v_cache = (const double*)lancius_kv_cache_v_buffer(cache, NULL);

            if (!k_cache || !v_cache || active_seq == 0) {
                lancius_set_error(LANCIUS_ERROR_NULL_PTR);
                return;
            }
            // Hostile fix: cache heads/dim must match query, else OOB read in kernel
            if (lancius_kv_cache_n_heads(cache) != n_heads ||
                lancius_kv_cache_head_dim(cache) != head_dim) {
                lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
                return;
            }

            kernel_attention_kv_cache(
                n->runtime_data,
                q,
                k_cache,
                v_cache,
                active_seq,
                n_heads,
                head_dim
            );
            return;
        }

        double* k = k_node ? k_node->runtime_data : NULL;
        double* v = v_node ? v_node->runtime_data : NULL;
        if (!k || !v) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        // Hostile fix: validate K/V heads/dim match Q before kernel
        if (k_node->shape[1] != n_heads || k_node->shape[2] != head_dim ||
            v_node->shape[1] != n_heads || v_node->shape[2] != head_dim) {
            lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
            return;
        }

        size_t kv_seq = k_node->shape[0];

        if (q_seq == kv_seq) {
            kernel_attention(n->runtime_data, q, k, v, q_seq, n_heads, head_dim);
        } else {
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            return; /* v11S: do not abort — return error to caller */
        }
    }

    else if (n->op == LANCIUS_OP_RMSNORM) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        double* gamma = n->inputs[1]->runtime_data;
        if(!in || !gamma) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

        size_t hidden = 0;
        if (!lancius_node_elements_checked(n->inputs[1], &hidden) || hidden == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        size_t total = 0;
        if (!lancius_node_elements_checked(n, &total) || total % hidden != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        /* Despot V6 truth: input elems must equal output elems (was OOB). */
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != total) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        size_t num_instances = total / hidden;

        kernel_rmsnorm(n->runtime_data, in, gamma, num_instances, hidden, LANCIUS_NORM_EPS);
    }
        else if (n->op == LANCIUS_OP_SWIGLU) {
            if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
            double* gate = n->inputs[0]->runtime_data;
            double* up = n->inputs[1]->runtime_data;
            if(!gate || !up) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
            {
                size_t ge = 0, ue = 0;
                if (!lancius_node_elements_checked(n->inputs[0], &ge) || !lancius_node_elements_checked(n->inputs[1], &ue) || ge != elements || ue != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            }
            kernel_swiglu(n->runtime_data, gate, up, elements);
        }
        else if (n->op == LANCIUS_OP_GQA) {
            /* Despot V6 truth: validate inputs before deref (was NULL-deref). */
            if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
            double* q = n->inputs[0]->runtime_data;
            double* k = n->inputs[1]->runtime_data;
            double* v = n->inputs[2]->runtime_data;
            if(!q || !k || !v) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
            // Hostile fix: validate Q/K/V shapes match declared heads/dim before kernel (prevents OOB)
            {
                const lancius_node* qn = n->inputs[0];
                const lancius_node* kn = n->inputs[1];
                const lancius_node* vn = n->inputs[2];
                size_t seq = n->shape[0], hq = n->kernel_h, hk = n->kernel_w, dim = n->shape[2];
                if (qn->ndim != 3 || kn->ndim != 3 || vn->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
                if (qn->shape[0] != seq || qn->shape[1] != hq || qn->shape[2] != dim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
                if (kn->shape[0] != seq || kn->shape[1] != hk || kn->shape[2] != dim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
                if (vn->shape[0] != seq || vn->shape[1] != hk || vn->shape[2] != dim) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
                if (hq % (hk ? hk : 1) != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            }
            kernel_gqa(n->runtime_data, q, k, v, n->shape[0], n->kernel_h, n->kernel_w, n->shape[2]);
        }
    else if (n->op == LANCIUS_OP_ROPE) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* qk = n->inputs[0]->runtime_data;
        if(!qk) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t seq_len = n->shape[0];
        size_t n_heads = n->shape[1];
        size_t head_dim_x2 = n->shape[2];
        if (seq_len == 0 || n_heads == 0 || head_dim_x2 == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (head_dim_x2 % 2 != 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        size_t head_dim = head_dim_x2 / 2;
        if (head_dim % 2 != 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        size_t elems = 0;
        if (!lancius_node_elements_checked(n, &elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (elems > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        /* Despot truth: qk must hold >= elems; offsets must not wrap; no (int) truncation. */
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != elems) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        if (n_heads > SIZE_MAX / head_dim_x2) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        if (seq_len > 0 && (n_heads * head_dim_x2) > SIZE_MAX / seq_len) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        if (seq_len > (size_t)INT32_MAX) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        memcpy(n->runtime_data, qk, elems * sizeof(double)); // Preserve SSA

        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < n_heads; h++) {
                double* head_base = n->runtime_data + s * (n_heads * head_dim_x2) + h * head_dim_x2;
                double* q = head_base;
                double* k = head_base + head_dim;
                kernel_rope(q, k, 1, 1, 1, head_dim, (int)s);
            }
        }
    }

    else if (n->op == LANCIUS_OP_PERMUTE) {
        execute_permute(n);
        return;
    }
    else if (n->op == LANCIUS_OP_MATMUL_BATCHED) {
        execute_matmul_batched(n);
        return;
    }

    /* v11A1 repair: transformer ops are not vision ops */
    if (n->op == LANCIUS_OP_LAYERNORM ||
        n->op == LANCIUS_OP_GELU ||
        n->op == LANCIUS_OP_ATTENTION ||
        n->op == LANCIUS_OP_RMSNORM ||
        n->op == LANCIUS_OP_SWIGLU ||
        n->op == LANCIUS_OP_GQA ||
        n->op == LANCIUS_OP_ROPE) {
        return;
    }

    /* v12R2 generic trainable primitives. New ids sort after CONV2D, so like
     * CROSS_ENTROPY they must be handled before the vision-op router below.
     * Framework only: bounded activation + regression loss, no truth semantics. */
    if (n->op == LANCIUS_OP_TANH) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data;
        if (!a || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        for (size_t k = 0; k < elements; k++) n->runtime_data[k] = tanh(a[k]);
        return;
    } else if (n->op == LANCIUS_OP_TANH_BWD) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* g = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!g || !y || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            size_t ge = 0, ye = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ge) || !lancius_node_elements_checked(n->inputs[1], &ye) || ge != elements || ye != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        for (size_t k = 0; k < elements; k++) { double d = 1.0 - y[k] * y[k]; n->runtime_data[k] = g[k] * d; }
        return;
    } else if (n->op == LANCIUS_OP_MSE) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* p = n->inputs[0]->runtime_data; double* t = n->inputs[1]->runtime_data;
        if (!p || !t || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t pe = 0, te = 0;
        if (!lancius_node_elements_checked(n->inputs[0], &pe) || !lancius_node_elements_checked(n->inputs[1], &te) || pe != te || pe == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        double acc = 0.0;
        for (size_t k = 0; k < pe; k++) { double d = p[k] - t[k]; acc += d * d; }
        n->runtime_data[0] = acc / (double)pe;
        return;
    } else if (n->op == LANCIUS_OP_MSE_BWD) {
        if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* p = n->inputs[0]->runtime_data; double* t = n->inputs[1]->runtime_data; double* g = n->inputs[2]->runtime_data;
        if (!p || !t || !g || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t pe = 0, te = 0, ge = 0;
        if (!lancius_node_elements_checked(n->inputs[0], &pe) || !lancius_node_elements_checked(n->inputs[1], &te) || pe != te || pe == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        if (!lancius_node_elements_checked(n->inputs[2], &ge) || ge != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        double scale = (2.0 * g[0]) / (double)pe;
        for (size_t k = 0; k < pe; k++) n->runtime_data[k] = scale * (p[k] - t[k]);
        return;
    }
    /* R3-1: N-dim reduction sorts after CONV2D, so like TANH/MSE it must
     * be handled before the vision-op router below (router rejects it). */
    else if (n->op == LANCIUS_OP_SUM_AXIS_ND) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in = n->inputs[0];
        if (in->ndim < 1 || in->ndim > 4 || n->ndim != in->ndim) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        uint32_t axis = n->axes[0];
        if (axis >= in->ndim) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        for (uint8_t i = 0; i < in->ndim; i++) {
            size_t want = (i == axis) ? 1 : in->shape[i];
            if (n->shape[i] != want) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        size_t in_elems = 0, out_elems = 0;
        if (!lancius_node_elements_checked(in, &in_elems) || in_elems == 0) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (!lancius_node_elements_checked(n, &out_elems) || out_elems == 0) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        /* Row-major strides with overflow guards. */
        size_t in_str[4] = {0,0,0,0}, out_str[4] = {0,0,0,0};
        in_str[in->ndim - 1] = 1;
        for (int i = (int)in->ndim - 2; i >= 0; i--) {
            size_t d = in->shape[i + 1];
            if (d != 0 && in_str[i + 1] > SIZE_MAX / d) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            in_str[i] = in_str[i + 1] * d;
        }
        out_str[n->ndim - 1] = 1;
        for (int i = (int)n->ndim - 2; i >= 0; i--) {
            size_t d = n->shape[i + 1];
            if (d != 0 && out_str[i + 1] > SIZE_MAX / d) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            out_str[i] = out_str[i + 1] * d;
        }
        size_t ax_extent = in->shape[axis];
        if (ax_extent == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (out_elems > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memset(n->runtime_data, 0, out_elems * sizeof(double));
        /* Strides are overflow-safe here: every partial product divides the
         * validated in_elems/out_elems totals (<=100M), and every dim >= 1
         * (zero-element tensors rejected above). */
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < out_elems; o++) {
            size_t rem = o;
            size_t base = 0;
            for (uint8_t i = 0; i < n->ndim; i++) {
                size_t coord = rem / out_str[i];
                rem %= out_str[i];
                base += coord * in_str[i];
            }
            double s = 0.0;
            for (size_t k = 0; k < ax_extent; k++) s += a[base + k * in_str[axis]];
            n->runtime_data[o] = s;
        }
    }
    else if (n->op == LANCIUS_OP_TRANSPOSE_BATCHED) {
        /* R3-2: 3D batched transpose [B,M,K] -> [B,K,M]. */
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* xa = n->inputs[0]->runtime_data; if (!xa || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* xa_in = n->inputs[0];
        if (xa_in->ndim != 3 || n->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t tB = xa_in->shape[0], tM = xa_in->shape[1], tK = xa_in->shape[2];
        if (n->shape[0] != tB || n->shape[1] != tK || n->shape[2] != tM) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        for (size_t b = 0; b < tB; b++)
            for (size_t k = 0; k < tK; k++)
                for (size_t m = 0; m < tM; m++)
                    n->runtime_data[(b * tK + k) * tM + m] = xa[(b * tM + m) * tK + k];
        return;
    }
    else if (n->op == LANCIUS_OP_LAYERNORM_BWD || n->op == LANCIUS_OP_LAYERNORM_BWD_GAMMA ||
             n->op == LANCIUS_OP_LAYERNORM_BWD_BETA || n->op == LANCIUS_OP_RMSNORM_BWD ||
             n->op == LANCIUS_OP_RMSNORM_BWD_GAMMA || n->op == LANCIUS_OP_GELU_BWD) {
        /* R3-2 norm/activation backwards. Instance split mirrors forward:
         * hidden = gamma elems, instances = total / hidden. */
        bool nln_dx = (n->op == LANCIUS_OP_LAYERNORM_BWD);
        bool nln_g = (n->op == LANCIUS_OP_LAYERNORM_BWD_GAMMA);
        bool nln_b = (n->op == LANCIUS_OP_LAYERNORM_BWD_BETA);
        bool nrms_dx = (n->op == LANCIUS_OP_RMSNORM_BWD);
        bool nrms_g = (n->op == LANCIUS_OP_RMSNORM_BWD_GAMMA);
        uint32_t bwd_need = (nln_dx) ? 4 : ((nln_g || nrms_dx || nrms_g) ? 3 : 2);
        if (!n->inputs || n->input_count < bwd_need) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        for (uint32_t bi = 0; bi < bwd_need; bi++) {
            if (!n->inputs[bi]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        }
        double* gd = n->inputs[0]->runtime_data;
        if (!gd || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (n->op == LANCIUS_OP_GELU_BWD) {
            double* xb = n->inputs[1]->runtime_data;
            if (!xb) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
            size_t bwd_ge = 0, bwd_xe = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &bwd_ge) ||
                !lancius_node_elements_checked(n->inputs[1], &bwd_xe) || bwd_ge != bwd_xe || bwd_ge == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            kernel_gelu_bwd(n->runtime_data, gd, xb, bwd_ge);
            return;
        }
        if (nln_b) {
            size_t bwd_hidden = 0, bwd_total = 0;
            if (!lancius_node_elements_checked(n->inputs[1], &bwd_hidden) || bwd_hidden == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            if (!lancius_node_elements_checked(n->inputs[0], &bwd_total) || bwd_total == 0 || bwd_total % bwd_hidden != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            size_t bwd_oe = 0;
            if (!lancius_node_elements_checked(n, &bwd_oe) || bwd_oe != bwd_hidden) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            kernel_layernorm_bwd_beta(n->runtime_data, gd, bwd_total / bwd_hidden, bwd_hidden);
            return;
        }
        double* xb = n->inputs[1]->runtime_data;
        double* gam = n->inputs[2]->runtime_data;
        if (!xb || !gam) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t bwd_hidden = 0, bwd_total = 0;
        if (!lancius_node_elements_checked(n->inputs[2], &bwd_hidden) || bwd_hidden == 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        if (!lancius_node_elements_checked(n->inputs[0], &bwd_total) || bwd_total == 0 || bwd_total % bwd_hidden != 0) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        {
            size_t bwd_xe = 0;
            if (!lancius_node_elements_checked(n->inputs[1], &bwd_xe) || bwd_xe != bwd_total) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        size_t bwd_ninst = bwd_total / bwd_hidden;
        if (nln_dx || nrms_dx) {
            /* dx carries the input shape (total elems). */
            size_t bwd_oe = 0;
            if (!lancius_node_elements_checked(n, &bwd_oe) || bwd_oe != bwd_total) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            if (nln_dx) { kernel_layernorm_bwd(n->runtime_data, gd, xb, gam, bwd_ninst, bwd_hidden, LANCIUS_NORM_EPS); return; }
            kernel_rmsnorm_bwd(n->runtime_data, gd, xb, gam, bwd_ninst, bwd_hidden, LANCIUS_NORM_EPS);
            return;
        }
        /* Gamma outputs carry hidden elems in gamma's own shape. */
        {
            size_t bwd_oe = 0;
            if (!lancius_node_elements_checked(n, &bwd_oe) || bwd_oe != bwd_hidden) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        if (nln_g) { kernel_layernorm_bwd_gamma(n->runtime_data, gd, xb, gam, bwd_ninst, bwd_hidden, LANCIUS_NORM_EPS); return; }
        if (nrms_g) { kernel_rmsnorm_bwd_gamma(n->runtime_data, gd, xb, gam, bwd_ninst, bwd_hidden, LANCIUS_NORM_EPS); return; }
        lancius_set_error(LANCIUS_ERROR_INTERNAL);
        return;
    }

    if (n->op >= LANCIUS_OP_CONV2D) { lancius_execute_vision_op(n); return; }

    if (n->op == LANCIUS_OP_ADD) {
        /* Despot V6 truth: validate inputs before deref (was NULL-deref). */
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        int8_t* b_int8 = n->inputs[1]->runtime_data_int8;

        /* Despot truth: INT8 Add is row-bias only ([1,N] bias + [R,N] act).
         * Any other broadcast with INT8 bias falls through to exact FP64
         * broadcast below; never miscompute via cols/shape[1] assumption. */
        if (b_int8 && a) {
            bool int8_row_bias = false;
            if (n->ndim == 2 && n->inputs[0]->ndim == 2 && n->inputs[1]->ndim == 2) {
                size_t cols = n->shape[1], rows = n->shape[0];
                size_t aR = n->inputs[0]->shape[0], aC = n->inputs[0]->shape[1];
                size_t bR = n->inputs[1]->shape[0], bC = n->inputs[1]->shape[1];
                size_t b_elems = 0;
                if (lancius_node_elements_checked(n->inputs[1], &b_elems) &&
                    rows > 0 && cols > 0 && aR == rows && aC == cols &&
                    bR == 1 && bC == cols && b_elems == cols &&
                    n->shape[0] == rows && n->shape[1] == cols) {
                    int8_row_bias = true;
                }
            }
            if (int8_row_bias) {
                double scale_b = n->inputs[1]->scale;
                size_t cols = n->shape[1];
                size_t rows = n->shape[0];
                for(size_t r=0; r<rows; r++) {
                    for(size_t cc=0; cc<cols; cc++) {
                        n->runtime_data[r*cols + cc] = a[r*cols + cc] + (double)b_int8[cc] * scale_b;
                    }
                }
                return;
            }
            /* else: fall through to exact FP64 broadcast; INT8 fast path refused. */
        }

        if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            bool same = (n->inputs[0]->ndim == n->inputs[1]->ndim) && (n->inputs[0]->ndim == n->ndim);
            if (same) {
                for (uint8_t i = 0; i < n->ndim; i++)
                    if (n->inputs[0]->shape[i] != n->inputs[1]->shape[i]) { same = false; break; }
            } else same = false;
            if (same) { for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] + b[k]; }
            else execute_broadcast_binary(n, 0);
        }
    }
    else if (n->op == LANCIUS_OP_MATMUL) {
        /* Despot V6 truth: validate inputs before deref (was NULL-deref). */
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        /*
         * v11A2 Section 11:
         * FP32 matmul execution path.
         */
        if (n->dtype == LANCIUS_DTYPE_FP32 &&
            n->inputs[0]->dtype == LANCIUS_DTYPE_FP32 &&
            n->inputs[1]->dtype == LANCIUS_DTYPE_FP32 &&
            n->inputs[0]->runtime_data_f32 &&
            n->inputs[1]->runtime_data_f32 &&
            n->runtime_data_f32) {
            if (n->inputs[0]->ndim != 2 || n->inputs[1]->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
            size_t M = n->inputs[0]->shape[0];
            size_t K = n->inputs[0]->shape[1];
            size_t N = n->inputs[1]->shape[1];
            /* Despot truth: FP32 path never checked K (FP64 did) — OOB read. */
            if (n->inputs[0]->shape[1] != n->inputs[1]->shape[0]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }

            kernel_matmul_f32(
                n->runtime_data_f32,
                n->inputs[0]->runtime_data_f32,
                n->inputs[1]->runtime_data_f32,
                M,
                K,
                N
            );
            return;
        }

        double* a = n->inputs[0]->runtime_data;
        double* b = n->inputs[1]->runtime_data;
        int8_t* b_int8 = n->inputs[1]->runtime_data_int8;

        // V10S ONNX Mixed-Precision MatMul
        if (b_int8 && a) {
            if (n->inputs[0]->ndim != 2 || n->inputs[1]->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
            size_t M = n->inputs[0]->shape[0]; size_t K = n->inputs[0]->shape[1]; size_t N = n->inputs[1]->shape[1];
            if (M == 0 || K == 0 || N == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
            /* Despot truth: INT8 path used K from A and N from B, never checked B rows == K. */
            if (n->inputs[1]->shape[0] != K) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            if (K > SIZE_MAX / (N ? N : 1)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            if (M > SIZE_MAX / (K ? K : 1)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            if (N && M > SIZE_MAX / N) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            if (M * N > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            double scale_b = n->inputs[1]->scale;
            double max_a = 0.0;
            size_t elems_a = M * K;
            for(size_t i=0; i<elems_a; i++) { double v = fabs(a[i]); if(v>max_a) max_a = v; }
            /* Despot truth: all-zero activations quantize to all-zero with any
             * scale; use 1.0 (not 1e-8 lie) so scale stays consistent with the
             * offline quantizer which skips all-zero tensors. Result is 0 either way. */
            double scale_a = (max_a > 0.0) ? (max_a / 127.0) : 1.0;

            int8_t* a_int8 = (int8_t*)malloc(elems_a ? elems_a : 1);
            if (!a_int8) { lancius_set_error(LANCIUS_ERROR_OOM); return; }
            for(size_t i=0; i<elems_a; i++) {
                double q = round(a[i] / scale_a);
                if (q > 127.0) q = 127.0;
                if (q < -128.0) q = -128.0;
                a_int8[i] = (int8_t)q;
            }

            memset(n->runtime_data, 0, M * N * sizeof(double));
            double final_scale = scale_a * scale_b;
            for(size_t r=0; r<M; r++) {
                for(size_t c=0; c<N; c++) {
                    int64_t sum = 0;
                    for(size_t k=0; k<K; k++) {
                        sum += (int64_t)a_int8[r*K + k] * (int64_t)b_int8[k*N + c];
                    }
                    n->runtime_data[r*N + c] = (double)sum * final_scale;
                }
            }
            free(a_int8);
            return;
        }

        if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (n->inputs[0]->ndim != 2 || n->inputs[1]->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t M = n->inputs[0]->shape[0]; size_t K = n->inputs[0]->shape[1]; size_t N = n->inputs[1]->shape[1];
        if (M == 0 || K == 0 || N == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (n->inputs[0]->shape[1] != n->inputs[1]->shape[0]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        /* Despot truth: a[r*K+k] needs M*K, b[k*N+c] needs K*N (only M*N was checked). */
        if (M > SIZE_MAX / K) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        if (K > SIZE_MAX / N) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        if (M > SIZE_MAX / N || M * N > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memset(n->runtime_data, 0, M * N * sizeof(double));
        // IKJ loop order: perfectly contiguous for AVX2 SIMD vectorization
        for(size_t r=0; r<M; r++) {
            for(size_t k=0; k<K; k++) {
                double val = a[r*K + k];
                #pragma omp simd
                for(size_t c=0; c<N; c++) {
                    n->runtime_data[r*N + c] += val * b[k*N + c];
                }
            }
        }
    }
    else if (n->op == LANCIUS_OP_MUL) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            bool same = (n->inputs[0]->ndim == n->inputs[1]->ndim) && (n->inputs[0]->ndim == n->ndim);
            if (same) {
                for (uint8_t i = 0; i < n->ndim; i++)
                    if (n->inputs[0]->shape[i] != n->inputs[1]->shape[i]) { same = false; break; }
            } else same = false;
            if (same) { for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] * b[k]; }
            else execute_broadcast_binary(n, 2);
        }
    }
    else if (n->op == LANCIUS_OP_SUB) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            bool same = (n->inputs[0]->ndim == n->inputs[1]->ndim) && (n->inputs[0]->ndim == n->ndim);
            if (same) {
                for (uint8_t i = 0; i < n->ndim; i++)
                    if (n->inputs[0]->shape[i] != n->inputs[1]->shape[i]) { same = false; break; }
            } else same = false;
            if (same) { for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] - b[k]; }
            else execute_broadcast_binary(n, 1);
        }
    }
    else if (n->op == LANCIUS_OP_TRANSPOSE) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (!n->inputs[0] || n->inputs[0]->ndim != 2 || n->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (n->shape[0] != C || n->shape[1] != R) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c*R + r] = a[r*C + c];
    }
    else if (n->op == LANCIUS_OP_SUM) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        /* Despot truth: never abort on corrupt shapes. */
        size_t elems = 0;
        if (!lancius_node_elements_checked(n->inputs[0], &elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        double sum = 0.0; for(size_t k=0; k<elems; k++) sum += a[k];
        n->runtime_data[0] = sum;
    }
    else if (n->op == LANCIUS_OP_SUM_AXIS0) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (!n->inputs[0] || n->inputs[0]->ndim != 2 || n->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (n->shape[0] != 1 || n->shape[1] != C) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        if (C > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memset(n->runtime_data, 0, C * sizeof(double));
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + c];
    }
    else if (n->op == LANCIUS_OP_SUM_AXIS1) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (!n->inputs[0] || n->inputs[0]->ndim != 2 || n->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        if (n->shape[0] != R || n->shape[1] != 1) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        for(size_t r=0; r<R; r++) { double s=0; for(size_t c=0; c<C; c++) s += a[r*C + c]; n->runtime_data[r] = s; }
    }
    else if (n->op == LANCIUS_OP_BROADCAST) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t in_elems = 0;
        if (!lancius_node_elements_checked(n->inputs[0], &in_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (in_elems == 1) {
            double val = a[0];
            for(size_t k=0; k<elements; k++) n->runtime_data[k] = val;
        } else {
            uint8_t out_ndim = n->ndim;
            uint8_t in_ndim = n->inputs[0]->ndim;
            if (in_ndim > out_ndim || out_ndim == 0 || out_ndim > 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }

            /* Despot truth: broadcast stride products are checked. */
            size_t in_strides_raw[4] = {0};
            in_strides_raw[in_ndim - 1] = 1;
            for (int i = (int)in_ndim - 2; i >= 0; i--) {
                size_t d = n->inputs[0]->shape[i + 1];
                if (d != 0 && in_strides_raw[i + 1] > SIZE_MAX / d) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                in_strides_raw[i] = in_strides_raw[i + 1] * d;
            }

            size_t out_strides[4] = {0};
            out_strides[out_ndim - 1] = 1;
            for (int i = (int)out_ndim - 2; i >= 0; i--) {
                size_t d = n->shape[i + 1];
                if (d != 0 && out_strides[i + 1] > SIZE_MAX / d) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                out_strides[i] = out_strides[i + 1] * d;
            }

            size_t effective_in_stride[4] = {0};
            int offset = (int)out_ndim - (int)in_ndim;
            for (uint8_t k = 0; k < out_ndim; k++) {
                if ((int)k < offset) {
                    effective_in_stride[k] = 0;
                } else {
                    uint8_t in_k = k - offset;
                    size_t in_dim = n->inputs[0]->shape[in_k];
                    size_t out_dim = n->shape[k];
                    if (in_dim == out_dim) {
                        effective_in_stride[k] = in_strides_raw[in_k];
                    } else if (in_dim == 1) {
                        effective_in_stride[k] = 0;
                    } else {
                        lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH);
                        return;
                    }
                }
            }

            for (size_t k = 0; k < elements; k++) {
                size_t rem = k;
                size_t in_idx = 0;
                for (uint8_t d = 0; d < out_ndim; d++) {
                    size_t coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    in_idx += coord * effective_in_stride[d];
                }
                n->runtime_data[k] = a[in_idx];
            }
        }
    }
    else if (n->op == LANCIUS_OP_RELU_BWD) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* grad = n->inputs[0]->runtime_data; double* fwd_a = n->inputs[1]->runtime_data;
        if (!grad || !fwd_a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            size_t ge = 0, fe = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ge) || !lancius_node_elements_checked(n->inputs[1], &fe) || ge != elements || fe != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = fwd_a[k] > 0.0 ? grad[k] : 0.0;
    }
    else if (n->op == LANCIUS_OP_SOFTMAX) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (n->ndim != 2 || n->inputs[0]->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t R = n->shape[0]; size_t C = n->shape[1];
        /* Despot truth: R==0/C==0 guards (a[r*C] OOB when C==0); Inf sum is NUMERICAL. */
        if (R == 0 || C == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        for(size_t r=0; r<R; r++) {
            double max_val = a[r*C];
            for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
            double sum = 0.0;
            for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c] - max_val); sum += n->runtime_data[r*C+c]; }
            if (!(sum > 0.0) || !isfinite(sum)) { lancius_set_error(LANCIUS_ERROR_NUMERICAL); return; }
            for(size_t c=0; c<C; c++) n->runtime_data[r*C+c] /= sum;
        }
    }
    else if (n->op == LANCIUS_OP_SOFTMAX_BWD) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* dy = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!dy || !y) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        if (n->ndim != 2 || n->inputs[0]->ndim != 2 || n->inputs[1]->ndim != 2) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        {
            size_t de = 0, ye = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &de) || !lancius_node_elements_checked(n->inputs[1], &ye) || de != elements || ye != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        size_t R = n->shape[0]; size_t C = n->shape[1];
        for(size_t r=0; r<R; r++) {
            double dot = 0.0;
            for(size_t c=0; c<C; c++) dot += dy[r*C + c] * y[r*C + c];
            for(size_t c=0; c<C; c++) n->runtime_data[r*C + c] = y[r*C + c] * (dy[r*C + c] - dot);
        }
    }
    else if (n->op == LANCIUS_OP_RELU) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* a = n->inputs[0]->runtime_data; if (!a) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        {
            size_t ie = 0;
            if (!lancius_node_elements_checked(n->inputs[0], &ie) || ie != elements) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] > 0.0 ? a[k] : 0.0;
    }
}

static int plan_pool_offset(lancius_schedule* schedule, const lancius_node* n, size_t nbytes, size_t* out_off) {
    if (!schedule || !schedule->plan || !n || !out_off) return 0;
    lancius_memory_plan* plan = schedule->plan;
    if (!plan->offsets || !plan->is_pooled) return 0;
    if (n->id >= plan->max_id) return 0;
    if (!plan->is_pooled[n->id]) return 0;
    if (!schedule->static_pool) return 0;
    size_t off = plan->offsets[n->id];
    if (off > plan->peak_memory) return 0;
    if (nbytes > plan->peak_memory - off) return 0;
    *out_off = off;
    return 1;
}

static void lancius_schedule_prepare_buffers(lancius_schedule* schedule) {
    if (!schedule) return;

    /*
     * v11A2 Section 9:
     *
     * Execution must be repeatable without requiring callers to manually
     * null intermediate tensors.
     *
     * Rules:
     *   - external buffers are preserved
     *   - owned heap buffers are preserved
     *   - arena-owned buffers are invalidated before execution
     *   - pool-owned buffers are rebound when a valid plan is attached
     */
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];

        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (!n || !n->rt) continue;

            size_t nbytes = 0;
            if (lancius_node_bytes_checked(n, &nbytes) && nbytes > 0) {
                size_t off = 0;
                if (plan_pool_offset(schedule, n, nbytes, &off)) {
                    /* Despot truth: FP32 nodes own f32 buffers (was: FP64
                     * pointer set for every dtype, confusing later checks). */
                    if (n->dtype == LANCIUS_DTYPE_FP32) {
                        n->runtime_data = NULL;
                        n->rt->buffer = NULL;
                        n->runtime_data_f32 = (float*)((uint8_t*)schedule->static_pool + off);
                        n->rt->buffer_f32 = n->runtime_data_f32;
                        n->rt->f32_owner = LANCIUS_MEMORY_POOL;
                    } else {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + off);
                        n->rt->buffer = n->runtime_data;
                        n->rt->buffer_owner = LANCIUS_MEMORY_POOL;
                    }
                    n->rt->offset = off;
                    n->rt->owner = LANCIUS_MEMORY_POOL;
                    continue;
                }
            } else if (schedule->plan && schedule->static_pool) {
                /* Plan attached but size invalid: do not trust pool, fall through to invalidate. */
                lancius_set_error(LANCIUS_ERROR_LIMIT);
            }

            if (n->rt->buffer_owner == LANCIUS_MEMORY_ARENA ||
                n->rt->buffer_owner == LANCIUS_MEMORY_POOL) {
                n->runtime_data = NULL;
                n->rt->buffer = NULL;
                n->rt->buffer_owner = LANCIUS_MEMORY_EXTERNAL;
                n->rt->offset = 0;
            }
            if (n->rt->f32_owner == LANCIUS_MEMORY_ARENA ||
                n->rt->f32_owner == LANCIUS_MEMORY_POOL) {
                n->runtime_data_f32 = NULL;
                n->rt->buffer_f32 = NULL;
                n->rt->f32_owner = LANCIUS_MEMORY_EXTERNAL;
                n->rt->offset = 0;
            }
        }
    }
}

void lancius_schedule_execute(lancius_schedule* schedule, lancius_arena* scratch) {
    if (!schedule) return;

    /* A1: mirror legacy runtime pointers into runtime state before execution */
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_runtime_sync_from_legacy(wave->nodes[i]);
        }
    }

    lancius_schedule_prepare_buffers(schedule);

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            /* Despot truth: never abort on corrupt shapes on hot path. */
            size_t elements = 0;
            size_t nbytes_checked = 0;
            if (!lancius_node_elements_checked(n, &elements)) { lancius_set_error(LANCIUS_ERROR_LIMIT); continue; }
            if (!lancius_node_bytes_checked(n, &nbytes_checked)) { lancius_set_error(LANCIUS_ERROR_LIMIT); continue; }

            if (n->op == LANCIUS_OP_INPUT) continue;
            if (n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes
            if (n->op == LANCIUS_OP_CONST) {
                if(!n->runtime_data) {
                    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, nbytes_checked, 32); /* A3 */
                    }
                    /* A2: record buffer ownership */
                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                    if (!n->runtime_data) { lancius_set_error(LANCIUS_ERROR_OOM); }
                    else for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }

            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) {
                    float* buf_f32 = NULL;

                    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        buf_f32 = (float*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        buf_f32 = (float*)lancius_arena_alloc(scratch, nbytes_checked, 32);
                    }

                    n->runtime_data_f32 = buf_f32;
                    if (n->rt) n->rt->buffer_f32 = buf_f32;

                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);

                    if (!buf_f32) {
                        lancius_set_error(LANCIUS_ERROR_OOM);
                        continue;
                    }
                }
            } else if (!n->runtime_data) {
                if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, nbytes_checked, 32); /* A3 */
                    }
                /* A2: record buffer ownership */
                lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                if (!n->runtime_data) {
                    lancius_set_error(LANCIUS_ERROR_OOM); /* A4 OOM */
                    continue;
                }
            }

            execute_node_math(n);
        }
    }
}

typedef struct {
    lancius_node* node;
    lancius_error error;
} lancius_parallel_task;

static void execute_node_math_trampoline(void* arg) {
    lancius_parallel_task* task = (lancius_parallel_task*)arg;
    execute_node_math(task->node);
    task->error = lancius_get_error();
}

void lancius_schedule_execute_parallel(lancius_schedule* schedule, lancius_arena* scratch, lancius_pool* pool) {
    if (!schedule || !pool) return;

    /* A1: mirror legacy runtime pointers into runtime state before parallel execution */
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_runtime_sync_from_legacy(wave->nodes[i]);
        }
    }

    lancius_schedule_prepare_buffers(schedule);

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            /* Despot truth: never abort on corrupt shapes on hot path. */
            size_t elements = 0;
            size_t nbytes_checked = 0;
            if (!lancius_node_elements_checked(n, &elements)) { lancius_set_error(LANCIUS_ERROR_LIMIT); continue; }
            if (!lancius_node_bytes_checked(n, &nbytes_checked)) { lancius_set_error(LANCIUS_ERROR_LIMIT); continue; }
            if (n->op == LANCIUS_OP_INPUT) continue;
            if (n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes
            if (n->op == LANCIUS_OP_CONST) {
                if(!n->runtime_data) {
                    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, nbytes_checked, 32); /* A3 */
                    }
                    /* A2: record buffer ownership */
                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                    if (!n->runtime_data) { lancius_set_error(LANCIUS_ERROR_OOM); }
                    else for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }
            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) {
                    float* buf_f32 = NULL;

                    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        buf_f32 = (float*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        buf_f32 = (float*)lancius_arena_alloc(scratch, nbytes_checked, 32);
                    }

                    n->runtime_data_f32 = buf_f32;
                    if (n->rt) n->rt->buffer_f32 = buf_f32;

                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);

                    if (!buf_f32) {
                        lancius_set_error(LANCIUS_ERROR_OOM);
                        continue;
                    }
                }
            } else if (!n->runtime_data) {
                if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, nbytes_checked, 32); /* A3 */
                    }
                /* A2: record buffer ownership */
                lancius_node_set_owner(n, (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled && n->id < schedule->plan->max_id && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                if (!n->runtime_data) {
                    lancius_set_error(LANCIUS_ERROR_OOM);
                    continue;
                }
            }
        }

        uint32_t task_count = 0;
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;
            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); continue; }
            } else {
                if (!n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); continue; }
            }
            task_count++;
        }

        if (task_count == 0) { lancius_pool_wait(pool, 0); continue; }

        lancius_parallel_task* tasks = (lancius_parallel_task*)malloc(sizeof(lancius_parallel_task) * task_count);
        if (!tasks) { lancius_set_error(LANCIUS_ERROR_OOM); return; }

        uint32_t t = 0;
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;
            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) continue;
            } else {
                if (!n->runtime_data) continue;
            }
            /* Buffers verified above; missing here means OOM path already set error. */
            tasks[t].node = n;
            tasks[t].error = LANCIUS_ERROR_OK;
            lancius_pool_submit(pool, execute_node_math_trampoline, &tasks[t]);
            t++;
        }
        lancius_pool_wait(pool, 0);

        for (uint32_t i = 0; i < task_count; i++) {
            if (tasks[i].error != LANCIUS_ERROR_OK) {
                lancius_set_error(tasks[i].error);
                break;
            }
        }
        free(tasks);
    }
}


void lancius_schedule_attach_pool(lancius_schedule* schedule, void* flat_buffer, lancius_memory_plan* plan) {
    if (!schedule) return;
    schedule->static_pool = flat_buffer;
    schedule->plan = plan;
}

void lancius_schedule_destroy(lancius_schedule* schedule) {
    if (!schedule) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) free(schedule->waves[w].nodes);
    free(schedule->waves); free(schedule);
}

size_t lancius_schedule_peak_memory(lancius_schedule* schedule) {
    if (!schedule) return 0;
    size_t peak = 0;
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        size_t wave_mem = 0;
        lancius_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (!n) continue;
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) {
                size_t b = 0;
                /* Despot truth: 0 meant both corrupt and empty; set error so
                 * execute_static_bounded cannot treat it as "any buffer fits". */
                if (!lancius_node_bytes_checked(n, &b)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return 0; }
                if (wave_mem > SIZE_MAX - b) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return 0; }
                wave_mem += b;
            }
        }
        if (wave_mem > peak) peak = wave_mem;
    }
    return peak;
}

// =====================================================================
// PREP FOR PHASE 3: STATIC MEMORY LIVENESS ANALYZER
// =====================================================================
lancius_liveness_profile lancius_analyze_liveness(lancius_graph* g) {
    lancius_liveness_profile profile = {0, 0, 0};
    if (!g) return profile;

    lancius_schedule* sched = lancius_ir_schedule(g);
    if (!sched) return profile;

    for(uint32_t w=0; w<sched->wave_count; w++) {
        size_t wave_mem = 0;
        for(uint32_t i=0; i<sched->waves[w].node_count; i++) {
            lancius_node* n = sched->waves[w].nodes[i];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) {
                size_t b = 0;
                if (!lancius_node_bytes_checked(n, &b)) { lancius_set_error(LANCIUS_ERROR_LIMIT); lancius_schedule_destroy(sched); return profile; }
                if (b > SIZE_MAX - wave_mem) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); lancius_schedule_destroy(sched); return profile; }
                wave_mem += b;
                profile.tensor_count++;
            }
        }
        if (wave_mem > profile.peak_memory_bytes) profile.peak_memory_bytes = wave_mem;
        profile.total_allocs += sched->waves[w].node_count;
    }
    lancius_schedule_destroy(sched);
    return profile;
}

// =====================================================================
// V9C STATIC FLAT BUFFER EXECUTOR (Microcontroller Mandate)
// =====================================================================
void lancius_schedule_execute_static(lancius_schedule* schedule, void* flat_buffer) {
    if (!schedule || !flat_buffer) return;
    /* Despot V6 truth: 32B AVX2 alignment. Plan path requires aligned base
     * (offsets are 32-aligned); bump path aligns start internally. */
    uintptr_t _base = (uintptr_t)flat_buffer;
    size_t _base_pad = (32u - (_base & 31u)) & 31u;

    /* v11A1 Task 8: use attached memory plan when available. */
    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled) {
        if (_base_pad != 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        for (uint32_t w = 0; w < schedule->wave_count; w++) {
            lancius_wave* wave = &schedule->waves[w];

            for (uint32_t i = 0; i < wave->node_count; i++) {
                lancius_node* n = wave->nodes[i];
                if (!n) continue;
                if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;
                if (n->id >= schedule->plan->max_id) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); continue; }
                if (!schedule->plan->is_pooled[n->id]) continue;
                size_t nbytes = 0;
                if (!lancius_node_bytes_checked(n, &nbytes)) { lancius_set_error(LANCIUS_ERROR_LIMIT); continue; }
                size_t off = schedule->plan->offsets[n->id];
                if (off > schedule->plan->peak_memory || nbytes > schedule->plan->peak_memory - off) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); continue; }
                if (n->dtype == LANCIUS_DTYPE_FP32) {
                    n->runtime_data_f32 = (float*)((uint8_t*)flat_buffer + off);
                } else {
                    n->runtime_data = (double*)((uint8_t*)flat_buffer + off);
                }
                lancius_node_set_owner(n, LANCIUS_MEMORY_POOL);
                lancius_runtime_sync_from_legacy(n);
            }

            for (uint32_t i = 0; i < wave->node_count; i++) {
                lancius_node* n = wave->nodes[i];
                if (!n) continue;
                if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;
                if (n->id >= schedule->plan->max_id) continue;
                if (!schedule->plan->is_pooled[n->id]) continue;
                /* Despot truth: never execute a node whose pool assignment
                 * failed above (stale/NULL buffer). */
                if (n->dtype == LANCIUS_DTYPE_FP32) { if (!n->runtime_data_f32) continue; }
                else { if (!n->runtime_data) continue; }
                execute_node_math(n);
            }
        }
    return;
    }


    // Simple bump pointer over the flat buffer
    // Despot V6: start at aligned delta so (base+offset) stays 32B-aligned.
    size_t offset = _base_pad;

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];

        // 1. Assign memory to all intermediate nodes in this wave
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (!n) continue;
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;

            size_t size = 0;
            if (!lancius_node_bytes_checked(n, &size)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }

            // Align to 32 bytes for AVX2/SIMD
            size_t aligned = (offset + 31) & ~(size_t)31;
            if (aligned < offset) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            offset = aligned;
            if (size > SIZE_MAX - offset) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }

            // Assign the flat buffer pointer directly to the node
            if (n->dtype == LANCIUS_DTYPE_FP32) {
                n->runtime_data_f32 = (float*)((char*)flat_buffer + offset);
                n->runtime_data = NULL;
            } else {
                n->runtime_data = (double*)((char*)flat_buffer + offset);
            }

                /* A1: mirror static pool buffer into runtime state */
                if (n->rt) {
                    if (n->dtype == LANCIUS_DTYPE_FP32) {
                        n->rt->buffer = NULL;
                        n->rt->buffer_f32 = n->runtime_data_f32;
                    } else {
                        n->rt->buffer = n->runtime_data;
                    }
                    n->rt->offset = offset;
                }
                lancius_node_set_owner(n, LANCIUS_MEMORY_POOL); /* A2 static */

                offset += size;
        }

        // 2. Execute the math for the wave
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (!n) continue;
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;
            if (n->dtype == LANCIUS_DTYPE_FP32) { if (!n->runtime_data_f32) continue; }
            else { if (!n->runtime_data) continue; }

            // Reuse the exact same math router from the standard executor
            execute_node_math(n);
        }
    }
}

/* v11S H2 fix: bounded static executor validates buffer size */
void lancius_schedule_execute_static_bounded(lancius_schedule* schedule, void* flat_buffer, size_t buffer_size) {
    if (!schedule || !flat_buffer) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    size_t required = lancius_schedule_static_memory_required(schedule);
    /* Despot truth: required==0 with sticky error means sizing failed;
     * executing anyway would under-alloc and OOB. */
    if (lancius_get_error() != LANCIUS_ERROR_OK) return;
    if (buffer_size < required) {
        lancius_set_error(LANCIUS_ERROR_OVERFLOW);
        return;
    }
    lancius_schedule_execute_static(schedule, flat_buffer);
}

static void execute_permute(lancius_node* n) {
    if (!n || !n->inputs || n->input_count == 0) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

    const lancius_node* in = n->inputs[0];
    const double* x = in ? in->runtime_data : NULL;
    double* y = n->runtime_data;

    if (!x || !y) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    if (!in || in->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }

    size_t in_shape[4] = {1, 1, 1, 1};
    size_t out_shape[4] = {1, 1, 1, 1};

    for (uint8_t i = 0; i < in->ndim && i < 4; i++) in_shape[i] = in->shape[i];
    for (uint8_t i = 0; i < n->ndim && i < 4; i++) out_shape[i] = n->shape[i];

    /* Despot truth: stride products are checked; corrupt shapes fail loud. */
    size_t in_stride[4];
    in_stride[3] = 1;
    in_stride[2] = in_shape[3];
    if (in_shape[2] != 0 && in_stride[2] > SIZE_MAX / in_shape[2]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    in_stride[1] = in_shape[2] * in_stride[2];
    if (in_shape[1] != 0 && in_stride[1] > SIZE_MAX / in_shape[1]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    in_stride[0] = in_shape[1] * in_stride[1];

    uint32_t axes[4] = {
        n->axes[0],
        n->axes[1],
        n->axes[2],
        n->axes[3]
    };

    if (lancius_validate_permutation(axes, 4) != LANCIUS_ERROR_OK) { lancius_set_error(LANCIUS_ERROR_INVALID_PERMUTATION); return; }
    /* Despot truth: out shape must correspond to in shape via axes; else OOB. */
    for (int d = 0; d < 4; d++) {
        if (out_shape[d] != in_shape[axes[d]]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
    }

    for (size_t i0 = 0; i0 < out_shape[0]; i0++) {
        for (size_t i1 = 0; i1 < out_shape[1]; i1++) {
            for (size_t i2 = 0; i2 < out_shape[2]; i2++) {
                for (size_t i3 = 0; i3 < out_shape[3]; i3++) {

                    size_t out_idx, in_idx = 0;
                    size_t out_indices[4] = {i0, i1, i2, i3};
                    /* Checked: ((i0*S1+i1)*S2+i2)*S3+i3 must not wrap. */
                    if (out_shape[1] && i0 > (SIZE_MAX - i1) / out_shape[1]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                    {
                        size_t t0 = i0 * out_shape[1] + i1;
                        if (out_shape[2] && t0 > (SIZE_MAX - i2) / out_shape[2]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                        size_t t1 = t0 * out_shape[2] + i2;
                        if (out_shape[3] && t1 > (SIZE_MAX - i3) / out_shape[3]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                        out_idx = t1 * out_shape[3] + i3;
                    }

                    for (int d = 0; d < 4; d++) {
                        size_t add;
                        if (out_indices[d] && in_stride[axes[d]] > SIZE_MAX / out_indices[d]) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                        add = out_indices[d] * in_stride[axes[d]];
                        if (in_idx > SIZE_MAX - add) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
                        in_idx += add;
                    }

                    y[out_idx] = x[in_idx];
                }
            }
        }
    }
}

static void execute_matmul_batched(lancius_node* n) {
    if (!n || n->input_count < 2) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

    const lancius_node* a = n->inputs[0];
    const lancius_node* b = n->inputs[1];

    const double* A = a ? a->runtime_data : NULL;
    const double* B = b ? b->runtime_data : NULL;
    double* C = n->runtime_data;

    if (!A || !B || !C) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    if (!a || !b || a->ndim != 3 || b->ndim != 3) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
    if (n->shape[0] != a->shape[0] || n->shape[1] != a->shape[1] || n->shape[2] != b->shape[2]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }

    size_t batches = a->shape[0];
    size_t M = a->shape[1];
    size_t K = a->shape[2];
    size_t N = b->shape[2];
    if (batches == 0 || M == 0 || K == 0 || N == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
    if (M > SIZE_MAX / K || M * K > SIZE_MAX / N) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    /* Despot truth: batch offsets are checked; overflow fails loud, never wraps. */
    size_t a_batch_elems = 0, b_batch_elems = 0, c_batch_elems = 0;
    if (!lancius_checked_mul_size(M, K, &a_batch_elems)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    if (!lancius_checked_mul_size(K, N, &b_batch_elems)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    if (!lancius_checked_mul_size(M, N, &c_batch_elems)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    size_t a_total = 0, b_total = 0, c_total = 0;
    if (!lancius_checked_mul_size(batches, a_batch_elems, &a_total)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    if (!lancius_checked_mul_size(batches, b_batch_elems, &b_total)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    if (!lancius_checked_mul_size(batches, c_batch_elems, &c_total)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
    (void)a_total; (void)b_total; (void)c_total;

    for (size_t batch = 0; batch < batches; batch++) {
        size_t a_off = 0, b_off = 0, c_off = 0;
        if (!lancius_checked_mul_size(batch, a_batch_elems, &a_off) ||
            !lancius_checked_mul_size(batch, b_batch_elems, &b_off) ||
            !lancius_checked_mul_size(batch, c_batch_elems, &c_off)) {
            lancius_set_error(LANCIUS_ERROR_OVERFLOW);
            return;
        }
        const double* A_batch = A + a_off;
        const double* B_batch = B + b_off;
        double* C_batch = C + c_off;

        kernel_matmul(C_batch, A_batch, B_batch, M, K, N);
    }
}

size_t lancius_schedule_static_memory_required(lancius_schedule* schedule) {
    if (!schedule) return 0;

    size_t offset = 0;

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];

        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (!n) continue;

            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) {
                continue;
            }

            size_t bytes = 0;
            /* Despot truth: 0 is ambiguous (corrupt vs empty); set error so
             * bounded executor cannot treat it as "any buffer fits". */
            if (!lancius_node_bytes_checked(n, &bytes)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return 0; }

            if (offset > SIZE_MAX - 31) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return 0; }
            offset = (offset + 31) & ~(size_t)31;

            if (bytes > SIZE_MAX - offset) {
                lancius_set_error(LANCIUS_ERROR_OVERFLOW);
                return 0;
            }

            offset += bytes;
        }
    }

    return offset;
}

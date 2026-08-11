#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_memory_planner.h"
#include "lancius/lancius_transformer.h"
#include "lancius/lancius_vision_ops.h"
#include "lancius/lancius_kernels.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

lancius_schedule* lancius_ir_schedule(lancius_graph* g) {
    if (!g || g->node_count == 0) return NULL;
    lancius_schedule* sched = (lancius_schedule*)calloc(1, sizeof(lancius_schedule));
    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    lancius_node** queue = (lancius_node**)malloc(sizeof(lancius_node*) * g->node_count);

    for (uint32_t i = 0; i < g->node_count; i++) in_degree[g->nodes[i]->id] = g->nodes[i]->input_count;

    uint32_t wave_cap = 16;
    sched->waves = (lancius_wave*)malloc(sizeof(lancius_wave) * wave_cap);
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
            fprintf(stderr, "[SCHEDULER FATAL] Cycle or disconnect! Processed %u / %u\n", processed, g->node_count);
            break;
        }
        if (sched->wave_count >= wave_cap) {
            wave_cap *= 2;
            sched->waves = (lancius_wave*)realloc(sched->waves, sizeof(lancius_wave) * wave_cap);
        }
        lancius_wave* w = &sched->waves[sched->wave_count++];
        w->node_count = q_tail;
        w->nodes = (lancius_node**)malloc(sizeof(lancius_node*) * q_tail);
        memcpy(w->nodes, queue, sizeof(lancius_node*) * q_tail);

        for (uint32_t i = 0; i < g->node_count; i++) {
            lancius_node* n = g->nodes[i];
            if (in_degree[n->id] == UINT32_MAX) continue;
            bool ready = true;
            for (uint32_t j = 0; j < n->input_count; j++) {
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

static void execute_node_math(lancius_node* n) {
    if (!n) return;
    lancius_runtime_sync_from_legacy(n); /* A1 */

    /*
     * v11A2 Section 10/11:
     * FP32 tensors use runtime_data_f32.
     */
    if (n->dtype == LANCIUS_DTYPE_FP32) {
        if (!n->runtime_data_f32) return;

        if (n->op != LANCIUS_OP_MATMUL &&
            n->op != LANCIUS_OP_INPUT &&
            n->op != LANCIUS_OP_CONST) {
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_DTYPE);
            return;
        }
    } else if (!n->runtime_data) {
        return;
    }

    size_t elements = lancius_node_elements(n); (void)elements;

    // CRITICAL FIX: Execute Cross-Entropy BEFORE the Vision Ops router hijacks it
    if (n->op == LANCIUS_OP_CROSS_ENTROPY) {
        double* x = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!x || !y) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        double total_loss = 0.0;
        for(size_t r=0; r<R; r++) {
            double max_val = x[r*C];
            for(size_t c=1; c<C; c++) if(x[r*C+c] > max_val) max_val = x[r*C+c];
            double sum_exp = 0.0;
            for(size_t c=0; c<C; c++) sum_exp += exp(x[r*C+c] - max_val);
            /* v11S H3 fix: guard against degenerate logits */
            if (sum_exp <= 0.0 || sum_exp != sum_exp) { total_loss += 1e30; continue; }
            double log_sum_exp = log(sum_exp) + max_val;
            for(size_t c=0; c<C; c++) {
                if (y[r*C+c] == 1.0) total_loss -= (x[r*C+c] - log_sum_exp);
            }
        }
        n->runtime_data[0] = total_loss / R;
        if (n->runtime_data[0] < 0.0 && n->runtime_data[0] > -1e-12) n->runtime_data[0] = 0.0;
        return;
    

    }
    else if (n->op == LANCIUS_OP_CROSS_ENTROPY_BWD) {
        double* x = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data; double* g = n->inputs[2]->runtime_data;
        if (!x || !y || !g) return;
        size_t R = n->shape[0]; size_t C = n->shape[1];
        double scale = g[0] / R;
        for(size_t r=0; r<R; r++) {
            double max_val = x[r*C];
            for(size_t c=1; c<C; c++) if(x[r*C+c] > max_val) max_val = x[r*C+c];
            double sum_exp = 0.0;
            for(size_t c=0; c<C; c++) sum_exp += exp(x[r*C+c] - max_val);
            for(size_t c=0; c<C; c++) {
                double sm = exp(x[r*C+c] - max_val) / sum_exp;
                n->runtime_data[r*C+c] = (sm - y[r*C+c]) * scale;
            }
        }
        return;
    }

    
    else if (n->op == LANCIUS_OP_LAYERNORM) {
        double* in = n->inputs[0]->runtime_data;
        double* gamma = n->inputs[1]->runtime_data;
        double* beta = n->inputs[2]->runtime_data;
        if(!in || !gamma || !beta) return;

        /*
         * v11A2 transformer fix:
         *
         * LayerNorm may receive:
         *   [batch, hidden]
         * or:
         *   [batch, heads, dim]
         *
         * In both cases the normalization domain is all elements after
         * the leading batch dimension.
         */
        size_t batch = n->shape[0];
        size_t total = lancius_node_elements(n);
        size_t hidden = (batch ? total / batch : 0);

        kernel_layernorm(n->runtime_data, in, gamma, beta, batch, hidden, 1e-5);
    }
    else if (n->op == LANCIUS_OP_GELU) {
        double* in = n->inputs[0]->runtime_data;
        if(!in) return;
        kernel_gelu(n->runtime_data, in, elements);
    }

    
    else if (n->op == LANCIUS_OP_ATTENTION) {
        const lancius_node* q_node = n->inputs[0];
        const lancius_node* k_node = n->inputs[1];
        const lancius_node* v_node = n->inputs[2];

        double* q = q_node->runtime_data;
        if (!q) return;

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
        if (!k || !v) return;

        size_t kv_seq = k_node->shape[0];

        if (q_seq == 1 && kv_seq >= 1) {
            kernel_attention_kv_cache(n->runtime_data, q, k, v, kv_seq, n_heads, head_dim);
        } else if (q_seq == kv_seq) {
            kernel_attention(n->runtime_data, q, k, v, q_seq, n_heads, head_dim);
        } else {
            fprintf(stderr, "[LANCIUS EXEC FATAL] unsupported attention shape: q_seq=%zu kv_seq=%zu\n", q_seq, kv_seq);
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            return; /* v11S: do not abort — return error to caller */
        }
    }

    else if (n->op == LANCIUS_OP_RMSNORM) {
            double* in = n->inputs[0]->runtime_data;
            double* gamma = n->inputs[1]->runtime_data;
            if(!in || !gamma) return;
            kernel_rmsnorm(n->runtime_data, in, gamma, n->shape[0], n->shape[1], 1e-5);
        }
        else if (n->op == LANCIUS_OP_SWIGLU) {
            double* gate = n->inputs[0]->runtime_data;
            double* up = n->inputs[1]->runtime_data;
            if(!gate || !up) return;
            kernel_swiglu(n->runtime_data, gate, up, elements);
        }
        else if (n->op == LANCIUS_OP_GQA) {
            double* q = n->inputs[0]->runtime_data;
            double* k = n->inputs[1]->runtime_data;
            double* v = n->inputs[2]->runtime_data;
            if(!q || !k || !v) return;
            kernel_gqa(n->runtime_data, q, k, v, n->shape[0], n->kernel_h, n->kernel_w, n->shape[2]);
        }
else if (n->op == LANCIUS_OP_ROPE) {
        double* qk = n->inputs[0]->runtime_data;
        if(!qk) return;
        size_t seq_len = n->shape[0];
        size_t n_heads = n->shape[1];
        size_t head_dim_x2 = n->shape[2];
        size_t head_dim = head_dim_x2 / 2;
        size_t elems = seq_len * n_heads * head_dim_x2;
        memcpy(n->runtime_data, qk, elems * sizeof(double)); // Preserve SSA
        double* q = n->runtime_data;
        double* k = n->runtime_data + (seq_len * n_heads * head_dim);
        kernel_rope(q, k, 1, seq_len, n_heads, head_dim, 0);
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

    if (n->op >= LANCIUS_OP_CONV2D) { lancius_execute_vision_op(n); return; }

    if (n->op == LANCIUS_OP_ADD) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        int8_t* b_int8 = n->inputs[1]->runtime_data_int8;

        // V10S ONNX Mixed-Precision Add (Biases)
        if (b_int8 && a) {
            double scale_b = n->inputs[1]->scale;
            size_t elements = lancius_node_elements(n); (void)elements;
            size_t cols = n->shape[1];
            size_t rows = n->shape[0];
            for(size_t r=0; r<rows; r++) {
                for(size_t cc=0; cc<cols; cc++) {
                    n->runtime_data[r*cols + cc] = a[r*cols + cc] + (double)b_int8[cc] * scale_b;
                }
            }
            return;
        }

        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] + b[k];
    }
    else if (n->op == LANCIUS_OP_MATMUL) {
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
            size_t M = n->inputs[0]->shape[0];
            size_t K = n->inputs[0]->shape[1];
            size_t N = n->inputs[1]->shape[1];

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
            size_t M = n->inputs[0]->shape[0]; size_t K = n->inputs[0]->shape[1]; size_t N = n->inputs[1]->shape[1];
            double scale_b = n->inputs[1]->scale;
            double max_a = 0.0;
            size_t elems_a = M * K;
            for(size_t i=0; i<elems_a; i++) { double v = fabs(a[i]); if(v>max_a) max_a = v; }
            double scale_a = (max_a > 0.0) ? (max_a / 127.0) : 1e-8;

            int8_t* a_int8 = (int8_t*)malloc(elems_a);
            for(size_t i=0; i<elems_a; i++) a_int8[i] = (int8_t)round(a[i] / scale_a);

            memset(n->runtime_data, 0, M * N * sizeof(double));
            double final_scale = scale_a * scale_b;
            for(size_t r=0; r<M; r++) {
                for(size_t c=0; c<N; c++) {
                    int32_t sum = 0;
                    for(size_t k=0; k<K; k++) {
                        sum += (int32_t)a_int8[r*K + k] * (int32_t)b_int8[k*N + c];
                    }
                    n->runtime_data[r*N + c] = (double)sum * final_scale;
                }
            }
            free(a_int8);
            return;
        }

        if (!a || !b) return;
        size_t M = n->inputs[0]->shape[0]; size_t K = n->inputs[0]->shape[1]; size_t N = n->inputs[1]->shape[1];
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
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] * b[k];
    }
    else if (n->op == LANCIUS_OP_SUB) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] - b[k];
    }
    else if (n->op == LANCIUS_OP_TRANSPOSE) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c*R + r] = a[r*C + c];
    }
    else if (n->op == LANCIUS_OP_SUM) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t elems = lancius_node_elements(n->inputs[0]);
        double sum = 0.0; for(size_t k=0; k<elems; k++) sum += a[k];
        n->runtime_data[0] = sum;
    }
    else if (n->op == LANCIUS_OP_SUM_AXIS0) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        memset(n->runtime_data, 0, C * sizeof(double));
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + c];
    }
    else if (n->op == LANCIUS_OP_SUM_AXIS1) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        for(size_t r=0; r<R; r++) { double s=0; for(size_t c=0; c<C; c++) s += a[r*C + c]; n->runtime_data[r] = s; }
    }
    else if (n->op == LANCIUS_OP_BROADCAST) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t in_elems = lancius_node_elements(n->inputs[0]);
        if (in_elems == 1) {
            double val = a[0];
            for(size_t k=0; k<elements; k++) n->runtime_data[k] = val;
        } else {
            size_t cols = n->shape[1];
            for(size_t r=0; r<n->shape[0]; r++) for(size_t c=0; c<cols; c++) n->runtime_data[r*cols + c] = a[c];
        }
    }
    else if (n->op == LANCIUS_OP_RELU_BWD) {
        double* grad = n->inputs[0]->runtime_data; double* fwd_a = n->inputs[1]->runtime_data;
        if (!grad || !fwd_a) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = fwd_a[k] > 0.0 ? grad[k] : 0.0;
    }
    else if (n->op == LANCIUS_OP_SOFTMAX) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->shape[0]; size_t C = n->shape[1];
        for(size_t r=0; r<R; r++) {
            double max_val = a[r*C];
            for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
            double sum = 0.0;
            for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c] - max_val); sum += n->runtime_data[r*C+c]; }
            for(size_t c=0; c<C; c++) n->runtime_data[r*C+c] /= sum;
        }
    }
    else if (n->op == LANCIUS_OP_SOFTMAX_BWD) {
        double* dy = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!dy || !y) return;
        size_t R = n->shape[0]; size_t C = n->shape[1];
        for(size_t r=0; r<R; r++) {
            double dot = 0.0;
            for(size_t c=0; c<C; c++) dot += dy[r*C + c] * y[r*C + c];
            for(size_t c=0; c<C; c++) n->runtime_data[r*C + c] = y[r*C + c] * (dy[r*C + c] - dot);
        }
    }
    else if (n->op == LANCIUS_OP_RELU) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] > 0.0 ? a[k] : 0.0;
    }
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

            if (schedule->plan &&
                schedule->plan->offsets &&
                schedule->plan->is_pooled &&
                schedule->plan->is_pooled[n->id] &&
                schedule->static_pool) {
                n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                n->rt->buffer = n->runtime_data;
                n->rt->offset = schedule->plan->offsets[n->id];
                n->rt->buffer_owner = LANCIUS_MEMORY_POOL;
                n->rt->owner = LANCIUS_MEMORY_POOL;
                continue;
            }

            if (n->rt->buffer_owner == LANCIUS_MEMORY_ARENA ||
                n->rt->buffer_owner == LANCIUS_MEMORY_POOL) {
                n->runtime_data = NULL;
                n->rt->buffer = NULL;
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
            size_t elements = lancius_node_elements(n); (void)elements;

            if (n->op == LANCIUS_OP_INPUT) continue;
            if (n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes
            if (n->op == LANCIUS_OP_CONST) {
                if(!n->runtime_data) {
                    if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32); /* A3 */
                    }
                    /* A2: record buffer ownership */
                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                    if(n->runtime_data) for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }

            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) {
                    float* buf_f32 = NULL;

                    if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        buf_f32 = (float*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        buf_f32 = (float*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32);
                    }

                    n->runtime_data_f32 = buf_f32;
                    if (n->rt) n->rt->buffer_f32 = buf_f32;

                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);

                    if (!buf_f32) {
                        lancius_set_error(LANCIUS_ERROR_OOM);
                        fprintf(stderr, "[EXEC FATAL] FP32 OOM at Node %u (op=%d) size=%zu\n", n->id, n->op, lancius_node_bytes(n));
                        continue;
                    }
                }
            } else if (!n->runtime_data) {
                if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32); /* A3 */
                    }
                /* A2: record buffer ownership */
                lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                if (!n->runtime_data) {
                    lancius_set_error(LANCIUS_ERROR_OOM); /* A4 OOM */
                    fprintf(stderr, "[EXEC FATAL] OOM at Node %u (op=%d) size=%zu\n", n->id, n->op, elements * sizeof(double));
                    continue;
                }
            }

            execute_node_math(n);
        }
    }
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
            size_t elements = lancius_node_elements(n); (void)elements;
            if (n->op == LANCIUS_OP_INPUT) continue;
            if (n->op == LANCIUS_OP_NOP) continue; // V9 Fix: Skip neutralized nodes
            if (n->op == LANCIUS_OP_CONST) {
                if(!n->runtime_data) {
                    if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32); /* A3 */
                    }
                    /* A2: record buffer ownership */
                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
                    if(n->runtime_data) for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }
            if (n->dtype == LANCIUS_DTYPE_FP32) {
                if (!n->runtime_data_f32) {
                    float* buf_f32 = NULL;

                    if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        buf_f32 = (float*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        buf_f32 = (float*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32);
                    }

                    n->runtime_data_f32 = buf_f32;
                    if (n->rt) n->rt->buffer_f32 = buf_f32;

                    lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);

                    if (!buf_f32) {
                        lancius_set_error(LANCIUS_ERROR_OOM);
                        fprintf(stderr, "[EXEC FATAL] FP32 OOM at Node %u (op=%d) size=%zu\n", n->id, n->op, lancius_node_bytes(n));
                        continue;
                    }
                }
            } else if (!n->runtime_data) {
                if (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) {
                        n->runtime_data = (double*)((uint8_t*)schedule->static_pool + schedule->plan->offsets[n->id]);
                    } else {
                        n->runtime_data = (double*)lancius_arena_alloc(scratch, lancius_node_bytes(n), 32); /* A3 */
                    }
                /* A2: record buffer ownership */
                lancius_node_set_owner(n, (schedule->plan && schedule->plan->is_pooled[n->id] && schedule->static_pool) ? LANCIUS_MEMORY_POOL : LANCIUS_MEMORY_ARENA);
            }
        }
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;
            if (!n->runtime_data) continue;
            lancius_pool_submit(pool, (lancius_task_fn)execute_node_math, n);
        }
        lancius_pool_wait(pool);
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
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) {
                wave_mem += lancius_node_bytes(n); /* A3 */
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
                size_t elems = lancius_node_elements(n);
                wave_mem += elems * sizeof(double);
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

    /* v11A1 Task 8: use attached memory plan when available. */
    if (schedule->plan && schedule->plan->offsets && schedule->plan->is_pooled) {
        for (uint32_t w = 0; w < schedule->wave_count; w++) {
            lancius_wave* wave = &schedule->waves[w];

            for (uint32_t i = 0; i < wave->node_count; i++) {
                lancius_node* n = wave->nodes[i];
                if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;
                if (!schedule->plan->is_pooled[n->id]) continue;
                if (n->dtype == LANCIUS_DTYPE_FP32) {
                    n->runtime_data_f32 = (float*)((uint8_t*)flat_buffer + schedule->plan->offsets[n->id]);
                } else {
                    n->runtime_data = (double*)((uint8_t*)flat_buffer + schedule->plan->offsets[n->id]);
                }
                lancius_node_set_owner(n, LANCIUS_MEMORY_POOL);
                lancius_runtime_sync_from_legacy(n);
            }

            for (uint32_t i = 0; i < wave->node_count; i++) {
                lancius_node* n = wave->nodes[i];
                if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) continue;
                if (!schedule->plan->is_pooled[n->id]) continue;
                execute_node_math(n);
            }
        }
    return;
    }


    // Simple bump pointer over the flat buffer
    size_t offset = 0;

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        lancius_wave* wave = &schedule->waves[w];

        // 1. Assign memory to all intermediate nodes in this wave
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;

            size_t elements = lancius_node_elements(n); (void)elements; /* A3 static */
            size_t size = lancius_node_bytes(n); /* A3 static */

            // Align to 32 bytes for AVX2/SIMD
            offset = (offset + 31) & ~(size_t)31;
            if (size > SIZE_MAX - offset) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }

            // Assign the flat buffer pointer directly to the node
                n->runtime_data = (double*)((char*)flat_buffer + offset);

                /* A1: mirror static pool buffer into runtime state */
                if (n->rt) {
                    n->rt->buffer = n->runtime_data;
                    n->rt->offset = offset;
                }
                lancius_node_set_owner(n, LANCIUS_MEMORY_POOL); /* A2 static */

                offset += size;
        }

        // 2. Execute the math for the wave
        for (uint32_t i = 0; i < wave->node_count; i++) {
            lancius_node* n = wave->nodes[i];
            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;
            if (!n->runtime_data) continue;

            // Reuse the exact same math router from the standard executor
            execute_node_math(n);
        }
    }
}

/* v11S H2 fix: bounded static executor validates buffer size */
void lancius_schedule_execute_static_bounded(lancius_schedule* schedule, void* flat_buffer, size_t buffer_size) {
    if (!schedule || !flat_buffer) return;
    size_t required = lancius_schedule_static_memory_required(schedule);
    if (buffer_size < required) {
        lancius_set_error(LANCIUS_ERROR_OVERFLOW);
        fprintf(stderr, "[EXEC FATAL] static buffer too small: need %zu, got %zu\n", required, buffer_size);
        return;
    }
    lancius_schedule_execute_static(schedule, flat_buffer);
}

static void execute_permute(lancius_node* n) {
    if (!n || n->input_count == 0) return;

    const lancius_node* in = n->inputs[0];
    const double* x = in ? in->runtime_data : NULL;
    double* y = n->runtime_data;

    if (!x || !y) return;

    size_t in_shape[4] = {1, 1, 1, 1};
    size_t out_shape[4] = {1, 1, 1, 1};

    for (uint8_t i = 0; i < in->ndim && i < 4; i++) in_shape[i] = in->shape[i];
    for (uint8_t i = 0; i < n->ndim && i < 4; i++) out_shape[i] = n->shape[i];

    size_t in_stride[4];
    in_stride[3] = 1;
    in_stride[2] = in_shape[3];
    in_stride[1] = in_shape[2] * in_stride[2];
    in_stride[0] = in_shape[1] * in_stride[1];

    uint32_t axes[4] = {
        n->axes[0],
        n->axes[1],
        n->axes[2],
        n->axes[3]
    };

    for (int i = 0; i < 4; i++) {
        if (axes[i] >= 4) axes[i] = (uint32_t)i;
    }

    for (size_t i0 = 0; i0 < out_shape[0]; i0++) {
        for (size_t i1 = 0; i1 < out_shape[1]; i1++) {
            for (size_t i2 = 0; i2 < out_shape[2]; i2++) {
                for (size_t i3 = 0; i3 < out_shape[3]; i3++) {

                    size_t out_idx =
                        ((i0 * out_shape[1] + i1) * out_shape[2] + i2) * out_shape[3] + i3;

                    size_t out_indices[4] = {i0, i1, i2, i3};
                    size_t in_idx = 0;

                    for (int d = 0; d < 4; d++) {
                        in_idx += out_indices[d] * in_stride[axes[d]];
                    }

                    y[out_idx] = x[in_idx];
                }
            }
        }
    }
}

static void execute_matmul_batched(lancius_node* n) {
    if (!n || n->input_count < 2) return;

    const lancius_node* a = n->inputs[0];
    const lancius_node* b = n->inputs[1];

    const double* A = a ? a->runtime_data : NULL;
    const double* B = b ? b->runtime_data : NULL;
    double* C = n->runtime_data;

    if (!A || !B || !C) return;

    size_t batches = a->shape[0];
    size_t M = a->shape[1];
    size_t K = a->shape[2];
    size_t N = b->shape[2];

    for (size_t batch = 0; batch < batches; batch++) {
        const double* A_batch = A + batch * M * K;
        const double* B_batch = B + batch * K * N;
        double* C_batch = C + batch * M * N;

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

            if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST || n->op == LANCIUS_OP_NOP) {
                continue;
            }

            size_t bytes = lancius_node_bytes(n);

            offset = (offset + 31) & ~(size_t)31;

            if (bytes > SIZE_MAX - offset) {
                return 0;
            }

            offset += bytes;
        }
    }

    return offset;
}

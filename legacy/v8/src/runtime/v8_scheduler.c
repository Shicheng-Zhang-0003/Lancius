#include "basis/v8_scheduler.h"
#include "basis/v8_vision_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

v8_schedule* v8_ir_schedule(v8_graph* g) {
    if (!g || g->node_count == 0) return NULL;
    v8_schedule* sched = (v8_schedule*)calloc(1, sizeof(v8_schedule));
    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    v8_node** queue = (v8_node**)malloc(sizeof(v8_node*) * g->node_count);

    for (uint32_t i = 0; i < g->node_count; i++) in_degree[g->nodes[i]->id] = g->nodes[i]->input_count;

    uint32_t wave_cap = 16;
    sched->waves = (v8_wave*)malloc(sizeof(v8_wave) * wave_cap);
    uint32_t processed = 0;

    while (processed < g->node_count) {
        uint32_t q_tail = 0;
        for (uint32_t i = 0; i < g->node_count; i++) {
            v8_node* n = g->nodes[i];
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
            sched->waves = (v8_wave*)realloc(sched->waves, sizeof(v8_wave) * wave_cap);
        }
        v8_wave* w = &sched->waves[sched->wave_count++];
        w->node_count = q_tail;
        w->nodes = (v8_node**)malloc(sizeof(v8_node*) * q_tail);
        memcpy(w->nodes, queue, sizeof(v8_node*) * q_tail);

        for (uint32_t i = 0; i < g->node_count; i++) {
            v8_node* n = g->nodes[i];
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

static void execute_node_math(v8_node* n) {
    if (!n || !n->runtime_data) return;
    size_t elements = v8_node_elements(n);

    // CRITICAL FIX: Execute Cross-Entropy BEFORE the Vision Ops router hijacks it
    if (n->op == V8_OP_CROSS_ENTROPY) {
        double* x = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!x || !y) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        double total_loss = 0.0;
        for(size_t r=0; r<R; r++) {
            double max_val = x[r*C];
            for(size_t c=1; c<C; c++) if(x[r*C+c] > max_val) max_val = x[r*C+c];
            double sum_exp = 0.0;
            for(size_t c=0; c<C; c++) sum_exp += exp(x[r*C+c] - max_val);
            double log_sum_exp = log(sum_exp) + max_val;
            for(size_t c=0; c<C; c++) {
                if (y[r*C+c] == 1.0) total_loss -= (x[r*C+c] - log_sum_exp);
            }
        }
        n->runtime_data[0] = total_loss / R;
        return;
    }
    else if (n->op == V8_OP_CROSS_ENTROPY_BWD) {
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

    if (n->op >= V8_OP_CONV2D) { v8_execute_vision_op(n); return; }

    if (n->op == V8_OP_ADD) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] + b[k];
    }
    else if (n->op == V8_OP_MATMUL) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
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
    else if (n->op == V8_OP_MUL) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] * b[k];
    }
    else if (n->op == V8_OP_SUB) {
        double* a = n->inputs[0]->runtime_data; double* b = n->inputs[1]->runtime_data;
        if (!a || !b) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] - b[k];
    }
    else if (n->op == V8_OP_TRANSPOSE) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c*R + r] = a[r*C + c];
    }
    else if (n->op == V8_OP_SUM) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t elems = v8_node_elements(n->inputs[0]);
        double sum = 0.0; for(size_t k=0; k<elems; k++) sum += a[k];
        n->runtime_data[0] = sum;
    }
    else if (n->op == V8_OP_SUM_AXIS0) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        memset(n->runtime_data, 0, C * sizeof(double));
        for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + c];
    }
    else if (n->op == V8_OP_SUM_AXIS1) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t R = n->inputs[0]->shape[0]; size_t C = n->inputs[0]->shape[1];
        for(size_t r=0; r<R; r++) { double s=0; for(size_t c=0; c<C; c++) s += a[r*C + c]; n->runtime_data[r] = s; }
    }
    else if (n->op == V8_OP_BROADCAST) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        size_t in_elems = v8_node_elements(n->inputs[0]);
        if (in_elems == 1) {
            double val = a[0];
            for(size_t k=0; k<elements; k++) n->runtime_data[k] = val;
        } else {
            size_t cols = n->shape[1];
            for(size_t r=0; r<n->shape[0]; r++) for(size_t c=0; c<cols; c++) n->runtime_data[r*cols + c] = a[c];
        }
    }
    else if (n->op == V8_OP_RELU_BWD) {
        double* grad = n->inputs[0]->runtime_data; double* fwd_a = n->inputs[1]->runtime_data;
        if (!grad || !fwd_a) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = fwd_a[k] > 0.0 ? grad[k] : 0.0;
    }
    else if (n->op == V8_OP_SOFTMAX) {
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
    else if (n->op == V8_OP_SOFTMAX_BWD) {
        double* dy = n->inputs[0]->runtime_data; double* y = n->inputs[1]->runtime_data;
        if (!dy || !y) return;
        size_t R = n->shape[0]; size_t C = n->shape[1];
        for(size_t r=0; r<R; r++) {
            double dot = 0.0;
            for(size_t c=0; c<C; c++) dot += dy[r*C + c] * y[r*C + c];
            for(size_t c=0; c<C; c++) n->runtime_data[r*C + c] = y[r*C + c] * (dy[r*C + c] - dot);
        }
    }
    else if (n->op == V8_OP_RELU) {
        double* a = n->inputs[0]->runtime_data; if (!a) return;
        for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] > 0.0 ? a[k] : 0.0;
    }
}

void v8_schedule_execute(v8_schedule* schedule, v8_arena* scratch) {
    if (!schedule) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        v8_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            v8_node* n = wave->nodes[i];
            size_t elements = v8_node_elements(n);

            if (n->op == V8_OP_INPUT) continue;
            if (n->op == V8_OP_CONST) {
                if(!n->runtime_data) {
                    n->runtime_data = (double*)v8_arena_alloc(scratch, elements * sizeof(double), 32);
                    if(n->runtime_data) for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }

            if (!n->runtime_data) {
                n->runtime_data = (double*)v8_arena_alloc(scratch, elements * sizeof(double), 32);
                if (!n->runtime_data) {
                    fprintf(stderr, "[EXEC FATAL] OOM at Node %u (op=%d) size=%zu\n", n->id, n->op, elements * sizeof(double));
                    continue;
                }
            }

            execute_node_math(n);
        }
    }
}

void v8_schedule_execute_parallel(v8_schedule* schedule, v8_arena* scratch, v8_pool* pool) {
    if (!schedule || !pool) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        v8_wave* wave = &schedule->waves[w];
        for (uint32_t i = 0; i < wave->node_count; i++) {
            v8_node* n = wave->nodes[i];
            size_t elements = v8_node_elements(n);
            if (n->op == V8_OP_INPUT) continue;
            if (n->op == V8_OP_CONST) {
                if(!n->runtime_data) {
                    n->runtime_data = (double*)v8_arena_alloc(scratch, elements * sizeof(double), 32);
                    if(n->runtime_data) for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                }
                continue;
            }
            if (!n->runtime_data) {
                n->runtime_data = (double*)v8_arena_alloc(scratch, elements * sizeof(double), 32);
            }
        }
        for (uint32_t i = 0; i < wave->node_count; i++) {
            v8_node* n = wave->nodes[i];
            if (n->op == V8_OP_INPUT || n->op == V8_OP_CONST) continue;
            if (!n->runtime_data) continue;
            v8_pool_submit(pool, (v8_task_fn)execute_node_math, n);
        }
        v8_pool_wait(pool);
    }
}

void v8_schedule_destroy(v8_schedule* schedule) {
    if (!schedule) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) free(schedule->waves[w].nodes);
    free(schedule->waves); free(schedule);
}

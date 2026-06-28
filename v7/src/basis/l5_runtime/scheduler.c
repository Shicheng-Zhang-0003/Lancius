#include "basis/l5_runtime/scheduler.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include <stdint.h>



basis_schedule* basis_ir_schedule(basis_ir_graph* g) {
    if (!g || g->node_count == 0) return NULL;

    basis_schedule* sched = (basis_schedule*)calloc(1, sizeof(basis_schedule));
    if (!sched) return NULL;

    // Allocate metadata for Kahn's Algorithm
    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    basis_ir_node** queue = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * g->node_count);

    // 1. Compute in-degrees
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        in_degree[n->id] = n->input_count;
    }

    // 2. Extract Waves (Level-based Topological Sort)
    uint32_t wave_cap = 16;
    sched->waves = (basis_wave*)malloc(sizeof(basis_wave) * wave_cap);
    sched->wave_count = 0;

    uint32_t processed = 0;

    while (processed < g->node_count) {
        uint32_t q_tail = 0;

        // Find all nodes with 0 in-degree (ready to execute)
        for (uint32_t i = 0; i < g->node_count; i++) {
            basis_ir_node* n = g->nodes[i];
            // We use a trick: mark processed nodes by setting in_degree to UINT32_MAX
            if (in_degree[n->id] == 0) {
                queue[q_tail++] = n;
                in_degree[n->id] = UINT32_MAX; // Mark as processed
            }
        }

        if (q_tail == 0) break; // Should not happen in a valid DAG

        // Create a new Wave
        if (sched->wave_count >= wave_cap) {
            wave_cap *= 2;
            sched->waves = (basis_wave*)realloc(sched->waves, sizeof(basis_wave) * wave_cap);
        }

        basis_wave* w = &sched->waves[sched->wave_count++];
        w->node_count = q_tail;
        w->nodes = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * q_tail);
        memcpy(w->nodes, queue, sizeof(basis_ir_node*) * q_tail);

        // Decrement in-degrees of children
        for (uint32_t i = 0; i < g->node_count; i++) {
            basis_ir_node* n = g->nodes[i];
            if (in_degree[n->id] == UINT32_MAX) continue; // Already processed

            bool all_inputs_ready = true;
            for (uint32_t j = 0; j < n->input_count; j++) {
                if (in_degree[n->inputs[j]->id] != UINT32_MAX) {
                    all_inputs_ready = false;
                    break;
                }
            }
            if (all_inputs_ready) {
                in_degree[n->id] = 0; // Ready for next wave
            }
        }

        processed += q_tail;
    }

    free(in_degree);
    free(queue);
    return sched;
}

void basis_schedule_execute(basis_schedule* schedule, basis_arena* scratch) {
    if (!schedule) return;

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        basis_wave* wave = &schedule->waves[w];

        for (uint32_t i = 0; i < wave->node_count; i++) {
            basis_ir_node* n = wave->nodes[i];
            size_t elements = n->rows * n->cols;

            if (n->op == BASIS_IR_OP_INPUT) {
                if(!n->runtime_data) {
                    n->runtime_data = (double*)basis_arena_alloc(scratch, elements * sizeof(double), 32);
                    if(n->runtime_data) memset(n->runtime_data, 0, elements * sizeof(double));
                }
                continue;
            }
            if (n->op == BASIS_IR_OP_CONST) {
                if(!n->runtime_data) {
                    n->runtime_data = (double*)basis_arena_alloc(scratch, elements * sizeof(double), 32);
                    if(n->runtime_data) {
                        for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val;
                    }
                }
                continue;
            }

            if (!n->runtime_data) {
                n->runtime_data = (double*)basis_arena_alloc(scratch, elements * sizeof(double), 32);
                if (!n->runtime_data) continue; // OOM Guard
            }

            if (n->op == BASIS_IR_OP_ADD) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                if (!a || !b || !n->runtime_data) continue; // UPSTREAM GUARD
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] + b[k];
            }
            else if (n->op == BASIS_IR_OP_MATMUL) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                if (!a || !b || !n->runtime_data) continue; // UPSTREAM GUARD
                size_t M = n->inputs[0]->rows;
                size_t K = n->inputs[0]->cols;
                size_t N = n->inputs[1]->cols;
                for(size_t r=0; r<M; r++) {
                    for(size_t c=0; c<N; c++) {
                        double sum = 0.0;
                        for(size_t k=0; k<K; k++) sum += a[r*K + k] * b[k*N + c];
                        n->runtime_data[r*N + c] = sum;
                    }
                }
            }
            else if (n->op == BASIS_IR_OP_MUL) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                if (!a || !b || !n->runtime_data) continue; // UPSTREAM GUARD
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] * b[k];
            }
            else if (n->op == BASIS_IR_OP_SUB) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                if (!a || !b || !n->runtime_data) continue; // UPSTREAM GUARD
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] - b[k];
            }
            else if (n->op == BASIS_IR_OP_TRANSPOSE) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                size_t R = n->inputs[0]->rows;
                size_t C = n->inputs[0]->cols;
                for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c*R + r] = a[r*C + c];
            }
            else if (n->op == BASIS_IR_OP_SUM) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                size_t elems = n->inputs[0]->rows * n->inputs[0]->cols;
                double sum = 0.0;
                for(size_t k=0; k<elems; k++) sum += a[k];
                n->runtime_data[0] = sum;
            }
            else if (n->op == BASIS_IR_OP_SUM_AXIS0) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                size_t R = n->inputs[0]->rows; size_t C = n->inputs[0]->cols;
                memset(n->runtime_data, 0, C * sizeof(double));
                for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + c];
            }
            else if (n->op == BASIS_IR_OP_SUM_AXIS1) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                size_t R = n->inputs[0]->rows; size_t C = n->inputs[0]->cols;
                for(size_t r=0; r<R; r++) { double s=0; for(size_t c=0; c<C; c++) s += a[r*C + c]; n->runtime_data[r] = s; }
            }
            else if (n->op == BASIS_IR_OP_BROADCAST) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                double val = a[0];
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = val;
            }
            else if (n->op == BASIS_IR_OP_RELU_BWD) {
                double* grad = n->inputs[0]->runtime_data;
                double* fwd_a = n->inputs[1]->runtime_data;
                if (!grad || !fwd_a || !n->runtime_data) continue;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = fwd_a[k] > 0.0 ? grad[k] : 0.0;
            }
            else if (n->op == BASIS_IR_OP_SOFTMAX) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                size_t R = n->rows;
                size_t C = n->cols;
                for(size_t r=0; r<R; r++) {
                    double max_val = a[r*C];
                    for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
                    double sum = 0.0;
                    for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c] - max_val); sum += n->runtime_data[r*C+c]; }
                    for(size_t c=0; c<C; c++) n->runtime_data[r*C+c] /= sum;
                }
            }
            else if (n->op == BASIS_IR_OP_SOFTMAX_BWD) {
                double* dy = n->inputs[0]->runtime_data;
                double* y = n->inputs[1]->runtime_data;
                if (!dy || !y || !n->runtime_data) continue;
                size_t R = n->rows; size_t C = n->cols;
                for(size_t r=0; r<R; r++) {
                    double dot = 0.0;
                    for(size_t c=0; c<C; c++) dot += dy[r*C + c] * y[r*C + c];
                    for(size_t c=0; c<C; c++) n->runtime_data[r*C + c] = y[r*C + c] * (dy[r*C + c] - dot);
                }
            }
            else if (n->op == BASIS_IR_OP_RELU) {
                double* a = n->inputs[0]->runtime_data;
                if (!a || !n->runtime_data) continue;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] > 0.0 ? a[k] : 0.0;
            }
        }
    }
}


void basis_schedule_print(basis_schedule* schedule) {
    if (!schedule) return;
    printf("=== Execution Schedule (%u Waves) ===\n", schedule->wave_count);
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        printf("Wave %u (%u parallel nodes): [ ", w, schedule->waves[w].node_count);
        for (uint32_t i = 0; i < schedule->waves[w].node_count; i++) {
            printf("%%%u ", schedule->waves[w].nodes[i]->id);
        }
        printf("]\n");
    }
}

void basis_schedule_destroy(basis_schedule* schedule) {
    if (!schedule) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        free(schedule->waves[w].nodes);
    }
    free(schedule->waves);
    free(schedule);
}

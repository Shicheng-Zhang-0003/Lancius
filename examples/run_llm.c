#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

int main() {
    printf("================================================================\n");
    printf("  Lancius v12R1: TRANSFORMER KERNEL DEMO (Experimental)            \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();
    if (!g) { fprintf(stderr, "FATAL: graph create failed\n"); return 1; }

    // Mock LLM dimensions: Seq=2, Heads=2, HeadDim=4
    size_t seq = 2, heads = 2, dim = 4;

    /* Despot truth: hand-mutated ndim/shape desynced element counts
     * (was: lancius_input 2D + manual 3D poke). Use the 3D builder. */
    lancius_node* Q_in = lancius_input_3d(g, seq, heads, dim);
    lancius_node* K_in = lancius_input_3d(g, seq, heads, dim);
    lancius_node* V_in = lancius_input_3d(g, seq, heads, dim);
    if (!Q_in || !K_in || !V_in) { fprintf(stderr, "FATAL: QKV inputs failed\n"); lancius_graph_destroy(g); return 1; }

    // 1. Multi-Head Attention
    lancius_node* attn = lancius_attention(g, Q_in, K_in, V_in);
    if (!attn) { fprintf(stderr, "FATAL: attention node failed\n"); lancius_graph_destroy(g); return 1; }

    // 2. LayerNorm (gamma=1, beta=0)
    lancius_node* gamma = lancius_input(g, 1, heads * dim);
    lancius_node* beta = lancius_input(g, 1, heads * dim);
    if (!gamma || !beta) { fprintf(stderr, "FATAL: norm inputs failed\n"); lancius_graph_destroy(g); return 1; }
    lancius_node* ln = lancius_layernorm(g, attn, gamma, beta);
    if (!ln) { fprintf(stderr, "FATAL: layernorm node failed\n"); lancius_graph_destroy(g); return 1; }

    // 3. GELU Activation
    lancius_node* gelu = lancius_gelu(g, ln);
    if (!gelu) { fprintf(stderr, "FATAL: gelu node failed\n"); lancius_graph_destroy(g); return 1; }

    // Allocate dummy data
    /* Despot truth: unchecked calloc + wrapping arena size (were: deref/OOM). */
    size_t qkv_sz = 0, gb_sz = 0;
    if (heads > SIZE_MAX / dim) { fprintf(stderr, "FATAL: dims overflow\n"); lancius_graph_destroy(g); return 1; }
    gb_sz = heads * dim;
    if (seq > SIZE_MAX / gb_sz) { fprintf(stderr, "FATAL: dims overflow\n"); lancius_graph_destroy(g); return 1; }
    qkv_sz = seq * gb_sz;
    Q_in->runtime_data = (double*)calloc(qkv_sz ? qkv_sz : 1, sizeof(double));
    K_in->runtime_data = (double*)calloc(qkv_sz ? qkv_sz : 1, sizeof(double));
    V_in->runtime_data = (double*)calloc(qkv_sz ? qkv_sz : 1, sizeof(double));
    gamma->runtime_data = (double*)calloc(gb_sz ? gb_sz : 1, sizeof(double));
    beta->runtime_data = (double*)calloc(gb_sz ? gb_sz : 1, sizeof(double));
    if (!Q_in->runtime_data || !K_in->runtime_data || !V_in->runtime_data ||
        !gamma->runtime_data || !beta->runtime_data) {
        fprintf(stderr, "FATAL: OOM dummy data\n");
        free(Q_in->runtime_data); free(K_in->runtime_data); free(V_in->runtime_data);
        free(gamma->runtime_data); free(beta->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }

    // V10S FIX: Inject variance so LayerNorm doesn't collapse to 0.0
for(size_t i=0; i<qkv_sz; i++) {
    Q_in->runtime_data[i] = 0.5 + 0.1 * i;
    K_in->runtime_data[i] = 0.5 - 0.05 * i;
    V_in->runtime_data[i] = 1.0 + 0.2 * i;
}
    for(size_t i=0; i<heads*dim; i++) { gamma->runtime_data[i] = 1.0; beta->runtime_data[i] = 0.0; }

    printf("[1/2] Compiling Transformer Schedule...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);
    if (!sched) { fprintf(stderr, "FATAL: schedule failed\n"); free(Q_in->runtime_data); free(K_in->runtime_data); free(V_in->runtime_data); free(gamma->runtime_data); free(beta->runtime_data); lancius_graph_destroy(g); return 1; }
    size_t peak_mem = lancius_schedule_peak_memory(sched);
    printf("  Liveness Analyzer: Peak Memory = %zu bytes\n", peak_mem);

    /* Despot truth: peak*2+1MB wrapped size_t; NULL scratch executed. */
    size_t arena_need = 0;
    if (peak_mem > (SIZE_MAX - 1024*1024) / 2) { fprintf(stderr, "FATAL: arena size overflow\n"); lancius_schedule_destroy(sched); free(Q_in->runtime_data); free(K_in->runtime_data); free(V_in->runtime_data); free(gamma->runtime_data); free(beta->runtime_data); lancius_graph_destroy(g); return 1; }
    arena_need = peak_mem * 2 + 1024*1024;
    lancius_arena* scratch = lancius_arena_create(arena_need ? arena_need : 1);
    if (!scratch) { fprintf(stderr, "FATAL: OOM scratch arena\n"); lancius_schedule_destroy(sched); free(Q_in->runtime_data); free(K_in->runtime_data); free(V_in->runtime_data); free(gamma->runtime_data); free(beta->runtime_data); lancius_graph_destroy(g); return 1; }

    printf("[2/2] Executing Transformer Block...\n");
    lancius_schedule_execute(sched, scratch);

    /* Despot truth: gelu[i<8] read unchecked (was: OOB/NULL). */
    {
        size_t ge = 0;
        if (!gelu->runtime_data || !lancius_node_elements_checked(gelu, &ge) || ge < 8) {
            fprintf(stderr, "FATAL: gelu output missing/short\n");
            lancius_schedule_destroy(sched); lancius_graph_destroy(g); lancius_arena_destroy(scratch);
            free(Q_in->runtime_data); free(K_in->runtime_data); free(V_in->runtime_data);
            free(gamma->runtime_data); free(beta->runtime_data);
            return 1;
        }
    }
    printf("  ✅ Transformer Block Executed Successfully!\n");
    printf("  Output GELU Logits (first 8 elements):\n  [");
    for(int i=0; i<8; i++) printf("%.4f ", gelu->runtime_data[i]);
    printf("]\n");

    // Extract heap pointers before destroying the graph Arena (which invalidates the node pointers)
    double* q_data = Q_in->runtime_data;
    double* k_data = K_in->runtime_data;
    double* v_data = V_in->runtime_data;
    double* gamma_data = gamma->runtime_data;
    double* beta_data = beta->runtime_data;

    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);

    free(q_data); free(k_data); free(v_data);
    free(gamma_data); free(beta_data);

    printf("\n================================================================\n");
    printf("  LANCIUS v12R1 TRANSFORMER KERNEL DEMO EXECUTED.\n");
    printf("================================================================\n");
    return 0;
}

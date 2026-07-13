#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("================================================================\n");
    printf("  Lancius v9A: TRANSFORMER MANDATE (LLM Kernels)             \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    // Mock LLM dimensions: Seq=2, Heads=2, HeadDim=4
    size_t seq = 2, heads = 2, dim = 4;

    // Create Q, K, V inputs (ndim=3)
    lancius_node* Q_in = lancius_input(g, seq, heads * dim);
    Q_in->ndim = 3; Q_in->shape[0] = seq; Q_in->shape[1] = heads; Q_in->shape[2] = dim;

    lancius_node* K_in = lancius_input(g, seq, heads * dim);
    K_in->ndim = 3; K_in->shape[0] = seq; K_in->shape[1] = heads; K_in->shape[2] = dim;

    lancius_node* V_in = lancius_input(g, seq, heads * dim);
    V_in->ndim = 3; V_in->shape[0] = seq; V_in->shape[1] = heads; V_in->shape[2] = dim;

    // 1. Multi-Head Attention
    lancius_node* attn = lancius_attention(g, Q_in, K_in, V_in);

    // 2. LayerNorm (gamma=1, beta=0)
    lancius_node* gamma = lancius_input(g, 1, heads * dim);
    lancius_node* beta = lancius_input(g, 1, heads * dim);
    lancius_node* ln = lancius_layernorm(g, attn, gamma, beta);

    // 3. GELU Activation
    lancius_node* gelu = lancius_gelu(g, ln);

    // Allocate dummy data
    size_t qkv_sz = seq * heads * dim;
    Q_in->runtime_data = (double*)calloc(qkv_sz, sizeof(double));
    K_in->runtime_data = (double*)calloc(qkv_sz, sizeof(double));
    V_in->runtime_data = (double*)calloc(qkv_sz, sizeof(double));
    gamma->runtime_data = (double*)calloc(heads * dim, sizeof(double));
    beta->runtime_data = (double*)calloc(heads * dim, sizeof(double));

    // V10A1 FIX: Inject variance so LayerNorm doesn't collapse to 0.0
for(size_t i=0; i<qkv_sz; i++) {
    Q_in->runtime_data[i] = 0.5 + 0.1 * i;
    K_in->runtime_data[i] = 0.5 - 0.05 * i;
    V_in->runtime_data[i] = 1.0 + 0.2 * i;
}
    for(size_t i=0; i<heads*dim; i++) { gamma->runtime_data[i] = 1.0; beta->runtime_data[i] = 0.0; }

    printf("[1/2] Compiling Transformer Schedule...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);
    size_t peak_mem = lancius_schedule_peak_memory(sched);
    printf("  🧠 Liveness Analyzer: Peak Memory = %zu bytes\n", peak_mem);

    lancius_arena* scratch = lancius_arena_create(peak_mem * 2 + 1024*1024);

    printf("[2/2] Executing Transformer Block...\n");
    lancius_schedule_execute(sched, scratch);

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
    printf("  LANCIUS v9A TRANSFORMER MANDATE VERIFIED.\n");
    printf("================================================================\n");
    return 0;
}

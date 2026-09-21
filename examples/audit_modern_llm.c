#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS v10S STABLE: MODERN LLM PARITY AUDIT (LLAMA 3 / MISTRAL)\n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();

    // 1. RMSNorm Test
    printf("[1/3] Testing RMSNorm (Llama 3 / Mistral Standard)...\n");
    lancius_node* x = lancius_input(g, 4, 8); // seq=4, hidden=8
    lancius_node* gamma = lancius_input(g, 1, 8);
    lancius_node* rms = lancius_rmsnorm(g, x, gamma);

    x->runtime_data = (double*)calloc(32, sizeof(double));
    gamma->runtime_data = (double*)malloc(8 * sizeof(double));
    for(int i=0; i<32; i++) x->runtime_data[i] = 1.0;
    for(int i=0; i<8; i++) gamma->runtime_data[i] = 2.0;

    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024);
    lancius_schedule_execute(sched, scratch);

    // Expected: RMS of [1,1..1] is 1.0. inv_rms = 1.0. out = 1.0 * 2.0 = 2.0
    int rms_pass = 1;
    for(int i=0; i<32; i++) {
        if (fabs(rms->runtime_data[i] - 2.0) > 1e-5) { rms_pass = 0; break; }
    }
    if (rms_pass) printf("  ✅ RMSNorm Math Verified (Output = 2.0)\n");
    else printf("  ❌ RMSNorm Math Failed\n");

    lancius_schedule_destroy(sched);
    lancius_arena_reset(scratch);

    // 2. SwiGLU Test
    printf("[2/3] Testing SwiGLU (Modern FFN Activation)...\n");
    lancius_graph* g2 = lancius_graph_create();
    lancius_node* gate = lancius_input(g2, 2, 4);
    lancius_node* up = lancius_input(g2, 2, 4);
    lancius_node* swiglu = lancius_swiglu(g2, gate, up);

    gate->runtime_data = (double*)calloc(8, sizeof(double));
    up->runtime_data = (double*)calloc(8, sizeof(double));
    for(int i=0; i<8; i++) { gate->runtime_data[i] = 0.0; up->runtime_data[i] = 5.0; }

    lancius_schedule* sched2 = lancius_ir_schedule(g2);
    lancius_schedule_execute(sched2, scratch);

    // SiLU(0) = 0 * 0.5 = 0. 0 * 5.0 = 0.0
    int swiglu_pass = 1;
    for(int i=0; i<8; i++) {
        if (fabs(swiglu->runtime_data[i] - 0.0) > 1e-5) { swiglu_pass = 0; break; }
    }
    if (swiglu_pass) printf("  ✅ SwiGLU Math Verified (SiLU(0) * 5.0 = 0.0)\n");
    else printf("  ❌ SwiGLU Math Failed\n");

    lancius_schedule_destroy(sched2);
    lancius_arena_reset(scratch);

    // 3. GQA Test
    printf("[3/3] Testing Grouped-Query Attention (GQA)...\n");
    lancius_graph* g3 = lancius_graph_create();
    size_t seq = 2, hq = 4, hk = 2, dim = 4;
    lancius_node* Q = lancius_input(g3, seq, hq * dim);
    Q->ndim = 3; Q->shape[0] = seq; Q->shape[1] = hq; Q->shape[2] = dim;
    lancius_node* K = lancius_input(g3, seq, hk * dim);
    K->ndim = 3; K->shape[0] = seq; K->shape[1] = hk; K->shape[2] = dim;
    lancius_node* V = lancius_input(g3, seq, hk * dim);
    V->ndim = 3; V->shape[0] = seq; V->shape[1] = hk; V->shape[2] = dim;

    lancius_node* gqa = lancius_gqa(g3, Q, K, V, hq, hk);
    (void)gqa; // Suppress -Werror=unused-variable

    Q->runtime_data = (double*)calloc(seq * hq * dim, sizeof(double));
    K->runtime_data = (double*)calloc(seq * hk * dim, sizeof(double));
    V->runtime_data = (double*)calloc(seq * hk * dim, sizeof(double));

    for(size_t i=0; i<seq*hq*dim; i++) Q->runtime_data[i] = 1.0;
    for(size_t i=0; i<seq*hk*dim; i++) K->runtime_data[i] = 1.0;
    for(size_t i=0; i<seq*hk*dim; i++) V->runtime_data[i] = 1.0;

    lancius_schedule* sched3 = lancius_ir_schedule(g3);
    lancius_schedule_execute(sched3, scratch);

    /* Despot truth: GQA with uniform 1.0 inputs must yield 1.0 outputs
       (equal softmax 0.5/0.5 over two identical V rows). Execution alone is not parity. */
    int gqa_pass = 1;
    if (!gqa || !gqa->runtime_data) gqa_pass = 0;
    else {
        for (size_t i = 0; i < seq * hq * dim; i++) {
            if (fabs(gqa->runtime_data[i] - 1.0) > 1e-9) { gqa_pass = 0; break; }
        }
    }
    if (gqa_pass) printf("  ✅ GQA Values Verified (uniform-1.0 -> 1.0, Heads: %zu Q, %zu KV)\n", hq, hk);
    else printf("  ❌ GQA Values Wrong (expected all 1.0)\n");

    lancius_schedule_destroy(sched3);
    lancius_graph_destroy(g3);
    lancius_graph_destroy(g2);
    lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);

    printf("\n================================================================\n");
    if (rms_pass && swiglu_pass && gqa_pass) {
    printf("  MODERN LLM PARITY VERIFIED. READY FOR v10S STABLE.           \n");
    } else {
    printf("  MODERN LLM PARITY FAILED.                                   \n");
    }
    printf("================================================================\n");
    return (rms_pass && swiglu_pass && gqa_pass) ? 0 : 1;
}

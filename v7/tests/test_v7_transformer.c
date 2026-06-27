#include "basis.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Reference Transformer Block (Attention+FFN) \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    size_t SEQ = 4, D = 4;
    double scale_factor = 1.0 / sqrt((double)D);

    printf("=== 1. Tracing Transformer Block (Deferred Execution) ===\n");

    basis_v7_tensor* X   = basis_v7_input(g, SEQ, D, NULL);
    basis_v7_tensor* Wq  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* Wk  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* Wv  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* W1  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* W2  = basis_v7_input(g, D, D, NULL);

    basis_v7_tensor* Scale = basis_v7_const(g, scale_factor, SEQ, SEQ);

    basis_v7_tensor* Q = basis_v7_matmul(g, X, Wq);
    basis_v7_tensor* K = basis_v7_matmul(g, X, Wk);
    basis_v7_tensor* V = basis_v7_matmul(g, X, Wv);

    basis_v7_tensor* K_T     = basis_v7_transpose(g, K);
    basis_v7_tensor* Scores  = basis_v7_matmul(g, Q, K_T);
    basis_v7_tensor* Scaled  = basis_v7_mul(g, Scores, Scale);
    basis_v7_tensor* Probs   = basis_v7_softmax(g, Scaled);
    basis_v7_tensor* AttnOut = basis_v7_matmul(g, Probs, V);

    basis_v7_tensor* H1      = basis_v7_matmul(g, AttnOut, W1);
    basis_v7_tensor* H1_Act  = basis_v7_relu(g, H1);
    basis_v7_tensor* FFN_Out = basis_v7_matmul(g, H1_Act, W2);

    basis_v7_tensor* BlockOut = basis_v7_add(g, AttnOut, FFN_Out);
    (void)BlockOut;

    printf("Raw Graph Nodes: %u\n", g->node_count);

    printf("\n=== 2. Running Compiler Passes ===\n");
    basis_ir_graph* g_opt = basis_ir_run_passes(g);
    printf("Optimized Graph Nodes: %u\n", g_opt->node_count);

    printf("\n=== 3. Extracting Parallel Execution Waves ===\n");
    basis_schedule* sched = basis_ir_schedule(g_opt);
    basis_schedule_print(sched);

    bool qkv_parallel = false;
    for(uint32_t w=0; w<sched->wave_count; w++) {
        if(sched->waves[w].node_count == 3) {
            qkv_parallel = true; break;
        }
    }
    if(qkv_parallel) {
        printf("\n  [PASS] Scheduler automatically parallelized Q, K, V projections!\n");
    } else {
        printf("\n  [FAIL] Q, K, V were not grouped in the same parallel wave.\n");
        failures++;
    }

    printf("\n=== 4. Executing Transformer Block ===\n");
    basis_schedule_execute(sched);

    basis_ir_node* opt_out = g_opt->nodes[g_opt->node_count - 1];
    basis_ir_node* opt_probs = NULL;
    for(uint32_t i=0; i<g_opt->node_count; i++) {
        if(g_opt->nodes[i]->op == BASIS_IR_OP_SOFTMAX) {
            opt_probs = g_opt->nodes[i];
            break;
        }
    }

    if (opt_out->rows == SEQ && opt_out->cols == D) {
        printf("  [PASS] Final output shape is correct [%zu x %zu]\n", SEQ, D);
    } else {
        printf("  [FAIL] Final output shape mismatch\n");
        failures++;
    }

    if (opt_probs && opt_probs->runtime_data) {
        double* probs_data = opt_probs->runtime_data;
        double row_sum = 0.0;
        for(size_t i=0; i<SEQ; i++) row_sum += probs_data[i];
        if (fabs(row_sum - 1.0) < 1e-6) {
            printf("  [PASS] Attention probabilities sum to 1.0 (Softmax verified)\n");
        } else {
            printf("  [FAIL] Softmax probabilities sum to %f\n", row_sum);
            failures++;
        }
    } else {
        printf("  [FAIL] Could not find executed Softmax node\n");
        failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g_opt);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ V7 REFERENCE TRANSFORMER SUCCESSFULLY EXECUTED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

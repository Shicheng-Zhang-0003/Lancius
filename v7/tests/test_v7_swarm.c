#include "basis.h"
#include <stdio.h>

// Helper to build a simple forward-only expert graph
basis_ir_graph* build_expert_graph(double multiplier) {
    basis_ir_graph* g = basis_ir_graph_create();
    basis_v7_tensor* X = basis_v7_input(g, 2, 2, NULL);
    basis_v7_tensor* W = basis_v7_const(g, multiplier, 2, 2);
    basis_v7_tensor* Out = basis_v7_mul(g, X, W); // Element-wise multiply
    (void)Out;
    return g;
}

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L7 Micro-Model Swarm & Planner              \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // 1. Build the Experts (Micro-Models)
    printf("[1/4] Compiling Micro-Models...\n");
        // Actually, let's make Expert A multiply by 10.0, Expert B multiply by -1.0
    basis_ir_graph* g_A = build_expert_graph(10.0);
    basis_ir_graph* g_B = build_expert_graph(-1.0);

    basis_schedule* sched_A = basis_ir_schedule(g_A);
    basis_schedule* sched_B = basis_ir_schedule(g_B);

    // 2. Initialize the Swarm Registry
    printf("[2/4] Registering Experts in the Swarm...\n");
    basis_swarm* swarm = basis_swarm_create();
    basis_swarm_register(swarm, "Expert_A_Positive", sched_A, g_A);
    basis_swarm_register(swarm, "Expert_B_Negative", sched_B, g_B);

    // 3. The Planner (Structured Routing Protocol)
    printf("[3/4] Executing Planner Routing Logic...\n");
    double inputs[4][4] = {
        { 1.0,  2.0,  3.0,  4.0},  // Positive -> Route to A (* 10)
        {-5.0, -6.0, -7.0, -8.0},  // Negative -> Route to B (* -1)
        { 0.5,  0.5,  0.5,  0.5},  // Positive -> Route to A (* 10)
        {-2.0,  2.0, -2.0,  2.0}   // Mixed -> Route based on first element (Negative -> B)
    };
    double outputs[4][4];

    basis_expert* exp_A = basis_swarm_get(swarm, "Expert_A_Positive");
    basis_expert* exp_B = basis_swarm_get(swarm, "Expert_B_Negative");

    for(int i=0; i<4; i++) {
        // The Planner inspects the tensor and routes it
        if (inputs[i][0] > 0.0) {
            basis_swarm_dispatch(exp_A, inputs[i], outputs[i]);
        } else {
            basis_swarm_dispatch(exp_B, inputs[i], outputs[i]);
        }
    }

    // 4. Verify Structured Inter-Model Protocol
    printf("[4/4] Verifying Outputs...\n");

    // Input 0: [1,2,3,4] * 10 = [10, 20, 30, 40]
    if (outputs[0][0] == 10.0 && outputs[0][3] == 40.0) printf("  [PASS] Planner correctly routed positive tensor to Expert A\n");
    else { printf("  [FAIL] Expert A routing/math failed\n"); failures++; }

    // Input 1: [-5,-6,-7,-8] * -1 = [5, 6, 7, 8]
    if (outputs[1][0] == 5.0 && outputs[1][3] == 8.0) printf("  [PASS] Planner correctly routed negative tensor to Expert B\n");
    else { printf("  [FAIL] Expert B routing/math failed\n"); failures++; }

    basis_swarm_destroy(swarm);
    basis_ir_graph_destroy(g_A);
    basis_ir_graph_destroy(g_B);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ V7.4 MICRO-MODEL SWARM SUCCESSFULLY EXECUTED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

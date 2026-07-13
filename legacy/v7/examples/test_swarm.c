#include <basis.h>
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
    printf("=== BASIS V7: Standalone Swarm Execution ===\n\n");

    // 1. Compile Micro-Models (Experts)
    basis_ir_graph* g_A = build_expert_graph(10.0);
    basis_schedule* sched_A = basis_ir_schedule(g_A);

    // 2. Register in the Swarm
    basis_swarm* swarm = basis_swarm_create();
    basis_swarm_register(swarm, "Expert_A", sched_A, g_A);

    // 3. Planner Routes Tensors
    basis_expert* exp = basis_swarm_get(swarm, "Expert_A");
    double in[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];

    basis_swarm_dispatch(exp, in, out);

    printf("Input:  [%.1f, %.1f, %.1f, %.1f]\n", in[0], in[1], in[2], in[3]);
    printf("Output: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);

    // 4. Teardown
    basis_swarm_destroy(swarm);
    basis_ir_graph_destroy(g_A);

    printf("\n✅ Standalone Swarm Execution Complete.\n");
    return 0;
}

#include "basis.h"
#include <stdio.h>
#include <stdbool.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L4 Compiler Pass Manager Validation         \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // 1. Build an intentionally wasteful, unoptimized graph
    printf("=== Building Unoptimized Graph ===\n");
    basis_ir_graph* g_raw = basis_ir_graph_create();

    basis_v7_tensor* A = basis_v7_input(g_raw, 4, 4, NULL);

    // Wasteful Math: Add constants at runtime
    basis_v7_tensor* C1 = basis_v7_const(g_raw, 5.0, 4, 4);
    basis_v7_tensor* C2 = basis_v7_const(g_raw, 3.0, 4, 4);
    basis_v7_tensor* SumC = basis_v7_add(g_raw, C1, C2); // Should fold to Const(8.0)

    // Wasteful Math: Add zero
    basis_v7_tensor* Zero = basis_v7_const(g_raw, 0.0, 4, 4);
    basis_v7_tensor* AddZero = basis_v7_add(g_raw, A, Zero); // Should simplify to A

    // Final operation to use the results
    basis_v7_tensor* Out = basis_v7_add(g_raw, SumC, AddZero);
    (void)Out; // Suppress unused variable warning

    printf("Raw Graph Node Count: %u\n", g_raw->node_count);
    basis_ir_print(g_raw);

    // 2. Run Compiler Passes
    printf("\n=== Running Compiler Passes (Folding & Elimination) ===\n");
    basis_ir_graph* g_opt = basis_ir_run_passes(g_raw);

    printf("Optimized Graph Node Count: %u\n", g_opt->node_count);
    basis_ir_print(g_opt);

    // 3. Verify Optimizations
    printf("\n=== Verification ===\n");

    // The raw graph had: A, C1, C2, SumC, Zero, AddZero, Out = 7 nodes
    // The optimized graph should have: A, SumC(folded to Const 8), Out = 3 nodes
    // (C1, C2, Zero, and AddZero should be completely eliminated)

    if (g_opt->node_count == 3) {
        printf("  [PASS] Node count reduced from 7 to 3 (Dead code eliminated)\n");
    } else {
        printf("  [FAIL] Node count is %u (Expected 3)\n", g_opt->node_count);
        failures++;
    }

    // Verify Constant Folding: Node 1 should be CONST with value 8.0
    basis_ir_node* folded_node = g_opt->nodes[1];
    if (folded_node->op == BASIS_IR_OP_CONST && folded_node->attr_val == 8.0) {
        printf("  [PASS] Constant Folding: Add(Const 5, Const 3) -> Const 8\n");
    } else {
        printf("  [FAIL] Constant folding failed\n");
        failures++;
    }

    // Verify Identity Elimination: The final ADD should take A and the folded Const
    // (Order-agnostic check since addition is commutative)
    basis_ir_node* final_node = g_opt->nodes[2];
    bool has_input = (final_node->inputs[0]->op == BASIS_IR_OP_INPUT || final_node->inputs[1]->op == BASIS_IR_OP_INPUT);
    bool has_const = (final_node->inputs[0]->op == BASIS_IR_OP_CONST || final_node->inputs[1]->op == BASIS_IR_OP_CONST);

    if (final_node->op == BASIS_IR_OP_ADD && has_input && has_const) {
        printf("  [PASS] Identity Elimination: Add(A, 0) simplified directly to A\n");
    } else {
        printf("  [FAIL] Identity elimination failed\n");
        failures++;
    }

    // 4. Cleanup
    basis_ir_graph_destroy(g_raw);
    basis_ir_graph_destroy(g_opt);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 COMPILER PASS TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

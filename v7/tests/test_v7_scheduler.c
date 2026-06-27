#include "basis.h"
#include <stdio.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L5 Deterministic Scheduler Validation       \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // Build a "Diamond" DAG to prove parallel wave extraction
    //      A (Input)
    //     / \
    //    B   C   (Both depend only on A -> Should be in Wave 1 together)
    //     \ /
    //      D     (Depends on B and C -> Should be in Wave 2)

    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    basis_v7_tensor* A = basis_v7_input(g, 2, 2, A_data);
    basis_v7_tensor* B = basis_v7_relu(g, A);
    basis_v7_tensor* C = basis_v7_add(g, A, A);
    basis_v7_tensor* D = basis_v7_add(g, B, C);

    printf("=== Graph Topology ===\n");
    basis_ir_print(g);

    printf("\n=== Scheduling Waves ===\n");
    basis_schedule* sched = basis_ir_schedule(g);
    basis_schedule_print(sched);

    // Verify Wave Count (Should be 3: Wave 0=[A], Wave 1=[B,C], Wave 2=[D])
    if (sched->wave_count == 3) {
        printf("\n  [PASS] Correctly identified 3 execution waves\n");
    } else {
        printf("\n  [FAIL] Expected 3 waves, got %u\n", sched->wave_count);
        failures++;
    }

    // Verify Wave 1 has 2 parallel nodes (B and C)
    if (sched->wave_count >= 2 && sched->waves[1].node_count == 2) {
        printf("  [PASS] Wave 1 correctly groups 2 independent nodes for parallel execution\n");
    } else {
        printf("  [FAIL] Wave 1 parallel grouping failed\n");
        failures++;
    }

    printf("\n=== Executing Schedule ===\n");
    basis_schedule_execute(sched);

    // Math check:
    // A = [1, 2, 3, 4]
    // B = ReLU(A) = [1, 2, 3, 4]
    // C = A + A = [2, 4, 6, 8]
    // D = B + C = [3, 6, 9, 12]

    double* d_data = D->node->runtime_data;
    if (d_data[0] == 3.0 && d_data[3] == 12.0) {
        printf("  [PASS] Wave-based execution math is correct!\n");
    } else {
        printf("  [FAIL] Math mismatch in wave execution\n");
        failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 SCHEDULER TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

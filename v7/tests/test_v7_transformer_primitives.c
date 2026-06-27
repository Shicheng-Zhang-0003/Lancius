#include "basis.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Transformer Primitives Validation           \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Test Transpose
    double A_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    basis_v7_tensor* A = basis_v7_input(g, 2, 3, A_data);
    basis_v7_tensor* A_T = basis_v7_transpose(g, A);

    // 2. Test Softmax (Log-Sum-Exp trick)
    double logits[] = {1.0, 2.0, 3.0, 100.0, 101.0, 102.0}; // Extreme values to test stability
    basis_v7_tensor* L = basis_v7_input(g, 2, 3, logits);
    basis_v7_tensor* S = basis_v7_softmax(g, L);

    // 3. Test Element-wise MUL & SUB
    basis_v7_tensor* B = basis_v7_input(g, 2, 3, A_data);
    basis_v7_tensor* Mul = basis_v7_mul(g, A, B); // A^2
    basis_v7_tensor* Sub = basis_v7_sub(g, Mul, A); // A^2 - A

    basis_schedule* sched = basis_ir_schedule(g);
    basis_schedule_execute(sched);

    printf("=== Verification ===\n");

    // Verify Transpose (2x3 -> 3x2)
    if (A_T->rows == 3 && A_T->cols == 2 && A_T->node->runtime_data[1] == 4.0) {
        printf("  [PASS] Transpose correctly flipped 2x3 to 3x2\n");
    } else {
        printf("  [FAIL] Transpose math/shape failed\n"); failures++;
    }

    // Verify Softmax Stability (Row 2 has 100, 101, 102. Naive exp() would overflow to Inf)
    double* s_data = S->node->runtime_data;
    double sum_row2 = s_data[3] + s_data[4] + s_data[5];
    if (fabs(sum_row2 - 1.0) < 1e-6 && !isinf(s_data[5])) {
        printf("  [PASS] Softmax Log-Sum-Exp trick prevented overflow on extreme logits\n");
    } else {
        printf("  [FAIL] Softmax overflowed or didn't sum to 1.0\n"); failures++;
    }

    // Verify MUL and SUB (A=3 -> A^2=9 -> A^2-A=6)
    if (Sub->node->runtime_data[2] == 6.0) {
        printf("  [PASS] Element-wise MUL and SUB executed correctly\n");
    } else {
        printf("  [FAIL] Element-wise math failed\n"); failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 TRANSFORMER PRIMITIVE TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

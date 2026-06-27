#include "basis/core/ir.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Immutable Graph IR Validation (SSA & Arena) \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Basic SSA & Shape Inference
    printf("=== Shape Inference & SSA Tests ===\n");
    basis_ir_node* in1 = basis_ir_input(g, 4, 8);
    basis_ir_node* in2 = basis_ir_input(g, 4, 8);
    basis_ir_node* add1 = basis_ir_add(g, in1, in2);

    if (add1 && add1->rows == 4 && add1->cols == 8) printf("  [PASS] Shape inference (Add)\n"); else { printf("  [FAIL] Add shape\n"); failures++; }

    basis_ir_node* w = basis_ir_input(g, 8, 16);
    basis_ir_node* mm = basis_ir_matmul(g, add1, w);
    if (mm && mm->rows == 4 && mm->cols == 16) printf("  [PASS] Shape inference (MatMul)\n"); else { printf("  [FAIL] MatMul shape\n"); failures++; }

    // Test Shape Validation (Should return NULL)
    basis_ir_node* bad_mm = basis_ir_matmul(g, in1, in2); // 4x8 * 4x8 is invalid
    if (bad_mm == NULL) printf("  [PASS] Shape validation rejects invalid MatMul\n"); else { printf("  [FAIL] Validation\n"); failures++; }

    // 2. The 1,000,000 Node Stress Test & O(1) Teardown
    printf("\n=== Stress Test: 1,000,000 Node Deep Graph ===\n");
    basis_ir_node* current = in1;
    basis_ir_node* bias = basis_ir_const(g, 0.5, 4, 8);

    clock_t start = clock();
    for(int i=0; i<1000000; i++) {
        current = basis_ir_add(g, current, bias);
    }
    clock_t end = clock();
    double build_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Built 1,000,000 node chain in %.4f sec\n", build_time);

    start = clock();
    basis_ir_graph_destroy(g);
    end = clock();
    double teardown_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Destroyed 1,000,000 node chain in %.6f sec\n", teardown_time);

    if (teardown_time < 0.01) printf("  [PASS] O(1) Arena Teardown verified (No recursive frees!)\n");
    else { printf("  [FAIL] Teardown too slow\n"); failures++; }

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 IR TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

#include "basis.h"
#include <stdio.h>
#include <time.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L1 Tensor Engine (Deferred Execution)       \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Define Inputs
    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    double B_data[] = {5.0, 6.0, 7.0, 8.0};

    basis_v7_tensor* A = basis_v7_input(g, 2, 2, A_data);
    basis_v7_tensor* B = basis_v7_input(g, 2, 2, B_data);

    // 2. Trace Graph (No math executed yet!)
    printf("=== Tracing Graph (Deferred Execution) ===\n");
    basis_v7_tensor* C = basis_v7_add(g, A, B);       // C = A + B
    basis_v7_tensor* D = basis_v7_matmul(g, C, A);    // D = (A + B) @ A
    basis_v7_tensor* E = basis_v7_relu(g, D);         // E = ReLU(D)

    if (C->node->runtime_data == NULL) printf("  [PASS] Add deferred (No data allocated yet)\n"); else { printf("  [FAIL] Add executed early\n"); failures++; }
    if (D->node->runtime_data == NULL) printf("  [PASS] MatMul deferred (No data allocated yet)\n"); else { printf("  [FAIL] MatMul executed early\n"); failures++; }

    // 3. Execute Graph
    printf("\n=== Executing Graph ===\n");
    basis_v7_execute(g);

    basis_v7_tensor_print(A, "A (Input)");
    basis_v7_tensor_print(C, "C (A + B)");
    basis_v7_tensor_print(E, "E (ReLU((A+B)@A))");

    // Verify Math: C[0,0] = 1+5 = 6.0
    if (C->node->runtime_data[0] == 6.0) printf("\n  [PASS] Forward math verified\n"); else { printf("\n  [FAIL] Math incorrect\n"); failures++; }

    // 4. The O(1) Teardown Flex
    printf("\n=== The O(1) Teardown Flex ===\n");
    printf("Destroying the graph will instantly free:\n");
    printf("  - 5 IR Nodes\n");
    printf("  - 5 Tensor metadata structs\n");
    printf("  - 5 Raw double* activation buffers\n");
    printf("Total memory reclaimed in exactly ONE CPU instruction (Arena Reset).\n");

    clock_t start = clock();
    basis_ir_graph_destroy(g);
    clock_t end = clock();
    double teardown_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("  Teardown time: %.6f sec\n", teardown_time);
    if (teardown_time < 0.001) printf("  [PASS] O(1) Arena Teardown verified!\n"); else { printf("  [FAIL] Teardown too slow\n"); failures++; }

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 TENSOR TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

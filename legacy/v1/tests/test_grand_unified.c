#include "basis/compiler.h"
#include "basis/geometry.h"
#include <stdio.h>

int main() {
    printf("Final Grand Unified Frontier Demo\n");
    printf("--------------------------------\n");

    // 1. Symbolic Model Definition
    Symbol* x_sym = sym_var("x");
    Symbol* w_sym = sym_var("w");
    Symbol* L_sym = sym_mul(x_sym, w_sym); // Simple linear model Y = x*w

    printf("Step 1: Symbolic Math Defined: ");
    sym_print(L_sym);
    printf("\n");

    // 2. Symbolic Reasoning (Chain Rule)
    Symbol* grad_sym = sym_diff(L_sym, "w");
    printf("Step 2: Symbolic Gradient (dL/dw): ");
    sym_print(grad_sym);
    printf("\n");

    // 3. Compilation to Numeric Execution
    Value* x_val = val_new(5.0);
    Value* w_val = val_new(2.0);
    Compiler* comp = compiler_new();
    compiler_map(comp, "x", x_val);
    compiler_map(comp, "w", w_val);

    Value* Y_numeric = compiler_compile(comp, L_sym);
    printf("Step 3: Compiled Numeric Output (Y = 5 * 2): %f\n", Y_numeric->data);

    // 4. Information Geometry
    // We treat X as our 1x1 input tensor and W as 1x1 weight
    Tensor* X_tensor = tensor_new(1, 1);
    tensor_set(X_tensor, 0, 0, 5.0);
    Tensor* W_tensor = tensor_new(1, 1);
    tensor_set(W_tensor, 0, 0, 2.0);

    Metric* FIM = metric_fisher(W_tensor, X_tensor, NULL);
    printf("Step 4: Information Geometry (Fisher Metric F): %f\n", FIM->matrix->data[0]->data);
    printf("       (F represents the 'curvature' of the model at this point)\n");

    printf("\nMission Accomplished: Math, Programming, and Geometry are unified.\n");

    // Cleanup
    sym_free(L_sym);
    sym_free(grad_sym);
    compiler_free(comp);
    metric_free(FIM);
    tensor_free(X_tensor);
    tensor_free(W_tensor);

    return 0;
}

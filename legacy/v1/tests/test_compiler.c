#include "basis/compiler.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing Grand Unified Frontier: Symbolic-to-Numeric Compiler...\n");

    // 1. Symbolic Definition
    Symbol* x_sym = sym_var("x");
    Symbol* w_sym = sym_var("w");
    Symbol* b_sym = sym_var("b");
    Symbol* L_sym = sym_add(sym_mul(x_sym, w_sym), b_sym);

    // 2. Symbolic Differentiation
    Symbol* dLdw_sym = sym_diff(L_sym, "w");
    printf("Symbolic dL/dw: ");
    sym_print(dLdw_sym);
    printf("\n");

    // 3. Numeric Mapping
    Value* x_val = val_new(2.0);
    Value* w_val = val_new(3.0);
    Value* b_val = val_new(10.0);

    Compiler* comp = compiler_new();
    compiler_map(comp, "x", x_val);
    compiler_map(comp, "w", w_val);
    compiler_map(comp, "b", b_val);

    // 4. Compilation
    // We compile the symbolic derivative into a numeric calculation node.
    Value* dLdw_numeric = compiler_compile(comp, dLdw_sym);
    
    printf("Numeric dL/dw (Compiled): %f (Expected 2.0)\n", dLdw_numeric->data);
    assert(fabs(dLdw_numeric->data - 2.0) < 1e-6);

    // 5. Comparison with Autograd
    Value* L_numeric = val_add(val_mul(x_val, w_val), b_val);
    val_backward(L_numeric);
    printf("Numeric dL/dw (Autograd): %f (Expected 2.0)\n", w_val->grad);
    assert(fabs(w_val->grad - 2.0) < 1e-6);

    printf("Grand Unified Test Passed! Symbolic Math == Numeric Code.\n");

    // Cleanup
    sym_free(L_sym);
    sym_free(dLdw_sym);
    compiler_free(comp);
    // In a real system, we'd free the Value graph properly
    
    return 0;
}

#include "basis/compiler.h"
#include "basis/symbolic.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing Symbolic-to-Numeric Compiler (V4)...\n");
    symbol* x_symbol = symbol_variable("x"); symbol* w_symbol = symbol_variable("w"); symbol* b_symbol = symbol_variable("b");
    symbol* L_symbol = symbol_addition(symbol_multiplication(x_symbol, w_symbol), b_symbol);
    symbol* dLdw_symbol = symbol_differentiation(L_symbol, "w");

    printf("Symbolic dL/dw: "); symbol_print(dLdw_symbol); printf("\n");

    value* x_value = value_new(2.0); value* w_value = value_new(3.0); value* b_value = value_new(10.0);

    compiler* compiler_instance = compiler_new();
    compiler_map(compiler_instance, "x", x_value);
    compiler_map(compiler_instance, "w", w_value);
    compiler_map(compiler_instance, "b", b_value);

    value* dLdw_numeric = compiler_compile(compiler_instance, dLdw_symbol);
    printf("Numeric dL/dw (Compiled): %f (Expected 2.0)\n", dLdw_numeric->data);
    assert(fabs(dLdw_numeric->data - 2.0) < 1e-6);

    value* L_numeric = value_addition(value_multiplication(x_value, w_value), b_value);
    value_backward_propagation(L_numeric);
    printf("Numeric dL/dw (Autograd): %f (Expected 2.0)\n", w_value->gradient);
    assert(fabs(w_value->gradient - 2.0) < 1e-6);

    printf("Compiler Test Passed! Symbolic Math == Numeric Code.\n");
    value_free(dLdw_numeric); value_free(L_numeric);
    value_free(x_value); value_free(w_value); value_free(b_value);
    symbol_free(L_symbol); symbol_free(dLdw_symbol); compiler_free(compiler_instance);
    return 0;
}

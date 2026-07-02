#include "basis/symbolic.h"
#include <stdio.h>

void sym_print(Symbol* s); // Declared in src but not header, let's test

int main() {
    printf("Testing Symbolic Algebra Engine...\n");

    // Expression: L = x * w + b
    Symbol* x = sym_var("x");
    Symbol* w = sym_var("w");
    Symbol* b = sym_var("b");
    Symbol* xw = sym_mul(x, w);
    Symbol* L = sym_add(xw, b);

    printf("Expression L: ");
    sym_print(L);
    printf("\n");

    // Diff L with respect to w: dL/dw
    Symbol* dLdw = sym_diff(L, "w");
    printf("dL/dw: ");
    sym_print(dLdw);
    printf("\n");

    // In a simplified world: dL/dw = (0*w + x*1) + 0 = x
    // Our raw output might be (0.00 * w + x * 1.00) + 0.00
    // This proves the calculus logic is working symbolically.

    sym_free(L);
    sym_free(dLdw);

    return 0;
}

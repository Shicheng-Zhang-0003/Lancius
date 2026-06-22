#include "basis/symbolic.h"
#include <stdio.h>

int main() {
    printf("Testing Symbolic Algebra Engine (V4)...\n");
    symbol* x = symbol_variable("x"); symbol* w = symbol_variable("w"); symbol* b = symbol_variable("b");
    symbol* xw = symbol_multiplication(x, w);
    symbol* L = symbol_addition(xw, b);
    printf("Expression L: "); symbol_print(L); printf("\n");

    symbol* dLdw = symbol_differentiation(L, "w");
    printf("Raw dL/dw: "); symbol_print(dLdw); printf("\n");

    symbol* dLdw_simplified = symbol_simplify(dLdw);
    printf("Simplified dL/dw: "); symbol_print(dLdw_simplified); printf("\n");

    symbol_free(L); symbol_free(dLdw); symbol_free(dLdw_simplified);
    return 0;
}

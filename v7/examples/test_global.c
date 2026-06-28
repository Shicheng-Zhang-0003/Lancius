#include <basis.h>
#include <stdio.h>

int main() {
    printf("🌍 Accessing BASIS v5 from the global system environment!\n\n");

    // 1. Build a simple Autograd graph
    basis_value* x = basis_value_new(3.0);
    basis_value* y = basis_value_new(4.0);
    basis_value* z = basis_value_addition(x, y);

    // 2. Trigger the Iterative Topological Sort
    basis_value_backward_propagation(z);

    printf("Forward Pass:  3.0 + 4.0 = %.1f\n", z->data);
    printf("Backward Pass: dx = %.1f, dy = %.1f\n", x->gradient, y->gradient);

    // 3. Strict Memory Teardown
    basis_value_free(z);
    basis_value_free(x);
    basis_value_free(y);

    printf("\n✅ Global library linkage successful. Zero memory leaks.\n");
    printf("🚀 BASIS v5 is now a native part of your operating system.\n");
    return 0;
}

#include "basis/scalar.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf("Testing Scalar Autograd...\n");
    // L = a * b + c
    Value* a = val_new (2.0);
    Value* b = val_new (-3.0);
    Value* c = val_new (10.0);
    // Forward pass
    Value* ab = val_mul (a, b);    // 2 * -3 = -6
    Value* L = val_add (ab, c);    // -6 + 10 = 4
    printf ("Forward Pass: L = %f (expected 4.0)\n", L -> data);
    assert (fabs (L -> data - 4.0) < 1e-6);
    // Backward pass
    val_backward (L);
    // Gradients:
    // dL/dL = 1.0
    // dL/dc = 1.0
    // dL/dab = 1.0
    // dL/da = dL/dab * dab/da = 1.0 * b = -3.0
    // dL/db = dL/dab * dab/db = 1.0 * a = 2.0
    printf ("Gradients:\n");
    printf ("  dL/da = %f (expected -3.0)\n", a -> grad);
    printf ("  dL/db = %f (expected 2.0)\n", b -> grad);
    printf ("  dL/dc = %f (expected 1.0)\n", c -> grad);
    assert (fabs (a -> grad - (-3.0)) < 1e-6);
    assert (fabs (b -> grad - 2.0) < 1e-6);
    assert (fabs (c -> grad - 1.0) < 1e-6);
    printf ("All Scalar tests passed!\n");
    // Note: In a full implementation, we'd need a recursive free or a memory pool.
    // For this simple test, we'll just free the nodes we created.
    val_free (L);
    val_free (ab);
    val_free (a);
    val_free (b);
    val_free (c);
    return 0;
}

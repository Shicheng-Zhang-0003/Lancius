#include "basis/scalar.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing Scalar Autograd (V4)...\n");
    value* first_value = value_new(2.0);
    value* second_value = value_new(-3.0);
    value* third_value = value_new(10.0);

    value* product_value = value_multiplication(first_value, second_value);
    value* loss_value = value_addition(product_value, third_value);

    printf("Forward Pass: L = %f (expected 4.0)\n", loss_value->data);
    assert(fabs(loss_value->data - 4.0) < 1e-6);

    value_backward_propagation(loss_value);
    printf("Gradients:\n");
    printf("  dL/da = %f (expected -3.0)\n", first_value->gradient);
    printf("  dL/db = %f (expected 2.0)\n", second_value->gradient);
    printf("  dL/dc = %f (expected 1.0)\n", third_value->gradient);

    assert(fabs(first_value->gradient - (-3.0)) < 1e-6);
    assert(fabs(second_value->gradient - 2.0) < 1e-6);
    assert(fabs(third_value->gradient - 1.0) < 1e-6);

    printf("All Scalar tests passed!\n");
    value_free(loss_value); value_free(product_value);
    value_free(first_value); value_free(second_value); value_free(third_value);
    return 0;
}

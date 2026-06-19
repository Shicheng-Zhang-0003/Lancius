#include "basis/tensor.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Tensor MatMul and Autograd...\n");
    tensor* input_tensor = tensor_new (1, 2);
    tensor_set (input_tensor, 0, 0, 1.0);
    tensor_set (input_tensor, 0, 1, 2.0);
    tensor* weight_tensor = tensor_new (2, 2);
    tensor_set (weight_tensor, 0, 0, 3.0);
    tensor_set (weight_tensor, 0, 1, 4.0);
    tensor_set (weight_tensor, 1, 0, 5.0);
    tensor_set (weight_tensor, 1, 1, 6.0);
    tensor* output_tensor = tensor_matrix_multiplication (input_tensor, weight_tensor);
    tensor_print (output_tensor, "Y = X * W");
    assert (fabs (output_tensor -> data [0] -> data - 13.0) < 1e-6);
    assert (fabs (output_tensor -> data [1] -> data - 16.0) < 1e-6);
    value* loss = value_addition (output_tensor -> data [0], output_tensor -> data [1]);
    printf ("Loss: %f (expected 29.0)\n", loss -> data);
    assert (fabs (loss -> data - 29.0) < 1e-6);
    value_backward_propagation (loss);
    printf ("Gradients of Weights (W):\n");
    for (size_t i = 0; i < weight_tensor -> row_count; i ++) {
        for (size_t j = 0; j < weight_tensor -> column_count; j ++) {
            double gradient = weight_tensor -> data [i * weight_tensor -> column_count + j] -> gradient;
            printf ("  dW[%zu,%zu] = %f\n", i, j, gradient);
            double expected;
            if (i == 0) {expected = 1.0;} else {expected = 2.0;}
            assert (fabs (gradient - expected) < 1e-6);
        }
    }
    printf ("All Tensor tests passed!\n");
    value_free (loss);
    tensor_free (output_tensor);
    tensor_free (input_tensor);
    tensor_free (weight_tensor);
    return 0;
}

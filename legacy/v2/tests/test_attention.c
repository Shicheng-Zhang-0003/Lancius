#include "basis/tensor.h"
#include "basis/sequence.h"
#include "basis/learning.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Transformer Attention Math...\n");
    tensor* query_tensor = tensor_new (1, 2);
    tensor_set (query_tensor, 0, 0, 1.0);
    tensor_set (query_tensor, 0, 1, 0.0);
    tensor* key_tensor = tensor_new (1, 2);
    tensor_set (key_tensor, 0, 0, 1.0);
    tensor_set (key_tensor, 0, 1, 0.0);
    tensor* value_tensor = tensor_new (1, 2);
    tensor_set (value_tensor, 0, 0, 10.0);
    tensor_set (value_tensor, 0, 1, 20.0);
    tensor* output_tensor = tensor_attention (query_tensor, key_tensor, value_tensor);
    tensor_print (output_tensor, "Attention Output");
    assert (fabs (output_tensor -> data [0] -> data - 10.0) < 1e-6);
    assert (fabs (output_tensor -> data [1] -> data - 20.0) < 1e-6);
    value* output_sum = tensor_sum (output_tensor);
    value_backward_propagation (output_sum);
    value_free (output_sum);
    printf ("Gradient dOutput/dV[0,0] = %f (expected 1.0)\n", value_tensor -> data [0] -> gradient);
    assert (fabs (value_tensor -> data [0] -> gradient - 1.0) < 1e-6);
    printf ("All Attention tests passed!\n");
    tensor_free (output_tensor);
    tensor_free (query_tensor);
    tensor_free (key_tensor);
    tensor_free (value_tensor);
    return 0;
}

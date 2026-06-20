#include "basis/tensor.h"
#include "basis/learning.h"
#include "basis/optim.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

int main () {
    printf ("Frontier Convergence Test: Learning x * 2 = 10\n");
    tensor* weight_tensor = tensor_new (1, 1);
    tensor_set (weight_tensor, 0, 0, 0.5);

    tensor* input_tensor = tensor_new (1, 1);
    tensor_set (input_tensor, 0, 0, 5.0);

    adam* optimizer = adam_new (1, 1, 0.1);

    for (int epoch = 0; epoch < 50; epoch ++) {
        tensor* output_tensor = tensor_matrix_multiplication (input_tensor, weight_tensor);
        value* target_value = value_new (-10.0);
        value* difference = value_addition (output_tensor -> data [0], target_value);
        value* loss = value_power (difference, 2.0);

        if (epoch % 10 == 0) {
            printf ("Epoch %d: Loss = %f, W = %f\n", epoch, loss -> data, weight_tensor -> data [0] -> data);
        }

        value_backward_propagation (loss);
        value_free (loss);
        value_free (difference);
        value_free (target_value);
        adam_optimization_step (optimizer, weight_tensor);
        tensor_free (output_tensor);
    }

    printf ("Final W: %f (Expected ~2.0)\n", weight_tensor -> data [0] -> data);
    assert (fabs (weight_tensor -> data [0] -> data - 2.0) < 0.1);

    adam_free (optimizer);
    tensor_free (weight_tensor);
    tensor_free (input_tensor);
    return 0;
}

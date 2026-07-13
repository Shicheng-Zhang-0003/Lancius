#include "basis/tensor.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing Tensor MatMul and Autograd...\n");

    // X: 1x2 Input
    Tensor* X = tensor_new(1, 2);
    tensor_set(X, 0, 0, 1.0);
    tensor_set(X, 0, 1, 2.0);

    // W: 2x2 Weights
    Tensor* W = tensor_new(2, 2);
    tensor_set(W, 0, 0, 3.0);
    tensor_set(W, 0, 1, 4.0);
    tensor_set(W, 1, 0, 5.0);
    tensor_set(W, 1, 1, 6.0);

    // Y = X * W (Result: 1x2)
    Tensor* Y = tensor_matmul(X, W);
    
    // Y[0,0] = 1*3 + 2*5 = 13
    // Y[0,1] = 1*4 + 2*6 = 16
    tensor_print(Y, "Y = X * W");
    assert(fabs(Y->data[0]->data - 13.0) < 1e-6);
    assert(fabs(Y->data[1]->data - 16.0) < 1e-6);

    // Loss = sum(Y) = 13 + 16 = 29
    Value* loss = val_add(Y->data[0], Y->data[1]);
    printf("Loss: %f (expected 29.0)\n", loss->data);
    assert(fabs(loss->data - 29.0) < 1e-6);

    // Backward pass
    val_backward(loss);

    // We expect dLoss/dW_ij = X_i
    // dLoss/dW_00 = X_0 = 1.0
    // dLoss/dW_10 = X_1 = 2.0
    // dLoss/dW_01 = X_0 = 1.0
    // dLoss/dW_11 = X_1 = 2.0

    printf("Gradients of Weights (W):\n");
    for (size_t i = 0; i < W->rows; i++) {
        for (size_t j = 0; j < W->cols; j++) {
            double grad = W->data[i * W->cols + j]->grad;
            printf("  dW[%zu,%zu] = %f\n", i, j, grad);
            double expected = (i == 0) ? 1.0 : 2.0;
            assert(fabs(grad - expected) < 1e-6);
        }
    }

    printf("All Tensor tests passed!\n");

    // Clean up (Simplified)
    // Note: val_backward builds a graph that we'd ideally clean up with a 
    // recursive free. For this test, we just free the tensor wrappers.
    tensor_free(Y);
    tensor_free(X);
    tensor_free(W);
    // val_free(loss); // loss is inside the graph

    return 0;
}

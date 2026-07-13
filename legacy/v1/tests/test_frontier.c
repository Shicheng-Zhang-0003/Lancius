#include "basis/tensor.h"
#include "basis/optim.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("Frontier Convergence Test: Learning x * 2 = 10\n");

    // W: 1x1 Weight matrix (init to random-ish small value)
    Tensor* W = tensor_new(1, 1);
    tensor_set(W, 0, 0, 0.5);

    // X: Input (5.0), Target (10.0)
    Tensor* X = tensor_new(1, 1);
    tensor_set(X, 0, 0, 5.0);

    Adam* opt = adam_new(1, 1, 0.1);

    for (int epoch = 0; epoch < 50; epoch++) {
        // Forward: Y = X * W
        Tensor* Y = tensor_matmul(X, W);
        
        // Stability: Apply LayerNorm (over 1 element it's simple, but tests the logic)
        Tensor* Y_norm = tensor_layernorm(Y, 1e-5);

        // Loss: MSE = (Y - 10.0)^2
        Value* diff = val_add(Y->data[0], val_new(-10.0));
        Value* loss = val_pow(diff, 2.0);

        if (epoch % 10 == 0) {
            printf("Epoch %d: Loss = %f, W = %f\n", epoch, loss->data, W->data[0]->data);
        }

        // Backward
        val_backward(loss);

        // Optimization Step
        adam_step(opt, W);

        // Cleanup graph (In real system, we'd have an arena allocator)
        // tensor_free(Y); tensor_free(Y_norm);
    }

    printf("Final W: %f (Expected ~2.0)\n", W->data[0]->data);

    adam_free(opt);
    tensor_free(W);
    tensor_free(X);

    return 0;
}

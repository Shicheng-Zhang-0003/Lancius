#include "basis/tensor.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing Transformer Attention Math...\n");

    // 1. Setup Query (Q), Key (K), and Value (V)
    // For simplicity, let's use 1 token of dimension 2
    Tensor* Q = tensor_new(1, 2);
    tensor_set(Q, 0, 0, 1.0);
    tensor_set(Q, 0, 1, 0.0);

    Tensor* K = tensor_new(1, 2);
    tensor_set(K, 0, 0, 1.0);
    tensor_set(K, 0, 1, 0.0);

    Tensor* V = tensor_new(1, 2);
    tensor_set(V, 0, 0, 10.0);
    tensor_set(V, 0, 1, 20.0);

    // 2. Compute Attention
    // formula: softmax(Q * K^T / sqrt(d_k)) * V
    // d_k = 2, sqrt(d_k) approx 1.414
    
    Tensor* KT = tensor_transpose(K);
    Tensor* QKT = tensor_matmul(Q, KT);
    Tensor* scaled = tensor_scalar_mul(QKT, 1.0 / sqrt(2.0));
    Tensor* weights = tensor_softmax(scaled);
    Tensor* output = tensor_matmul(weights, V);

    tensor_print(output, "Attention Output");

    // Since we only have 1 token, the softmax of [x] is always [1.0]
    // So output should be 1.0 * V = [10.0, 20.0]
    assert(fabs(output->data[0]->data - 10.0) < 1e-6);
    assert(fabs(output->data[1]->data - 20.0) < 1e-6);

    // 3. Verify Gradient Flow
    // dOutput/dV should be 1.0
    val_backward(output->data[0]);
    printf("Gradient dOutput[0]/dV[0,0] = %f (expected 1.0)\n", V->data[0]->grad);
    assert(fabs(V->data[0]->grad - 1.0) < 1e-6);

    printf("All Attention tests passed!\n");

    // Cleanup
    tensor_free(Q);
    tensor_free(K);
    tensor_free(KT);
    tensor_free(QKT);
    tensor_free(scaled);
    tensor_free(weights);
    tensor_free(output);
    tensor_free(V);

    return 0;
}

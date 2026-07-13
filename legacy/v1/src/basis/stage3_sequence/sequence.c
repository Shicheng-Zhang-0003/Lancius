#include "basis/stage3_sequence/sequence.h"
#include <math.h>
Tensor* tensor_transpose (Tensor* t) {
    Tensor* out = tensor_new (t -> cols, t -> rows);
    for (size_t i = 0; i < t -> rows; i++) {
        for (size_t j = 0; j < t -> cols; j++) {
            val_free (out->data [j * out -> cols + i]);
            out -> data [j * out -> cols + i] = t -> data [i * t -> cols + j];
        }
    } return out;
} Tensor* tensor_softmax (Tensor* t) {
    Tensor* out = tensor_new (t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows; i++) {
        Value* sum = val_new (0.0);
        for (size_t j = 0; j < t -> cols; j++) {sum = val_add (sum, val_exp (t -> data [i * t -> cols + j]));}
        Value* inv_sum = val_pow (sum, -1.0);
        for (size_t j = 0; j < t -> cols; j++) {
            val_free (out -> data [i * t -> cols + j]);
            out -> data [i * t -> cols + j] = val_mul (val_exp (t -> data [i * t -> cols + j]), inv_sum);
        }
    } return out;
} Tensor* tensor_attention (Tensor* Q, Tensor* K, Tensor* V) {
    Tensor* KT = tensor_transpose (K);
    Tensor* QKT = tensor_matmul (Q, KT);
    Tensor* scaled = tensor_scalar_mul (QKT, 1.0 / sqrt ((double) K -> cols));
    Tensor* weights = tensor_softmax (scaled);
    Tensor* output = tensor_matmul (weights, V);
    // In a managed system we'd cleanup intermediate tensors
    return output;
}

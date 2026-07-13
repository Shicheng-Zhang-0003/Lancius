#ifndef BASIS_STAGE3_SEQUENCE_H
#define BASIS_STAGE3_SEQUENCE_H
#include "basis/stage2_linear/tensor.h"
Tensor* tensor_transpose (Tensor* t);
Tensor* tensor_softmax (Tensor* t);
// Attention logic lives here mathematically
Tensor* tensor_attention (Tensor* Q, Tensor* K, Tensor* V);
#endif

#ifndef BASIS_STAGE4_LEARNING_H
#define BASIS_STAGE4_LEARNING_H
#include "basis/stage2_linear/tensor.h"
#include "basis/stage4_learning/optim.h"
Tensor* tensor_layernorm (Tensor* t, double eps);
Tensor* tensor_rope (Tensor* t, int pos, double base);
Value* tensor_sum (Tensor* t);
Tensor* tensor_log (Tensor* t);
Tensor* tensor_mul (Tensor* a, Tensor* b); // Element-wise
#endif

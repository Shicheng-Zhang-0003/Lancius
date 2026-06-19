#ifndef basis_stage4_learning_h
#define basis_stage4_learning_h
#include "basis/stage2_linear/tensor.h"
#include "basis/stage4_learning/optim.h"
tensor* tensor_layer_normalization (tensor* target_tensor, double epsilon);
tensor* tensor_rotary_positional_embedding (tensor* target_tensor, int position, double base);
value* tensor_sum (tensor* target_tensor);
tensor* tensor_logarithm (tensor* target_tensor);
tensor* tensor_multiplication (tensor* first_tensor, tensor* second_tensor);
#endif

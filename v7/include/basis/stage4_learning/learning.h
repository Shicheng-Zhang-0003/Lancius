
#ifndef basis_stage4_learning_h
#define basis_stage4_learning_h
#include "basis/stage2_linear/tensor.h"
basis_tensor* basis_tensor_layer_normalization(basis_tensor* target_tensor, double epsilon);
basis_tensor* basis_tensor_rotary_positional_embedding(basis_tensor* target_tensor, int position, double base);
basis_value* basis_tensor_sum(basis_tensor* target_tensor);
basis_tensor* basis_tensor_logarithm(basis_tensor* target_tensor);
basis_tensor* basis_tensor_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor);
#endif

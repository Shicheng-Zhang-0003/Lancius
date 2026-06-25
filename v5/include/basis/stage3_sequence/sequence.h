
#ifndef basis_stage3_sequence_h
#define basis_stage3_sequence_h
#include "basis/stage2_linear/tensor.h"
basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor);
basis_tensor* basis_tensor_softmax(basis_tensor* target_tensor);
basis_tensor* basis_tensor_log_softmax(basis_tensor* target_tensor);
basis_tensor* basis_tensor_attention(basis_tensor* query_tensor, basis_tensor* key_tensor, basis_tensor* basis_value_tensor);
#endif

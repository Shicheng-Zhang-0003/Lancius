#ifndef basis_stage3_sequence_h
#define basis_stage3_sequence_h

#include "basis/stage2_linear/tensor.h"

tensor* tensor_transpose(tensor* target_tensor);
tensor* tensor_softmax(tensor* target_tensor);
tensor* tensor_attention(tensor* query_tensor, tensor* key_tensor, tensor* value_tensor);

#endif

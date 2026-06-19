#ifndef basis_stage2_tensor_h
#define basis_stage2_tensor_h
#include "basis/stage1_atomic/scalar.h"
typedef struct tensor {
    value** data;
    size_t row_count;
    size_t column_count;
} tensor;
tensor* tensor_new (size_t row_count, size_t column_count);
void tensor_free (tensor* target_tensor);
void tensor_fill (tensor* target_tensor, double fill_value);
void tensor_set (tensor* target_tensor, size_t row_index, size_t column_index, double data_value);
tensor* tensor_addition (tensor* first_tensor, tensor* second_tensor);
tensor* tensor_matrix_multiplication (tensor* first_tensor, tensor* second_tensor);
tensor* tensor_scalar_multiplication (tensor* target_tensor, double scalar_value);
tensor* tensor_rectified_linear_unit (tensor* target_tensor);
void tensor_print (tensor* target_tensor, const char* name);
#endif

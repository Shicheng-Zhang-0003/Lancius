#ifndef basis_stage2_tensor_h
#define basis_stage2_tensor_h
#include "basis/stage1_atomic/scalar.h"
#include <stdbool.h>

typedef struct basis_tensor {
    basis_value** data;
    size_t flat_size; // Total allocated elements in root data array

    size_t row_count;
    size_t column_count;

    size_t row_stride;
    size_t col_stride;
    size_t offset;

    bool is_view;
    struct basis_tensor* parent;
    size_t ref_count;
} basis_tensor;

#define BASIS_TENSOR_AT(t, i, j) ((t)->data[(t)->offset + (i) * (t)->row_stride + (j) * (t)->col_stride])

basis_tensor* basis_tensor_new(size_t row_count, size_t column_count);
basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor);
basis_tensor* basis_tensor_slice_rows(basis_tensor* target_tensor, size_t start_row, size_t num_rows);
void basis_tensor_free(basis_tensor* target_tensor);
void basis_tensor_fill(basis_tensor* target_tensor, double fill_value);
void basis_tensor_set(basis_tensor* target_tensor, size_t row_index, size_t column_index, double data_value);
basis_tensor* basis_tensor_addition(basis_tensor* first_tensor, basis_tensor* second_tensor);
basis_tensor* basis_tensor_matrix_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor);
basis_tensor* basis_tensor_scalar_multiplication(basis_tensor* target_tensor, double scalar_value);
basis_tensor* basis_tensor_rectified_linear_unit(basis_tensor* target_tensor);
basis_tensor* basis_tensor_tanh(basis_tensor* target_tensor);
basis_tensor* basis_tensor_broadcast_view(basis_tensor* target_tensor, size_t target_rows, size_t target_cols);
basis_tensor* basis_tensor_add(basis_tensor* first_tensor, basis_tensor* second_tensor);
void basis_tensor_matmul_backward_cblas(basis_tensor* grad_output, basis_tensor* A, basis_tensor* B);
void basis_tensor_save_binary(basis_tensor* target_tensor, const char* filename);
basis_tensor* basis_tensor_load_binary(const char* filename);
void basis_tensor_print(basis_tensor* target_tensor, const char* name);
#endif

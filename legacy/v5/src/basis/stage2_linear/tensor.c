#include "basis/stage2_linear/tensor.h"
#include "basis/core/error.h"
#ifdef BASIS_USE_CBLAS
#include <cblas.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static char* tensor_string_duplicate(const char* source) {
    if (!source) return NULL;
    size_t len = strlen(source) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, source, len);
    return copy;
}

basis_tensor* basis_tensor_new(size_t row_count, size_t column_count) {
    if (row_count == 0 || column_count == 0) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Tensor dimensions must be > 0");
        return NULL;
    }
    basis_tensor* new_tensor = (basis_tensor*)malloc(sizeof(basis_tensor));
    new_tensor->row_count = row_count;
    new_tensor->column_count = column_count;
    new_tensor->flat_size = row_count * column_count;
    new_tensor->data = (basis_value**)malloc(sizeof(basis_value*) * new_tensor->flat_size);
    for (size_t i = 0; i < new_tensor->flat_size; i++) {
        new_tensor->data[i] = basis_value_new(0.0);
    }
    new_tensor->row_stride = column_count;
    new_tensor->col_stride = 1;
    new_tensor->offset = 0;
    new_tensor->is_view = false;
    new_tensor->parent = NULL;
    new_tensor->ref_count = 1;
    return new_tensor;
}

basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = target_tensor->column_count;
    view->column_count = target_tensor->row_count;
    view->row_stride = target_tensor->col_stride;
    view->col_stride = target_tensor->row_stride;
    view->offset = target_tensor->offset;
    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++; // Keep parent alive
    return view;
}

basis_tensor* basis_tensor_slice_rows(basis_tensor* target_tensor, size_t start_row, size_t num_rows) {
    BASIS_CHECK_NULL(target_tensor);
    if (start_row + num_rows > target_tensor->row_count) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Slice out of bounds");
        return NULL;
    }
    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = num_rows;
    view->column_count = target_tensor->column_count;
    view->row_stride = target_tensor->row_stride;
    view->col_stride = target_tensor->col_stride;
    view->offset = target_tensor->offset + (start_row * target_tensor->row_stride);
    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++;
    return view;
}

void basis_tensor_free(basis_tensor* target_tensor) {
    if (!target_tensor) return;
    target_tensor->ref_count--;
    if (target_tensor->ref_count > 0) return;

    if (target_tensor->is_view) {
        basis_tensor_free(target_tensor->parent);
    } else {
        if (target_tensor->data) {
            for (size_t i = 0; i < target_tensor->flat_size; i++) {
                basis_value_free(target_tensor->data[i]);
            }
            free(target_tensor->data);
        }
    }
    free(target_tensor);
}

void basis_tensor_fill(basis_tensor* target_tensor, double fill_value) {
    if (!target_tensor) return;
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            BASIS_TENSOR_AT(target_tensor, i, j)->data = fill_value;
}

void basis_tensor_set(basis_tensor* target_tensor, size_t row_index, size_t column_index, double data_value) {
    if (!target_tensor || row_index >= target_tensor->row_count || column_index >= target_tensor->column_count) return;
    BASIS_TENSOR_AT(target_tensor, row_index, column_index)->data = data_value;
}

basis_tensor* basis_tensor_addition(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);
    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count; i++)
        for (size_t j = 0; j < first_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_addition(BASIS_TENSOR_AT(first_tensor, i, j), BASIS_TENSOR_AT(second_tensor, i, j));
    return output_tensor;
}

static void backward_matmul_element(basis_value* output_value) {
    size_t K = output_value->previous_node_count / 2;
    double grad = output_value->gradient;
    for (size_t k = 0; k < K; k++) {
        basis_value* a_ik = output_value->previous_nodes[k];
        basis_value* b_kj = output_value->previous_nodes[K + k];
        a_ik->gradient += grad * b_kj->data;
        b_kj->gradient += grad * a_ik->data;
    }
}

basis_tensor* basis_tensor_matrix_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->column_count == second_tensor->row_count);
    size_t M = first_tensor->row_count;
    size_t K = first_tensor->column_count;
    size_t N = second_tensor->column_count;
    basis_tensor* output_tensor = basis_tensor_new(M, N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            basis_value* c_ij = basis_value_new(0.0);
            free(c_ij->operation);
            c_ij->operation = tensor_string_duplicate("matmul_fused");
            c_ij->previous_node_count = 2 * K;
            c_ij->previous_nodes = (basis_value**)malloc(sizeof(basis_value*) * 2 * K);
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                basis_value* a_ik = BASIS_TENSOR_AT(first_tensor, i, k);
                basis_value* b_kj = BASIS_TENSOR_AT(second_tensor, k, j);
                c_ij->previous_nodes[k] = a_ik;
                c_ij->previous_nodes[K + k] = b_kj;
                basis_value_retain(a_ik);
                basis_value_retain(b_kj);
                sum += a_ik->data * b_kj->data;
            }
            c_ij->data = sum;
            c_ij->backward_function = backward_matmul_element;
            basis_value_free(output_tensor->data[i * N + j]);
            output_tensor->data[i * N + j] = c_ij;
        }
    }
    return output_tensor;
}

basis_tensor* basis_tensor_scalar_multiplication(basis_tensor* target_tensor, double scalar_value) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    basis_value* scalar = basis_value_new(scalar_value);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_multiplication(BASIS_TENSOR_AT(target_tensor, i, j), scalar);
    basis_value_free(scalar);
    return output_tensor;
}

basis_tensor* basis_tensor_rectified_linear_unit(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_rectified_linear_unit(BASIS_TENSOR_AT(target_tensor, i, j));
    return output_tensor;
}


basis_tensor* basis_tensor_broadcast_view(basis_tensor* target_tensor, size_t target_rows, size_t target_cols) {
    BASIS_CHECK_NULL(target_tensor);
    // Validate broadcasting rules
    bool row_compat = (target_tensor->row_count == target_rows) || (target_tensor->row_count == 1);
    bool col_compat = (target_tensor->column_count == target_cols) || (target_tensor->column_count == 1);
    if (!row_compat || !col_compat) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Broadcast shape mismatch");
        return NULL;
    }

    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = target_rows;
    view->column_count = target_cols;

    // The Math: stride=0 on singleton dimensions
    view->row_stride = (target_tensor->row_count == 1) ? 0 : target_tensor->row_stride;
    view->col_stride = (target_tensor->column_count == 1) ? 0 : target_tensor->col_stride;
    view->offset = target_tensor->offset;

    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++;
    return view;
}

basis_tensor* basis_tensor_add(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);

    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count; i++) {
        for (size_t j = 0; j < first_tensor->column_count; j++) {
            basis_value_free(output_tensor->data[i * output_tensor->column_count + j]);
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_addition(
                BASIS_TENSOR_AT(first_tensor, i, j),
                BASIS_TENSOR_AT(second_tensor, i, j)
            );
        }
    }
    return output_tensor;
}

void basis_tensor_matmul_backward_cblas(basis_tensor* grad_output, basis_tensor* A, basis_tensor* B) {
#ifdef BASIS_USE_CBLAS
    // Future Phase K: This will be natively hooked into the Tensor-level Tape.
    // For now, it serves as the high-performance gradient scatter utility.
    size_t M = A->row_count; size_t K = A->column_count; size_t N = B->column_count;
    double* dC = malloc(M * N * sizeof(double));
    double* A_flat = malloc(M * K * sizeof(double));
    double* B_flat = malloc(K * N * sizeof(double));
    double* dA = calloc(M * K, sizeof(double));
    double* dB = calloc(K * N, sizeof(double));

    for(size_t i=0; i<M; i++) for(size_t j=0; j<N; j++) dC[i*N+j] = BASIS_TENSOR_AT(grad_output, i, j)->gradient;
    for(size_t i=0; i<M; i++) for(size_t k=0; k<K; k++) A_flat[i*K+k] = BASIS_TENSOR_AT(A, i, k)->data;
    for(size_t k=0; k<K; k++) for(size_t j=0; j<N; j++) B_flat[k*N+j] = BASIS_TENSOR_AT(B, k, j)->data;

    // dA = dC * B^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0, dC, N, B_flat, N, 0.0, dA, K);
    // dB = A^T * dC
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0, A_flat, K, dC, N, 0.0, dB, N);

    for(size_t i=0; i<M; i++) for(size_t k=0; k<K; k++) BASIS_TENSOR_AT(A, i, k)->gradient += dA[i*K+k];
    for(size_t k=0; k<K; k++) for(size_t j=0; j<N; j++) BASIS_TENSOR_AT(B, k, j)->gradient += dB[k*N+j];

    free(dC); free(A_flat); free(B_flat); free(dA); free(dB);
#else
    (void)grad_output; (void)A; (void)B;
    fprintf(stderr, "[BASIS] CBLAS backward pass called, but BASIS_USE_CBLAS is not defined.\n");
#endif
}


basis_tensor* basis_tensor_tanh(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(output_tensor->data[i * output_tensor->column_count + j]);
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_tanh(BASIS_TENSOR_AT(target_tensor, i, j));
        }
    return output_tensor;
}


void basis_tensor_save_binary(basis_tensor* target_tensor, const char* filename) {
    if (!target_tensor || !filename) return;
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&target_tensor->row_count, sizeof(size_t), 1, f);
    fwrite(&target_tensor->column_count, sizeof(size_t), 1, f);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            double val = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            fwrite(&val, sizeof(double), 1, f);
        }
    }
    fclose(f);
}

basis_tensor* basis_tensor_load_binary(const char* filename) {
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    size_t r, c;
    size_t dummy_r = fread(&r, sizeof(size_t), 1, f); (void)dummy_r;
    size_t dummy_c = fread(&c, sizeof(size_t), 1, f); (void)dummy_c;
    basis_tensor* t = basis_tensor_new(r, c);
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            double val;
            size_t dummy_val = fread(&val, sizeof(double), 1, f); (void)dummy_val;
            BASIS_TENSOR_AT(t, i, j)->data = val;
        }
    }
    fclose(f);
    return t;
}

void basis_tensor_print(basis_tensor* target_tensor, const char* name) {
    if (!target_tensor) return;
    printf("Tensor %s (%zu x %zu):\n", name, target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        printf("  [");
        for (size_t j = 0; j < target_tensor->column_count; j++)
            printf(" %8.4f ", BASIS_TENSOR_AT(target_tensor, i, j)->data);
        printf("]\n");
    }
}

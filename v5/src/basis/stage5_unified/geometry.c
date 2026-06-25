#include "basis/stage5_unified/geometry.h"
#include "basis/stage3_sequence/sequence.h"
#include "basis/core/error.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

basis_metric* basis_metric_fisher_information(basis_tensor* weight_tensor, basis_tensor* input_tensor, basis_tensor* output_tensor) {
    (void)weight_tensor; (void)output_tensor;
    BASIS_CHECK_NULL(input_tensor);
    basis_tensor* transposed_input = basis_tensor_transpose(input_tensor);
    basis_tensor* fisher_matrix = basis_tensor_matrix_multiplication(transposed_input, input_tensor);
    basis_metric* new_metric = (basis_metric*)malloc(sizeof(basis_metric));
    new_metric->matrix = fisher_matrix;
    basis_tensor_free(transposed_input);
    return new_metric;
}

void basis_metric_free(basis_metric* target_metric) {
    if (!target_metric) return;
    basis_tensor_free(target_metric->matrix);
    free(target_metric);
}

// Gaussian elimination with partial pivoting to solve Ax = b
static int basis_matrix_solve(double* A, double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double max_val = fabs(A[i * n + i]);
        size_t max_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(A[k * n + i]) > max_val) {
                max_val = fabs(A[k * n + i]);
                max_row = k;
            }
        }
        if (max_val < 1e-12) return -1; // Singular

        if (max_row != i) {
            for (size_t j = 0; j < n; j++) {
                double tmp = A[i * n + j];
                A[i * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
            double tmp = b[i]; b[i] = b[max_row]; b[max_row] = tmp;
        }

        for (size_t k = i + 1; k < n; k++) {
            double c = -A[k * n + i] / A[i * n + i];
            for (size_t j = i; j < n; j++) {
                if (i == j) A[k * n + j] = 0;
                else A[k * n + j] += c * A[i * n + j];
            }
            b[k] += c * b[i];
        }
    }

    for (int i = (int)n - 1; i >= 0; i--) {
        b[i] /= A[i * n + i];
        for (int k = i - 1; k >= 0; k--) {
            b[k] -= A[k * n + i] * b[i];
        }
    }
    return 0;
}

void basis_metric_ngd_step(basis_metric* fisher_metric, basis_tensor* weight_tensor, double learning_rate) {
    if (!fisher_metric || !weight_tensor) return;

    size_t n = weight_tensor->row_count * weight_tensor->column_count;
    size_t f_rows = fisher_metric->matrix->row_count;
    size_t f_cols = fisher_metric->matrix->column_count;

    // The Fisher matrix must be an NxN square matrix, where N is the number of weights
    if (f_rows != n || f_cols != n) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Fisher matrix must be NxN where N is the number of weights");
        return;
    }

    double* F = (double*)malloc(n * n * sizeof(double));
    double* grad = (double*)malloc(n * sizeof(double));

    for(size_t i = 0; i < n * n; i++) F[i] = fisher_metric->matrix->data[i]->data;
    for(size_t i = 0; i < n; i++) grad[i] = weight_tensor->data[i]->gradient;

    // Tikhonov regularization (damping) for numerical stability
    double damping = 1e-12; // Lowered to prevent bias on ill-conditioned matrices
    for(size_t i = 0; i < n; i++) F[i * n + i] += damping;

    int status = basis_matrix_solve(F, grad, n);

    if (status == 0) {
        for(size_t i = 0; i < n; i++) {
            weight_tensor->data[i]->data -= learning_rate * grad[i];
            weight_tensor->data[i]->gradient = 0.0;
        }
    } else {
        fprintf(stderr, "[BASIS NGD] Fisher matrix is singular, falling back to standard GD.\n");
        for(size_t i = 0; i < n; i++) {
            weight_tensor->data[i]->data -= learning_rate * weight_tensor->data[i]->gradient;
            weight_tensor->data[i]->gradient = 0.0;
        }
    }

    free(F);
    free(grad);
}

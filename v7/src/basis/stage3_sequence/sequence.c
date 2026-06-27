
#include "basis/stage3_sequence/sequence.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>



basis_tensor* basis_tensor_softmax(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        double maximum_data = target_tensor->data[i * target_tensor->column_count]->data;
        for (size_t j = 1; j < target_tensor->column_count; j++) {
            if (BASIS_TENSOR_AT(target_tensor, i, j)->data > maximum_data) {
                maximum_data = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            }
        }
        basis_value* negative_maximum = basis_value_new(-maximum_data);
        basis_value* sum = basis_value_new(0.0);
        basis_value** exponentials = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* shifted = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_maximum);
            exponentials[j] = basis_value_exponential(shifted);
            basis_value* new_sum = basis_value_addition(sum, exponentials[j]);
            basis_value_free(sum); basis_value_free(shifted);
            sum = new_sum;
        }
        basis_value_free(negative_maximum);
        basis_value* inverse_sum = basis_value_power(sum, -1.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_multiplication(exponentials[j], inverse_sum);
            basis_value_free(exponentials[j]);
        }
        basis_value_free(sum); basis_value_free(inverse_sum); free(exponentials);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_log_softmax(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        double maximum_data = target_tensor->data[i * target_tensor->column_count]->data;
        for (size_t j = 1; j < target_tensor->column_count; j++) {
            if (BASIS_TENSOR_AT(target_tensor, i, j)->data > maximum_data) {
                maximum_data = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            }
        }
        basis_value* negative_maximum = basis_value_new(-maximum_data);
        basis_value* sum = basis_value_new(0.0);

        basis_value** shifted_vals = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);
        basis_value** exponentials = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);

        for (size_t j = 0; j < target_tensor->column_count; j++) {
            shifted_vals[j] = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_maximum);
            exponentials[j] = basis_value_exponential(shifted_vals[j]);
            basis_value* new_sum = basis_value_addition(sum, exponentials[j]);
            basis_value_free(sum);
            sum = new_sum;
        }
        basis_value_free(negative_maximum);

        basis_value* log_sum = basis_value_logarithm(sum);
        basis_value_free(sum); // sum tree destroyed, drops ref to exponentials. Array holds the other ref.

        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            // LogSoftmax(x) = (x - max) - log(sum(exp(x - max)))
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_subtraction(shifted_vals[j], log_sum);
            basis_value_free(shifted_vals[j]);
            basis_value_free(exponentials[j]);
        }
        basis_value_free(log_sum);
        free(shifted_vals);
        free(exponentials);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_attention(basis_tensor* query_tensor, basis_tensor* key_tensor, basis_tensor* basis_value_tensor) {
    BASIS_CHECK_NULL(query_tensor); BASIS_CHECK_NULL(key_tensor); BASIS_CHECK_NULL(basis_value_tensor);
    BASIS_CHECK_SHAPE(query_tensor->column_count == key_tensor->column_count && key_tensor->row_count == basis_value_tensor->row_count);

    basis_tensor* transposed_key_tensor = basis_tensor_transpose(key_tensor);
    basis_tensor* query_key_product = basis_tensor_matrix_multiplication(query_tensor, transposed_key_tensor);
    basis_tensor* scaled_tensor = basis_tensor_scalar_multiplication(query_key_product, 1.0 / sqrt((double)key_tensor->column_count));
    basis_tensor* attention_weights = basis_tensor_softmax(scaled_tensor);
    basis_tensor* output_tensor = basis_tensor_matrix_multiplication(attention_weights, basis_value_tensor);

    basis_tensor_free(transposed_key_tensor); basis_tensor_free(query_key_product);
    basis_tensor_free(scaled_tensor); basis_tensor_free(attention_weights);
    return output_tensor;
}

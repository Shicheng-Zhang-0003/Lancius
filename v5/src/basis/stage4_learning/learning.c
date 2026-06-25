
#include "basis/stage4_learning/learning.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>

basis_tensor* basis_tensor_layer_normalization(basis_tensor* target_tensor, double epsilon) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        basis_value* sum = basis_value_new(0.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* new_sum = basis_value_addition(sum, BASIS_TENSOR_AT(target_tensor, i, j));
            basis_value_free(sum); sum = new_sum;
        }
        basis_value* mean = basis_value_multiplication(sum, basis_value_new(1.0 / target_tensor->column_count));
        basis_value_free(sum);
        basis_value* variance_sum = basis_value_new(0.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* negative_mean = basis_value_multiplication(mean, basis_value_new(-1.0));
            basis_value* difference = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_mean);
            basis_value_free(negative_mean);
            basis_value* squared = basis_value_power(difference, 2.0);
            basis_value_free(difference);
            basis_value* new_variance_sum = basis_value_addition(variance_sum, squared);
            basis_value_free(variance_sum); basis_value_free(squared);
            variance_sum = new_variance_sum;
        }
        basis_value* variance = basis_value_multiplication(variance_sum, basis_value_new(1.0 / target_tensor->column_count));
        basis_value_free(variance_sum);
        basis_value* epsilon_value = basis_value_new(epsilon);
        basis_value* variance_epsilon = basis_value_addition(variance, epsilon_value);
        basis_value_free(variance); basis_value_free(epsilon_value);
        basis_value* inverse_standard_deviation = basis_value_power(variance_epsilon, -0.5);
        basis_value_free(variance_epsilon);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* negative_mean = basis_value_multiplication(mean, basis_value_new(-1.0));
            basis_value* difference = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_mean);
            basis_value_free(negative_mean);
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_multiplication(difference, inverse_standard_deviation);
            basis_value_free(difference);
        }
        basis_value_free(mean); basis_value_free(inverse_standard_deviation);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_rotary_positional_embedding(basis_tensor* target_tensor, int position, double base) {
    BASIS_CHECK_NULL(target_tensor);
    BASIS_CHECK_SHAPE(target_tensor->column_count % 2 == 0);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    size_t half_dim = target_tensor->column_count / 2;
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < half_dim; j++) {
            double theta = pow(base, -(double)(2 * j) / target_tensor->column_count);
            double angle = position * theta;
            basis_value* angle_value = basis_value_new(angle);
            basis_value* cosine_value = basis_value_cosine(angle_value);
            basis_value* sine_value = basis_value_sine(angle_value);
            basis_value_free(angle_value);
            size_t idx1 = i * target_tensor->column_count + j;
            size_t idx2 = i * target_tensor->column_count + j + half_dim;
            basis_value* first_element = BASIS_TENSOR_AT(target_tensor, 0, idx1);
            basis_value* second_element = BASIS_TENSOR_AT(target_tensor, 0, idx2);
            basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, idx1)); basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, idx2));
            basis_value* negative_one = basis_value_new(-1.0);
            basis_value* negative_sine = basis_value_multiplication(sine_value, negative_one);
            basis_value_free(negative_one);
            basis_value* term1 = basis_value_multiplication(first_element, cosine_value);
            basis_value* term2 = basis_value_multiplication(second_element, negative_sine);
            BASIS_TENSOR_AT(output_tensor, 0, idx1) = basis_value_addition(term1, term2);
            basis_value_free(term1); basis_value_free(term2); basis_value_free(negative_sine);
            basis_value* term3 = basis_value_multiplication(first_element, sine_value);
            basis_value* term4 = basis_value_multiplication(second_element, cosine_value);
            BASIS_TENSOR_AT(output_tensor, 0, idx2) = basis_value_addition(term3, term4);
            basis_value_free(term3); basis_value_free(term4);
            basis_value_free(cosine_value); basis_value_free(sine_value);
        }
    }
    return output_tensor;
}

basis_value* basis_tensor_sum(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_value* sum = basis_value_new(0.0);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* new_sum = basis_value_addition(sum, BASIS_TENSOR_AT(target_tensor, i, j));
            basis_value_free(sum);
            sum = new_sum;
        }
    }
    return sum;
}

basis_tensor* basis_tensor_logarithm(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count * target_tensor->column_count; i++) {
        basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, i));
        BASIS_TENSOR_AT(output_tensor, 0, i) = basis_value_logarithm(BASIS_TENSOR_AT(target_tensor, 0, i));
    }
    return output_tensor;
}

basis_tensor* basis_tensor_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);
    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count * first_tensor->column_count; i++) {
        basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, i));
        BASIS_TENSOR_AT(output_tensor, 0, i) = basis_value_multiplication(BASIS_TENSOR_AT(first_tensor, 0, i), BASIS_TENSOR_AT(second_tensor, 0, i));
    }
    return output_tensor;
}

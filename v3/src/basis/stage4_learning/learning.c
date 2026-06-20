#include "basis/stage4_learning/learning.h"
#include <math.h>
#include <stdlib.h>

tensor* tensor_layer_normalization (tensor* target_tensor, double epsilon) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        value* sum = value_new (0.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* new_sum = value_addition (sum, target_tensor -> data [i * target_tensor -> column_count + j]);
            value_free (sum);
            sum = new_sum;
        }
        value* mean = value_multiplication (sum, value_new (1.0 / target_tensor -> column_count));
        value_free (sum);

        value* variance_sum = value_new (0.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* negative_mean = value_multiplication (mean, value_new (-1.0));
            value* difference = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_mean);
            value_free (negative_mean);
            value* squared = value_power (difference, 2.0);
            value_free (difference);
            value* new_variance_sum = value_addition (variance_sum, squared);
            value_free (variance_sum);
            value_free (squared);
            variance_sum = new_variance_sum;
        }
        value* variance = value_multiplication (variance_sum, value_new (1.0 / target_tensor -> column_count));
        value_free (variance_sum);

        value* epsilon_value = value_new (epsilon);
        value* variance_epsilon = value_addition (variance, epsilon_value);
        value_free (variance);
        value* inverse_standard_deviation = value_power (variance_epsilon, -0.5);
        value_free (variance_epsilon);
        value_free (epsilon_value);

        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* negative_mean = value_multiplication (mean, value_new (-1.0));
            value* difference = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_mean);
            value_free (negative_mean);
            value_free (output_tensor -> data [i * target_tensor -> column_count + j]);
            output_tensor -> data [i * target_tensor -> column_count + j] = value_multiplication (difference, inverse_standard_deviation);
            value_free (difference);
        }
        value_free (mean);
        value_free (inverse_standard_deviation);
    }
    return output_tensor;
}

tensor* tensor_rotary_positional_embedding (tensor* target_tensor, int position, double base) {
    if ((!target_tensor) || (target_tensor -> column_count % 2 != 0)) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    size_t half_dim = target_tensor->column_count / 2;

    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        for (size_t j = 0; j < half_dim; j ++) {
            // Standard frequency formula: theta = base^(-2j/d)
            double theta = pow(base, -(double)(2 * j) / target_tensor->column_count);
            double angle = position * theta;

            value* angle_value = value_new (angle);
            value* cosine_value = value_cosine (angle_value);
            value* sine_value = value_sine (angle_value);
            value_free (angle_value);

            // Split dimensions: x[..., :d/2] and x[..., d/2:]
            size_t idx1 = i * target_tensor->column_count + j;
            size_t idx2 = i * target_tensor->column_count + j + half_dim;

            value* first_element = target_tensor->data[idx1];
            value* second_element = target_tensor->data[idx2];

            value_free (output_tensor->data[idx1]);
            value_free (output_tensor->data[idx2]);

            value* negative_one = value_new(-1.0);
            value* negative_sine = value_multiplication(sine_value, negative_one);
            value_free(negative_one);

            // out[..., :d/2] = x1 * cos - x2 * sin
            value* term1 = value_multiplication(first_element, cosine_value);
            value* term2 = value_multiplication(second_element, negative_sine);
            output_tensor->data[idx1] = value_addition(term1, term2);
            value_free(term1); value_free(term2); value_free(negative_sine);

            // out[..., d/2:] = x2 * cos + x1 * sin
            value* term3 = value_multiplication(first_element, sine_value);
            value* term4 = value_multiplication(second_element, cosine_value);
            output_tensor->data[idx2] = value_addition(term3, term4);
            value_free(term3); value_free(term4);

            value_free(cosine_value);
            value_free(sine_value);
        }
    }
    return output_tensor;
}

value* tensor_sum (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    value* sum = value_new (0.0);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value* new_sum = value_addition (sum, target_tensor -> data [i]);
        value_free (sum);
        sum = new_sum;
    }
    return sum;
}

tensor* tensor_logarithm (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_logarithm (target_tensor -> data [i]);
    }
    return output_tensor;
}

tensor* tensor_multiplication (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || ((first_tensor -> row_count) != (second_tensor -> row_count)) || ((first_tensor -> column_count) != (second_tensor -> column_count))) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, first_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count * first_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_multiplication (first_tensor -> data [i], second_tensor -> data [i]);
    }
    return output_tensor;
}

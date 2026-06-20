#include "basis/stage2_linear/tensor.h"
#include <stdio.h>
#include <assert.h>

tensor* tensor_new (size_t row_count, size_t column_count) {
    tensor* new_tensor = (tensor*) malloc (sizeof (tensor));
    new_tensor -> row_count = row_count;
    new_tensor -> column_count = column_count;
    new_tensor -> data = (value **) malloc (sizeof (value*) * row_count * column_count);
    for (size_t i = 0; i < row_count * column_count; i ++) {new_tensor -> data [i] = value_new (0.0);}
    return new_tensor;
}

void tensor_free (tensor* target_tensor) {
    if (!target_tensor) {return;}
    if (target_tensor -> data) {
        for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {value_free (target_tensor -> data [i]);}
        free (target_tensor -> data);
    }
    free (target_tensor);
}

void tensor_fill (tensor* target_tensor, double fill_value) {
    if (!target_tensor) {return;}
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {target_tensor -> data [i] -> data = fill_value;}
}

void tensor_set (tensor* target_tensor, size_t row_index, size_t column_index, double data_value) {
    if ((!target_tensor) || (row_index >= target_tensor -> row_count) || (column_index >= target_tensor -> column_count)) {return;}
    target_tensor -> data [row_index * target_tensor -> column_count + column_index] -> data = data_value;
}

tensor* tensor_addition (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || ((first_tensor -> row_count) != (second_tensor -> row_count)) || ((first_tensor -> column_count) != (second_tensor -> column_count))) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, first_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count * first_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_addition (first_tensor -> data [i], second_tensor -> data [i]);
    }
    return output_tensor;
}

tensor* tensor_matrix_multiplication (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || (first_tensor -> column_count != second_tensor -> row_count)) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, second_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count; i ++) {
        for (size_t j = 0; j < second_tensor -> column_count; j ++) {
            value* sum = value_new (0.0);
            for (size_t k = 0; k < first_tensor -> column_count; k ++) {
                value* product = value_multiplication (first_tensor -> data [i * first_tensor -> column_count + k], second_tensor -> data [k * second_tensor -> column_count + j]);
                value* new_sum = value_addition (sum, product);
                value_free (sum);
                value_free (product);
                sum = new_sum;
            }
            value_free (output_tensor -> data [i * second_tensor -> column_count + j]);
            output_tensor -> data [i * second_tensor -> column_count + j] = sum;
        }
    }
    return output_tensor;
}

tensor* tensor_scalar_multiplication (tensor* target_tensor, double scalar_value) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    value* scalar = value_new (scalar_value);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_multiplication (target_tensor -> data [i], scalar);
    }
    value_free (scalar);
    return output_tensor;
}

tensor* tensor_rectified_linear_unit (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_rectified_linear_unit (target_tensor -> data [i]);
    }
    return output_tensor;
}

void tensor_print (tensor* target_tensor, const char* name) {
    if (!target_tensor) {return;}
    printf ("Tensor %s (%zu x %zu):\n", name, target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        printf ("  [ ");
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {printf ("%8.4f ", target_tensor -> data [i * target_tensor -> column_count + j] -> data);}
        printf ("]\n");
    }
}

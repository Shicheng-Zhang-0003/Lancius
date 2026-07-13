#include "basis/stage3_sequence/sequence.h"
#include <math.h>
#include <stdlib.h>

tensor* tensor_transpose (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = (tensor*) malloc (sizeof (tensor));
    output_tensor -> row_count = target_tensor -> column_count;
    output_tensor -> column_count = target_tensor -> row_count;
    output_tensor -> data = (value **) malloc (sizeof (value*) * output_tensor -> row_count * output_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* current_value = target_tensor -> data [i * target_tensor -> column_count + j];
            value_retain (current_value);
            output_tensor -> data [j * output_tensor -> column_count + i] = current_value;
        }
    }
    return output_tensor;
}

tensor* tensor_softmax (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        double maximum_data = target_tensor -> data [i * target_tensor -> column_count] -> data;
        for (size_t j = 1; j < target_tensor -> column_count; j ++) {
            if (target_tensor -> data [i * target_tensor -> column_count + j] -> data > maximum_data) {maximum_data = target_tensor -> data [i * target_tensor -> column_count + j] -> data;}
        }
        value* negative_maximum = value_new (-maximum_data);
        value* sum = value_new (0.0);
        value** exponentials = (value **) malloc (sizeof (value*) * target_tensor -> column_count);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* shifted = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_maximum);
            exponentials [j] = value_exponential (shifted);
            value* new_sum = value_addition (sum, exponentials [j]);
            value_free (sum);
            value_free (shifted);
            sum = new_sum;
        }
        value_free (negative_maximum);
        value* inverse_sum = value_power (sum, -1.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value_free (output_tensor -> data [i * target_tensor -> column_count + j]);
            output_tensor -> data [i * target_tensor -> column_count + j] = value_multiplication (exponentials [j], inverse_sum);
            value_free (exponentials [j]);
        }
        value_free (sum);
        value_free (inverse_sum);
        free (exponentials);
    }
    return output_tensor;
}

tensor* tensor_attention (tensor* query_tensor, tensor* key_tensor, tensor* value_tensor) {
    if ((!query_tensor) || (!key_tensor) || (!value_tensor)) {return NULL;}
    if ((query_tensor -> column_count != key_tensor -> column_count) || (key_tensor -> row_count != value_tensor -> row_count)) {return NULL;}

    tensor* transposed_key_tensor = tensor_transpose (key_tensor);
    if (!transposed_key_tensor) {return NULL;}

    tensor* query_key_product = tensor_matrix_multiplication (query_tensor, transposed_key_tensor);
    if (!query_key_product) {
        tensor_free (transposed_key_tensor);
        return NULL;
    }

    tensor* scaled_tensor = tensor_scalar_multiplication (query_key_product, 1.0 / sqrt ((double) key_tensor -> column_count));
    if (!scaled_tensor) {
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }

    tensor* attention_weights = tensor_softmax (scaled_tensor);
    if (!attention_weights) {
        tensor_free (scaled_tensor);
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }

    tensor* output_tensor = tensor_matrix_multiplication (attention_weights, value_tensor);
    if (!output_tensor) {
        tensor_free (attention_weights);
        tensor_free (scaled_tensor);
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }

    tensor_free (transposed_key_tensor);
    tensor_free (query_key_product);
    tensor_free (scaled_tensor);
    tensor_free (attention_weights);
    return output_tensor;
}

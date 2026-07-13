#include "basis/stage5_unified/geometry.h"
#include "basis/stage3_sequence/sequence.h"
#include <stdlib.h>

metric* metric_fisher_information (tensor* weight_tensor, tensor* input_tensor, tensor* output_tensor) {
    (void) weight_tensor; (void) output_tensor;
    if (!input_tensor) {return NULL;}
    tensor* transposed_input = tensor_transpose (input_tensor);
    if (!transposed_input) {return NULL;}
    tensor* fisher_matrix = tensor_matrix_multiplication (transposed_input, input_tensor);
    if (!fisher_matrix) {
        tensor_free (transposed_input);
        return NULL;
    }
    metric* new_metric = (metric*) malloc (sizeof (metric));
    if (!new_metric) {
        tensor_free (fisher_matrix);
        tensor_free (transposed_input);
        return NULL;
    }
    new_metric -> matrix = fisher_matrix;
    tensor_free (transposed_input);
    return new_metric;
}

void metric_free (metric* target_metric) {
    if (!target_metric) {return;}
    tensor_free (target_metric -> matrix);
    free (target_metric);
}

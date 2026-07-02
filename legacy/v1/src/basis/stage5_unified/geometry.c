#include "basis/stage5_unified/geometry.h"
#include "basis/stage3_sequence/sequence.h"
#include <stdio.h>

Metric* metric_fisher(Tensor* weights, Tensor* inputs, Tensor* outputs) {
    Tensor* IT = tensor_transpose(inputs);
    Tensor* F = tensor_matmul(IT, inputs);
    Metric* m = (Metric*)malloc(sizeof(Metric));
    m->matrix = F;
    tensor_free(IT);
    return m;
}

void metric_free(Metric* m) {
    tensor_free(m->matrix);
    free(m);
}

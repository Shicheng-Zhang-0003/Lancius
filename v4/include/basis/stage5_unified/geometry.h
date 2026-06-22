#ifndef basis_stage5_unified_geometry_h
#define basis_stage5_unified_geometry_h

#include "basis/stage2_linear/tensor.h"

typedef struct {
    tensor* matrix;
} metric;

metric* metric_fisher_information(tensor* weight_tensor, tensor* input_tensor, tensor* output_tensor);
void metric_free(metric* target_metric);

#endif

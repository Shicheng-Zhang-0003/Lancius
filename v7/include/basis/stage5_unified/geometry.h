#ifndef basis_stage5_unified_geometry_h
#define basis_stage5_unified_geometry_h
#include "basis/stage2_linear/tensor.h"

typedef struct {
    basis_tensor* matrix;
} basis_metric;

basis_metric* basis_metric_fisher_information(basis_tensor* weight_tensor, basis_tensor* input_tensor, basis_tensor* output_tensor);
void basis_metric_free(basis_metric* target_metric);
void basis_metric_ngd_step(basis_metric* fisher_metric, basis_tensor* weight_tensor, double learning_rate);

#endif

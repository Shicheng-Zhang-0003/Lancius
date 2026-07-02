#ifndef BASIS_STAGE5_GEOMETRY_H
#define BASIS_STAGE5_GEOMETRY_H
#include "basis/stage2_linear/tensor.h"
typedef struct {Tensor* matrix;} Metric;
Metric* metric_fisher (Tensor* weights, Tensor* inputs, Tensor* outputs);
void metric_free (Metric* m);
#endif

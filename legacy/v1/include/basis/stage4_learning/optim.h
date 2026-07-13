#ifndef BASIS_STAGE4_OPTIM_H
#define BASIS_STAGE4_OPTIM_H
#include "basis/stage2_linear/tensor.h"
typedef struct Adam {
    double lr;
    double beta1;
    double beta2;
    double eps;
    size_t t;
    Tensor* m;
    Tensor* v;
} Adam;
Adam* adam_new (size_t rows, size_t cols, double lr);
void adam_step (Adam* opt, Tensor* W);
void adam_free (Adam* opt);
#endif

#ifndef basis_stage4_optim_h
#define basis_stage4_optim_h

#include "basis/stage2_linear/tensor.h"

typedef struct adam {
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    size_t time_step;
    double* first_moment;  // V4: Decoupled from autograd graph
    double* second_moment; // V4: Decoupled from autograd graph
    size_t element_count;
} adam;

adam* adam_new(size_t row_count, size_t column_count, double learning_rate);
void adam_optimization_step(adam* optimizer, tensor* weight_tensor);
void adam_free(adam* optimizer);

#endif


#ifndef basis_stage4_optim_h
#define basis_stage4_optim_h
#include "basis/stage2_linear/tensor.h"
typedef struct basis_adam {
    double learning_rate; double beta1; double beta2; double epsilon; size_t time_step;
    double* first_moment; double* second_moment; size_t element_count;
} basis_adam;
basis_adam* basis_adam_new(size_t row_count, size_t column_count, double learning_rate);
void basis_adam_optimization_step(basis_adam* optimizer, basis_tensor* weight_tensor);
void basis_adam_free(basis_adam* optimizer);
#endif

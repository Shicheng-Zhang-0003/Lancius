
#include "basis/stage4_learning/optim.h"
#include <math.h>
#include <stdlib.h>
basis_adam* basis_adam_new(size_t row_count, size_t column_count, double learning_rate) {
    basis_adam* optimizer = (basis_adam*)malloc(sizeof(basis_adam));
    optimizer->learning_rate = learning_rate; optimizer->beta1 = 0.9; optimizer->beta2 = 0.999;
    optimizer->epsilon = 1e-8; optimizer->time_step = 0; optimizer->element_count = row_count * column_count;
    optimizer->first_moment = (double*)calloc(optimizer->element_count, sizeof(double));
    optimizer->second_moment = (double*)calloc(optimizer->element_count, sizeof(double));
    return optimizer;
}
void basis_adam_optimization_step(basis_adam* optimizer, basis_tensor* weight_tensor) {
    optimizer->time_step++;
    for (size_t i = 0; i < optimizer->element_count; i++) {
        double gradient = weight_tensor->data[i]->gradient;
        optimizer->first_moment[i] = optimizer->beta1 * optimizer->first_moment[i] + (1.0 - optimizer->beta1) * gradient;
        optimizer->second_moment[i] = optimizer->beta2 * optimizer->second_moment[i] + (1.0 - optimizer->beta2) * gradient * gradient;
        double first_moment_hat = optimizer->first_moment[i] / (1.0 - pow(optimizer->beta1, optimizer->time_step));
        double second_moment_hat = optimizer->second_moment[i] / (1.0 - pow(optimizer->beta2, optimizer->time_step));
        weight_tensor->data[i]->data -= optimizer->learning_rate * first_moment_hat / (sqrt(second_moment_hat) + optimizer->epsilon);
        weight_tensor->data[i]->gradient = 0.0;
    }
}
void basis_adam_free(basis_adam* optimizer) {
    if (!optimizer) return;
    free(optimizer->first_moment); free(optimizer->second_moment); free(optimizer);
}

#include "basis/stage4_learning/optim.h"
#include <math.h>
#include <stdlib.h>
adam* adam_new (size_t row_count, size_t column_count, double learning_rate) {
    adam* optimizer = (adam*) malloc (sizeof (adam));
    optimizer -> learning_rate = learning_rate;
    optimizer -> beta1 = 0.9;
    optimizer -> beta2 = 0.999;
    optimizer -> epsilon = 1e-8;
    optimizer -> time_step = 0;
    optimizer -> first_moment = tensor_new (row_count, column_count);
    optimizer -> second_moment = tensor_new (row_count, column_count);
    tensor_fill (optimizer -> first_moment, 0.0);
    tensor_fill (optimizer -> second_moment, 0.0);
    return optimizer;
} void adam_optimization_step (adam* optimizer, tensor* weight_tensor) {
    optimizer -> time_step ++;
    for (size_t i = 0; i < weight_tensor -> row_count * weight_tensor -> column_count; i ++) {
        double gradient = weight_tensor -> data [i] -> gradient;
        optimizer -> first_moment -> data [i] -> data = optimizer -> beta1 * optimizer -> first_moment -> data [i] -> data + (1.0 - optimizer -> beta1) * gradient;
        optimizer -> second_moment -> data [i] -> data = optimizer -> beta2 * optimizer -> second_moment -> data [i] -> data + (1.0 - optimizer -> beta2) * gradient * gradient;
        double first_moment_hat = optimizer -> first_moment -> data [i] -> data / (1.0 - pow (optimizer -> beta1, optimizer -> time_step));
        double second_moment_hat = optimizer -> second_moment -> data [i] -> data / (1.0 - pow (optimizer -> beta2, optimizer -> time_step));
        weight_tensor -> data [i] -> data -= optimizer -> learning_rate * first_moment_hat / (sqrt (second_moment_hat) + optimizer -> epsilon);
        weight_tensor -> data [i] -> gradient = 0.0;
    }
} void adam_free (adam* optimizer) {
    if (!optimizer) {return;}
    tensor_free (optimizer -> first_moment);
    tensor_free (optimizer -> second_moment);
    free (optimizer);
}

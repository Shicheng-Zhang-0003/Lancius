#include "basis/stage4_learning/learning.h"
#include <math.h>
Tensor* tensor_layernorm (Tensor* t, double eps) {
    Tensor* out = tensor_new (t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows; i++) {
        Value* sum = val_new (0.0);
        for (size_t j = 0; j < t -> cols; j++) {sum = val_add (sum, t -> data [i * t -> cols + j]);}
        Value* mean = val_mul (sum, val_new (1.0 / t -> cols));
        Value* var_sum = val_new (0.0);
        for (size_t j = 0; j < t -> cols; j++) {
            Value* diff = val_add (t -> data [i * t -> cols + j], val_mul (mean, val_new (-1.0)));
            var_sum = val_add (var_sum, val_pow (diff, 2.0));
        } Value* var = val_mul (var_sum, val_new (1.0 / t -> cols));
        Value* inv_std = val_pow (val_add (var, val_new (eps)), -0.5);
        for (size_t j = 0; j < t -> cols; j++) {
            Value* diff = val_add (t -> data [i * t -> cols + j], val_mul (mean, val_new (-1.0)));
            val_free (out -> data [i * t -> cols + j]);
            out -> data [i * t -> cols + j] = val_mul (diff, inv_std);
        }
    } return out;
} Tensor* tensor_rope (Tensor* t, int pos, double base) {
    if (t -> cols % 2 != 0) return NULL;
    Tensor* out = tensor_new (t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows; i++) {
        for (size_t j = 0; j < t -> cols; j += 2) {
            double theta = pow (base, -((double) j / t -> cols));
            double angle = pos * theta;
            Value* cos_val = val_cos (val_new (angle));
            Value* sin_val = val_sin (val_new (angle));
            Value* x1 = t -> data [i * t -> cols + j];
            Value* x2 = t -> data [i * t -> cols + j + 1];
            val_free (out -> data [i * t -> cols + j]);
            val_free (out -> data [i * t -> cols + j + 1]);
            out -> data[i * t -> cols + j] = val_add (val_mul (x1, cos_val), val_mul (x2, val_mul (sin_val, val_new (-1.0))));
            out -> data[i * t -> cols + j + 1] = val_add (val_mul (x1, sin_val), val_mul (x2, cos_val));
        }
    } return out;
} Value* tensor_sum (Tensor* t) {
    Value* sum = val_new (0.0);
    for (size_t i = 0; i < t -> rows * t -> cols; i++) {sum = val_add (sum, t -> data [i]);}
    return sum;
} Tensor* tensor_log (Tensor* t) {
    Tensor* out = tensor_new (t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows * t -> cols; i++) {
        val_free (out -> data [i]);
        out -> data [i] = val_log (t -> data [i]);
    } return out;
} Tensor* tensor_mul (Tensor* a, Tensor* b) {
    if ((a -> rows != b -> rows) || (a -> cols != b -> cols)) {return NULL;}
    Tensor* out = tensor_new (a -> rows, a -> cols);
    for (size_t i = 0; i < a -> rows * a -> cols; i++) {
        val_free (out -> data [i]);
        out -> data [i] = val_mul (a -> data [i], b -> data [i]);
    } return out;
}

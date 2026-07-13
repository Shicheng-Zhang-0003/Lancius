#include "basis/stage2_linear/tensor.h"
#include <stdio.h>
Tensor* tensor_new (size_t rows, size_t cols) {
    Tensor* t = (Tensor*) malloc (sizeof (Tensor));
    t -> rows = rows;
    t -> cols = cols;
    t -> data = (Value**) malloc (sizeof (Value*) * rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {t -> data [i] = val_new (0.0);}
    return t;
} void tensor_fill (Tensor* t, double val) {for (size_t i = 0; i < t -> rows * t -> cols; i++) {t -> data [i] -> data = val;}}
void tensor_set (Tensor* t, size_t r, size_t c, double val) {t -> data [r * t -> cols + c] -> data = val;}
Tensor* tensor_add (Tensor* a, Tensor* b) {
    if ((a -> rows != b -> rows) || (a -> cols != b -> cols)) {return NULL;}
    Tensor* out = tensor_new (a -> rows, a -> cols);
    for (size_t i = 0; i < a -> rows * a -> cols; i++) {
        val_free (out -> data [i]);
        out -> data [i] = val_add (a -> data [i], b -> data [i]);
    } return out;
} Tensor* tensor_matmul (Tensor* a, Tensor* b) {
    if (a -> cols != b -> rows) return NULL;
    Tensor* out = tensor_new (a -> rows, b -> cols);
    for (size_t i = 0; i < a -> rows; i++) {
        for (size_t j = 0; j < b -> cols; j++) {
            Value* sum = val_new (0.0);
            for (size_t k = 0; k < a -> cols; k++) {
                Value* prod = val_mul (a -> data [i * a -> cols + k], b -> data [k * b -> cols + j]);
                sum = val_add (sum, prod);
            } val_free (out -> data [i * b -> cols + j]);
            out -> data [i * b -> cols + j] = sum;
        }
    } return out;
} Tensor* tensor_scalar_mul (Tensor* t, double scalar) {
    Tensor* out = tensor_new(t -> rows, t -> cols);
    Value* s = val_new (scalar);
    for (size_t i = 0; i < t -> rows * t -> cols; i++) {
        val_free (out -> data [i]);
        out -> data [i] = val_mul (t -> data [i], s);
    } return out;
} Tensor* tensor_relu (Tensor* t) {
    Tensor* out = tensor_new (t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows * t -> cols; i++) {
        val_free (out -> data [i]);
        out -> data [i] = val_relu (t -> data [i]);
    } return out;
} void tensor_print (Tensor* t, char* name) {
    printf ("Tensor %s (%zu x %zu):\n", name, t -> rows, t -> cols);
    for (size_t i = 0; i < t -> rows; i++) {
        printf("  [ ");
        for (size_t j = 0; j < t -> cols; j++) {printf("%8.4f ", t -> data [i * t -> cols + j] -> data);}
        printf ("]\n");
    }
} void tensor_free (Tensor* t) {
    free (t -> data);
    free (t);
}

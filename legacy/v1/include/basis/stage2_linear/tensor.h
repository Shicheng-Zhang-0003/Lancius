#ifndef BASIS_STAGE2_TENSOR_H
#define BASIS_STAGE2_TENSOR_H
#include "basis/stage1_atomic/scalar.h"
typedef struct Tensor {
    Value** data;
    size_t rows;
    size_t cols;
} Tensor;
Tensor* tensor_new (size_t rows, size_t cols);
void tensor_free (Tensor* t);
void tensor_fill (Tensor* t, double val);
void tensor_set (Tensor* t, size_t r, size_t c, double val);
Tensor* tensor_add (Tensor* a, Tensor* b);
Tensor* tensor_matmul (Tensor* a, Tensor* b);
Tensor* tensor_scalar_mul (Tensor* t, double scalar);
Tensor* tensor_relu (Tensor* t);
void tensor_print (Tensor* t, char* name);
#endif

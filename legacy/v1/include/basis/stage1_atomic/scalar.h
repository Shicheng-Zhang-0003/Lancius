#ifndef BASIS_STAGE1_SCALAR_H
#define BASIS_STAGE1_SCALAR_H
#include <stdlib.h>
typedef struct Value {
    double data;
    double grad;
    void (*backward) (struct Value* self);
    struct Value** prev;
    size_t prev_count;
    char* op;
    double constant;
} Value;
Value* val_new (double data);
void val_free (Value* v);
Value* val_add (Value* a, Value* b);
Value* val_mul (Value* a, Value* b);
Value* val_pow (Value* a, double exponent);
Value* val_exp (Value* a);
Value* val_log (Value* a);
Value* val_sin (Value* a);
Value* val_cos (Value* a);
Value* val_relu (Value* a);
void val_backward (Value* root);
#endif

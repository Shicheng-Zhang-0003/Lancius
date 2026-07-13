#include "basis/stage1_atomic/scalar.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
Value* val_new (double data) {
    Value* v = (Value*) malloc (sizeof (Value));
    v -> data = data;
    v -> grad = 0.0;
    v -> backward = NULL;
    v -> prev = NULL;
    v -> prev_count = 0;
    v -> op = "";
    v -> constant = 0.0;
    return v;
} static Value* val_op_init (Value* a, Value* b, char* op, size_t count) {
    Value* out = val_new (0.0);
    out -> op = op;
    out -> prev_count = count;
    out -> prev = (Value**) malloc (sizeof (Value*) * count);
    if (a) {out -> prev [0] = a;}
    if ((b) && (count > 1)) {out -> prev [1] = b;}
    return out;
} static void backward_add (Value* self) {
    self -> prev [0] -> grad += 1.0 * self -> grad;
    self -> prev [1] -> grad += 1.0 * self -> grad;
} Value* val_add (Value* a, Value* b) {
    Value* out = val_op_init (a, b, "+", 2);
    out -> data = a -> data + b -> data;
    out -> backward = backward_add;
    return out;
} static void backward_mul (Value* self) {
    self -> prev [0] -> grad += self -> prev [1] -> data * self -> grad;
    self -> prev [1] -> grad += self -> prev [0] -> data * self -> grad;
} Value* val_mul (Value* a, Value* b) {
    Value* out = val_op_init (a, b, "*", 2);
    out -> data = a -> data * b -> data;
    out -> backward = backward_mul;
    return out;
} static void backward_pow (Value* self) {
    double a = self -> prev [0] -> data;
    double n = self -> constant;
    self -> prev [0] -> grad += (n * pow (a, n - 1.0)) * self -> grad;
} Value* val_pow (Value* a, double exponent) {
    Value* out = val_op_init (a, NULL, "**", 1);
    out -> data = pow (a -> data, exponent);
    out -> constant = exponent;
    out -> backward = backward_pow;
    return out;
} static void backward_exp (Value* self) {self -> prev [0] -> grad += self -> data * self -> grad;}
Value* val_exp (Value* a) {
    Value* out = val_op_init (a, NULL, "exp", 1);
    out -> data = exp (a -> data);
    out -> backward = backward_exp;
    return out;
} static void backward_log (Value* self) {self -> prev [0] -> grad += (1.0 / self -> prev [0] -> data) * self -> grad;}
Value* val_log (Value* a) {
    Value* out = val_op_init (a, NULL, "log", 1);
    out -> data = log (a -> data);
    out -> backward = backward_log;
    return out;
} static void backward_sin (Value* self) {self -> prev [0] -> grad += cos (self -> prev [0] -> data) * self -> grad;}
Value* val_sin (Value* a) {
    Value* out = val_op_init (a, NULL, "sin", 1);
    out -> data = sin (a -> data);
    out -> backward = backward_sin;
    return out;
} static void backward_cos(Value* self) {self -> prev [0] -> grad += -sin (self -> prev [0] -> data) * self -> grad;}
Value* val_cos (Value* a) {
    Value* out = val_op_init (a, NULL, "cos", 1);
    out -> data = cos (a -> data);
    out -> backward = backward_cos;
    return out;
} static void backward_relu (Value* self) {self -> prev [0] -> grad += (self -> data > 0 ? 1.0 : 0.0) * self -> grad;}
Value* val_relu(Value* a) {
    Value* out = val_op_init (a, NULL, "ReLU", 1);
    out -> data = a -> data > 0 ? a -> data : 0.0;
    out -> backward = backward_relu;
    return out;
} static void build_topo (Value* v, Value*** topo, size_t* size, size_t* cap, Value** visited, size_t* v_count) {
    for (size_t i = 0; i < *v_count; i++) {if (visited [i] == v) {return;}}
    visited [(*v_count)++] = v;
    for (size_t i = 0; i < v->prev_count; i++) {build_topo (v -> prev [i], topo, size, cap, visited, v_count);}
    if (*size >= *cap) {
        *cap *= 2;
        *topo = (Value**) realloc (*topo, sizeof (Value*) * (*cap));
    } (*topo) [(*size)++] = v;
} void val_backward (Value* root) {
    size_t cap = 128;
    size_t size = 0;
    Value** topo = (Value**) malloc (sizeof (Value*) * cap);
    Value** visited = (Value**) malloc (sizeof (Value*) * 8192);
    size_t v_count = 0;
    build_topo (root, &topo, &size, &cap, visited, &v_count);
    root -> grad = 1.0;
    for (int i = (int) size - 1; i >= 0; i--) {if (topo [i] -> backward) {topo[i] -> backward (topo [i]);}}
    free (topo);
    free (visited);
} void val_free (Value* v) {
    if (v -> prev) {free(v -> prev);}
    free (v);
}

#include "lancius/lancius_checked.h"

int lancius_checked_add_size(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

int lancius_checked_mul_size(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

int lancius_checked_product_shape(const size_t* shape, uint8_t ndim, size_t* out) {
    if (!out) return 0;
    if (ndim > 4) return 0;
    size_t e = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        size_t d = shape ? shape[i] : 1;
        if (!lancius_checked_mul_size(e, d, &e)) return 0;
    }
    *out = e;
    return 1;
}

int lancius_checked_bytes(size_t elems, size_t elem_size, size_t* out) {
    return lancius_checked_mul_size(elems, elem_size, out);
}

int lancius_checked_add_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return 0;
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

int lancius_checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return 0;
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

#ifndef LANCIUS_CHECKED_H
#define LANCIUS_CHECKED_H

#include <stddef.h>
#include <stdint.h>

/* Checked-arithmetic primitives. Return 1 on success, 0 on overflow. */
int lancius_checked_add_size(size_t a, size_t b, size_t* out);
int lancius_checked_mul_size(size_t a, size_t b, size_t* out);
int lancius_checked_product_shape(const size_t* shape, uint8_t ndim, size_t* out);
int lancius_checked_bytes(size_t elems, size_t elem_size, size_t* out);
int lancius_checked_add_u64(uint64_t a, uint64_t b, uint64_t* out);
int lancius_checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out);

#endif /* LANCIUS_CHECKED_H */

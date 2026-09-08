#ifndef LANCIUS_VALIDATE_H
#define LANCIUS_VALIDATE_H

#include <stddef.h>
#include <stdint.h>
#include "lancius/lancius_error.h"

/* Central tensor/shape validation. Returns LANCIUS_ERROR_OK on success. */
lancius_error lancius_validate_rank(uint8_t ndim);
lancius_error lancius_validate_shape(const size_t* shape, uint8_t ndim);
lancius_error lancius_validate_permutation(const uint32_t* axes, uint8_t rank);
lancius_error lancius_validate_matmul(const size_t* a_shape, uint8_t a_ndim,
                                      const size_t* b_shape, uint8_t b_ndim);
lancius_error lancius_validate_reshape(size_t in_elems, const size_t* out_shape, uint8_t out_ndim);
lancius_error lancius_validate_binary_same_shape(const size_t* a, const size_t* b, uint8_t ndim);
lancius_error lancius_validate_conv2d(size_t H_in, size_t W_in, size_t K_h, size_t K_w,
                                      uint32_t stride, uint32_t pad);
lancius_error lancius_validate_gqa(uint32_t n_heads_q, uint32_t n_heads_kv);
lancius_error lancius_validate_stride_nonzero(uint32_t stride);

#endif /* LANCIUS_VALIDATE_H */

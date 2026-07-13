#ifndef V8_KERNELS_H
#define V8_KERNELS_H

#include <stddef.h>
#include <stdint.h>

// PURE MATH KERNELS. No IR, no Arena, no Graph. Just pointers and dimensions.
// This is the single source of truth for all V8 math.

void kernel_matmul(double* out, const double* a, const double* b, size_t M, size_t K, size_t N);
void kernel_conv2d_fwd(double* out, const double* in, const double* w,
                       size_t N, size_t C_in, size_t H_in, size_t W_in,
                       size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad);
void kernel_conv2d_bwd_in(double* out, const double* grad, const double* w,
                          size_t N, size_t C_in, size_t H_in, size_t W_in,
                          size_t C_out, size_t H_out, size_t W_out,
                          size_t K_h, size_t K_w, size_t stride, size_t pad);
void kernel_conv2d_bwd_w(double* out, const double* grad, const double* in,
                         size_t N, size_t C_in, size_t H_in, size_t W_in,
                         size_t C_out, size_t H_out, size_t W_out,
                         size_t K_h, size_t K_w, size_t stride, size_t pad);

void kernel_conv2d_relu_fwd(double* out, const double* in, const double* w,
                            size_t N, size_t C_in, size_t H_in, size_t W_in,
                            size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad);

#endif

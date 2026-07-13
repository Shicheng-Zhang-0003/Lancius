#ifndef LANCIUS_KERNELS_H
#define LANCIUS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

// PURE MATH KERNELS. No IR, no Arena, no Graph. Just pointers and dimensions.
// This is the single source of truth for all LANCIUS math.

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

void kernel_conv2d_int8_fwd(double* out, const int8_t* in, const int8_t* w, double scale_in, double scale_w,
                            size_t N, size_t C_in, size_t H_in, size_t W_in,
                            size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad);


// V13 Transformer Kernels
void kernel_layernorm(double* out, const double* in, const double* gamma, const double* beta, size_t batch_size, size_t hidden_size, double eps);
void kernel_gelu(double* out, const double* in, size_t elements);
void kernel_rope(double* q, double* k, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim, int pos_offset);


void kernel_attention(double* out, const double* q, const double* k, const double* v, size_t seq_len, size_t n_heads, size_t head_dim);


void kernel_attention_kv_cache(double* out, const double* q, const double* k_cache, const double* v_cache, size_t seq_len, size_t n_heads, size_t head_dim);


void kernel_rmsnorm(double* out, const double* in, const double* gamma, size_t seq_len, size_t hidden_size, double eps);
void kernel_swiglu(double* out, const double* gate, const double* up, size_t elements);
void kernel_gqa(double* out, const double* q, const double* k, const double* v, size_t seq_len, size_t n_heads_q, size_t n_heads_kv, size_t head_dim);

// =====================================================================
// V9.5 PERFORMANCE MANDATE: FP32 & SIMD KERNELS
// =====================================================================
void kernel_matmul_f32(float* out, const float* a, const float* b, size_t M, size_t K, size_t N);
void kernel_matmul_f64_simd(double* out, const double* a, const double* b, size_t M, size_t K, size_t N);

// Dtype-aware dispatch macro (Zero-cost abstraction)
#ifdef LANCIUS_USE_F32
    #define kernel_matmul_dispatch(out, a, b, M, K, N) \
        kernel_matmul_f32((float*)(out), (const float*)(a), (const float*)(b), (M), (K), (N))
#else
    #define kernel_matmul_dispatch(out, a, b, M, K, N) \
        kernel_matmul_f64_simd((double*)(out), (const double*)(a), (const double*)(b), (M), (K), (N))
#endif

#endif

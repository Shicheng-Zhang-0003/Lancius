#include "lancius/lancius_kernels.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void kernel_matmul(double* out, const double* a, const double* b, size_t M, size_t K, size_t N) {
    memset(out, 0, M * N * sizeof(double));
    for(size_t r=0; r<M; r++) {
        for(size_t k=0; k<K; k++) {
            double val = a[r*K + k];
            #pragma omp simd
            for(size_t c=0; c<N; c++) {
                out[r*N + c] += val * b[k*N + c];
            }
        }
    }
}

void kernel_conv2d_fwd(double* out, const double* in, const double* w,
                       size_t N, size_t C_in, size_t H_in, size_t W_in,
                       size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad) {
    size_t H_out = (H_in + 2*pad - K_h)/stride + 1;
    size_t W_out = (W_in + 2*pad - K_w)/stride + 1;
    memset(out, 0, N*C_out*H_out*W_out*sizeof(double));

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t ni=0; ni<N; ni++) {
        for(size_t co=0; co<C_out; co++) {
            for(size_t ho=0; ho<H_out; ho++) {
                for(size_t wo=0; wo<W_out; wo++) {
                    double sum = 0.0;
                    for(size_t ci=0; ci<C_in; ci++) {
                        for(size_t kh=0; kh<K_h; kh++) {
                            for(size_t kw=0; kw<K_w; kw++) {
                                int ih = (int)(ho*stride) - (int)pad + (int)kh;
                                int iw = (int)(wo*stride) - (int)pad + (int)kw;
                                if(ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                    size_t in_idx = ni*(C_in*H_in*W_in) + ci*(H_in*W_in) + ih*W_in + iw;
                                    size_t w_idx = co*(C_in*K_h*K_w) + ci*(K_h*K_w) + kh*K_w + kw;
                                    sum += in[in_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    out[out_idx] = sum;
                }
            }
        }
    }
}

void kernel_conv2d_bwd_in(double* out, const double* grad, const double* w,
                          size_t N, size_t C_in, size_t H_in, size_t W_in,
                          size_t C_out, size_t H_out, size_t W_out,
                          size_t K_h, size_t K_w, size_t stride, size_t pad) {
    memset(out, 0, N*C_in*H_in*W_in*sizeof(double));
    #pragma omp parallel for schedule(static)
    for(size_t ni=0; ni<N; ni++) {
        for(size_t co=0; co<C_out; co++) {
            for(size_t ho=0; ho<H_out; ho++) {
                for(size_t wo=0; wo<W_out; wo++) {
                    size_t grad_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    double g = grad[grad_idx];
                    for(size_t ci=0; ci<C_in; ci++) {
                        for(size_t kh=0; kh<K_h; kh++) {
                            for(size_t kw=0; kw<K_w; kw++) {
                                int ih = (int)(ho*stride) - (int)pad + (int)kh;
                                int iw = (int)(wo*stride) - (int)pad + (int)kw;
                                if(ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                    size_t in_idx = ni*(C_in*H_in*W_in) + ci*(H_in*W_in) + ih*W_in + iw;
                                    size_t w_idx = co*(C_in*K_h*K_w) + ci*(K_h*K_w) + kh*K_w + kw;
                                    out[in_idx] += g * w[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void kernel_conv2d_bwd_w(double* out, const double* grad, const double* in,
                         size_t N, size_t C_in, size_t H_in, size_t W_in,
                         size_t C_out, size_t H_out, size_t W_out,
                         size_t K_h, size_t K_w, size_t stride, size_t pad) {
    size_t w_elems = C_out*C_in*K_h*K_w;
    memset(out, 0, w_elems * sizeof(double));

    // TITANIUM THREAD-LOCAL ACCUMULATOR: Prevents OpenMP race conditions on weight gradients
    #pragma omp parallel
    {
        double* local_w = (double*)calloc(w_elems, sizeof(double));
        if(local_w) {
            #pragma omp for collapse(2) schedule(static)
            for(size_t ni=0; ni<N; ni++) {
                for(size_t co=0; co<C_out; co++) {
                    for(size_t ho=0; ho<H_out; ho++) {
                        for(size_t wo=0; wo<W_out; wo++) {
                            size_t grad_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                            double g = grad[grad_idx];
                            for(size_t ci=0; ci<C_in; ci++) {
                                for(size_t kh=0; kh<K_h; kh++) {
                                    for(size_t kw=0; kw<K_w; kw++) {
                                        int ih = (int)(ho*stride) - (int)pad + (int)kh;
                                        int iw = (int)(wo*stride) - (int)pad + (int)kw;
                                        if(ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                            size_t in_idx = ni*(C_in*H_in*W_in) + ci*(H_in*W_in) + ih*W_in + iw;
                                            size_t w_idx = co*(C_in*K_h*K_w) + ci*(K_h*K_w) + kh*K_w + kw;
                                            local_w[w_idx] += g * in[in_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            {
                for(size_t i=0; i<w_elems; i++) out[i] += local_w[i];
            }
            free(local_w);
        }
    }
}


// V9 FUSION: Conv2D + ReLU
void kernel_conv2d_relu_fwd(double* out, const double* in, const double* w,
                            size_t N, size_t C_in, size_t H_in, size_t W_in,
                            size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad) {
    size_t H_out = (H_in + 2*pad - K_h)/stride + 1;
    size_t W_out = (W_in + 2*pad - K_w)/stride + 1;
    memset(out, 0, N*C_out*H_out*W_out*sizeof(double));

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t ni=0; ni<N; ni++) {
        for(size_t co=0; co<C_out; co++) {
            for(size_t ho=0; ho<H_out; ho++) {
                for(size_t wo=0; wo<W_out; wo++) {
                    double sum = 0.0;
                    for(size_t ci=0; ci<C_in; ci++) {
                        for(size_t kh=0; kh<K_h; kh++) {
                            for(size_t kw=0; kw<K_w; kw++) {
                                int ih = (int)(ho*stride) - (int)pad + (int)kh;
                                int iw = (int)(wo*stride) - (int)pad + (int)kw;
                                if(ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                    size_t in_idx = ni*(C_in*H_in*W_in) + ci*(H_in*W_in) + ih*W_in + iw;
                                    size_t w_idx = co*(C_in*K_h*K_w) + ci*(K_h*K_w) + kh*K_w + kw;
                                    sum += in[in_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    // V9 FUSION: Apply ReLU immediately before writing to memory
                    size_t out_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    out[out_idx] = sum > 0.0 ? sum : 0.0;
                }
            }
        }
    }
}


// V11 INT8 FUSION: int8 * int8 -> int32 accumulation
void kernel_conv2d_int8_fwd(double* out, const int8_t* in, const int8_t* w, double scale_in, double scale_w,
                            size_t N, size_t C_in, size_t H_in, size_t W_in,
                            size_t C_out, size_t K_h, size_t K_w, size_t stride, size_t pad) {
    size_t H_out = (H_in + 2*pad - K_h)/stride + 1;
    size_t W_out = (W_in + 2*pad - K_w)/stride + 1;
    memset(out, 0, N*C_out*H_out*W_out*sizeof(double));
    double final_scale = scale_in * scale_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t ni=0; ni<N; ni++) {
        for(size_t co=0; co<C_out; co++) {
            for(size_t ho=0; ho<H_out; ho++) {
                for(size_t wo=0; wo<W_out; wo++) {
                    int32_t sum = 0; // 32-bit accumulator prevents overflow
                    for(size_t ci=0; ci<C_in; ci++) {
                        for(size_t kh=0; kh<K_h; kh++) {
                            for(size_t kw=0; kw<K_w; kw++) {
                                int ih = (int)(ho*stride) - (int)pad + (int)kh;
                                int iw = (int)(wo*stride) - (int)pad + (int)kw;
                                if(ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                    size_t in_idx = ni*(C_in*H_in*W_in) + ci*(H_in*W_in) + ih*W_in + iw;
                                    size_t w_idx = co*(C_in*K_h*K_w) + ci*(K_h*K_w) + kh*K_w + kw;
                                    sum += (int32_t)in[in_idx] * (int32_t)w[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    out[out_idx] = (double)sum * final_scale;
                }
            }
        }
    }
}


// =====================================================================
// V13 TRANSFORMER MANDATE: LLM MATH KERNELS
// =====================================================================

void kernel_layernorm(double* out, const double* in, const double* gamma, const double* beta,
                      size_t batch_size, size_t hidden_size, double eps) {
    for(size_t b=0; b<batch_size; b++) {
        const double* x = in + b * hidden_size;
        double* y = out + b * hidden_size;

        double mean = 0.0;
        for(size_t i=0; i<hidden_size; i++) mean += x[i];
        mean /= hidden_size;

        double var = 0.0;
        for(size_t i=0; i<hidden_size; i++) var += (x[i] - mean) * (x[i] - mean);
        var /= hidden_size;

        double inv_std = 1.0 / sqrt(var + eps);
        for(size_t i=0; i<hidden_size; i++) {
            y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

void kernel_gelu(double* out, const double* in, size_t elements) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const double sqrt_2_over_pi = 0.7978845608028654;
    for(size_t i=0; i<elements; i++) {
        double x = in[i];
        out[i] = 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
    }
}

void kernel_rope(double* q, double* k, size_t batch_size, size_t seq_len, size_t n_heads, size_t head_dim, int pos_offset) {
    // Rotary Position Embedding (RoPE)
    for(size_t b=0; b<batch_size; b++) {
        for(size_t s=0; s<seq_len; s++) {
            int pos = s + pos_offset;
            for(size_t h=0; h<n_heads; h++) {
                for(size_t d=0; d<head_dim; d+=2) {
                    double freq = 1.0 / pow(10000.0, (double)d / (double)head_dim);
                    double angle = pos * freq;
                    double cos_val = cos(angle);
                    double sin_val = sin(angle);

                    size_t idx = b*(seq_len*n_heads*head_dim) + s*(n_heads*head_dim) + h*head_dim + d;
                    double q0 = q[idx], q1 = q[idx+1];
                    double k0 = k[idx], k1 = k[idx+1];

                    q[idx]   = q0 * cos_val - q1 * sin_val;
                    q[idx+1] = q0 * sin_val + q1 * cos_val;
                    k[idx]   = k0 * cos_val - k1 * sin_val;
                    k[idx+1] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
    }
}


// =====================================================================
// V13 TRANSFORMER MANDATE: MULTI-HEAD ATTENTION (CAUSAL)
// =====================================================================
void kernel_attention(double* out, const double* q, const double* k, const double* v,
                      size_t seq_len, size_t n_heads, size_t head_dim) {
    double scale = 1.0 / sqrt((double)head_dim);
    size_t hidden_size = n_heads * head_dim;

    double* scores = (double*)malloc(seq_len * seq_len * sizeof(double));
    if (!scores) return;

    for(size_t h=0; h<n_heads; h++) {
        // 1. Q * K^T
        for(size_t i=0; i<seq_len; i++) {
            for(size_t j=0; j<seq_len; j++) {
                double sum = 0.0;
                for(size_t d=0; d<head_dim; d++) {
                    size_t q_idx = i * hidden_size + h * head_dim + d;
                    size_t k_idx = j * hidden_size + h * head_dim + d;
                    sum += q[q_idx] * k[k_idx];
                }
                scores[i * seq_len + j] = sum * scale;

                // CAUSAL MASK: Cannot attend to future tokens
                if (j > i) scores[i * seq_len + j] = -1e9;
            }
        }

        // 2. Softmax over rows (j)
        for(size_t i=0; i<seq_len; i++) {
            double max_val = -1e9;
            for(size_t j=0; j<=i; j++) {
                if(scores[i*seq_len + j] > max_val) max_val = scores[i*seq_len + j];
            }
            double sum_exp = 0.0;
            for(size_t j=0; j<=i; j++) {
                scores[i*seq_len + j] = exp(scores[i*seq_len + j] - max_val);
                sum_exp += scores[i*seq_len + j];
            }
            for(size_t j=0; j<=i; j++) scores[i*seq_len + j] /= sum_exp;
            for(size_t j=i+1; j<seq_len; j++) scores[i*seq_len + j] = 0.0;
        }

        // 3. Scores * V
        for(size_t i=0; i<seq_len; i++) {
            for(size_t d=0; d<head_dim; d++) {
                double sum = 0.0;
                for(size_t j=0; j<seq_len; j++) {
                    size_t v_idx = j * hidden_size + h * head_dim + d;
                    sum += scores[i * seq_len + j] * v[v_idx];
                }
                size_t out_idx = i * hidden_size + h * head_dim + d;
                out[out_idx] = sum;
            }
        }
    }
    free(scores);
}


// =====================================================================
// V13 TRANSFORMER MANDATE: AUTOREGRESSIVE KV-CACHE ATTENTION
// =====================================================================
void kernel_attention_kv_cache(double* out, const double* q, const double* k_cache, const double* v_cache,
                               size_t seq_len, size_t n_heads, size_t head_dim) {
    double scale = 1.0 / sqrt((double)head_dim);
    size_t hidden_size = n_heads * head_dim;

    memset(out, 0, hidden_size * sizeof(double));
    double* scores = (double*)malloc(seq_len * sizeof(double));

    for(size_t h=0; h<n_heads; h++) {
        double max_val = -1e9;
        // 1. Q (1 x head_dim) * K_cache^T (head_dim x seq_len)
        for(size_t j=0; j<seq_len; j++) {
            double sum = 0.0;
            for(size_t d=0; d<head_dim; d++) {
                size_t q_idx = h * head_dim + d;
                size_t k_idx = j * hidden_size + h * head_dim + d;
                sum += q[q_idx] * k_cache[k_idx];
            }
            scores[j] = sum * scale;
            if (scores[j] > max_val) max_val = scores[j];
        }

        // 2. Softmax
        double sum_exp = 0.0;
        for(size_t j=0; j<seq_len; j++) {
            scores[j] = exp(scores[j] - max_val);
            sum_exp += scores[j];
        }
        for(size_t j=0; j<seq_len; j++) scores[j] /= sum_exp;

        // 3. Scores (1 x seq_len) * V_cache (seq_len x head_dim)
        for(size_t d=0; d<head_dim; d++) {
            double sum = 0.0;
            for(size_t j=0; j<seq_len; j++) {
                size_t v_idx = j * hidden_size + h * head_dim + d;
                sum += scores[j] * v_cache[v_idx];
            }
            out[h * head_dim + d] = sum;
        }
    }
    free(scores);
}

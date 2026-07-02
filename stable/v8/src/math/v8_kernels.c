#include "basis/v8_kernels.h"
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

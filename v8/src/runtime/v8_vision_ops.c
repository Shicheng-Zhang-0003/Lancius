#include "basis/v8_ir.h"
#include "basis/v8_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void v8_execute_vision_op(v8_node* n) {
    if (!n || !n->runtime_data) return;

    if (n->op == V8_OP_CONV2D) {
        double* in = n->inputs[0]->runtime_data;
        double* w = n->inputs[1]->runtime_data;
        if (!in || !w) return;
        const v8_node* in_node = n->inputs[0];
        const v8_node* w_node = n->inputs[1];
        kernel_conv2d_fwd(n->runtime_data, in, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);
    }
    else if (n->op == V8_OP_MAXPOOL2D) {
        double* in = n->inputs[0]->runtime_data;
        if (!in) return;
        const v8_node* in_node = n->inputs[0];
        size_t N = in_node->shape[0], C = in_node->shape[1], H_in = in_node->shape[2], W_in = in_node->shape[3];
        size_t K = n->kernel_h;
        size_t stride = n->stride;
        size_t H_out = (H_in - K) / stride + 1;
        size_t W_out = (W_in - K) / stride + 1;

        #pragma omp parallel for collapse(2) schedule(static)
        for(size_t ni=0; ni<N; ni++) {
            for(size_t c=0; c<C; c++) {
                for(size_t ho=0; ho<H_out; ho++) {
                    for(size_t wo=0; wo<W_out; wo++) {
                        double max_val = -1e9;
                        for(size_t kh=0; kh<K; kh++) {
                            for(size_t kw=0; kw<K; kw++) {
                                size_t ih = ho*stride + kh;
                                size_t iw = wo*stride + kw;
                                size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + ih*W_in + iw;
                                if (in[in_idx] > max_val) max_val = in[in_idx];
                            }
                        }
                        size_t out_idx = ni*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        n->runtime_data[out_idx] = max_val;
                    }
                }
            }
        }
    }
    else if (n->op == V8_OP_FLATTEN) {
        double* in = n->inputs[0]->runtime_data;
        if (!in) return;
        size_t elems = v8_node_elements(n);
        memcpy(n->runtime_data, in, elems * sizeof(double));
    }
    else if (n->op == V8_OP_CONV2D_BWD) {
        double* grad = n->inputs[0]->runtime_data;
        double* w = n->inputs[2]->runtime_data;
        if (!grad || !w) return;
        const v8_node* in_node = n->inputs[1];
        const v8_node* grad_node = n->inputs[0];
        const v8_node* w_node = n->inputs[2];
        kernel_conv2d_bwd_in(n->runtime_data, grad, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], grad_node->shape[2], grad_node->shape[3],
            w_node->shape[2], w_node->shape[3], n->stride, n->pad);
    }
    else if (n->op == V8_OP_CONV2D_BWD_W) {
        double* grad = n->inputs[0]->runtime_data;
        double* in = n->inputs[1]->runtime_data;
        if (!grad || !in) return;
        const v8_node* in_node = n->inputs[1];
        const v8_node* grad_node = n->inputs[0];
        kernel_conv2d_bwd_w(n->runtime_data, grad, in,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            n->shape[0], grad_node->shape[2], grad_node->shape[3],
            n->shape[2], n->shape[3], n->stride, n->pad);
    }
    else if (n->op == V8_OP_MAXPOOL2D_BWD) {
        double* grad = n->inputs[0]->runtime_data;
        double* fwd_in = n->inputs[1]->runtime_data;
        if (!grad || !fwd_in) return;
        const v8_node* in_node = n->inputs[1];
        const v8_node* grad_node = n->inputs[0];

        size_t N = in_node->shape[0], C = in_node->shape[1], H_in = in_node->shape[2], W_in = in_node->shape[3];
        size_t K = n->kernel_h;
        size_t stride = n->stride;
        size_t H_out = grad_node->shape[2], W_out = grad_node->shape[3];

        memset(n->runtime_data, 0, N*C*H_in*W_in*sizeof(double));

        #pragma omp parallel for schedule(static)
        for(size_t ni=0; ni<N; ni++) {
            for(size_t c=0; c<C; c++) {
                for(size_t ho=0; ho<H_out; ho++) {
                    for(size_t wo=0; wo<W_out; wo++) {
                        size_t grad_idx = ni*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        double g = grad[grad_idx];

                        double max_val = -1e9;
                        size_t max_ih = 0, max_iw = 0;
                        for(size_t kh=0; kh<K; kh++) {
                            for(size_t kw=0; kw<K; kw++) {
                                size_t ih = ho*stride + kh;
                                size_t iw = wo*stride + kw;
                                size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + ih*W_in + iw;
                                if (fwd_in[in_idx] > max_val) {
                                    max_val = fwd_in[in_idx];
                                    max_ih = ih; max_iw = iw;
                                }
                            }
                        }
                        size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + max_ih*W_in + max_iw;
                        n->runtime_data[in_idx] += g;
                    }
                }
            }
        }
    }
}

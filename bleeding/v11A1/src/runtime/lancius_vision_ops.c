#include "lancius/lancius_ir.h"
#include "lancius/lancius_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

void lancius_execute_vision_op(lancius_node* n) {
    if (!n || !n->runtime_data) return;

    /* v11A1: fail loudly on unhandled vision-range ops */
    switch (n->op) {
        case LANCIUS_OP_CONV2D:
        case LANCIUS_OP_CONV2D_RELU_FUSED:
        case LANCIUS_OP_MAXPOOL2D:
        case LANCIUS_OP_FLATTEN:
        case LANCIUS_OP_RESHAPE:
        case LANCIUS_OP_CONV2D_BWD:
        case LANCIUS_OP_CONV2D_BWD_W:
        case LANCIUS_OP_MAXPOOL2D_BWD:
            break;

        default:
            fprintf(stderr,
                "[LANCIUS VISION FATAL] unsupported vision-range op %d\n",
                (int)n->op);
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            abort();
    }

    // v11A1 Task 9: no hidden execution-time quantization side effects.
    // INT8 Conv2D runs only when both activation and weight INT8 buffers already exist.
    bool can_use_int8 =
        n->input_count >= 2 &&
        n->inputs[0] &&
        n->inputs[1] &&
        n->inputs[0]->runtime_data_int8 &&
        n->inputs[1]->runtime_data_int8;

    if ((n->op == LANCIUS_OP_CONV2D || n->op == LANCIUS_OP_CONV2D_RELU_FUSED) && can_use_int8) {
        const lancius_node* in_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[1];

        const int8_t* in = in_node->runtime_data_int8;
        const int8_t* w = w_node->runtime_data_int8;

        if (!in || !w) return;

        double scale_in = (in_node->scale != 0.0) ? in_node->scale : 1e-8;
        double scale_w = (w_node->scale != 0.0) ? w_node->scale : 1e-8;

        kernel_conv2d_int8_fwd(n->runtime_data, in, w, scale_in, scale_w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);

        if (n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            size_t elems = lancius_node_elements(n);
            for (size_t i = 0; i < elems; i++) {
                if (n->runtime_data[i] < 0.0) n->runtime_data[i] = 0.0;
            }
        }

        return;
    }

    // If a Conv2D weight is INT8-only but activations are not INT8, fail loudly.
    if ((n->op == LANCIUS_OP_CONV2D || n->op == LANCIUS_OP_CONV2D_RELU_FUSED) &&
        n->input_count >= 2 && n->inputs[1] &&
        n->inputs[1]->dtype == LANCIUS_DTYPE_INT8 &&
        !n->inputs[1]->runtime_data) {
        fprintf(stderr, "[LANCIUS VISION FATAL] INT8 Conv2D requires explicit INT8 activation buffers in v11A1\n");
        abort();
    }

    // Standard FP64 Routing

    if (n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
        double* in = n->inputs[0]->runtime_data;
        double* w = n->inputs[1]->runtime_data;
        if (!in || !w) return;
        const lancius_node* in_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[1];
        kernel_conv2d_relu_fwd(n->runtime_data, in, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_CONV2D) {
        double* in = n->inputs[0]->runtime_data;
        double* w = n->inputs[1]->runtime_data;
        if (!in || !w) return;
        const lancius_node* in_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[1];
        kernel_conv2d_fwd(n->runtime_data, in, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_MAXPOOL2D) {
        double* in = n->inputs[0]->runtime_data;
        if (!in) return;
        const lancius_node* in_node = n->inputs[0];
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
    else if (n->op == LANCIUS_OP_FLATTEN) {
        double* in = n->inputs[0]->runtime_data;
        if (!in) return;
        size_t elems = lancius_node_elements(n);
        memcpy(n->runtime_data, in, elems * sizeof(double));
    }
    else if (n->op == LANCIUS_OP_RESHAPE) {
        // V9 Fix: Route RESHAPE backward pass (Flatten gradient)
        double* in = n->inputs[0]->runtime_data;
        if (!in) return;
        size_t elems = lancius_node_elements(n);
        memcpy(n->runtime_data, in, elems * sizeof(double));
    }
    else if (n->op == LANCIUS_OP_CONV2D_BWD) {
        double* grad = n->inputs[0]->runtime_data;
        double* w = n->inputs[2]->runtime_data;
        if (!grad || !w) return;
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[2];
        kernel_conv2d_bwd_in(n->runtime_data, grad, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], grad_node->shape[2], grad_node->shape[3],
            w_node->shape[2], w_node->shape[3], n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_CONV2D_BWD_W) {
        double* grad = n->inputs[0]->runtime_data;
        double* in = n->inputs[1]->runtime_data;
        if (!grad || !in) return;
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];
        kernel_conv2d_bwd_w(n->runtime_data, grad, in,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            n->shape[0], grad_node->shape[2], grad_node->shape[3],
            n->shape[2], n->shape[3], n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_MAXPOOL2D_BWD) {
        double* grad = n->inputs[0]->runtime_data;
        double* fwd_in = n->inputs[1]->runtime_data;
        if (!grad || !fwd_in) return;
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];

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

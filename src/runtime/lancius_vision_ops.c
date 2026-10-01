#include "lancius/lancius_ir.h"
#include "lancius/lancius_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

void lancius_execute_vision_op(lancius_node* n) {
    if (!n || !n->runtime_data) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
    /* Despot truth: n->inputs[0] with inputs==NULL is NULL+0 deref. */
    if (!n->inputs && n->input_count > 0) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }

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
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            return;
    }

    // v11A1 Task 9: no hidden execution-time quantization side effects.
    // INT8 Conv2D runs only when both activation and weight INT8 buffers already exist.
    bool can_use_int8 =
        n->inputs &&
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

        if (!in || !w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }

        double scale_in = in_node->scale;
        double scale_w = w_node->scale;
        /* Despot truth: zero INT8 scale is numerically degenerate (dequant
         * would collapse to zeros). Report NUMERICAL, not shape mismatch. */
        if (!(scale_in > 0.0) || !(scale_w > 0.0) || scale_in != scale_in || scale_w != scale_w) {
            lancius_set_error(LANCIUS_ERROR_NUMERICAL);
            return;
        }

        kernel_conv2d_int8_fwd(n->runtime_data, in, w, scale_in, scale_w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);

        if (n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            size_t elems = 0;
            if (!lancius_node_elements_checked(n, &elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
            for (size_t i = 0; i < elems; i++) {
                if (n->runtime_data[i] < 0.0) n->runtime_data[i] = 0.0;
            }
        }

        return;
    }

    // If a Conv2D weight is INT8-only but activations are not INT8, fail loudly.
    if ((n->op == LANCIUS_OP_CONV2D || n->op == LANCIUS_OP_CONV2D_RELU_FUSED) &&
        n->inputs && n->input_count >= 2 && n->inputs[1] &&
        n->inputs[1]->dtype == LANCIUS_DTYPE_INT8 &&
        !n->inputs[1]->runtime_data) {
        lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_DTYPE);
        return;
    }

    // Standard FP64 Routing

    /* Despot truth: every branch validates input_count/inputs first; expected
     * H_out/W_out are verified against n->shape so a corrupt shape cannot
     * OOB-write past n->runtime_data sized from n->shape. */
    if (n->op == LANCIUS_OP_CONV2D_RELU_FUSED) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        double* w = n->inputs[1]->runtime_data;
        if (!in || !w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[1];
        if (in_node->ndim != 4 || w_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        {
            size_t H_in = in_node->shape[2], W_in = in_node->shape[3];
            size_t K_h = w_node->shape[2], K_w = w_node->shape[3];
            if (n->stride == 0 || K_h == 0 || K_w == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_STRIDE); return; }
            if (n->pad <= (SIZE_MAX - H_in) / 2 && n->pad <= (SIZE_MAX - W_in) / 2 &&
                H_in + 2 * (size_t)n->pad >= K_h && W_in + 2 * (size_t)n->pad >= K_w) {
                size_t eH = (H_in + 2 * (size_t)n->pad - K_h) / n->stride + 1;
                size_t eW = (W_in + 2 * (size_t)n->pad - K_w) / n->stride + 1;
                if (eH != n->shape[2] || eW != n->shape[3]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            } else { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        }
        kernel_conv2d_relu_fwd(n->runtime_data, in, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_CONV2D) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        double* w = n->inputs[1]->runtime_data;
        if (!in || !w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[1];
        if (in_node->ndim != 4 || w_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        {
            size_t H_in = in_node->shape[2], W_in = in_node->shape[3];
            size_t K_h = w_node->shape[2], K_w = w_node->shape[3];
            if (n->stride == 0 || K_h == 0 || K_w == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_STRIDE); return; }
            if (n->pad <= (SIZE_MAX - H_in) / 2 && n->pad <= (SIZE_MAX - W_in) / 2 &&
                H_in + 2 * (size_t)n->pad >= K_h && W_in + 2 * (size_t)n->pad >= K_w) {
                size_t eH = (H_in + 2 * (size_t)n->pad - K_h) / n->stride + 1;
                size_t eW = (W_in + 2 * (size_t)n->pad - K_w) / n->stride + 1;
                if (eH != n->shape[2] || eW != n->shape[3]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            } else { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        }
        kernel_conv2d_fwd(n->runtime_data, in, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], w_node->shape[2], w_node->shape[3],
            n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_MAXPOOL2D) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[0];
        if (in_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        size_t N = in_node->shape[0], C = in_node->shape[1], H_in = in_node->shape[2], W_in = in_node->shape[3];
        size_t K = n->kernel_h;
        size_t stride = n->stride;
        if (stride == 0 || K == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_STRIDE); return; }
        if (H_in < K || W_in < K) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
        size_t H_out = (H_in - K) / stride + 1;
        size_t W_out = (W_in - K) / stride + 1;
        if (H_out != n->shape[2] || W_out != n->shape[3]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        /* Despot truth: ih=in_idx products must not wrap. */
        if (C && H_in > SIZE_MAX / W_in) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }

        #pragma omp parallel for collapse(2) schedule(static)
        for(size_t ni=0; ni<N; ni++) {
            for(size_t c=0; c<C; c++) {
                for(size_t ho=0; ho<H_out; ho++) {
                    for(size_t wo=0; wo<W_out; wo++) {
                        double max_val = -INFINITY;
                        for(size_t kh=0; kh<K; kh++) {
                            for(size_t kw=0; kw<K; kw++) {
                                size_t ih = ho*stride + kh;
                                size_t iw = wo*stride + kw;
                                size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + ih*W_in + iw;
                                double v = in[in_idx];
                                if (v != v) { max_val = v; break; }
                                if (v > max_val) max_val = v;
                            }
                            if (max_val != max_val) break;
                        }
                        size_t out_idx = ni*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        n->runtime_data[out_idx] = max_val;
                    }
                }
            }
        }
    }
    else if (n->op == LANCIUS_OP_FLATTEN) {
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t elems = 0, in_elems = 0;
        if (!lancius_node_elements_checked(n, &elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (!lancius_node_elements_checked(n->inputs[0], &in_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (elems != in_elems) { lancius_set_error(LANCIUS_ERROR_RESHAPE_MISMATCH); return; }
        if (elems > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memcpy(n->runtime_data, in, elems * sizeof(double));
    }
    else if (n->op == LANCIUS_OP_RESHAPE) {
        // V9 Fix: Route RESHAPE backward pass (Flatten gradient)
        if (!n->inputs || n->input_count < 1 || !n->inputs[0]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* in = n->inputs[0]->runtime_data;
        if (!in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        size_t elems = 0, in_elems = 0;
        if (!lancius_node_elements_checked(n, &elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (!lancius_node_elements_checked(n->inputs[0], &in_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (elems != in_elems) { lancius_set_error(LANCIUS_ERROR_RESHAPE_MISMATCH); return; }
        if (elems > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memcpy(n->runtime_data, in, elems * sizeof(double));
    }
    else if (n->op == LANCIUS_OP_CONV2D_BWD) {
        if (!n->inputs || n->input_count < 3 || !n->inputs[0] || !n->inputs[1] || !n->inputs[2]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* grad = n->inputs[0]->runtime_data;
        double* w = n->inputs[2]->runtime_data;
        if (!grad || !w) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];
        const lancius_node* w_node = n->inputs[2];
        /* Despot V6 truth: mirror FWD rank/stride/H_out checks (was unchecked). */
        if (in_node->ndim != 4 || grad_node->ndim != 4 || w_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        if (n->stride == 0 || w_node->shape[2] == 0 || w_node->shape[3] == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_STRIDE); return; }
        {
            size_t H_in = in_node->shape[2], W_in = in_node->shape[3];
            size_t K_h = w_node->shape[2], K_w = w_node->shape[3];
            if (n->pad > (SIZE_MAX - H_in) / 2 || n->pad > (SIZE_MAX - W_in) / 2) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
            if (H_in + 2 * (size_t)n->pad < K_h || W_in + 2 * (size_t)n->pad < K_w) { lancius_set_error(LANCIUS_ERROR_INVALID_SHAPE); return; }
            size_t eH = (H_in + 2 * (size_t)n->pad - K_h) / n->stride + 1;
            size_t eW = (W_in + 2 * (size_t)n->pad - K_w) / n->stride + 1;
            if (eH != grad_node->shape[2] || eW != grad_node->shape[3]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
            if (in_node->shape[0] != grad_node->shape[0] || in_node->shape[0] != n->shape[0]) { lancius_set_error(LANCIUS_ERROR_SHAPE_MISMATCH); return; }
        }
        kernel_conv2d_bwd_in(n->runtime_data, grad, w,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            w_node->shape[0], grad_node->shape[2], grad_node->shape[3],
            w_node->shape[2], w_node->shape[3], n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_CONV2D_BWD_W) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* grad = n->inputs[0]->runtime_data;
        double* in = n->inputs[1]->runtime_data;
        if (!grad || !in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];
        /* Despot V6 truth: rank/stride guards (was unchecked). */
        if (in_node->ndim != 4 || grad_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }
        if (n->stride == 0 || n->shape[2] == 0 || n->shape[3] == 0) { lancius_set_error(LANCIUS_ERROR_INVALID_STRIDE); return; }
        kernel_conv2d_bwd_w(n->runtime_data, grad, in,
            in_node->shape[0], in_node->shape[1], in_node->shape[2], in_node->shape[3],
            n->shape[0], grad_node->shape[2], grad_node->shape[3],
            n->shape[2], n->shape[3], n->stride, n->pad);
    }
    else if (n->op == LANCIUS_OP_MAXPOOL2D_BWD) {
        if (!n->inputs || n->input_count < 2 || !n->inputs[0] || !n->inputs[1]) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); return; }
        double* grad = n->inputs[0]->runtime_data;
        double* fwd_in = n->inputs[1]->runtime_data;
        if (!grad || !fwd_in) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return; }
        const lancius_node* in_node = n->inputs[1];
        const lancius_node* grad_node = n->inputs[0];
        if (in_node->ndim != 4 || grad_node->ndim != 4 || n->ndim != 4) { lancius_set_error(LANCIUS_ERROR_INVALID_RANK); return; }

        size_t N = in_node->shape[0], C = in_node->shape[1], H_in = in_node->shape[2], W_in = in_node->shape[3];
        size_t K = n->kernel_h;
        size_t stride = n->stride;
        size_t H_out = grad_node->shape[2], W_out = grad_node->shape[3];

        /* Despot truth: output byte size is checked; corrupt shapes fail loud. */
        size_t out_elems = 0;
        if (!lancius_node_elements_checked(n, &out_elems)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return; }
        if (out_elems > SIZE_MAX / sizeof(double)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return; }
        memset(n->runtime_data, 0, out_elems * sizeof(double));

        /* v12R1 fix: with stride < K, overlapping pooling windows scatter
         * several output positions into the same input element, so the
         * gradient scatter `runtime_data[in_idx] += g` is a data race when
         * parallelized. Each thread now accumulates into a private buffer
         * and the partial sums are reduced under a critical section. */
        int omp_alloc_failed = 0;
        #pragma omp parallel
        {
            double* local_acc = (double*)calloc(out_elems ? out_elems : 1, sizeof(double));
            if (!local_acc) {
                #pragma omp atomic write
                omp_alloc_failed = 1;
            } else {
                #pragma omp for schedule(static)
                for(size_t ni=0; ni<N; ni++) {
                    for(size_t c=0; c<C; c++) {
                        for(size_t ho=0; ho<H_out; ho++) {
                            for(size_t wo=0; wo<W_out; wo++) {
                                size_t grad_idx = ni*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                                double g = grad[grad_idx];

                                double max_val = -INFINITY;
                                size_t max_ih = 0, max_iw = 0;
                                for(size_t kh=0; kh<K; kh++) {
                                    for(size_t kw=0; kw<K; kw++) {
                                        size_t ih = ho*stride + kh;
                                        size_t iw = wo*stride + kw;
                                        size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + ih*W_in + iw;
                                        double v = fwd_in[in_idx];
                                        if (v != v) { max_val = v; max_ih = ih; max_iw = iw; break; }
                                        if (v > max_val) {
                                            max_val = v;
                                            max_ih = ih; max_iw = iw;
                                        }
                                    }
                                    if (max_val != max_val) break;
                                }
                                size_t in_idx = ni*(C*H_in*W_in) + c*(H_in*W_in) + max_ih*W_in + max_iw;
                                local_acc[in_idx] += g;
                            }
                        }
                    }
                }
                #pragma omp critical
                {
                    for (size_t i = 0; i < out_elems; i++) n->runtime_data[i] += local_acc[i];
                }
                free(local_acc);
            }
        }
        if (omp_alloc_failed) { lancius_set_error(LANCIUS_ERROR_OOM); return; }
    }
}

#include "lancius/lancius_ir.h"
#include "lancius/lancius_error.h"
#include <stdlib.h>
#include <math.h>

/* v12R1 fix: quantizer reports through the error channel, not stdout. */
void lancius_quantize_graph(lancius_graph* g) {
    if (!g) return;
    int quantized_count = 0;
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n) continue;
        // Quantize 4D Weights (Conv2D filters)
        if (n->op == LANCIUS_OP_INPUT && n->runtime_data && n->ndim == 4 && n->dtype == LANCIUS_DTYPE_FP64) {
            size_t elems = 0;
            if (!lancius_node_elements_checked(n, &elems) || elems == 0) continue;
            double max_val = 0.0;
            for(size_t j=0; j<elems; j++) {
                double abs_v = fabs(n->runtime_data[j]);
                if (abs_v > max_val) max_val = abs_v;
            }
            /* Despot truth: all-zero stays FP64; non-finite max (Inf/NaN)
             * would make scale Inf and (int8_t)NaN UB — skip loud. */
            if (!(max_val > 0.0) || !isfinite(max_val)) continue;
            n->scale = max_val / 127.0;
            /* Despot truth: overwriting a prior owned int8 buffer leaks it. */
            if (n->runtime_data_int8 && n->rt && n->rt->int8_owner == LANCIUS_MEMORY_OWNED_HEAP) {
                free(n->runtime_data_int8);
                n->runtime_data_int8 = NULL;
                n->rt->buffer_int8 = NULL;
                n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
            }

            int8_t* q = (int8_t*)malloc(elems ? elems : 1);
            if (!q) continue;
            for(size_t j=0; j<elems; j++) {
                double v = round(n->runtime_data[j] / n->scale);
                if (v > 127.0) v = 127.0;
                if (v < -128.0) v = -128.0;
                q[j] = (int8_t)v;
            }
            n->runtime_data_int8 = q;
            n->dtype = LANCIUS_DTYPE_INT8;
            /* A2: quantized INT8 weights are owned heap buffers */
            lancius_node_bind_owned_heap_int8(n, n->runtime_data_int8);
            /* Despot truth: rt->scale was left stale at 1.0, mis-scaling any
             * rt-based dequant path. Sync it. */
            if (n->rt) n->rt->scale = n->scale;
            lancius_runtime_sync_from_legacy(n);
            quantized_count++;
        }
    }
    (void)quantized_count;
    lancius_set_error(LANCIUS_ERROR_OK);
}

/*
 * v12R1: per-channel INT8 quantization for 4D Conv2D weights.
 * One scale per output channel (shape[0]); scales live in
 * rt->scale_per_channel so dequantization can restore each channel exactly.
 */
void lancius_quantize_graph_per_channel(lancius_graph* g) {
    if (!g) return;
    int quantized_count = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n) continue;
        if (n->op != LANCIUS_OP_INPUT || !n->runtime_data || n->ndim != 4 ||
            n->dtype != LANCIUS_DTYPE_FP64) continue;

        size_t elems = 0;
        if (!lancius_node_elements_checked(n, &elems) || elems == 0) continue;
        size_t out_c = n->shape[0];
        if (out_c == 0 || out_c > elems || elems % out_c != 0) continue;
        size_t per_channel = elems / out_c;

        double* scales = (double*)malloc(out_c * sizeof(double));
        if (!scales) continue;

        int channel_ok = 1;
        for (size_t c = 0; c < out_c && channel_ok; c++) {
            double max_val = 0.0;
            for (size_t j = 0; j < per_channel; j++) {
                double abs_v = fabs(n->runtime_data[c * per_channel + j]);
                if (abs_v > max_val) max_val = abs_v;
            }
            /* All-zero / non-finite channel: skip the whole tensor loud. */
            if (!(max_val > 0.0) || !isfinite(max_val)) { channel_ok = 0; break; }
            scales[c] = max_val / 127.0;
        }
        if (!channel_ok) { free(scales); continue; }

        /* Despot truth: overwriting a prior owned int8 buffer leaks it. */
        if (n->runtime_data_int8 && n->rt && n->rt->int8_owner == LANCIUS_MEMORY_OWNED_HEAP) {
            free(n->runtime_data_int8);
            n->runtime_data_int8 = NULL;
            n->rt->buffer_int8 = NULL;
            n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
        }
        /* Drop any stale per-channel scales before installing new ones. */
        if (n->rt && n->rt->scale_per_channel) {
            free(n->rt->scale_per_channel);
            n->rt->scale_per_channel = NULL;
            n->rt->scale_channels = 0;
        }

        int8_t* q = (int8_t*)malloc(elems ? elems : 1);
        if (!q) { free(scales); continue; }
        for (size_t c = 0; c < out_c; c++) {
            for (size_t j = 0; j < per_channel; j++) {
                double v = round(n->runtime_data[c * per_channel + j] / scales[c]);
                if (v > 127.0) v = 127.0;
                if (v < -128.0) v = -128.0;
                q[c * per_channel + j] = (int8_t)v;
            }
        }
        n->runtime_data_int8 = q;
        n->dtype = LANCIUS_DTYPE_INT8;
        lancius_node_bind_owned_heap_int8(n, n->runtime_data_int8);
        if (n->rt) {
            n->rt->scale_per_channel = scales;
            n->rt->scale_channels = (uint32_t)out_c;
            /* Tensor-level scale mirrors the largest channel scale (info/compat). */
            double smax = 0.0;
            for (size_t c = 0; c < out_c; c++) if (scales[c] > smax) smax = scales[c];
            n->scale = smax;
            n->rt->scale = smax;
        } else {
            free(scales);
        }
        lancius_runtime_sync_from_legacy(n);
        quantized_count++;
    }
    (void)quantized_count;
    lancius_set_error(LANCIUS_ERROR_OK);
}

/*
 * v12R1: dequantize INT8 tensors back to FP64.
 * Uses per-channel scales when present (rt->scale_per_channel), otherwise
 * the per-tensor n->scale. The INT8 buffer is released and the node returns
 * to FP64 with an owned-heap dequantized copy.
 */
void lancius_dequantize_graph(lancius_graph* g) {
    if (!g) return;
    int dequantized_count = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n) continue;
        if (n->dtype != LANCIUS_DTYPE_INT8 || !n->runtime_data_int8) continue;

        size_t elems = 0;
        if (!lancius_node_elements_checked(n, &elems) || elems == 0) continue;
        if (elems > SIZE_MAX / sizeof(double)) continue;

        double* deq = (double*)malloc(elems * sizeof(double));
        if (!deq) continue;

        if (n->rt && n->rt->scale_per_channel && n->rt->scale_channels > 0) {
            size_t out_c = n->rt->scale_channels;
            if (out_c > elems || elems % out_c != 0) { free(deq); continue; }
            size_t per_channel = elems / out_c;
            for (size_t c = 0; c < out_c; c++) {
                double s = n->rt->scale_per_channel[c];
                for (size_t j = 0; j < per_channel; j++) {
                    deq[c * per_channel + j] =
                        (double)n->runtime_data_int8[c * per_channel + j] * s;
                }
            }
        } else {
            double s = n->scale;
            for (size_t j = 0; j < elems; j++) {
                deq[j] = (double)n->runtime_data_int8[j] * s;
            }
        }

        /* Release the INT8 buffer (owned heap) and per-channel scales. */
        if (n->rt && n->rt->int8_owner == LANCIUS_MEMORY_OWNED_HEAP) {
            free(n->runtime_data_int8);
        }
        n->runtime_data_int8 = NULL;
        if (n->rt) {
            n->rt->buffer_int8 = NULL;
            n->rt->int8_owner = LANCIUS_MEMORY_EXTERNAL;
            if (n->rt->scale_per_channel) {
                free(n->rt->scale_per_channel);
                n->rt->scale_per_channel = NULL;
                n->rt->scale_channels = 0;
            }
            n->rt->scale = 1.0;
        }
        n->scale = 1.0;

        n->runtime_data = deq;
        n->dtype = LANCIUS_DTYPE_FP64;
        lancius_node_bind_owned_heap(n, deq);
        lancius_runtime_sync_from_legacy(n);
        dequantized_count++;
    }
    (void)dequantized_count;
    lancius_set_error(LANCIUS_ERROR_OK);
}

#include "lancius/lancius_ir.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

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
            if (!(max_val > 0.0)) continue; /* all-zero weights: leave FP64, avoid 1e-8 lie */
            n->scale = max_val / 127.0;

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
            quantized_count++;
        }
    }
    printf("[V10S QUANTIZER] Compressed %d conv weight tensors to INT8.\n", quantized_count);
}

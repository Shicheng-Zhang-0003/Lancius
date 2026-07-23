#include "lancius/lancius_ir.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void lancius_quantize_graph(lancius_graph* g) {
    if (!g) return;
    int quantized_count = 0;
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        // Quantize 4D Weights (Conv2D filters)
        if (n->op == LANCIUS_OP_INPUT && n->runtime_data && n->ndim == 4) {
            size_t elems = lancius_node_elements(n);
            double max_val = 0.0;
            for(size_t j=0; j<elems; j++) {
                double abs_v = fabs(n->runtime_data[j]);
                if (abs_v > max_val) max_val = abs_v;
            }
            n->scale = max_val / 127.0;
            if (n->scale == 0.0) n->scale = 1e-8;

            n->runtime_data_int8 = (int8_t*)malloc(elems);
            for(size_t j=0; j<elems; j++) {
                n->runtime_data_int8[j] = (int8_t)round(n->runtime_data[j] / n->scale);
            }
            n->dtype = LANCIUS_DTYPE_INT8;
            /* A2: quantized INT8 weights are owned heap buffers */
            lancius_node_bind_owned_heap_int8(n, n->runtime_data_int8);
            quantized_count++;
        }
    }
    printf("[V10S QUANTIZER] Compressed %d weight tensors to INT8 (8x size reduction).\n", quantized_count);
}

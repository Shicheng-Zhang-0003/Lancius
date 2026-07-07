#include "lancius/lancius_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANCIUS_MAGIC 0x21434E41 // "LANC!"

// Bulletproof macros to silence GCC's aggressive warn_unused_result on fread/fwrite
#define SAFE_READ(ptr, size, nmemb, stream) do { size_t _r = fread(ptr, size, nmemb, stream); (void)_r; } while(0)
#define SAFE_WRITE(ptr, size, nmemb, stream) do { size_t _r = fwrite(ptr, size, nmemb, stream); (void)_r; } while(0)

void lancius_graph_save(lancius_graph* g, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = LANCIUS_MAGIC;
    SAFE_WRITE(&magic, sizeof(uint32_t), 1, f);
    SAFE_WRITE(&g->node_count, sizeof(uint32_t), 1, f);

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        SAFE_WRITE(&n->id, sizeof(uint32_t), 1, f);
        SAFE_WRITE(&n->op, sizeof(lancius_opcode), 1, f);
        SAFE_WRITE(&n->ndim, sizeof(uint8_t), 1, f);
        SAFE_WRITE(n->shape, sizeof(size_t), 4, f);
        SAFE_WRITE(&n->input_count, sizeof(uint32_t), 1, f);
        for(uint32_t j=0; j<n->input_count; j++) {
            SAFE_WRITE(&n->inputs[j]->id, sizeof(uint32_t), 1, f);
        }
        SAFE_WRITE(&n->attr_val, sizeof(double), 1, f);
        uint32_t meta[4] = {n->kernel_h, n->kernel_w, n->stride, n->pad};
        SAFE_WRITE(meta, sizeof(uint32_t), 4, f);
        SAFE_WRITE(n->axes, sizeof(uint32_t), 4, f);

        uint8_t has_weights = (n->op == LANCIUS_OP_INPUT && (n->runtime_data != NULL || n->runtime_data_int8 != NULL)) ? 1 : 0;
        SAFE_WRITE(&has_weights, sizeof(uint8_t), 1, f);
        if (has_weights) {
            size_t elems = lancius_node_elements(n);
                if (elems > 100000000) { fprintf(stderr, "[SERIAL FATAL] Tensor size exceeds sanity limit."); fclose(f); return; }
            uint8_t dtype = n->dtype;
            SAFE_WRITE(&dtype, sizeof(uint8_t), 1, f);
            SAFE_WRITE(&n->scale, sizeof(double), 1, f);
            if (dtype == LANCIUS_DTYPE_INT8) {
                SAFE_WRITE(n->runtime_data_int8, sizeof(int8_t), elems, f);
            } else {
                SAFE_WRITE(n->runtime_data, sizeof(double), elems, f);
            }
        }
    }
    fclose(f);
    printf("[LANCIUS SERIAL] Saved %u nodes to %s\n", g->node_count, path);
}

lancius_graph* lancius_graph_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, node_count;
    SAFE_READ(&magic, sizeof(uint32_t), 1, f);
    if (magic != LANCIUS_MAGIC) { fclose(f); return NULL; }
    if (fread(&node_count, sizeof(uint32_t), 1, f) != 1) { fprintf(stderr, "[SERIAL FATAL] Truncated file: failed to read node count.\n"); fclose(f); return NULL; }
    if (node_count > 1000000) { fprintf(stderr, "[SERIAL FATAL] Node count exceeds sanity limit.\n"); fclose(f); return NULL; }

    lancius_graph* g = lancius_graph_create();
    lancius_node** id_map = (lancius_node**)calloc(node_count + 1000, sizeof(lancius_node*));

    for (uint32_t i = 0; i < node_count; i++) {
        uint32_t id, op, input_count;
        uint8_t ndim;
        size_t shape[4];
        SAFE_READ(&id, sizeof(uint32_t), 1, f);
        SAFE_READ(&op, sizeof(lancius_opcode), 1, f);
        SAFE_READ(&ndim, sizeof(uint8_t), 1, f);
        SAFE_READ(shape, sizeof(size_t), 4, f);
        SAFE_READ(&input_count, sizeof(uint32_t), 1, f);

        uint32_t* in_ids = (uint32_t*)malloc(input_count * sizeof(uint32_t));
        SAFE_READ(in_ids, sizeof(uint32_t), input_count, f);

        double attr_val;
        SAFE_READ(&attr_val, sizeof(double), 1, f);

        uint32_t meta[4], axes[4];
        SAFE_READ(meta, sizeof(uint32_t), 4, f);
        SAFE_READ(axes, sizeof(uint32_t), 4, f);

        uint8_t has_weights;
        SAFE_READ(&has_weights, sizeof(uint8_t), 1, f);

        lancius_node* n = NULL;
        const lancius_node* in0 = input_count > 0 ? id_map[in_ids[0]] : NULL;
        const lancius_node* in1 = input_count > 1 ? id_map[in_ids[1]] : NULL;

        if (op == LANCIUS_OP_INPUT) {
            if (ndim == 4) n = lancius_input_4d(g, shape[0], shape[1], shape[2], shape[3]);
            else n = lancius_input(g, shape[0], shape[1]);
        } else if (op == LANCIUS_OP_CONST) n = lancius_const(g, attr_val, shape[0], shape[1]);
        else if (op == LANCIUS_OP_ADD) n = lancius_add(g, in0, in1);
        else if (op == LANCIUS_OP_SUB) n = lancius_sub(g, in0, in1);
        else if (op == LANCIUS_OP_MUL) n = lancius_mul(g, in0, in1);
        else if (op == LANCIUS_OP_MATMUL) n = lancius_matmul(g, in0, in1);
        else if (op == LANCIUS_OP_RELU) n = lancius_relu(g, in0);
        else if (op == LANCIUS_OP_SOFTMAX) n = lancius_softmax(g, in0);
        else if (op == LANCIUS_OP_SUM) n = lancius_sum(g, in0);
        else if (op == LANCIUS_OP_BROADCAST) n = lancius_broadcast(g, in0, shape[0], shape[1]);
        else if (op == LANCIUS_OP_TRANSPOSE) n = lancius_transpose(g, in0);
        else if (op == LANCIUS_OP_CONV2D) n = lancius_conv2d(g, in0, in1, meta[2], meta[3]);
        else if (op == LANCIUS_OP_MAXPOOL2D) n = lancius_maxpool2d(g, in0, meta[0], meta[2]);
        else if (op == LANCIUS_OP_FLATTEN) n = lancius_flatten(g, in0);
        else if (op == LANCIUS_OP_RESHAPE) n = lancius_reshape(g, in0, ndim, shape[0], shape[1], shape[2], shape[3]);
        else if (op == LANCIUS_OP_PERMUTE) n = lancius_permute(g, in0, axes[0], axes[1], axes[2], axes[3]);
        else if (op == LANCIUS_OP_MATMUL_BATCHED) n = lancius_matmul_batched(g, in0, in1);
        else if (op == LANCIUS_OP_NOP) n = NULL; // V10 Fix: Orphaned NOP
        else if (op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            n = lancius_conv2d(g, in0, in1, meta[2], meta[3]);
            if (n) n->op = LANCIUS_OP_CONV2D_RELU_FUSED; // Preserve fused opcode
        }
        else if (op == LANCIUS_OP_CROSS_ENTROPY) {
            const lancius_node* dummy_y = in1 ? in1 : lancius_const(g, 0.0, in0 ? in0->shape[0] : 1, in0 ? in0->shape[1] : 1);
            n = lancius_cross_entropy(g, in0, dummy_y);
        }
        else if (op == LANCIUS_OP_BROADCAST) n = lancius_broadcast(g, in0, shape[0], shape[1]);
        else if (op == LANCIUS_OP_SUB) n = lancius_sub(g, in0, in1);
        else if (op == LANCIUS_OP_MUL) n = lancius_mul(g, in0, in1);
        else if (op == LANCIUS_OP_SUM) n = lancius_sum(g, in0);
        else if (op == LANCIUS_OP_TRANSPOSE) n = lancius_transpose(g, in0);
        else if (op == LANCIUS_OP_SOFTMAX) n = lancius_softmax(g, in0);
        else if (op == LANCIUS_OP_RESHAPE) n = lancius_reshape(g, in0, ndim, shape[0], shape[1], shape[2], shape[3]);
        else if (op == LANCIUS_OP_SUM_AXIS0) n = lancius_sum_axis0(g, in0);
        else if (op == LANCIUS_OP_SUM_AXIS1) n = lancius_sum_axis1(g, in0);
        else { fprintf(stderr, "[LANCIUS SERIAL WARN] Unknown op %u during load, skipping\n", op); }

        if (n) {
            n->kernel_h = meta[0]; n->kernel_w = meta[1]; n->stride = meta[2]; n->pad = meta[3];
            memcpy(n->axes, axes, sizeof(uint32_t)*4);
            id_map[id] = n;

            if (has_weights) {
                size_t elems = lancius_node_elements(n);
                if (elems > 100000000) { fprintf(stderr, "[SERIAL FATAL] Tensor size exceeds sanity limit."); fclose(f); free(id_map); lancius_graph_destroy(g); return NULL; }
                uint8_t dtype;
                SAFE_READ(&dtype, sizeof(uint8_t), 1, f);
                n->dtype = (lancius_dtype)dtype;
                SAFE_READ(&n->scale, sizeof(double), 1, f);
                if (n->dtype == LANCIUS_DTYPE_INT8) {
                    n->runtime_data_int8 = (int8_t*)malloc(elems);
                    SAFE_READ(n->runtime_data_int8, sizeof(int8_t), elems, f);
                } else {
                    n->runtime_data = (double*)malloc(elems * sizeof(double));
                    SAFE_READ(n->runtime_data, sizeof(double), elems, f);
                }
            }
        }
        free(in_ids);
    }
    free(id_map);
    fclose(f);
    printf("[LANCIUS SERIAL] Loaded %u nodes from %s\n", node_count, path);
    return g;
}

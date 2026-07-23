#include "lancius/lancius_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANCIUS_MAGIC 0x21434E41 // "LANC!"
#define LANCIUS_FORMAT_VERSION 1

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
        // V9.5: Write view flag (0 for non-views, maintains backward compat)
        uint8_t is_view_flag = n->is_view ? 1 : 0;
        SAFE_WRITE(&is_view_flag, sizeof(uint8_t), 1, f);
        if (is_view_flag) {
            SAFE_WRITE(n->strides, sizeof(size_t), 4, f);
            uint32_t source_id = n->view_source ? n->view_source->id : UINT32_MAX;
            SAFE_WRITE(&source_id, sizeof(uint32_t), 1, f);
        }

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
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) { fclose(f); return NULL; }
    if (magic != LANCIUS_MAGIC) { fclose(f); return NULL; }
    if (fread(&node_count, sizeof(uint32_t), 1, f) != 1) { fclose(f); return NULL; }
    if (node_count > 1000000) { fprintf(stderr, "[SERIAL FATAL] Node count exceeds sanity limit.\n"); fclose(f); return NULL; }

    lancius_graph* g = lancius_graph_create();
    if (!g) { fclose(f); return NULL; }

    uint32_t map_size = node_count + 1000;
    lancius_node** id_map = (lancius_node**)calloc(map_size, sizeof(lancius_node*));
    if (!id_map) { lancius_graph_destroy(g); fclose(f); return NULL; }

    for (uint32_t i = 0; i < node_count; i++) {
        uint32_t id, op, input_count;
        uint8_t ndim;
        size_t shape[4];

        if (fread(&id, sizeof(uint32_t), 1, f) != 1) goto fail;
        if (id >= map_size) goto fail;

        if (fread(&op, sizeof(lancius_opcode), 1, f) != 1) goto fail;
        if (fread(&ndim, sizeof(uint8_t), 1, f) != 1) goto fail;
        if (fread(shape, sizeof(size_t), 4, f) != 4) goto fail;

        if (fread(&input_count, sizeof(uint32_t), 1, f) != 1) goto fail;
        if (input_count > 16) goto fail; // Hard limit: max 16 inputs per node

        uint32_t* in_ids = NULL;
        if (input_count > 0) {
            in_ids = (uint32_t*)malloc(input_count * sizeof(uint32_t));
            if (!in_ids) goto fail;
            if (fread(in_ids, sizeof(uint32_t), input_count, f) != input_count) {
                free(in_ids);
                goto fail;
            }
            for (uint32_t j = 0; j < input_count; j++) {
                if (in_ids[j] >= map_size) {
                    free(in_ids);
                    goto fail;
                }
            }
        }

        double attr_val;
        if (fread(&attr_val, sizeof(double), 1, f) != 1) { free(in_ids); goto fail; }

        uint32_t meta[4], axes[4];
        if (fread(meta, sizeof(uint32_t), 4, f) != 4) { free(in_ids); goto fail; }
        if (fread(axes, sizeof(uint32_t), 4, f) != 4) { free(in_ids); goto fail; }

        uint8_t is_view = 0;
        if (fread(&is_view, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }

        size_t view_strides[4] = {0};
        uint32_t view_source_id = UINT32_MAX;
        if (is_view) {
            if (fread(view_strides, sizeof(size_t), 4, f) != 4) { free(in_ids); goto fail; }
            if (fread(&view_source_id, sizeof(uint32_t), 1, f) != 1) { free(in_ids); goto fail; }
        }

        uint8_t has_weights;
        if (fread(&has_weights, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }

        lancius_node* n = NULL;
        const lancius_node* in0 = input_count > 0 ? id_map[in_ids[0]] : NULL;
        const lancius_node* in1 = input_count > 1 ? id_map[in_ids[1]] : NULL;
        const lancius_node* in2 = input_count > 2 ? id_map[in_ids[2]] : NULL;

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
        else if (op == LANCIUS_OP_CONV2D) {
            if (in0 && in1) n = lancius_conv2d(g, in0, in1, meta[2], meta[3]);
        }
        else if (op == LANCIUS_OP_MAXPOOL2D) {
            if (in0) n = lancius_maxpool2d(g, in0, meta[0], meta[2]);
        }
        else if (op == LANCIUS_OP_FLATTEN) {
            if (in0) n = lancius_flatten(g, in0);
        }
        else if (op == LANCIUS_OP_RESHAPE) {
            if (in0) n = lancius_reshape(g, in0, ndim, shape[0], shape[1], shape[2], shape[3]);
        }
        else if (op == LANCIUS_OP_CONV2D_RELU_FUSED) {
            if (in0 && in1) {
                n = lancius_conv2d(g, in0, in1, meta[2], meta[3]);
                if (n) n->op = LANCIUS_OP_CONV2D_RELU_FUSED;
            }
        }
        else if (op == LANCIUS_OP_CROSS_ENTROPY) {
            if (in0 && in1) n = lancius_cross_entropy(g, in0, in1);
        }
        else if (op == LANCIUS_OP_LAYERNORM) {
            if (in0 && in1 && in2) n = lancius_layernorm(g, in0, in1, in2);
        }
        else if (op == LANCIUS_OP_ATTENTION) {
            if (in0 && in1 && in2) n = lancius_attention(g, in0, in1, in2);
        }
        else {
            fprintf(stderr, "[LANCIUS SERIAL WARN] Unknown op %u during load, skipping\n", op);
        }

        if (n) {
            n->kernel_h = meta[0]; n->kernel_w = meta[1]; n->stride = meta[2]; n->pad = meta[3];
            memcpy(n->axes, axes, sizeof(uint32_t)*4);
            if (is_view) {
                n->is_view = 1;
                memcpy(n->strides, view_strides, sizeof(size_t)*4);
                if (view_source_id != UINT32_MAX && view_source_id < map_size) {
                    n->view_source = id_map[view_source_id];
                }
            }
            id_map[id] = n;
            if (has_weights) {
                size_t elems = lancius_node_elements(n);
                if (elems > SIZE_MAX / sizeof(double)) {
                    fprintf(stderr, "[SERIAL FATAL] Tensor size overflow.\n");
                    free(in_ids); fclose(f); free(id_map);
                    for (uint32_t k = 0; k < g->node_count; k++) {
                        if (g->nodes[k]->runtime_data) free(g->nodes[k]->runtime_data);
                        if (g->nodes[k]->runtime_data_int8) free(g->nodes[k]->runtime_data_int8);
                    }
                    lancius_graph_destroy(g); return NULL;
                }
                if (elems > 100000000) {
                    fprintf(stderr, "[SERIAL FATAL] Tensor size exceeds sanity limit.\n");
                    free(in_ids); fclose(f); free(id_map);
                    for (uint32_t k = 0; k < g->node_count; k++) {
                        if (g->nodes[k]->runtime_data) free(g->nodes[k]->runtime_data);
                        if (g->nodes[k]->runtime_data_int8) free(g->nodes[k]->runtime_data_int8);
                    }
                    lancius_graph_destroy(g); return NULL;
                }
                uint8_t dtype;
                if (fread(&dtype, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }
                n->dtype = (lancius_dtype)dtype;
                if (fread(&n->scale, sizeof(double), 1, f) != 1) { free(in_ids); goto fail; }
                if (n->dtype == LANCIUS_DTYPE_INT8) {
                    n->runtime_data_int8 = (int8_t*)malloc(elems);
                    if (!n->runtime_data_int8) { free(in_ids); goto fail; }
                    if (fread(n->runtime_data_int8, sizeof(int8_t), elems, f) != elems) { free(in_ids); goto fail; }
                } else {
                    n->runtime_data = (double*)malloc(elems * sizeof(double));
                    if (!n->runtime_data) { free(in_ids); goto fail; }
                    if (fread(n->runtime_data, sizeof(double), elems, f) != elems) { free(in_ids); goto fail; }
                }
            }
        }
        free(in_ids);
    }
    free(id_map);
    fclose(f);
    printf("[LANCIUS SERIAL] Loaded %u nodes from %s\n", node_count, path);
    return g;

fail:
    fprintf(stderr, "[SERIAL FATAL] Malformed or truncated .lancius file. Aborting load.\n");
    if (g) {
        for (uint32_t k = 0; k < g->node_count; k++) {
            if (g->nodes[k]->runtime_data) free(g->nodes[k]->runtime_data);
            if (g->nodes[k]->runtime_data_int8) free(g->nodes[k]->runtime_data_int8);
        }
        lancius_graph_destroy(g);
    }
    free(id_map);
    fclose(f);
    return NULL;
}

#include "basis/v8_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V8_MAGIC 0x56384221 // "V8B!"

// Bulletproof macros to silence GCC's aggressive warn_unused_result on fread/fwrite
#define SAFE_READ(ptr, size, nmemb, stream) do { size_t _r = fread(ptr, size, nmemb, stream); (void)_r; } while(0)
#define SAFE_WRITE(ptr, size, nmemb, stream) do { size_t _r = fwrite(ptr, size, nmemb, stream); (void)_r; } while(0)

void v8_graph_save(v8_graph* g, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    uint32_t magic = V8_MAGIC;
    SAFE_WRITE(&magic, sizeof(uint32_t), 1, f);
    SAFE_WRITE(&g->node_count, sizeof(uint32_t), 1, f);

    for (uint32_t i = 0; i < g->node_count; i++) {
        v8_node* n = g->nodes[i];
        SAFE_WRITE(&n->id, sizeof(uint32_t), 1, f);
        SAFE_WRITE(&n->op, sizeof(v8_opcode), 1, f);
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

        uint8_t has_weights = (n->op == V8_OP_INPUT && (n->runtime_data != NULL || n->runtime_data_int8 != NULL)) ? 1 : 0;
        SAFE_WRITE(&has_weights, sizeof(uint8_t), 1, f);
        if (has_weights) {
            size_t elems = v8_node_elements(n);
            uint8_t dtype = n->dtype;
            SAFE_WRITE(&dtype, sizeof(uint8_t), 1, f);
            SAFE_WRITE(&n->scale, sizeof(double), 1, f);
            if (dtype == V8_DTYPE_INT8) {
                SAFE_WRITE(n->runtime_data_int8, sizeof(int8_t), elems, f);
            } else {
                SAFE_WRITE(n->runtime_data, sizeof(double), elems, f);
            }
        }
    }
    fclose(f);
    printf("[V8 SERIAL] Saved %u nodes to %s\n", g->node_count, path);
}

v8_graph* v8_graph_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, node_count;
    SAFE_READ(&magic, sizeof(uint32_t), 1, f);
    if (magic != V8_MAGIC) { fclose(f); return NULL; }
    SAFE_READ(&node_count, sizeof(uint32_t), 1, f);

    v8_graph* g = v8_graph_create();
    v8_node** id_map = (v8_node**)calloc(node_count + 1000, sizeof(v8_node*));

    for (uint32_t i = 0; i < node_count; i++) {
        uint32_t id, op, input_count;
        uint8_t ndim;
        size_t shape[4];
        SAFE_READ(&id, sizeof(uint32_t), 1, f);
        SAFE_READ(&op, sizeof(v8_opcode), 1, f);
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

        v8_node* n = NULL;
        const v8_node* in0 = input_count > 0 ? id_map[in_ids[0]] : NULL;
        const v8_node* in1 = input_count > 1 ? id_map[in_ids[1]] : NULL;

        if (op == V8_OP_INPUT) {
            if (ndim == 4) n = v8_input_4d(g, shape[0], shape[1], shape[2], shape[3]);
            else n = v8_input(g, shape[0], shape[1]);
        } else if (op == V8_OP_CONST) n = v8_const(g, attr_val, shape[0], shape[1]);
        else if (op == V8_OP_ADD) n = v8_add(g, in0, in1);
        else if (op == V8_OP_SUB) n = v8_sub(g, in0, in1);
        else if (op == V8_OP_MUL) n = v8_mul(g, in0, in1);
        else if (op == V8_OP_MATMUL) n = v8_matmul(g, in0, in1);
        else if (op == V8_OP_RELU) n = v8_relu(g, in0);
        else if (op == V8_OP_SOFTMAX) n = v8_softmax(g, in0);
        else if (op == V8_OP_SUM) n = v8_sum(g, in0);
        else if (op == V8_OP_BROADCAST) n = v8_broadcast(g, in0, shape[0], shape[1]);
        else if (op == V8_OP_TRANSPOSE) n = v8_transpose(g, in0);
        else if (op == V8_OP_CONV2D) n = v8_conv2d(g, in0, in1, meta[2], meta[3]);
        else if (op == V8_OP_MAXPOOL2D) n = v8_maxpool2d(g, in0, meta[0], meta[2]);
        else if (op == V8_OP_FLATTEN) n = v8_flatten(g, in0);
        else if (op == V8_OP_RESHAPE) n = v8_reshape(g, in0, ndim, shape[0], shape[1], shape[2], shape[3]);
        else if (op == V8_OP_PERMUTE) n = v8_permute(g, in0, axes[0], axes[1], axes[2], axes[3]);
        else if (op == V8_OP_MATMUL_BATCHED) n = v8_matmul_batched(g, in0, in1);
        else if (op == V8_OP_NOP) n = NULL; // V10 Fix: Orphaned NOP
        else if (op == V8_OP_CONV2D_RELU_FUSED) {
            n = v8_conv2d(g, in0, in1, meta[2], meta[3]);
            if (n) n->op = V8_OP_CONV2D_RELU_FUSED; // Preserve fused opcode
        }
        else if (op == V8_OP_CROSS_ENTROPY) {
            const v8_node* dummy_y = in1 ? in1 : v8_const(g, 0.0, in0 ? in0->shape[0] : 1, in0 ? in0->shape[1] : 1);
            n = v8_cross_entropy(g, in0, dummy_y);
        }
        else if (op == V8_OP_BROADCAST) n = v8_broadcast(g, in0, shape[0], shape[1]);
        else if (op == V8_OP_SUB) n = v8_sub(g, in0, in1);
        else if (op == V8_OP_MUL) n = v8_mul(g, in0, in1);
        else if (op == V8_OP_SUM) n = v8_sum(g, in0);
        else if (op == V8_OP_TRANSPOSE) n = v8_transpose(g, in0);
        else if (op == V8_OP_SOFTMAX) n = v8_softmax(g, in0);
        else if (op == V8_OP_RESHAPE) n = v8_reshape(g, in0, ndim, shape[0], shape[1], shape[2], shape[3]);
        else if (op == V8_OP_SUM_AXIS0) n = v8_sum_axis0(g, in0);
        else if (op == V8_OP_SUM_AXIS1) n = v8_sum_axis1(g, in0);
        else { fprintf(stderr, "[V8 SERIAL WARN] Unknown op %u during load, skipping\n", op); }

        if (n) {
            n->kernel_h = meta[0]; n->kernel_w = meta[1]; n->stride = meta[2]; n->pad = meta[3];
            memcpy(n->axes, axes, sizeof(uint32_t)*4);
            id_map[id] = n;

            if (has_weights) {
                size_t elems = v8_node_elements(n);
                uint8_t dtype;
                SAFE_READ(&dtype, sizeof(uint8_t), 1, f);
                n->dtype = (v8_dtype)dtype;
                SAFE_READ(&n->scale, sizeof(double), 1, f);
                if (n->dtype == V8_DTYPE_INT8) {
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
    printf("[V8 SERIAL] Loaded %u nodes from %s\n", node_count, path);
    return g;
}

#include "lancius/lancius_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t node_count;
    uint32_t tensor_count;
    uint32_t attribute_count;
    uint32_t header_size;
    uint32_t reserved0;
    uint64_t weight_block_offset;
    uint32_t checksum_crc32;
    uint32_t reserved1;
} v2_header;

typedef struct __attribute__((packed)) {
    uint32_t id;
    uint32_t op;
    uint8_t ndim;
    uint64_t shape[4];
    uint32_t input_count;
    double attr;
    uint32_t meta[4];
    uint32_t axes[4];
    uint8_t flags;
    uint8_t dtype;
    uint8_t has_weights;
    double scale;
    uint64_t weight_elems;
} v2_node;

static int is_little_endian(void) {
    uint16_t x = 1;
    return *(const uint8_t*)&x == 1;
}

typedef struct {
    lancius_node** v;
    uint32_t cap;
} idmap;

static int map_set(idmap* m, uint32_t id, lancius_node* n) {
    if (id >= m->cap) {
        uint32_t nc = m->cap ? m->cap : 1024;
        while (nc <= id) {
            if (nc > 5000000u) return 0;
            nc *= 2u;
        }
        if (nc > 10000000u) return 0;

        lancius_node** nv = (lancius_node**)realloc(m->v, (size_t)nc * sizeof(lancius_node*));
        if (!nv) return 0;

        memset(nv + m->cap, 0, (size_t)(nc - m->cap) * sizeof(lancius_node*));
        m->v = nv;
        m->cap = nc;
    }

    m->v[id] = n;
    return 1;
}

static lancius_node* map_get(idmap* m, uint32_t id) {
    if (id >= m->cap) return NULL;
    return m->v[id];
}

int lancius_graph_save_v2(lancius_graph* g, const char* path) {
    if (!g || !path) return -1;
    if (!is_little_endian()) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    v2_header h;
    memset(&h, 0, sizeof(h));
    h.magic = LANCIUS_MODEL_MAGIC_V2;
    h.version = LANCIUS_MODEL_VERSION_V2;
    h.flags = LANCIUS_MODEL_FLAG_LITTLE_ENDIAN | LANCIUS_MODEL_FLAG_STATIC_GRAPH;
    h.node_count = g->node_count;
    h.tensor_count = g->node_count;
    h.attribute_count = 0;
    h.header_size = (uint32_t)sizeof(h);

    if (fwrite(&h, 1, sizeof(h), f) != sizeof(h)) {
        fclose(f);
        return -1;
    }

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        lancius_runtime_sync_from_legacy(n);

        v2_node rn;
        memset(&rn, 0, sizeof(rn));

        rn.id = n->id;
        rn.op = (uint32_t)n->op;
        rn.ndim = n->ndim;

        for (int s = 0; s < 4; s++) rn.shape[s] = (uint64_t)n->shape[s];

        rn.input_count = n->input_count;
        rn.attr = n->attr_val;

        rn.meta[0] = n->kernel_h;
        rn.meta[1] = n->kernel_w;
        rn.meta[2] = n->stride;
        rn.meta[3] = n->pad;

        for (int a = 0; a < 4; a++) rn.axes[a] = n->axes[a];

        rn.flags = 0;
        rn.dtype = (uint8_t)n->dtype;
        if (!lancius_dtype_is_valid(rn.dtype)) rn.dtype = LANCIUS_DTYPE_FP64;

        rn.has_weights = 0;
        rn.scale = n->scale;
        rn.weight_elems = 0;

        const void* data = NULL;
        size_t elem_size = sizeof(double);
        size_t elems = 0;

        if (n->op == LANCIUS_OP_INPUT) {
            if (rn.dtype == LANCIUS_DTYPE_INT8 && n->runtime_data_int8) {
                rn.has_weights = 1;
                data = n->runtime_data_int8;
                elem_size = sizeof(int8_t);
                elems = lancius_node_elements(n);
            } else if (n->runtime_data) {
                rn.has_weights = 1;
                rn.dtype = LANCIUS_DTYPE_FP64;
                data = n->runtime_data;
                elem_size = sizeof(double);
                elems = lancius_node_elements(n);
            }
        }

        if (rn.has_weights && elems > 100000000u) {
            fclose(f);
            return -1;
        }

        rn.weight_elems = (uint64_t)elems;

        if (fwrite(&rn, 1, sizeof(rn), f) != sizeof(rn)) {
            fclose(f);
            return -1;
        }

        for (uint32_t j = 0; j < n->input_count; j++) {
            uint32_t in_id = n->inputs[j] ? n->inputs[j]->id : UINT32_MAX;
            if (fwrite(&in_id, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
        }

        if (rn.has_weights && elems > 0) {
            if (fwrite(data, elem_size, elems, f) != elems) {
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}

lancius_graph* lancius_graph_load_v2(const char* path) {
    if (!path) return NULL;
    if (!is_little_endian()) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    v2_header h;
    if (fread(&h, 1, sizeof(h), f) != sizeof(h)) {
        fclose(f);
        return NULL;
    }

    if (h.magic != LANCIUS_MODEL_MAGIC_V2) {
        fclose(f);
        return NULL;
    }

    if (
        h.version != LANCIUS_MODEL_VERSION_V2 ||
        !(h.flags & LANCIUS_MODEL_FLAG_LITTLE_ENDIAN) ||
        h.header_size != sizeof(h) ||
        h.node_count > 1000000u
    ) {
        fclose(f);
        return NULL;
    }

    lancius_graph* g = lancius_graph_create();
    if (!g) {
        fclose(f);
        return NULL;
    }

    idmap map = {NULL, 0};
    uint32_t* in_ids = NULL;

    for (uint32_t i = 0; i < h.node_count; i++) {
        v2_node rn;

        if (fread(&rn, 1, sizeof(rn), f) != sizeof(rn)) goto fail;

        if (rn.ndim > 4) goto fail;
        if (rn.input_count > 16u) goto fail;
        if (rn.weight_elems > 100000000ull) goto fail;
        if (!lancius_dtype_is_valid(rn.dtype)) goto fail;
        if (rn.op > LANCIUS_MODEL_OP_GQA) goto fail;
        if (rn.dtype != LANCIUS_DTYPE_FP64 && rn.dtype != LANCIUS_DTYPE_INT8) goto fail;

        (void)rn.flags;

        size_t sh[4];
        for (int s = 0; s < 4; s++) sh[s] = (size_t)rn.shape[s];

        in_ids = NULL;
        if (rn.input_count > 0) {
            in_ids = (uint32_t*)malloc((size_t)rn.input_count * sizeof(uint32_t));
            if (!in_ids) goto fail;

            if (fread(in_ids, sizeof(uint32_t), rn.input_count, f) != rn.input_count) {
                goto fail;
            }
        }

        lancius_node* in0 = rn.input_count > 0 ? map_get(&map, in_ids[0]) : NULL;
        lancius_node* in1 = rn.input_count > 1 ? map_get(&map, in_ids[1]) : NULL;
        lancius_node* in2 = rn.input_count > 2 ? map_get(&map, in_ids[2]) : NULL;

        free(in_ids);
        in_ids = NULL;

        lancius_node* n = NULL;

        switch (rn.op) {
            case LANCIUS_MODEL_OP_NOP:
                n = NULL;
                break;

            case LANCIUS_MODEL_OP_INPUT:
                if (rn.ndim == 4) n = lancius_input_4d(g, sh[0], sh[1], sh[2], sh[3]);
                else n = lancius_input(g, sh[0], sh[1]);
                break;

            case LANCIUS_MODEL_OP_CONST:
                n = lancius_const(g, rn.attr, sh[0], sh[1]);
                break;

            case LANCIUS_MODEL_OP_ADD:
                n = lancius_add(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_SUB:
                n = lancius_sub(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_MUL:
                n = lancius_mul(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_MATMUL:
                n = lancius_matmul(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_RELU:
                n = lancius_relu(g, in0);
                break;

            case LANCIUS_MODEL_OP_SOFTMAX:
                n = lancius_softmax(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM:
                n = lancius_sum(g, in0);
                break;

            case LANCIUS_MODEL_OP_BROADCAST:
                if (rn.ndim == 4) n = lancius_broadcast_4d(g, in0, sh[0], sh[1], sh[2], sh[3]);
                else n = lancius_broadcast(g, in0, sh[0], sh[1]);
                break;

            case LANCIUS_MODEL_OP_TRANSPOSE:
                n = lancius_transpose(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM_AXIS0:
                n = lancius_sum_axis0(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM_AXIS1:
                n = lancius_sum_axis1(g, in0);
                break;

            case LANCIUS_MODEL_OP_CONV2D:
                n = lancius_conv2d(g, in0, in1, rn.meta[2], rn.meta[3]);
                break;

            case LANCIUS_MODEL_OP_MAXPOOL2D:
                n = lancius_maxpool2d(g, in0, rn.meta[0], rn.meta[2]);
                break;

            case LANCIUS_MODEL_OP_FLATTEN:
                n = lancius_flatten(g, in0);
                break;

            case LANCIUS_MODEL_OP_CONV2D_RELU_FUSED:
                n = lancius_conv2d(g, in0, in1, rn.meta[2], rn.meta[3]);
                if (n) n->op = LANCIUS_OP_CONV2D_RELU_FUSED;
                break;

            case LANCIUS_MODEL_OP_CROSS_ENTROPY:
                n = lancius_cross_entropy(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_PERMUTE:
                n = lancius_permute(g, in0, rn.axes[0], rn.axes[1], rn.axes[2], rn.axes[3]);
                break;

            case LANCIUS_MODEL_OP_MATMUL_BATCHED:
                n = lancius_matmul_batched(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_RESHAPE:
                n = lancius_reshape(g, in0, rn.ndim, sh[0], sh[1], sh[2], sh[3]);
                break;

            case LANCIUS_MODEL_OP_LAYERNORM:
                n = lancius_layernorm(g, in0, in1, in2);
                break;

            case LANCIUS_MODEL_OP_GELU:
                n = lancius_gelu(g, in0);
                break;

            case LANCIUS_MODEL_OP_ATTENTION:
                n = lancius_attention(g, in0, in1, in2);
                break;

            case LANCIUS_MODEL_OP_RMSNORM:
                n = lancius_rmsnorm(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_SWIGLU:
                n = lancius_swiglu(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_GQA:
                n = lancius_gqa(g, in0, in1, in2, rn.meta[0], rn.meta[1]);
                break;

            default:
                fprintf(stderr, "[SERIAL V2 WARN] unsupported op %u during load\n", rn.op);
                n = NULL;
                break;
        }

        if (n) {
            n->kernel_h = rn.meta[0];
            n->kernel_w = rn.meta[1];
            n->stride = rn.meta[2];
            n->pad = rn.meta[3];

            memcpy(n->axes, rn.axes, sizeof(rn.axes));

            n->dtype = (lancius_dtype)rn.dtype;
            n->scale = rn.scale;
        }

        if (rn.has_weights && rn.weight_elems > 0) {
            size_t elems = (size_t)rn.weight_elems;
            size_t elem_size = (rn.dtype == LANCIUS_DTYPE_INT8) ? sizeof(int8_t) : sizeof(double);
            size_t bytes = elems * elem_size;

            if (elem_size != 0 && bytes / elem_size != elems) goto fail;

            if (n) {
                if (rn.dtype == LANCIUS_DTYPE_INT8) {
                    int8_t* buf = (int8_t*)malloc(bytes);
                    if (!buf) goto fail;

                    if (fread(buf, 1, bytes, f) != bytes) {
                        free(buf);
                        goto fail;
                    }

                    n->runtime_data_int8 = buf;
                    lancius_node_bind_owned_heap_int8(n, buf);
                } else {
                    double* buf = (double*)malloc(bytes);
                    if (!buf) goto fail;

                    if (fread(buf, sizeof(double), elems, f) != elems) {
                        free(buf);
                        goto fail;
                    }

                    n->runtime_data = buf;
                    lancius_node_bind_owned_heap(n, buf);
                }
            } else {
                if (bytes > 0 && fseek(f, (long)bytes, SEEK_CUR) != 0) goto fail;
            }
        }

        if (!map_set(&map, rn.id, n)) goto fail;
    }

    free(map.v);
    fclose(f);

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_runtime_sync_from_legacy(g->nodes[i]);
    }

    return g;

fail:
    free(in_ids);
    free(map.v);
    if (g) lancius_graph_destroy(g);
    fclose(f);
    return NULL;
}

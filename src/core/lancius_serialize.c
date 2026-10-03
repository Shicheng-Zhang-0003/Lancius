#define _POSIX_C_SOURCE 200809L
#include "lancius/lancius_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LANCIUS_MAGIC 0x21434E41 // "LANC!"
/* v11A1 Task 6a: active format is v1. v2 lands in Task 6b. */
#define LANCIUS_FORMAT_VERSION 1

/* v11A1 Task 6b: v2 format entry points */
int lancius_graph_save_v2(lancius_graph* g, const char* path);
lancius_graph* lancius_graph_load_v2(const char* path);

/* Despot truth: every write is checked; any failure unlinks the partial file
 * (was: return values discarded, truncated files reported as success). */
#define CHECKED_WRITE(ptr, size, nmemb, stream) \
    do { if (fwrite(ptr, size, nmemb, stream) != (size_t)(nmemb)) goto wfail; } while(0)

/*
 * v12R1 fix: the v1 on-disk format is fixed-width little-endian.
 * size_t fields are serialized as uint64_t (portable across 32/64-bit
 * hosts), and every multi-byte field is byte-swapped on big-endian systems
 * so files load identically on either endianness.
 */
static int ser_is_little_endian(void) {
    uint16_t x = 1;
    return *(const uint8_t*)&x == 1;
}

static uint32_t ser_bswap32(uint32_t v) {
    return ((v & 0xFF000000u) >> 24) |
           ((v & 0x00FF0000u) >> 8) |
           ((v & 0x0000FF00u) << 8) |
           ((v & 0x000000FFu) << 24);
}

static uint64_t ser_bswap64(uint64_t v) {
    return ((uint64_t)ser_bswap32((uint32_t)(v & 0xFFFFFFFFu)) << 32) |
           (uint64_t)ser_bswap32((uint32_t)(v >> 32));
}

static double ser_bswap_double(double d) {
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    u = ser_bswap64(u);
    double r;
    memcpy(&r, &u, sizeof(r));
    return r;
}

/* Host -> disk (little-endian) and disk -> host conversions. */
static uint32_t ser_to_le32(uint32_t v) { return ser_is_little_endian() ? v : ser_bswap32(v); }
static uint64_t ser_to_le64(uint64_t v) { return ser_is_little_endian() ? v : ser_bswap64(v); }
static double ser_to_le_double(double v) { return ser_is_little_endian() ? v : ser_bswap_double(v); }
static uint32_t ser_from_le32(uint32_t v) { return ser_to_le32(v); }
static uint64_t ser_from_le64(uint64_t v) { return ser_to_le64(v); }
static double ser_from_le_double(double v) { return ser_to_le_double(v); }

int lancius_graph_save(lancius_graph* g, const char* path) {
    if (lancius_graph_save_v2(g, path) == 0) return 0;

    /* Despot truth: NULL graph/path derefed (was unguarded). */
    if (!g || !path || !g->nodes) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return -1; }
    /* Despot V6 truth: tmp+mkstemp+rename (was truncate-in-place). */
    char v1tmp[4096];
    if (snprintf(v1tmp, sizeof(v1tmp), "%s.tmp.XXXXXX", path) >= (int)sizeof(v1tmp)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return -1; }
    int v1fd = mkstemp(v1tmp);
    if (v1fd < 0) { lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    FILE* f = fdopen(v1fd, "wb");
    if (!f) { close(v1fd); unlink(v1tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    uint32_t magic = ser_to_le32(LANCIUS_MAGIC);
    CHECKED_WRITE(&magic, sizeof(uint32_t), 1, f);
    uint32_t node_count_le = ser_to_le32(g->node_count);
    CHECKED_WRITE(&node_count_le, sizeof(uint32_t), 1, f);

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        uint32_t meta[4];
        uint8_t is_view_flag;
        uint8_t has_weights;
        size_t elems = 0;
        uint8_t dtype;
        if (!n) goto wfail;
        lancius_runtime_sync_from_legacy(n); /* A1 */
        uint32_t id_le = ser_to_le32(n->id);
        CHECKED_WRITE(&id_le, sizeof(uint32_t), 1, f);
        uint32_t op_le = ser_to_le32((uint32_t)n->op);
        CHECKED_WRITE(&op_le, sizeof(uint32_t), 1, f);
        CHECKED_WRITE(&n->ndim, sizeof(uint8_t), 1, f);
        /* v12R1 fix: shape serialized as fixed-width uint64_t (was native size_t). */
        uint64_t shape_le[4];
        for (int s = 0; s < 4; s++) shape_le[s] = ser_to_le64((uint64_t)n->shape[s]);
        CHECKED_WRITE(shape_le, sizeof(uint64_t), 4, f);
        if (n->input_count > 16) goto wfail;
        if (n->input_count > 0 && !n->inputs) goto wfail;
        uint32_t input_count_le = ser_to_le32(n->input_count);
        CHECKED_WRITE(&input_count_le, sizeof(uint32_t), 1, f);
        for(uint32_t j=0; j<n->input_count; j++) {
            if (!n->inputs[j]) goto wfail;
            uint32_t in_id_le = ser_to_le32(n->inputs[j]->id);
            CHECKED_WRITE(&in_id_le, sizeof(uint32_t), 1, f);
        }
        double attr_le = ser_to_le_double(n->attr_val);
        CHECKED_WRITE(&attr_le, sizeof(double), 1, f);
        meta[0] = ser_to_le32(n->kernel_h); meta[1] = ser_to_le32(n->kernel_w);
        meta[2] = ser_to_le32(n->stride); meta[3] = ser_to_le32(n->pad);
        CHECKED_WRITE(meta, sizeof(uint32_t), 4, f);
        uint32_t axes_le[4];
        for (int a = 0; a < 4; a++) axes_le[a] = ser_to_le32(n->axes[a]);
        CHECKED_WRITE(axes_le, sizeof(uint32_t), 4, f);
        // V9.5: Write view flag (0 for non-views, maintains backward compat)
        is_view_flag = n->is_view ? 1 : 0;
        CHECKED_WRITE(&is_view_flag, sizeof(uint8_t), 1, f);
        if (is_view_flag) {
            uint32_t source_id = n->view_source ? n->view_source->id : UINT32_MAX;
            /* v12R1 fix: strides serialized as fixed-width uint64_t (was native size_t). */
            uint64_t strides_le[4];
            for (int s = 0; s < 4; s++) strides_le[s] = ser_to_le64((uint64_t)n->strides[s]);
            CHECKED_WRITE(strides_le, sizeof(uint64_t), 4, f);
            uint32_t source_id_le = ser_to_le32(source_id);
            CHECKED_WRITE(&source_id_le, sizeof(uint32_t), 1, f);
        }

        has_weights = (n->op == LANCIUS_OP_INPUT && (n->runtime_data != NULL || n->runtime_data_int8 != NULL || n->runtime_data_f32 != NULL)) ? 1 : 0;
        CHECKED_WRITE(&has_weights, sizeof(uint8_t), 1, f);
        if (has_weights) {
            /* Despot truth: never abort() from a library save path. */
            if (!lancius_node_elements_checked(n, &elems)) goto wfail;
                if (elems > 100000000) goto wfail;
            dtype = n->dtype;
            /* A3: clamp unknown dtypes to FP64 for serialization safety */
            if (!lancius_dtype_is_valid(dtype)) dtype = LANCIUS_DTYPE_FP64;
            CHECKED_WRITE(&dtype, sizeof(uint8_t), 1, f);
            double scale_le = ser_to_le_double(n->scale);
            CHECKED_WRITE(&scale_le, sizeof(double), 1, f);
            if (dtype == LANCIUS_DTYPE_INT8) {
                if (!n->runtime_data_int8) goto wfail;
                CHECKED_WRITE(n->runtime_data_int8, sizeof(int8_t), elems, f);
            } else if (dtype == LANCIUS_DTYPE_FP32) {
                /* Despot V6 truth: FP32 weights persist (were dropped as no-weights). */
                if (!n->runtime_data_f32) goto wfail;
                if (ser_is_little_endian()) {
                    CHECKED_WRITE(n->runtime_data_f32, sizeof(float), elems, f);
                } else {
                    for (size_t j = 0; j < elems; j++) {
                        uint32_t u; memcpy(&u, &n->runtime_data_f32[j], 4);
                        u = ser_to_le32(u);
                        CHECKED_WRITE(&u, 4, 1, f);
                    }
                }
            } else {
                if (!n->runtime_data) goto wfail;
                if (ser_is_little_endian()) {
                    CHECKED_WRITE(n->runtime_data, sizeof(double), elems, f);
                } else {
                    /* v12R1 fix: byte-swap FP64 payload on big-endian hosts. */
                    for (size_t j = 0; j < elems; j++) {
                        double sv = ser_bswap_double(n->runtime_data[j]);
                        CHECKED_WRITE(&sv, sizeof(double), 1, f);
                    }
                }
            }
        }
    }
    if (fflush(f) != 0) { fclose(f); unlink(v1tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    {
        int _fd = fileno(f);
        if (_fd >= 0 && fsync(_fd) != 0) { fclose(f); unlink(v1tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    }
    if (fclose(f) != 0) { unlink(v1tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    if (rename(v1tmp, path) != 0) { unlink(v1tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    return 0;
wfail:
    fclose(f);
    unlink(v1tmp);
    lancius_set_error(LANCIUS_ERROR_IO);
    return -1;
}

lancius_graph* lancius_graph_load(const char* path) {
    /* Despot truth: fopen(NULL) is UB (was unguarded). */
    if (!path) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    lancius_graph* g_v2 = lancius_graph_load_v2(path);
    if (g_v2) return g_v2;

    lancius_clear_error();

    FILE* f = fopen(path, "rb");
    if (!f) { lancius_set_error(LANCIUS_ERROR_IO); return NULL; }
    uint32_t magic, node_count;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) { lancius_set_error(LANCIUS_ERROR_INVALID_MODEL); fclose(f); return NULL; }
    magic = ser_from_le32(magic);
    if (magic != LANCIUS_MAGIC) { lancius_set_error(LANCIUS_ERROR_INVALID_MODEL); /* A4 magic */ fclose(f); return NULL; }
    if (fread(&node_count, sizeof(uint32_t), 1, f) != 1) { lancius_set_error(LANCIUS_ERROR_INVALID_MODEL); fclose(f); return NULL; }
    node_count = ser_from_le32(node_count);
    if (node_count > 1000000) { lancius_set_error(LANCIUS_ERROR_INVALID_MODEL); fclose(f); return NULL; }

    lancius_graph* g = lancius_graph_create();
    if (!g) { lancius_set_error(LANCIUS_ERROR_OOM); fclose(f); return NULL; }

    uint32_t map_size = node_count + 1000;
    lancius_node** id_map = (lancius_node**)calloc(map_size, sizeof(lancius_node*));
    if (!id_map) { lancius_set_error(LANCIUS_ERROR_OOM); lancius_graph_destroy(g); fclose(f); return NULL; }

    for (uint32_t i = 0; i < node_count; i++) {
        uint32_t id, op, input_count;
        uint8_t ndim;
        size_t shape[4];
        uint64_t shape_u64[4];

        if (fread(&id, sizeof(uint32_t), 1, f) != 1) goto fail;
        id = ser_from_le32(id);
        if (id >= map_size) goto fail;

        if (fread(&op, sizeof(uint32_t), 1, f) != 1) goto fail;
        op = ser_from_le32(op);
        if (op > LANCIUS_OP_MSE_BWD) goto fail;
        if (fread(&ndim, sizeof(uint8_t), 1, f) != 1) goto fail;
        if (ndim == 0 || ndim > 4) goto fail;
        /* v12R1 fix: shape read as fixed-width uint64_t (was native size_t). */
        if (fread(shape_u64, sizeof(uint64_t), 4, f) != 4) goto fail;
        for (int s = 0; s < 4; s++) {
            shape_u64[s] = ser_from_le64(shape_u64[s]);
            if (shape_u64[s] > (uint64_t)SIZE_MAX) goto fail;
            shape[s] = (size_t)shape_u64[s];
        }

        if (fread(&input_count, sizeof(uint32_t), 1, f) != 1) goto fail;
        input_count = ser_from_le32(input_count);
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
                in_ids[j] = ser_from_le32(in_ids[j]);
                if (in_ids[j] >= map_size) {
                    free(in_ids);
                    goto fail;
                }
            }
        }

        double attr_val;
        if (fread(&attr_val, sizeof(double), 1, f) != 1) { free(in_ids); goto fail; }
        attr_val = ser_from_le_double(attr_val);

        uint32_t meta[4], axes[4];
        if (fread(meta, sizeof(uint32_t), 4, f) != 4) { free(in_ids); goto fail; }
        if (fread(axes, sizeof(uint32_t), 4, f) != 4) { free(in_ids); goto fail; }
        for (int m = 0; m < 4; m++) meta[m] = ser_from_le32(meta[m]);
        for (int a = 0; a < 4; a++) axes[a] = ser_from_le32(axes[a]);

        uint8_t is_view = 0;
        if (fread(&is_view, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }

        size_t view_strides[4] = {0};
        uint64_t view_strides_u64[4] = {0};
        uint32_t view_source_id = UINT32_MAX;
        if (is_view) {
            /* v12R1 fix: view strides read as fixed-width uint64_t (was native size_t). */
            if (fread(view_strides_u64, sizeof(uint64_t), 4, f) != 4) { free(in_ids); goto fail; }
            if (fread(&view_source_id, sizeof(uint32_t), 1, f) != 1) { free(in_ids); goto fail; }
            view_source_id = ser_from_le32(view_source_id);
            for (int s = 0; s < 4; s++) {
                view_strides_u64[s] = ser_from_le64(view_strides_u64[s]);
                if (view_strides_u64[s] > (uint64_t)SIZE_MAX) { free(in_ids); goto fail; }
                view_strides[s] = (size_t)view_strides_u64[s];
            }
        }

        uint8_t has_weights;
        if (fread(&has_weights, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }

        lancius_node* n = NULL;
        const lancius_node* in0 = input_count > 0 ? id_map[in_ids[0]] : NULL;
        const lancius_node* in1 = input_count > 1 ? id_map[in_ids[1]] : NULL;
        const lancius_node* in2 = input_count > 2 ? id_map[in_ids[2]] : NULL;

        if (op == LANCIUS_OP_INPUT) {
            /* Despot truth: ndim 1/3 was silently coerced to 2D, dropping dims
             * and desyncing the weight stream. Only 2/3/4 load; else fail. */
            if (ndim == 4) n = lancius_input_4d(g, shape[0], shape[1], shape[2], shape[3]);
            else if (ndim == 3) n = lancius_input_3d(g, shape[0], shape[1], shape[2]);
            else if (ndim == 2) n = lancius_input(g, shape[0], shape[1]);
            else { free(in_ids); goto fail; }
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
            lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
            free(in_ids);
            goto fail;
        }

        if (!n) { free(in_ids); goto fail; }
        if (id_map[id] != NULL) { free(in_ids); goto fail; } /* duplicate id */
        {
            size_t ne = 0;
            if (!lancius_node_elements_checked(n, &ne)) { free(in_ids); goto fail; }
        }

        if (n) {
            n->kernel_h = meta[0]; n->kernel_w = meta[1]; n->stride = meta[2]; n->pad = meta[3];
            memcpy(n->axes, axes, sizeof(uint32_t)*4);
            if (is_view) {
                /* Despot truth: dangling/forward view_source derefed later (was
                 * stored unchecked, NULL or not-yet-loaded). */
                if (view_source_id == UINT32_MAX || view_source_id >= map_size ||
                    id_map[view_source_id] == NULL) { free(in_ids); goto fail; }
                n->is_view = 1;
                memcpy(n->strides, view_strides, sizeof(size_t)*4);
                n->view_source = id_map[view_source_id];
            }
            id_map[id] = n;
            if (has_weights) {
                size_t elems = 0;
                if (!lancius_node_elements_checked(n, &elems)) { free(in_ids); goto fail; }
                /* Despot truth: the manual free loops below double-freed:
                 * buffers are bound OWNED_HEAP, so graph_destroy releases them.
                 * Just destroy (v2 pattern). */
                if (elems > SIZE_MAX / sizeof(double)) {
                    free(in_ids); fclose(f); free(id_map);
                    lancius_graph_destroy(g); return NULL;
                }
                if (elems > 100000000) {
                    free(in_ids); fclose(f); free(id_map);
                    lancius_graph_destroy(g); return NULL;
                }
                uint8_t dtype;
                if (fread(&dtype, sizeof(uint8_t), 1, f) != 1) { free(in_ids); goto fail; }
                /* A3: validate serialized dtype */
                if (!lancius_dtype_is_valid(dtype)) {
                    free(in_ids);
                    goto fail;
                }
                n->dtype = (lancius_dtype)dtype;
                if (fread(&n->scale, sizeof(double), 1, f) != 1) { free(in_ids); goto fail; }
                n->scale = ser_from_le_double(n->scale);
                /* Despot V6 truth: INT8 scale validated (was unchecked). */
                if (n->dtype == LANCIUS_DTYPE_INT8) {
                    if (!(n->scale > 0.0)) { free(in_ids); goto fail; }
                }
                if (n->dtype == LANCIUS_DTYPE_INT8) {
                    n->runtime_data_int8 = (int8_t*)malloc(elems);
                    if (!n->runtime_data_int8) { free(in_ids); goto fail; }
                    if (fread(n->runtime_data_int8, sizeof(int8_t), elems, f) != elems) { free(in_ids); goto fail; }
                    lancius_node_bind_owned_heap_int8(n, n->runtime_data_int8); /* A2 */
                } else if (n->dtype == LANCIUS_DTYPE_FP32) {
                    /* Despot V6 truth: FP32 payload loads (was coerced to FP64). */
                    n->runtime_data_f32 = (float*)malloc(elems * sizeof(float));
                    if (!n->runtime_data_f32) { free(in_ids); goto fail; }
                    if (fread(n->runtime_data_f32, sizeof(float), elems, f) != elems) { free(in_ids); goto fail; }
                    if (!ser_is_little_endian()) {
                        for (size_t j = 0; j < elems; j++) {
                            uint32_t u; memcpy(&u, &n->runtime_data_f32[j], 4);
                            u = ser_from_le32(u);
                            memcpy(&n->runtime_data_f32[j], &u, 4);
                        }
                    }
                    lancius_node_bind_owned_heap_f32(n, n->runtime_data_f32);
                } else {
                    n->runtime_data = (double*)malloc(elems * sizeof(double));
                    if (!n->runtime_data) { free(in_ids); goto fail; }
                    if (fread(n->runtime_data, sizeof(double), elems, f) != elems) { free(in_ids); goto fail; }
                    if (!ser_is_little_endian()) {
                        /* v12R1 fix: byte-swap FP64 payload on big-endian hosts. */
                        for (size_t j = 0; j < elems; j++)
                            n->runtime_data[j] = ser_bswap_double(n->runtime_data[j]);
                    }
                    lancius_node_bind_owned_heap(n, n->runtime_data); /* A2 */
                }
            }
        }
        free(in_ids);
    }
    free(id_map);
    fclose(f);

    /* A1: mirror loaded legacy buffers into runtime state */
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_runtime_sync_from_legacy(g->nodes[i]);
    }

    return g;

fail:
    lancius_set_error(LANCIUS_ERROR_INVALID_MODEL); /* A4 fail */
    /* Despot truth: buffers are bound OWNED_HEAP; manual free + destroy
     * double-freed every bound node (was: free loops here). */
    if (g) {
        lancius_graph_destroy(g);
    }
    free(id_map);
    fclose(f);
    return NULL;
}

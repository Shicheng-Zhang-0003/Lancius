#define _POSIX_C_SOURCE 200809L
#include "lancius/lancius_ir.h"
#include "lancius/lancius_validate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <threads.h>

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

/*
 * v11A3 format freeze: CRC32 (ISO 3309 / zlib-compatible).
 * Used to integrity-check the model body (bytes after the 48-byte header).
 */
static uint32_t crc32_table[256];
static once_flag crc32_table_once = ONCE_FLAG_INIT;

/* v12R1 fix: the table was lazily initialized under a plain flag — a data
 * race when two threads loaded models concurrently. C11 call_once is both
 * thread-safe and wait-free after first init. */
static void crc32_table_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            if (c & 1u) c = 0xEDB88320u ^ (c >> 1);
            else c >>= 1;
        }
        crc32_table[i] = c;
    }
}

static uint32_t lancius_crc32(uint32_t crc, const uint8_t* data, size_t len) {
    call_once(&crc32_table_once, crc32_table_init);
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

/* v12R1 fix: fread wrapper that folds every body byte into the running CRC
 * as it is parsed, so integrity is verified in a single pass (was: a second
 * full re-read of the body after parsing). */
static size_t crc_fread(void* ptr, size_t size, size_t nmemb, FILE* f, uint32_t* crc) {
    size_t got = fread(ptr, size, nmemb, f);
    if (got > 0 && size != 0) {
        *crc = lancius_crc32(*crc, (const uint8_t*)ptr, got * size);
    }
    return got;
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
    char tmp[PATH_MAX];
    FILE* f;
    if (!g || !path) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return -1; }
    if (!is_little_endian()) { lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP); return -1; }

    /* Despot V6 truth: mkstemp+fsync+rename (was predictable .tmp race). */
    {
        if (snprintf(tmp, sizeof(tmp), "%s.tmp.XXXXXX", path) >= (int)sizeof(tmp)) { lancius_set_error(LANCIUS_ERROR_LIMIT); return -1; }
        int fd = mkstemp(tmp);
        if (fd < 0) { lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        f = fdopen(fd, "w+b");
        if (!f) { close(fd); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    }

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
        fclose(f); unlink(tmp);
        lancius_set_error(LANCIUS_ERROR_IO);
        return -1;
    }

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (!n) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_NULL_PTR); return -1; }
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
        /* Despot truth: invalid dtype was silently coerced to FP64. Fail loud. */
        if (!lancius_dtype_is_valid(rn.dtype)) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_INVALID_DTYPE); return -1; }
        /* Despot V6 truth: per-channel scales not in format; refuse silent drop. */
        if (n->rt && n->rt->scale_per_channel) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP); return -1; }

        rn.has_weights = 0;
        rn.scale = n->scale;
        rn.weight_elems = 0;

        const void* data = NULL;
        size_t elem_size = sizeof(double);
        size_t elems = 0;

        if (n->op == LANCIUS_OP_INPUT) {
            size_t ne = 0;
            if (!lancius_node_elements_checked(n, &ne)) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_LIMIT); return -1; }
            if (rn.dtype == LANCIUS_DTYPE_INT8 && n->runtime_data_int8) {
                rn.has_weights = 1;
                data = n->runtime_data_int8;
                elem_size = sizeof(int8_t);
                elems = ne;
            } else if (rn.dtype == LANCIUS_DTYPE_FP32 && n->runtime_data_f32) {
                rn.has_weights = 1;
                rn.dtype = LANCIUS_DTYPE_FP32;
                data = n->runtime_data_f32;
                elem_size = sizeof(float);
                elems = ne;
            } else if (n->runtime_data) {
                rn.has_weights = 1;
                rn.dtype = LANCIUS_DTYPE_FP64;
                data = n->runtime_data;
                elem_size = sizeof(double);
                elems = ne;
            }
        }

        if (rn.has_weights && elems > 100000000u) {
            fclose(f); unlink(tmp);
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }

        rn.weight_elems = (uint64_t)elems;

        if (fwrite(&rn, 1, sizeof(rn), f) != sizeof(rn)) {
            fclose(f); unlink(tmp);
            lancius_set_error(LANCIUS_ERROR_IO);
            return -1;
        }

        for (uint32_t j = 0; j < n->input_count; j++) {
            uint32_t in_id = n->inputs[j] ? n->inputs[j]->id : UINT32_MAX;
            if (fwrite(&in_id, sizeof(uint32_t), 1, f) != 1) {
                fclose(f); unlink(tmp);
                lancius_set_error(LANCIUS_ERROR_IO);
                return -1;
            }
        }

        if (rn.has_weights && elems > 0) {
            if (fwrite(data, elem_size, elems, f) != elems) {
                fclose(f); unlink(tmp);
                lancius_set_error(LANCIUS_ERROR_IO);
                return -1;
            }
        }
    }

    /* v11A3 format freeze: stream CRC32 over the model body (no 800MB malloc,
     * checked seeks, offsetof not magic 40). Despot V6: ftello/off_t (was
     * long truncation); empty graph (body 0) is savable. */
    {
        off_t body_start = (off_t)sizeof(v2_header);
        off_t body_end, body_size, left;
        uint32_t crc = 0;
        uint8_t chunk[65536];
        if (fflush(f) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        body_end = ftello(f);
        if (body_end < 0 || body_end < body_start) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        body_size = body_end - body_start;
        if (fseeko(f, body_start, SEEK_SET) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        left = body_size;
        while (left > 0) {
            size_t want = (left < (off_t)sizeof(chunk)) ? (size_t)left : sizeof(chunk);
            size_t got = fread(chunk, 1, want, f);
            if (got != want) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
            crc = lancius_crc32(crc, chunk, got);
            left -= (off_t)got;
        }
        if (crc == 0) crc = 1; /* 0 means legacy/unverified; never emit it */
        if (fseeko(f, (off_t)offsetof(v2_header, checksum_crc32), SEEK_SET) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        if (fwrite(&crc, sizeof(uint32_t), 1, f) != 1) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        if (fflush(f) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        {
            int fd2 = fileno(f);
            if (fd2 >= 0 && fsync(fd2) != 0) { fclose(f); unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
        }
        if (fclose(f) != 0) { unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    }

    if (rename(tmp, path) != 0) { unlink(tmp); lancius_set_error(LANCIUS_ERROR_IO); return -1; }
    return 0;
}

lancius_graph* lancius_graph_load_v2(const char* path) {
    if (!path) { lancius_set_error(LANCIUS_ERROR_NULL_PTR); return NULL; }
    if (!is_little_endian()) { lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP); return NULL; }

    FILE* f = fopen(path, "rb");
    if (!f) { lancius_set_error(LANCIUS_ERROR_IO); return NULL; }

    v2_header h;
    if (fread(&h, 1, sizeof(h), f) != sizeof(h)) {
        lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
        fclose(f);
        return NULL;
    }

    if (h.magic != LANCIUS_MODEL_MAGIC_V2) {
        lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
        fclose(f);
        return NULL;
    }

    if (
        h.version != LANCIUS_MODEL_VERSION_V2 ||
        !(h.flags & LANCIUS_MODEL_FLAG_LITTLE_ENDIAN) ||
        (h.flags & LANCIUS_MODEL_FLAG_EXTERNAL_WEIGHTS) || /* reserved, reject */
        h.header_size != sizeof(h) ||
        h.node_count > 1000000u ||
        h.reserved0 != 0 ||
        h.reserved1 != 0 ||
        h.weight_block_offset != 0 ||
        h.attribute_count != 0
    ) {
        lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
        fclose(f);
        return NULL;
    }

    lancius_graph* g = lancius_graph_create();
    if (!g) {
        lancius_set_error(LANCIUS_ERROR_OOM);
        fclose(f);
        return NULL;
    }

    idmap map = {NULL, 0};
    uint32_t* in_ids = NULL;
    /* v12R1 fix: running CRC over the model body, folded in as each record
     * is parsed (was: a second full re-read of the body after parsing). */
    uint32_t computed_crc = 0;
    // Hostile fix: seen_ids linear O(n^2) scan removed; map_get is the duplicate oracle (O(1)).
    // Bound sparse IDs to prevent 80MB realloc DoS: ids must be dense-ish.
    // Legit saves emit dense ids < next_id <= node_count + NOP slack.

    for (uint32_t i = 0; i < h.node_count; i++) {
        v2_node rn;

        if (crc_fread(&rn, 1, sizeof(rn), f, &computed_crc) != sizeof(rn)) goto fail;

        /* Despot V6 truth: ndim==0 rejected (was coerced to 2D). */
        if (rn.ndim == 0 || rn.ndim > 4) goto fail;
        if (rn.input_count > 16u) goto fail;
        if (rn.weight_elems > 100000000ull) goto fail;
        if (!lancius_dtype_is_valid(rn.dtype)) goto fail;
        if (rn.op > LANCIUS_MODEL_OP_SUM_AXIS_ND) goto fail;
        if (rn.dtype != LANCIUS_DTYPE_FP64 && rn.dtype != LANCIUS_DTYPE_INT8 && rn.dtype != LANCIUS_DTYPE_FP32) goto fail;
        // Hostile fix: bound sparse id (DoS via 10M-pointer realloc + O(n^2))
        if (rn.id >= 10000000u) goto fail;
        if (h.node_count > 0 && rn.id >= h.node_count * 16u + 1024u) goto fail;

        (void)rn.flags;

        /* Despot truth: u64->size_t narrowing is checked for 32-bit hosts. */
        size_t sh[4];
        for (int s = 0; s < 4; s++) {
            if (rn.shape[s] > (uint64_t)SIZE_MAX) goto fail;
            sh[s] = (size_t)rn.shape[s];
        }

        in_ids = NULL;
        if (rn.input_count > 0) {
            in_ids = (uint32_t*)malloc((size_t)rn.input_count * sizeof(uint32_t));
            if (!in_ids) goto fail;

            if (crc_fread(in_ids, sizeof(uint32_t), rn.input_count, f, &computed_crc) != rn.input_count) {
                goto fail;
            }
        }

        /* A3: reject forward/missing input references */
        for (uint32_t j = 0; j < rn.input_count; j++) {
            if (!map_get(&map, in_ids[j])) goto fail;
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
 /* v12R1-201: validate shape; reject oversized / zero-dim / bad-rank.
  * Despot V6: 0/1-D rejected (was coerced to 2D). Only 2/3/4-D persistable. */
if (rn.ndim == 4) {
    if (lancius_validate_shape(sh, 4) != LANCIUS_ERROR_OK) goto fail;
    n = lancius_input_4d(g, sh[0], sh[1], sh[2], sh[3]);
} else if (rn.ndim == 3) {
    if (lancius_validate_shape(sh, 3) != LANCIUS_ERROR_OK) goto fail;
    n = lancius_input_3d(g, sh[0], sh[1], sh[2]);
} else if (rn.ndim == 2) {
    if (lancius_validate_shape(sh, 2) != LANCIUS_ERROR_OK) goto fail;
    n = lancius_input(g, sh[0], sh[1]);
} else {
    goto fail;
}
break;

            case LANCIUS_MODEL_OP_CONST:
                /* Despot V6 truth: 1..4-D CONST round-trips (was 2D-only). */
                if (rn.ndim < 1 || rn.ndim > 4) goto fail;
                if (rn.ndim == 2) n = lancius_const(g, rn.attr, sh[0], sh[1]);
                else {
                    size_t cshape[4] = {sh[0], sh[1], sh[2], sh[3]};
                    if (lancius_validate_shape(cshape, rn.ndim) != LANCIUS_ERROR_OK) goto fail;
                    n = lancius_const_scalar(g, rn.attr, rn.ndim);
                    if (n) {
                        for (int _s = 0; _s < rn.ndim; _s++) n->shape[_s] = sh[_s];
                    }
                }
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

            case LANCIUS_MODEL_OP_BROADCAST: {
                /* Despot truth: BROADCAST persists for any 1..4-D shape
                 * (SUM grads use 1/3-D via broadcast_to_shape). */
                size_t bshape[4] = {sh[0], sh[1], sh[2], sh[3]};
                if (rn.ndim == 4) n = lancius_broadcast_4d(g, in0, sh[0], sh[1], sh[2], sh[3]);
                else if (rn.ndim == 2) n = lancius_broadcast(g, in0, sh[0], sh[1]);
                else if (rn.ndim >= 1 && rn.ndim <= 4) n = lancius_broadcast_to_shape(g, in0, bshape, rn.ndim);
                else goto fail;
                break;
            }

            case LANCIUS_MODEL_OP_TRANSPOSE:
                n = lancius_transpose(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM_AXIS0:
                n = lancius_sum_axis0(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM_AXIS1:
                n = lancius_sum_axis1(g, in0);
                break;

            case LANCIUS_MODEL_OP_SUM_AXIS_ND:
                /* R3-1: axis persisted in axes[0]; builder validates. */
                if (rn.axes[0] >= 4u) goto fail;
                n = lancius_sum_axis_nd(g, in0, rn.axes[0]);
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

            /* v12R2 generic primitives, forward only: training graphs with
             * _BWD nodes are execution artifacts, never persisted. */
            case LANCIUS_MODEL_OP_TANH:
                n = lancius_tanh(g, in0);
                break;

            case LANCIUS_MODEL_OP_MSE:
                n = lancius_mse(g, in0, in1);
                break;

            case LANCIUS_MODEL_OP_ROPE:
                /* Despot V6 truth: ROPE persistable (was save-ok/load-fail). */
                if (rn.ndim != 3) goto fail;
                {
                    size_t _hd2 = sh[2];
                    if (_hd2 % 2 != 0) goto fail;
                    n = lancius_rope(g, in0, sh[0], sh[1], _hd2 / 2);
                }
                break;

            default:
                lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
                n = NULL;
                break;
        }

        /* v12R1-201: non-NOP op that failed to reconstruct = corrupt model */
        if (rn.op != LANCIUS_MODEL_OP_NOP && !n) goto fail;
        if (n) {
            n->kernel_h = rn.meta[0];
            n->kernel_w = rn.meta[1];
            n->stride = rn.meta[2];
            n->pad = rn.meta[3];

            memcpy(n->axes, rn.axes, sizeof(rn.axes));

            n->dtype = (lancius_dtype)rn.dtype;
            /* Despot V6 truth: INT8 scale must be >0 finite (was unchecked). */
            if (rn.dtype == LANCIUS_DTYPE_INT8) {
                if (!(rn.scale > 0.0) || !isfinite(rn.scale)) goto fail;
            }
            n->scale = rn.scale;
        }

        /*
         * v11A2 Section 12:
         * Reject malformed models where file-provided weight length
         * disagrees with the node shape.
         */
        if (rn.has_weights && rn.weight_elems > 0 && n) {
            size_t expected_elems = 0;
            if (!lancius_node_elements_checked(n, &expected_elems)) goto fail;
            if ((uint64_t)expected_elems != rn.weight_elems) goto fail;
        }

        if (rn.has_weights && rn.weight_elems > 0) {
            size_t elems = (size_t)rn.weight_elems;
            size_t elem_size = (rn.dtype == LANCIUS_DTYPE_INT8) ? sizeof(int8_t) :
                               ((rn.dtype == LANCIUS_DTYPE_FP32) ? sizeof(float) : sizeof(double));
            size_t bytes = elems * elem_size;

            if (elem_size != 0 && bytes / elem_size != elems) goto fail;

            if (n) {
                if (rn.dtype == LANCIUS_DTYPE_INT8) {
                    int8_t* buf = (int8_t*)malloc(bytes);
                    if (!buf) goto fail;

                    if (crc_fread(buf, 1, bytes, f, &computed_crc) != bytes) {
                        free(buf);
                        goto fail;
                    }

                    n->runtime_data_int8 = buf;
                    lancius_node_bind_owned_heap_int8(n, buf);
                } else if (rn.dtype == LANCIUS_DTYPE_FP32) {
                    float* buf = (float*)malloc(bytes);
                    if (!buf) goto fail;
                    if (crc_fread(buf, sizeof(float), elems, f, &computed_crc) != elems) {
                        free(buf);
                        goto fail;
                    }
                    n->runtime_data_f32 = buf;
                    lancius_node_bind_owned_heap_f32(n, buf);
                } else {
                    double* buf = (double*)malloc(bytes);
                    if (!buf) goto fail;

                    if (crc_fread(buf, sizeof(double), elems, f, &computed_crc) != elems) {
                        free(buf);
                        goto fail;
                    }

                    n->runtime_data = buf;
                    lancius_node_bind_owned_heap(n, buf);
                }
            } else {
                // Hostile fix: no fseek with (long) truncation, no pipe failure. Stream-skip.
                size_t to_skip = bytes;
                uint8_t tmp[4096];
                while (to_skip > 0) {
                    size_t chunk = to_skip < sizeof(tmp) ? to_skip : sizeof(tmp);
                    if (crc_fread(tmp, 1, chunk, f, &computed_crc) != chunk) goto fail;
                    to_skip -= chunk;
                }
            }
        }

        /* A3: reject duplicate node ids.
         * v12R1 fix: NOP ids are skipped entirely — they are neither mapped
         * nor duplicate-checked (a NOP is invisible to map_get, so it can
         * never be referenced as an input and never collides). */
        if (n) {
            if (map_get(&map, rn.id)) goto fail;
            if (!map_set(&map, rn.id, n)) goto fail;
        }
    }


    /* v11A3 format freeze: CRC32 required. checksum==0 (legacy unverified) is
     * rejected by default — it lets an attacker zero 4 bytes to bypass integrity.
     * Set LANCIUS_ALLOW_LEGACY_UNVERIFIED=1 to opt into legacy loads. */
    {
        int allow_legacy = 0;
        const char* env = getenv("LANCIUS_ALLOW_LEGACY_UNVERIFIED");
        if (env && env[0] == '1') allow_legacy = 1;
        if (h.checksum_crc32 == 0 && !allow_legacy) goto fail;
    }
    /* v12R1 fix: the CRC was computed by re-reading the whole body after
     * parsing (extra I/O, plus a TOCTOU window on truncated files). It is
     * now folded into computed_crc while parsing; the save side normalizes
     * a zero CRC to 1, so mirror that here for a consistent comparison. */
    if (h.checksum_crc32 != 0) {
        if (computed_crc == 0) computed_crc = 1;
        if (computed_crc != h.checksum_crc32) goto fail;
    }

    free(map.v);
    fclose(f);

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_runtime_sync_from_legacy(g->nodes[i]);
    }
    lancius_clear_error();

    return g;

fail:
    lancius_set_error(LANCIUS_ERROR_INVALID_MODEL);
    free(in_ids);
    free(map.v);
    if (g) lancius_graph_destroy(g);
    fclose(f);
    return NULL;
}

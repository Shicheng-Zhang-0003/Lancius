#ifndef LANCIUS_MODEL_FORMAT_H
#define LANCIUS_MODEL_FORMAT_H

#include <stdint.h>

/* B1: Lancius model format definition */

#define LANCIUS_MODEL_MAGIC_V1 0x21434E41u
#define LANCIUS_MODEL_MAGIC_V2 0x32434E41u

#define LANCIUS_MODEL_VERSION_V1 1u
#define LANCIUS_MODEL_VERSION_V2 2u

#define LANCIUS_MODEL_HEADER_SIZE_V2 48u
#define LANCIUS_MODEL_MAX_RANK 4u

#define LANCIUS_MODEL_FLAG_LITTLE_ENDIAN   (1u << 0)
#define LANCIUS_MODEL_FLAG_STATIC_GRAPH    (1u << 1)
#define LANCIUS_MODEL_FLAG_EXTERNAL_WEIGHTS (1u << 2)

typedef enum {
    LANCIUS_MODEL_DTYPE_FP64 = 0,
    LANCIUS_MODEL_DTYPE_INT8 = 1,
    LANCIUS_MODEL_DTYPE_FP32 = 2,
    LANCIUS_MODEL_DTYPE_INT32 = 3
} lancius_model_dtype;

typedef enum {
    LANCIUS_MODEL_OP_NOP = 0,
    LANCIUS_MODEL_OP_INPUT = 1,
    LANCIUS_MODEL_OP_CONST = 2,
    LANCIUS_MODEL_OP_ADD = 3,
    LANCIUS_MODEL_OP_SUB = 4,
    LANCIUS_MODEL_OP_MUL = 5,
    LANCIUS_MODEL_OP_MATMUL = 6,
    LANCIUS_MODEL_OP_RELU = 7,
    LANCIUS_MODEL_OP_SOFTMAX = 8,
    LANCIUS_MODEL_OP_SUM = 9,
    LANCIUS_MODEL_OP_BROADCAST = 10,
    LANCIUS_MODEL_OP_TRANSPOSE = 11,
    LANCIUS_MODEL_OP_RELU_BWD = 12,
    LANCIUS_MODEL_OP_SOFTMAX_BWD = 13,
    LANCIUS_MODEL_OP_SUM_AXIS0 = 14,
    LANCIUS_MODEL_OP_SUM_AXIS1 = 15,
    LANCIUS_MODEL_OP_CONV2D = 16,
    LANCIUS_MODEL_OP_MAXPOOL2D = 17,
    LANCIUS_MODEL_OP_FLATTEN = 18,
    LANCIUS_MODEL_OP_CONV2D_RELU_FUSED = 19,
    LANCIUS_MODEL_OP_CROSS_ENTROPY = 20,
    LANCIUS_MODEL_OP_CROSS_ENTROPY_BWD = 21,
    LANCIUS_MODEL_OP_PERMUTE = 22,
    LANCIUS_MODEL_OP_MATMUL_BATCHED = 23,
    LANCIUS_MODEL_OP_CONV2D_BWD = 24,
    LANCIUS_MODEL_OP_CONV2D_BWD_W = 25,
    LANCIUS_MODEL_OP_MAXPOOL2D_BWD = 26,
    LANCIUS_MODEL_OP_RESHAPE = 27,
    LANCIUS_MODEL_OP_EMBEDDING = 28,
    LANCIUS_MODEL_OP_LAYERNORM = 29,
    LANCIUS_MODEL_OP_GELU = 30,
    LANCIUS_MODEL_OP_ROPE = 31,
    LANCIUS_MODEL_OP_ATTENTION = 32,
    LANCIUS_MODEL_OP_KV_CACHE_READ = 33,
    LANCIUS_MODEL_OP_KV_CACHE_WRITE = 34,
    LANCIUS_MODEL_OP_RMSNORM = 35,
    LANCIUS_MODEL_OP_SWIGLU = 36,
    LANCIUS_MODEL_OP_GQA = 37
} lancius_model_op;

/*
 * Logical v2 header layout.
 *
 * This struct describes the intended binary header.
 * Serialization should still use explicit fixed-width little-endian writes.
 */
typedef struct {
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
} lancius_model_header_v2;

_Static_assert(
    sizeof(lancius_model_header_v2) == LANCIUS_MODEL_HEADER_SIZE_V2,
    "lancius_model_header_v2 must be exactly 48 bytes"
);

#endif /* LANCIUS_MODEL_FORMAT_H */

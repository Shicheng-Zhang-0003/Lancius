#ifndef LANCIUS_TYPES_H
#define LANCIUS_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* A1: shared tensor/runtime contract types */

typedef enum {
    LANCIUS_DTYPE_FP64 = 0,
    LANCIUS_DTYPE_INT8 = 1,
    LANCIUS_DTYPE_FP32 = 2,
    LANCIUS_DTYPE_INT32 = 3
} lancius_dtype;

/* A3: dtype utilities */
static inline size_t lancius_dtype_size(lancius_dtype dtype) {
    switch (dtype) {
        case LANCIUS_DTYPE_FP32:  return sizeof(float);
        case LANCIUS_DTYPE_FP64:  return sizeof(double);
        case LANCIUS_DTYPE_INT8:  return sizeof(int8_t);
        case LANCIUS_DTYPE_INT32: return sizeof(int32_t);
        default:                  return sizeof(double);
    }
}

static inline const char* lancius_dtype_name(lancius_dtype dtype) {
    switch (dtype) {
        case LANCIUS_DTYPE_FP32:  return "fp32";
        case LANCIUS_DTYPE_FP64:  return "fp64";
        case LANCIUS_DTYPE_INT8:  return "int8";
        case LANCIUS_DTYPE_INT32: return "int32";
        default:                  return "unknown";
    }
}

static inline int lancius_dtype_is_valid(uint8_t code) {
    return code <= LANCIUS_DTYPE_INT32;
}

typedef enum {
    LANCIUS_MEMORY_EXTERNAL = 0,
    LANCIUS_MEMORY_ARENA = 1,
    LANCIUS_MEMORY_POOL = 2,
    LANCIUS_MEMORY_OWNED_HEAP = 3
} lancius_memory_owner;

typedef struct lancius_runtime_state {
    void* buffer;
    int8_t* buffer_int8;
    lancius_dtype dtype;
    double scale;
    lancius_memory_owner owner;

    /* A2: split ownership for legacy FP64 and INT8 buffers */
    lancius_memory_owner buffer_owner;
    lancius_memory_owner int8_owner;
    size_t offset;
    uint32_t flags;
} lancius_runtime_state;

/* A2: ownership flag bits */
#define LANCIUS_RT_FLAG_AUTOFREE_BUFFER (1u << 0)
#define LANCIUS_RT_FLAG_AUTOFREE_INT8   (1u << 1)

#endif /* LANCIUS_TYPES_H */

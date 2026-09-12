#include "lancius/lancius_stable_api.h"
#include "lancius/lancius_checked.h"
#include "lancius/lancius_ir.h" /* for LANCIUS_MAX_TENSOR_ELEMS */
#include "lancius/lancius_ir.h"
#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_arena.h"
#include <stdlib.h>
#include <string.h>

// Thread-Local Error State (Production Standard)
static _Thread_local lancius_status g_last_error = LANCIUS_OK;

static void set_error(lancius_status err) {
    g_last_error = err;
}

static lancius_status map_internal_error(lancius_error err) {
    switch (err) {
        case LANCIUS_ERROR_OK:               return LANCIUS_OK;
        case LANCIUS_ERROR_OOM:              return LANCIUS_ERR_OOM;
        case LANCIUS_ERROR_NULL_PTR:         return LANCIUS_ERR_NULL_PTR;
        case LANCIUS_ERROR_SHAPE_MISMATCH:   return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_UNSUPPORTED_OP:   return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_UNSUPPORTED_DTYPE:return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_INVALID_DTYPE:    return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_INVALID_MODEL:    return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_VERSION_MISMATCH: return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_IO:               return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_GRAPH_CYCLE:      return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_GRAPH_INVALID:    return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_OVERFLOW:         return LANCIUS_ERR_OOM;
        case LANCIUS_ERROR_LIMIT:            return LANCIUS_ERR_OOM;
        case LANCIUS_ERROR_INTERNAL:         return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_INVALID_HANDLE:   return LANCIUS_ERR_NULL_PTR;
        case LANCIUS_ERROR_LIFETIME:         return LANCIUS_ERR_NULL_PTR;
        case LANCIUS_ERROR_INVALID_RANK:     return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_SHAPE:    return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_STRIDE:   return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_PERMUTATION: return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_RESHAPE_MISMATCH: return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_NUMERICAL:        return LANCIUS_ERR_SHAPE_MISMATCH;
        default:                             return LANCIUS_ERR_UNSUPPORTED_OP;
    }
}

static void sync_internal_error(void) {
    lancius_error err = lancius_get_error();
    if (err != LANCIUS_ERROR_OK) {
        set_error(map_internal_error(err));
        lancius_clear_error();
    }
}


LANCIUS_EXPORT lancius_status lancius_get_last_error(void) {
    return g_last_error;
}

LANCIUS_EXPORT const char* lancius_get_error_string(lancius_status err) {
    switch(err) {
        case LANCIUS_OK: return "Success";
        case LANCIUS_ERR_OOM: return "Out of Memory";
        case LANCIUS_ERR_SHAPE_MISMATCH: return "Shape Mismatch";
        case LANCIUS_ERR_NULL_PTR: return "Null Pointer";
        case LANCIUS_ERR_UNSUPPORTED_OP: return "Unsupported Operation";
        case LANCIUS_ERR_BUFFER_TOO_SMALL: return "Buffer Too Small";
        case LANCIUS_ERR_IO: return "I/O Error";
        default: return "Unknown Error";
    }
}

typedef struct {
    lancius_arena* arena;
} lancius_context_internal;

// V1.0 FIX: Wrap the graph with its execution state (scratch arena & schedule)
// This prevents the dangling pointer segfault when reading outputs!
typedef struct {
    lancius_graph* g;
    lancius_arena* scratch;
    lancius_schedule* sched;
} lancius_graph_internal;

LANCIUS_EXPORT lancius_context lancius_create_context(void) {
    lancius_context_internal* ctx = (lancius_context_internal*)malloc(sizeof(lancius_context_internal));
    if (!ctx) { set_error(LANCIUS_ERR_OOM); return NULL; }
    ctx->arena = lancius_arena_create(64 * 1024 * 1024); // 64MB default scratch
    if (!ctx->arena) { free(ctx); set_error(LANCIUS_ERR_OOM); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_context)ctx;
}

LANCIUS_EXPORT void lancius_destroy_context(lancius_context ctx) {
    if (!ctx) return;
    lancius_context_internal* internal = (lancius_context_internal*)ctx;
    if (internal->arena) lancius_arena_destroy(internal->arena);
    free(internal);
}

LANCIUS_EXPORT lancius_graph_handle lancius_graph_create_stable(lancius_context ctx) {
    if (!ctx) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }

    lancius_graph_internal* wrapper = (lancius_graph_internal*)malloc(sizeof(lancius_graph_internal));
    if (!wrapper) { set_error(LANCIUS_ERR_OOM); return NULL; }

    wrapper->g = lancius_graph_create();
    if (!wrapper->g) { free(wrapper); sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_OOM); return NULL; }

    wrapper->scratch = lancius_arena_create(16 * 1024 * 1024); // 16MB execution scratch
    wrapper->sched = NULL;

    set_error(LANCIUS_OK);
    return (lancius_graph_handle)wrapper;
}

LANCIUS_EXPORT void lancius_graph_destroy_stable(lancius_graph_handle g) {
    if (!g) return;
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    if (wrapper->sched) lancius_schedule_destroy(wrapper->sched);
    if (wrapper->scratch) lancius_arena_destroy(wrapper->scratch);
    if (wrapper->g) lancius_graph_destroy(wrapper->g);
    free(wrapper);
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_input(lancius_graph_handle g, size_t rows, size_t cols) {
    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    /* v12R1-205: prevent abort() via stable API on oversized tensors */
    size_t elems = 0;
    if (!lancius_checked_mul_size(rows, cols, &elems) || elems > LANCIUS_MAX_TENSOR_ELEMS) {
        set_error(LANCIUS_ERR_SHAPE_MISMATCH);
        return NULL;
    }
    lancius_node* n = lancius_input(wrapper->g, rows, cols);
    if (!n) { set_error(LANCIUS_ERR_OOM); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_matmul(lancius_graph_handle g, lancius_tensor_handle a, lancius_tensor_handle b) {
    if (!g || !a || !b) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    lancius_node* n = lancius_matmul(wrapper->g, (lancius_node*)a, (lancius_node*)b);
    if (!n) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_SHAPE_MISMATCH); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_relu(lancius_graph_handle g, lancius_tensor_handle a) {
    if (!g || !a) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    lancius_node* n = lancius_relu(wrapper->g, (lancius_node*)a);
    if (!n) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_SHAPE_MISMATCH); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_status lancius_bind_data(lancius_tensor_handle t, double* data_ptr) {
    if (!t || !data_ptr) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_node* n = (lancius_node*)t;
    /* A2: external data binding is explicitly non-owned */
    lancius_node_bind_external(n, data_ptr);
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_compile_and_run(lancius_graph_handle g) {
    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;

    if (wrapper->sched) lancius_schedule_destroy(wrapper->sched);

    lancius_clear_error();
    wrapper->sched = lancius_ir_schedule(wrapper->g);
    if (!wrapper->sched) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_OOM); return g_last_error; }

    /* v11A3 fix: auto-size scratch arena from liveness analysis */
    {
        size_t peak = lancius_schedule_peak_memory(wrapper->sched);
        if (peak > SIZE_MAX - (1024 * 1024)) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
        size_t needed = peak + (1024 * 1024); /* 1MB headroom */
        if (needed > 16 * 1024 * 1024) {
            lancius_arena_destroy(wrapper->scratch);
            wrapper->scratch = lancius_arena_create(needed);
            if (!wrapper->scratch) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
        }
    }
    lancius_arena_reset(wrapper->scratch);
    lancius_clear_error();
    lancius_schedule_execute(wrapper->sched, wrapper->scratch);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        sync_internal_error();
        return g_last_error;
    }
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_read_output(lancius_tensor_handle t, double* out_buffer, size_t buffer_size) {
    if (!t || !out_buffer) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_node* n = (lancius_node*)t;
    if (n->dtype != LANCIUS_DTYPE_FP64) { set_error(LANCIUS_ERR_UNSUPPORTED_OP); return LANCIUS_ERR_UNSUPPORTED_OP; }
    if (!n->runtime_data) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }

    if (buffer_size % sizeof(double) != 0) {
        set_error(LANCIUS_ERR_SHAPE_MISMATCH);
        return LANCIUS_ERR_SHAPE_MISMATCH;
    }
    size_t elems = 0;
    if (!lancius_node_elements_checked(n, &elems)) { set_error(LANCIUS_ERR_SHAPE_MISMATCH); return LANCIUS_ERR_SHAPE_MISMATCH; }
    size_t buffer_elems = buffer_size / sizeof(double);
    /* v11A3 fix: reject silent truncation */
    if (buffer_elems < elems) {
        set_error(LANCIUS_ERR_BUFFER_TOO_SMALL);
        return LANCIUS_ERR_BUFFER_TOO_SMALL;
    }
    if (elems > SIZE_MAX / sizeof(double)) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
    memcpy(out_buffer, n->runtime_data, elems * sizeof(double));
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}


/* v11A3 stable API expansion: model I/O */
LANCIUS_EXPORT lancius_graph_handle lancius_graph_load_stable(lancius_context ctx, const char* path) {
    if (!ctx || !path) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph* g = lancius_graph_load(path);
    if (!g) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_IO); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)malloc(sizeof(lancius_graph_internal));
    if (!wrapper) { lancius_graph_destroy(g); set_error(LANCIUS_ERR_OOM); return NULL; }
    wrapper->g = g;
    wrapper->scratch = lancius_arena_create(64 * 1024 * 1024); /* 64MB default for loaded models */
    wrapper->sched = NULL;
    if (!wrapper->scratch) { lancius_graph_destroy(g); free(wrapper); set_error(LANCIUS_ERR_OOM); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_graph_handle)wrapper;
}

LANCIUS_EXPORT lancius_status lancius_graph_save_stable(lancius_graph_handle g, const char* path) {
    if (!g || !path) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    /* v11S H1 fix: propagate save failure to FFI consumers */
    if (lancius_graph_save(wrapper->g, path) != 0) {
        set_error(LANCIUS_ERR_IO);
        return LANCIUS_ERR_IO;
    }
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

/* v11A3 stable API expansion: tensor introspection */
LANCIUS_EXPORT size_t lancius_tensor_element_count(lancius_tensor_handle t) {
    if (!t) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    lancius_node* n = (lancius_node*)t;
    size_t elems = 0;
    if (!lancius_node_elements_checked(n, &elems)) { set_error(LANCIUS_ERR_SHAPE_MISMATCH); return 0; }
    set_error(LANCIUS_OK);
    return elems;
}

/* A3: dtype query */
LANCIUS_EXPORT int lancius_tensor_get_dtype(lancius_tensor_handle t) {
    if (!t) {
        set_error(LANCIUS_ERR_NULL_PTR);
        return -1;
    }

    lancius_node* n = (lancius_node*)t;
    set_error(LANCIUS_OK);
    return (int)n->dtype;
}
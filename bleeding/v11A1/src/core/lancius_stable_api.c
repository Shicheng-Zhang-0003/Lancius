#include "lancius/lancius_stable_api.h"
#include "lancius/lancius_ir.h"
#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_arena.h"
#include <stdlib.h>
#include <string.h>

// Thread-Local Error State (Production Standard)
static __thread lancius_status g_last_error = LANCIUS_OK;

static void set_error(lancius_status err) {
    g_last_error = err;
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
    if (!wrapper->g) { free(wrapper); set_error(LANCIUS_ERR_OOM); return NULL; }

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
    lancius_node* n = lancius_input(wrapper->g, rows, cols);
    if (!n) { set_error(LANCIUS_ERR_OOM); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_matmul(lancius_graph_handle g, lancius_tensor_handle a, lancius_tensor_handle b) {
    if (!g || !a || !b) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    lancius_node* n = lancius_matmul(wrapper->g, (lancius_node*)a, (lancius_node*)b);
    if (!n) { set_error(LANCIUS_ERR_SHAPE_MISMATCH); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_relu(lancius_graph_handle g, lancius_tensor_handle a) {
    if (!g || !a) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;
    lancius_node* n = lancius_relu(wrapper->g, (lancius_node*)a);
    if (!n) { set_error(LANCIUS_ERR_OOM); return NULL; }
    set_error(LANCIUS_OK);
    return (lancius_tensor_handle)n;
}

LANCIUS_EXPORT lancius_status lancius_bind_data(lancius_tensor_handle t, double* data_ptr) {
    if (!t || !data_ptr) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_node* n = (lancius_node*)t;
    n->runtime_data = data_ptr;
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_compile_and_run(lancius_graph_handle g) {
    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;

    if (wrapper->sched) lancius_schedule_destroy(wrapper->sched);

    wrapper->sched = lancius_ir_schedule(wrapper->g);
    if (!wrapper->sched) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }

    lancius_arena_reset(wrapper->scratch);
    lancius_schedule_execute(wrapper->sched, wrapper->scratch);

    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_read_output(lancius_tensor_handle t, double* out_buffer, size_t buffer_size) {
    if (!t || !out_buffer) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    lancius_node* n = (lancius_node*)t;
    if (!n->runtime_data) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }

    if (buffer_size % sizeof(double) != 0) {
        set_error(LANCIUS_ERR_SHAPE_MISMATCH);
        return LANCIUS_ERR_SHAPE_MISMATCH;
    }
    size_t elems = lancius_node_elements(n);
    size_t copy_size = (elems < buffer_size / sizeof(double)) ? elems : buffer_size / sizeof(double);
    memcpy(out_buffer, n->runtime_data, copy_size * sizeof(double));

    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

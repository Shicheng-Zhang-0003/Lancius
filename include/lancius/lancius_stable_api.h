/**
 * @file lancius_stable_api.h
 * @brief Stable FFI API: Opaque handles and thread-local error states for safe Python/Rust/C++ bindings.
 */
#ifndef LANCIUS_STABLE_API_H
#define LANCIUS_STABLE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #define LANCIUS_EXPORT __declspec(dllexport)
#else
  #define LANCIUS_EXPORT __attribute__((visibility("default")))
#endif

/* Opaque handles for FFI safety. Lifetimes (binding contract):
 * - context owns an arena; destroy with lancius_destroy_context when done.
 * - graph handles are independent of the context after creation (context may
 *   be destroyed first); destroy each graph with lancius_graph_destroy_stable.
 * - tensor handles borrow from their graph (do NOT free); they die with it.
 * - handles carry magic tags; wrong-type / stale / destroyed handles return
 *   INVALID_HANDLE instead of corrupting memory. NULL always fails loud. */
typedef void* lancius_context;
typedef void* lancius_graph_handle;
typedef void* lancius_tensor_handle;

// Production Error Codes
typedef enum {
    LANCIUS_OK = 0,
    LANCIUS_ERR_OOM = -1,
    LANCIUS_ERR_SHAPE_MISMATCH = -2,
    LANCIUS_ERR_NULL_PTR = -3,
    LANCIUS_ERR_UNSUPPORTED_OP = -4,
    LANCIUS_ERR_BUFFER_TOO_SMALL = -5,
    LANCIUS_ERR_IO = -6,
    // v12R1 hostile fix: preserve cycle/overflow/numerical/handle causes (were collapsed to SHAPE/OOM)
    LANCIUS_ERR_GRAPH_CYCLE = -7,
    LANCIUS_ERR_OVERFLOW = -8,
    LANCIUS_ERR_NUMERICAL = -9,
    LANCIUS_ERR_INVALID_HANDLE = -10,
    // v12R1 fix: distinguish dtype failures from unsupported-op failures
    LANCIUS_ERR_UNSUPPORTED_DTYPE = -11
} lancius_status;

// Error Handling
LANCIUS_EXPORT lancius_status lancius_get_last_error(void);
LANCIUS_EXPORT const char* lancius_get_error_string(lancius_status err);

// Context & Graph Lifecycle
LANCIUS_EXPORT lancius_context lancius_create_context(void);
LANCIUS_EXPORT void lancius_destroy_context(lancius_context ctx);

LANCIUS_EXPORT lancius_graph_handle lancius_graph_create_stable(lancius_context ctx);
LANCIUS_EXPORT void lancius_graph_destroy_stable(lancius_graph_handle g);

// Tensor Builders
LANCIUS_EXPORT lancius_tensor_handle lancius_add_input(lancius_graph_handle g, size_t rows, size_t cols);
LANCIUS_EXPORT lancius_tensor_handle lancius_add_matmul(lancius_graph_handle g, lancius_tensor_handle a, lancius_tensor_handle b);
LANCIUS_EXPORT lancius_tensor_handle lancius_add_relu(lancius_graph_handle g, lancius_tensor_handle a);

// Data Binding & Execution
LANCIUS_EXPORT lancius_status lancius_bind_data(lancius_tensor_handle t, double* data_ptr);
LANCIUS_EXPORT lancius_status lancius_compile_and_run(lancius_graph_handle g);
LANCIUS_EXPORT lancius_status lancius_read_output(lancius_tensor_handle t, double* out_buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif


/* v11A3 stable API expansion: model I/O */
LANCIUS_EXPORT lancius_graph_handle lancius_graph_load_stable(lancius_context ctx, const char* path);
LANCIUS_EXPORT lancius_status lancius_graph_save_stable(lancius_graph_handle g, const char* path);

/* v11A3 stable API expansion: tensor introspection */
LANCIUS_EXPORT size_t lancius_tensor_element_count(lancius_tensor_handle t);

/* A3: dtype query API */
LANCIUS_EXPORT int lancius_tensor_get_dtype(lancius_tensor_handle t);

#endif // LANCIUS_STABLE_API_H

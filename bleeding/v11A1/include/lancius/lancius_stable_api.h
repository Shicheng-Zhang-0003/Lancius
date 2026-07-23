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

// Opaque Handles for FFI Safety (Python/Rust/Go cannot corrupt internal structs)
typedef void* lancius_context;
typedef void* lancius_graph_handle;
typedef void* lancius_tensor_handle;

// Production Error Codes
typedef enum {
    LANCIUS_OK = 0,
    LANCIUS_ERR_OOM = -1,
    LANCIUS_ERR_SHAPE_MISMATCH = -2,
    LANCIUS_ERR_NULL_PTR = -3,
    LANCIUS_ERR_UNSUPPORTED_OP = -4
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

#endif // LANCIUS_STABLE_API_H

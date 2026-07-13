
#ifndef BASIS_ERROR_H
#define BASIS_ERROR_H
#include <stdio.h>
#include <stdbool.h>

typedef enum {
    BASIS_SUCCESS = 0,
    BASIS_NULL_POINTER,
    BASIS_OUT_OF_MEMORY,
    BASIS_INVALID_SHAPE,
    BASIS_INVALID_EXPRESSION,
    BASIS_DOMAIN_ERROR,
    BASIS_UNKNOWN_ERROR
} basis_error_t;

extern __thread basis_error_t _basis_last_error;
extern __thread const char* _basis_last_error_msg;
extern __thread bool _basis_error_quiet;

#define BASIS_SET_ERROR(err, msg) do { \
    _basis_last_error = err; \
    _basis_last_error_msg = msg; \
    if (!_basis_error_quiet) { \
        fprintf(stderr, "[BASIS ERROR] %s: %s (File: %s, Line: %d)\n", #err, msg, __FILE__, __LINE__); \
    } \
} while(0)

#define BASIS_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        BASIS_SET_ERROR(BASIS_NULL_POINTER, "Null pointer encountered"); \
        return NULL; \
    } \
} while(0)

#define BASIS_CHECK_SHAPE(cond) do { \
    if (!(cond)) { \
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Tensor shape mismatch or invalid dimensions"); \
        return NULL; \
    } \
} while(0)

static inline basis_error_t basis_get_last_error(void) { return _basis_last_error; }
static inline const char* basis_get_last_error_msg(void) { return _basis_last_error_msg; }
static inline void basis_clear_error(void) { _basis_last_error = BASIS_SUCCESS; _basis_last_error_msg = ""; }
static inline void basis_set_error_quiet(bool quiet) { _basis_error_quiet = quiet; }
#endif

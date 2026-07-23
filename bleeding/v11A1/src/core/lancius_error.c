#include "lancius/lancius_error.h"

/* A4: internal error model */

static _Thread_local lancius_error g_lancius_internal_error = LANCIUS_ERROR_OK;

void lancius_set_error(lancius_error err) {
    g_lancius_internal_error = err;
}

lancius_error lancius_get_error(void) {
    return g_lancius_internal_error;
}

void lancius_clear_error(void) {
    g_lancius_internal_error = LANCIUS_ERROR_OK;
}

const char* lancius_error_string(lancius_error err) {
    switch (err) {
        case LANCIUS_ERROR_OK:               return "success";
        case LANCIUS_ERROR_OOM:              return "out of memory";
        case LANCIUS_ERROR_NULL_PTR:         return "null pointer";
        case LANCIUS_ERROR_SHAPE_MISMATCH:   return "shape mismatch";
        case LANCIUS_ERROR_UNSUPPORTED_OP:   return "unsupported op";
        case LANCIUS_ERROR_UNSUPPORTED_DTYPE:return "unsupported dtype";
        case LANCIUS_ERROR_INVALID_MODEL:    return "invalid model";
        case LANCIUS_ERROR_VERSION_MISMATCH: return "version mismatch";
        case LANCIUS_ERROR_IO:               return "io error";
        case LANCIUS_ERROR_OVERFLOW:         return "overflow";
        case LANCIUS_ERROR_INTERNAL:         return "internal error";
        default:                             return "unknown error";
    }
}

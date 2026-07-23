#ifndef LANCIUS_ERROR_H
#define LANCIUS_ERROR_H

/* A4: internal error model */

typedef enum {
    LANCIUS_ERROR_OK = 0,
    LANCIUS_ERROR_OOM = -1,
    LANCIUS_ERROR_NULL_PTR = -2,
    LANCIUS_ERROR_SHAPE_MISMATCH = -3,
    LANCIUS_ERROR_UNSUPPORTED_OP = -4,
    LANCIUS_ERROR_UNSUPPORTED_DTYPE = -5,
    LANCIUS_ERROR_INVALID_MODEL = -6,
    LANCIUS_ERROR_VERSION_MISMATCH = -7,
    LANCIUS_ERROR_IO = -8,
    LANCIUS_ERROR_OVERFLOW = -9,
    LANCIUS_ERROR_INTERNAL = -10
} lancius_error;

void lancius_set_error(lancius_error err);
lancius_error lancius_get_error(void);
void lancius_clear_error(void);
const char* lancius_error_string(lancius_error err);

#define LANCIUS_CHECK(cond, err, ret) \
    do { \
        if (!(cond)) { \
            lancius_set_error(err); \
            return ret; \
        } \
    } while (0)

#endif /* LANCIUS_ERROR_H */

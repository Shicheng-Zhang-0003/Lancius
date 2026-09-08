#!/usr/bin/env python3
"""Phase 1.1: checked-arithmetic primitives (new files, additive)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

HEADER = '''#ifndef LANCIUS_CHECKED_H
#define LANCIUS_CHECKED_H

#include <stddef.h>
#include <stdint.h>

/* Checked-arithmetic primitives. Return 1 on success, 0 on overflow. */
int lancius_checked_add_size(size_t a, size_t b, size_t* out);
int lancius_checked_mul_size(size_t a, size_t b, size_t* out);
int lancius_checked_product_shape(const size_t* shape, uint8_t ndim, size_t* out);
int lancius_checked_bytes(size_t elems, size_t elem_size, size_t* out);
int lancius_checked_add_u64(uint64_t a, uint64_t b, uint64_t* out);
int lancius_checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out);

#endif /* LANCIUS_CHECKED_H */
'''

SOURCE = '''#include "lancius/lancius_checked.h"

int lancius_checked_add_size(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

int lancius_checked_mul_size(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

int lancius_checked_product_shape(const size_t* shape, uint8_t ndim, size_t* out) {
    if (!out) return 0;
    if (ndim > 4) return 0;
    size_t e = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        size_t d = shape ? shape[i] : 1;
        if (!lancius_checked_mul_size(e, d, &e)) return 0;
    }
    *out = e;
    return 1;
}

int lancius_checked_bytes(size_t elems, size_t elem_size, size_t* out) {
    return lancius_checked_mul_size(elems, elem_size, out);
}

int lancius_checked_add_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return 0;
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

int lancius_checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return 0;
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}
'''

def main():
    print("[101] checked arithmetic")
    se.create_file("include/lancius/lancius_checked.h", HEADER, "LANCIUS_CHECKED_H")
    se.create_file("src/core/lancius_checked.c", SOURCE, '#include "lancius/lancius_checked.h"')
    se.add_makefile_source("src/core/lancius_checked.c")
    print("[101] done")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 1.2: expand the internal error enum (additive, header-only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

ANCHOR = "LANCIUS_ERROR_INTERNAL = -10"
MARKER = "LANCIUS_ERROR_INVALID_HANDLE"

def main():
    print("[102] status codes")
    def transform(content):
        se.ensure_absent(content, MARKER, "status codes")
        new_codes = (
            ANCHOR + ",\n"
            "    LANCIUS_ERROR_INVALID_HANDLE = -11,\n"
            "    LANCIUS_ERROR_LIFETIME = -12,\n"
            "    LANCIUS_ERROR_INVALID_RANK = -13,\n"
            "    LANCIUS_ERROR_INVALID_SHAPE = -14,\n"
            "    LANCIUS_ERROR_INVALID_STRIDE = -15,\n"
            "    LANCIUS_ERROR_INVALID_DTYPE = -16,\n"
            "    LANCIUS_ERROR_INVALID_PERMUTATION = -17,\n"
            "    LANCIUS_ERROR_RESHAPE_MISMATCH = -18,\n"
            "    LANCIUS_ERROR_GRAPH_CYCLE = -19,\n"
            "    LANCIUS_ERROR_GRAPH_INVALID = -20,\n"
            "    LANCIUS_ERROR_NUMERICAL = -21,\n"
            "    LANCIUS_ERROR_LIMIT = -22"
        )
        return se.replace_once(content, ANCHOR, new_codes)
    se.edit_file("include/lancius/lancius_error.h", transform)
    print("[102] done")

if __name__ == "__main__":
    main()

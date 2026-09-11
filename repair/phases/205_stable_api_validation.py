#!/usr/bin/env python3
"""Phase 2.5: Close the abort-via-stable-API hole in lancius_add_input."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/core/lancius_stable_api.c"
MARKER = "v12R1-205"

def main():
    print("[205] stable API shape validation")
    content = se.read(se.p(FILE))
    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # 1. Add includes
    content = se.insert_after(
        content,
        '#include "lancius/lancius_stable_api.h"',
        '\n#include "lancius/lancius_checked.h"\n#include "lancius/lancius_ir.h" /* for LANCIUS_MAX_TENSOR_ELEMS */'
    )

    # 2. Patch lancius_add_input
    old_block = (
        "LANCIUS_EXPORT lancius_tensor_handle lancius_add_input(lancius_graph_handle g, size_t rows, size_t cols) {\n"
        "    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }\n"
        "    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;\n"
        "    lancius_node* n = lancius_input(wrapper->g, rows, cols);"
    )
    new_block = (
        "LANCIUS_EXPORT lancius_tensor_handle lancius_add_input(lancius_graph_handle g, size_t rows, size_t cols) {\n"
        "    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return NULL; }\n"
        "    lancius_graph_internal* wrapper = (lancius_graph_internal*)g;\n"
        "    /* v12R1-205: prevent abort() via stable API on oversized tensors */\n"
        "    size_t elems = 0;\n"
        "    if (!lancius_checked_mul_size(rows, cols, &elems) || elems > LANCIUS_MAX_TENSOR_ELEMS) {\n"
        "        set_error(LANCIUS_ERR_SHAPE_MISMATCH);\n"
        "        return NULL;\n"
        "    }\n"
        "    lancius_node* n = lancius_input(wrapper->g, rows, cols);"
    )
    content = se.replace_once(content, old_block, new_block)

    se.backup(se.p(FILE))
    se.write(se.p(FILE), content)
    print(f"  [edit] {FILE}")
    print("[205] done")

if __name__ == "__main__":
    main()

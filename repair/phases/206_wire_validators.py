#!/usr/bin/env python3
"""Phase 2.6: Wire up dead validators and fix Add/Sub/Mul broadcast OOB."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

def main():
    print("[206] wire validators & fix broadcast OOB")

    # 1. Add broadcast validator to lancius_validate.h and .c
    hdr_file = "include/lancius/lancius_validate.h"
    src_file = "src/core/lancius_validate.c"

    hdr_content = se.read(se.p(hdr_file))
    if "lancius_validate_binary_broadcast" not in hdr_content:
        hdr_content = se.insert_before(
            hdr_content,
            "#endif /* LANCIUS_VALIDATE_H */",
            "lancius_error lancius_validate_binary_broadcast(const lancius_node* a, const lancius_node* b);\n"
        )
        se.backup(se.p(hdr_file))
        se.write(se.p(hdr_file), hdr_content)

    src_content = se.read(se.p(src_file))
    if "lancius_validate_binary_broadcast" not in src_content:
        new_func = """
lancius_error lancius_validate_binary_broadcast(const lancius_node* a, const lancius_node* b) {
    if (!a || !b) return LANCIUS_ERROR_NULL_PTR;
    if (a->ndim != b->ndim) return LANCIUS_ERROR_SHAPE_MISMATCH;
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] && a->shape[i] != 1 && b->shape[i] != 1) {
            return LANCIUS_ERROR_SHAPE_MISMATCH;
        }
    }
    return LANCIUS_ERROR_OK;
}
"""
        src_content = src_content + new_func
        se.backup(se.p(src_file))
        se.write(se.p(src_file), src_content)

    # 2. Wire validators in lancius_ir.c
    ir_file = "src/ir/lancius_ir.c"
    ir_content = se.read(se.p(ir_file))

    # MatMul
    if "lancius_validate_matmul(" not in ir_content:
        ir_content = se.insert_before(
            ir_content,
            "size_t a_rows = a->shape[a->ndim - 2];",
            "if (lancius_validate_matmul(a->shape, a->ndim, b->shape, b->ndim) != LANCIUS_ERROR_OK) return NULL;\n    "
        )

    # Add/Sub/Mul
    for op in ["ADD", "SUB", "MUL"]:
        anchor = f"lancius_node* n = alloc_node(g, LANCIUS_OP_{op}, a->ndim, 2);"
        if anchor in ir_content and "lancius_validate_binary_broadcast" not in ir_content.split(anchor)[1][:200]:
            ir_content = se.insert_before(
                ir_content,
                anchor,
                f"if (lancius_validate_binary_broadcast(a, b) != LANCIUS_ERROR_OK) return NULL;\n    "
            )

    se.backup(se.p(ir_file))
    se.write(se.p(ir_file), ir_content)
    print(f"  [edit] {ir_file}")
    print("[206] done")

if __name__ == "__main__":
    main()

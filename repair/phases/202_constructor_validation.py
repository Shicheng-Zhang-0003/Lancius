#!/usr/bin/env python3
"""Phase 2.2: route IR constructors through the central validator for states that
are never valid: bad permutation, reshape non-conservation, zero stride/kernel,
invalid GQA heads, flatten overflow."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/ir/lancius_ir.c"

def main():
    print("[202] constructor validation")

    def transform(content):
        se.ensure_absent(content, "v12R1-202", "constructor validation")

        # includes
        content = se.replace_once(
            content,
            '#include "lancius/lancius_ir.h"\n#include <stdlib.h>',
            '#include "lancius/lancius_ir.h"\n'
            '#include "lancius/lancius_validate.h"\n'
            '#include "lancius/lancius_checked.h"\n'
            '#include <stdlib.h>'
        )

        # permute: validate axes BEFORE dims[axes] is read (kills the OOB segfault)
        content = se.insert_before(
            content,
            "lancius_node* n = alloc_node(g, LANCIUS_OP_PERMUTE, 4, 1);",
            "uint32_t perm_axes[4] = {a0, a1, a2, a3}; /* v12R1-202 */\n"
            "    if (lancius_validate_permutation(perm_axes, 4) != LANCIUS_ERROR_OK) return NULL;\n    "
        )

        # reshape: enforce element conservation
        content = se.insert_before(
            content,
            "lancius_node* n = alloc_node(g, LANCIUS_OP_RESHAPE, ndim, 1);",
            "size_t resh_out_shape[4] = {s0, s1, s2, s3};\n"
            "    size_t resh_in_elems = 0;\n"
            "    if (!lancius_node_elements_checked(in, &resh_in_elems)) return NULL;\n"
            "    if (lancius_validate_reshape(resh_in_elems, resh_out_shape, ndim) != LANCIUS_ERROR_OK) return NULL;\n    "
        )

        # conv2d: reject stride==0 / kernel==0 / spatial underflow
        content = se.insert_before(
            content,
            "size_t H_out = (H_in + 2*pad - K_h) / stride + 1;",
            "if (lancius_validate_conv2d(H_in, W_in, K_h, K_w, stride, pad) != LANCIUS_ERROR_OK) return NULL;\n    "
        )

        # maxpool2d: reject stride==0 / kernel==0
        content = se.insert_before(
            content,
            "size_t H_out = (H_in - kernel) / stride + 1;",
            "if (stride == 0 || kernel == 0) return NULL;\n    "
        )

        # gqa: reject kv==0, q<kv, q%kv!=0
        content = se.insert_before(
            content,
            "lancius_node* n = alloc_node(g, LANCIUS_OP_GQA, q->ndim, 3);",
            "if (lancius_validate_gqa(n_heads_q, n_heads_kv) != LANCIUS_ERROR_OK) return NULL;\n    "
        )

        # flatten: checked product instead of raw multiply
        content = se.replace_once(
            content,
            "size_t flat = in->shape[1] * in->shape[2] * in->shape[3];",
            "size_t flat_dims[3] = {in->shape[1], in->shape[2], in->shape[3]};\n"
            "    size_t flat = 0;\n"
            "    if (!lancius_checked_product_shape(flat_dims, 3, &flat)) return NULL;"
        )

        return content

    se.edit_file(FILE, transform)
    print("[202] done")

if __name__ == "__main__":
    main()

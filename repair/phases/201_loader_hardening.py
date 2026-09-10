#!/usr/bin/env python3
"""Phase 2.1: harden the v2 loader. Reject oversized/zero-dim input shapes and
NULL node reconstruction. Closes the abort-via-stable-API hole and the rank-3 bug."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/core/lancius_serialize_v2.c"
MARKER = "v12R1-201"

def main():
    print("[201] v2 loader hardening")
    content = se.read(se.p(FILE))

    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # --- 1. Add validate include ---
    inc_anchor = '#include "lancius/lancius_ir.h"'
    se.ensure_unique(content, inc_anchor, "validate include anchor")
    content = se.insert_after(
        content, inc_anchor,
        '\n#include "lancius/lancius_validate.h"'
    )

    # --- 2. Replace INPUT case using regex (whitespace-tolerant) ---
    # Find the INPUT case block: from "case LANCIUS_MODEL_OP_INPUT:" to the next "break;"
    pattern = re.compile(
        r'(case\s+LANCIUS_MODEL_OP_INPUT:\s*\n)'   # the case line
        r'((?:[ \t]*.*\n)*?)'                        # body lines (lazy)
        r'([ \t]*break;)',                           # the break
        re.MULTILINE
    )
    m = pattern.search(content)
    if not m:
        raise RuntimeError("[ABORT] could not locate LANCIUS_MODEL_OP_INPUT case block")

    # Detect indentation from the body
    body_lines = m.group(2).strip().split('\n')
    indent = "            "  # default 12 spaces
    if body_lines:
        leading = len(body_lines[0]) - len(body_lines[0].lstrip())
        indent = " " * leading

    # Detect case-line indentation
    case_start = m.start()
    line_start = content.rfind('\n', 0, case_start) + 1
    case_indent = " " * (case_start - line_start)

    new_block = (
        f"case LANCIUS_MODEL_OP_INPUT:\n"
        f"{indent}/* {MARKER}: validate shape; reject oversized / zero-dim / bad-rank */\n"
        f"{indent}if (rn.ndim == 4) {{\n"
        f"{indent}    if (lancius_validate_shape(sh, 4) != LANCIUS_ERROR_OK) goto fail;\n"
        f"{indent}    n = lancius_input_4d(g, sh[0], sh[1], sh[2], sh[3]);\n"
        f"{indent}}} else if (rn.ndim == 3) {{\n"
        f"{indent}    if (lancius_validate_shape(sh, 3) != LANCIUS_ERROR_OK) goto fail;\n"
        f"{indent}    n = lancius_input_3d(g, sh[0], sh[1], sh[2]);\n"
        f"{indent}}} else {{\n"
        f"{indent}    if (lancius_validate_shape(sh, 2) != LANCIUS_ERROR_OK) goto fail;\n"
        f"{indent}    n = lancius_input(g, sh[0], sh[1]);\n"
        f"{indent}}}\n"
        f"{indent}break;"
    )

    # Replace the entire matched region
    content = content[:m.start()] + new_block + content[m.end():]

    # --- 3. Reject NULL reconstruction for non-NOP ops ---
    # Find "if (n) {" followed by "n->kernel_h"
    guard_pattern = re.compile(
        r'([ \t]*)(if\s*\(n\)\s*\{\s*\n'
        r'[ \t]*n->kernel_h\s*=\s*rn\.meta\[0\];)'
    )
    gm = guard_pattern.search(content)
    if not gm:
        raise RuntimeError("[ABORT] could not locate node-field assignment guard")

    guard_indent = gm.group(1)
    old_guard = gm.group(0)
    new_guard = (
        f"{guard_indent}/* {MARKER}: non-NOP op that failed to reconstruct = corrupt model */\n"
        f"{guard_indent}if (rn.op != LANCIUS_MODEL_OP_NOP && !n) goto fail;\n"
        f"{old_guard}"
    )
    content = content[:gm.start()] + new_guard + content[gm.end():]

    # --- Write ---
    se.backup(se.p(FILE))
    se.write(se.p(FILE), content)
    print(f"  [edit] {FILE}")
    print("[201] done")

if __name__ == "__main__":
    main()

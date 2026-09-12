#!/usr/bin/env python3
"""Phase 2.9: Fix bytecode VM - zero registers, check second operand."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/compiler/lancius_bytecode.c"
MARKER = "v12R1-209"

def main():
    print("[209] bytecode VM hardening")
    content = se.read(se.p(FILE))
    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # 1. Zero the registers array after allocation
    old_regs = (
        "    double** regs = (double**)lancius_arena_alloc(scratch, prog->num_regs * sizeof(double*), 8);\n"
        "    if (!regs) return -1;"
    )
    new_regs = (
        f"    /* {MARKER}: zero registers to prevent garbage pointer reads */\n"
        "    double** regs = (double**)lancius_arena_alloc(scratch, prog->num_regs * sizeof(double*), 8);\n"
        "    if (!regs) return -1;\n"
        "    memset(regs, 0, prog->num_regs * sizeof(double*));"
    )
    content = se.replace_once(content, old_regs, new_regs)

    # 2. Add null check for second operand in binary ops
    # Find the "if (!a) continue;" line and add a check for b
    old_check = "    if (!a) continue;"
    new_check = (
        f"    /* {MARKER}: check both operands for binary ops */\n"
        "    if (!a) continue;\n"
        "    if (!is_unary && !b) continue;"
    )
    content = se.replace_once(content, old_check, new_check)

    # 3. Add checked arithmetic for element count calculation
    old_elems = "    size_t elements = prog->rows[r_out] * prog->cols[r_out];"
    new_elems = (
        f"    /* {MARKER}: checked element count */\n"
        "    size_t elements = 0;\n"
        "    if (!lancius_checked_mul_size(prog->rows[r_out], prog->cols[r_out], &elements)) return -1;"
    )
    content = se.replace_once(content, old_elems, new_elems)

    # 4. Add include for checked arithmetic
    old_include = '#include "lancius/lancius_bytecode.h"'
    new_include = (
        '#include "lancius/lancius_bytecode.h"\n'
        '#include "lancius/lancius_checked.h"'
    )
    content = se.replace_once(content, old_include, new_include)

    se.backup(se.p(FILE))
    se.write(se.p(FILE), content)
    print(f"  [edit] {FILE}")
    print("[209] done")

if __name__ == "__main__":
    main()

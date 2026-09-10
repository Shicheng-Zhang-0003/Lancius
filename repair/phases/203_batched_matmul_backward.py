#!/usr/bin/env python3
"""Phase 2.3 (corrected): MATMUL_BATCHED backward must fail loudly.
Uses regex to tolerate any whitespace in the actual file."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/math/lancius_autodiff.c"
MARKER = "v12R1-203"

def main():
    print("[203] batched matmul backward honesty (corrected)")
    path = se.p(FILE)
    content = se.read(path)

    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # Find the MATMUL_BATCHED backward block using regex.
    # Pattern: the else-if opener, the two accum_grad lines, up to the next else-if.
    pattern = re.compile(
        r'(\}\s*else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{)'  # opener
        r'((?:[^}]*?))'                                                                      # body (lazy)
        r'(\}\s*else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_PERMUTE)',                    # next else-if
        re.DOTALL
    )

    m = pattern.search(content)
    if not m:
        raise RuntimeError("[ABORT] could not locate MATMUL_BATCHED backward block")

    # Detect indentation from the body
    body = m.group(2)
    body_lines = [l for l in body.strip().split('\n') if l.strip()]
    indent = "            "
    if body_lines:
        leading = len(body_lines[0]) - len(body_lines[0].lstrip())
        indent = " " * leading

    # Build the fail-loudly replacement
    new_body = (
        f"\n{indent}/* {MARKER}: correct batched-matmul backward needs a batched\n"
        f"{indent}   transpose that the IR does not provide. Fail loudly rather\n"
        f"{indent}   than emit mathematically wrong gradients. */\n"
        f'{indent}fprintf(stderr, "[AUTODIFF FATAL] MATMUL_BATCHED backward is not implemented.\\n");\n'
        f"{indent}free(grad_map); free(fwd_to_full);\n"
        f"{indent}lancius_graph_destroy(tg->graph); free(tg);\n"
        f"{indent}return NULL;\n{indent}"
    )

    replacement = m.group(1) + new_body + m.group(3)
    content = content[:m.start()] + replacement + content[m.end():]

    se.backup(path)
    se.write(path, content)
    print(f"  [edit] {FILE}")
    print("[203] done")

if __name__ == "__main__":
    main()

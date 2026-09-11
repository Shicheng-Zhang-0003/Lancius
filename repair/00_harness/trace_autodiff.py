#!/usr/bin/env python3
"""
trace_autodiff.py — Injects deep tracing into lancius_autodiff.c,
forces a clean rebuild, runs the test, and shows exactly what the engine sees.
"""
import os, sys, re, subprocess, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTODIFF = os.path.join(REPO, "../src/math/lancius_autodiff.c")
BACKUP = AUTODIFF + ".trace_backup"

def main():
    print("="*72)
    print("AUTODIFF DEEP TRACE")
    print("="*72)

    shutil.copy2(AUTODIFF, BACKUP)

    with open(AUTODIFF, "r") as f:
        content = f.read()

    # 1. Trace Forward Pass Node Reconstruction
    content = re.sub(
        r'(lancius_node\* old = fwd_g->nodes\[i\];)',
        r'\1\n        fprintf(stderr, "[FWD] i=%u id=%u op=%d ndim=%u\\n", i, old->id, (int)old->op, old->ndim);',
        content, count=1
    )

    # Trace the result of the switch statement (n)
    content = re.sub(
        r'(fwd_to_full\[old->id\] = n;)',
        r'fprintf(stderr, "[FWD] -> reconstructed n=%p\\n", (void*)n);\n        \1',
        content, count=1
    )

    # 2. Trace Backward Pass Loop
    content = re.sub(
        r'(lancius_node\* grad_out = grad_map\[fwd_n->id\];)',
        r'\1\n        fprintf(stderr, "[BWD] i=%d id=%u op=%d grad_out=%p\\n", i, fwd_n->id, (int)fwd_n->op, (void*)grad_out);',
        content, count=1
    )

    # 3. Trace Specific Blocks
    content = re.sub(
        r'(\}\s*else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{)',
        r'\1\n        fprintf(stderr, "[BWD] !!! MATMUL_BATCHED BLOCK REACHED !!!\\n");',
        content, count=1
    )

    content = re.sub(
        r'(\}\s*else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_SUM\s*\)\s*\{)',
        r'\1\n        fprintf(stderr, "[BWD] !!! SUM BLOCK REACHED !!!\\n");',
        content, count=1
    )

    with open(AUTODIFF, "w") as f:
        f.write(content)

    print("\n[1/3] FORCING CLEAN REBUILD...")
    subprocess.run(["make", "clean"], cwd=REPO, capture_output=True)
    r = subprocess.run(["make", "audit_fault_injection"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED:")
        print(r.stderr)
        shutil.copy2(BACKUP, AUTODIFF)
        return

    print("\n[2/3] RUNNING INSTRUMENTED TEST...")
    r = subprocess.run(["./audit_fault_injection"], cwd=REPO, capture_output=True, text=True)

    print("\n" + "="*72)
    print("STDOUT (Test Results)")
    print("="*72)
    print(r.stdout)

    print("\n" + "="*72)
    print("STDERR (Engine Trace)")
    print("="*72)
    print(r.stderr)

    print("\n[3/3] RESTORING ORIGINAL SOURCE...")
    shutil.copy2(BACKUP, AUTODIFF)
    os.remove(BACKUP)
    subprocess.run(["make", "clean"], cwd=REPO, capture_output=True)
    subprocess.run(["make", "audit_fault_injection"], cwd=REPO, capture_output=True)
    print("Done. Original source restored and rebuilt clean.")

if __name__ == "__main__":
    main()

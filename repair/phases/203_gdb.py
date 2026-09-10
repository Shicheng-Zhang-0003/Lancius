#!/usr/bin/env python3
"""203_gdb v2 — path-fixed + decisive binary inspection + runtime trace."""
import os, sys, re, subprocess, shutil

def find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "makefile")) and os.path.isdir(os.path.join(d, "include", "lancius")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found (no makefile + include/lancius)")

REPO     = find_repo_root()
AUTODIFF = os.path.join(REPO, "src/math/lancius_autodiff.c")
LIB      = os.path.join(REPO, "liblancius.a")
TESTBIN  = os.path.join(REPO, "audit_fault_injection")
BACKUP   = AUTODIFF + ".gdb_backup"
FATAL    = "MATMUL_BATCHED backward is not implemented"

def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print("  " + t); print("=" * 72)

def grep_binary(path, needle):
    if not os.path.exists(path):
        return None
    r = subprocess.run(["grep", "-a", "-c", needle, path], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0

def runtime_trace():
    hr("[3] RUNTIME TRACE — instrumenting backward loop")
    shutil.copy2(AUTODIFF, BACKUP)
    content = open(AUTODIFF).read()

    def repl_loop(m):
        return (m.group(1) +
                '\n        fprintf(stderr, "[GDB] i=%d id=%u op=%d grad_out=%p\\n",'
                ' i, fwd_n->id, (int)fwd_n->op, (void*)grad_out);')
    pat_loop = r'(lancius_node\* fwd_n = fwd_g->nodes\[i\];\s*\n\s*lancius_node\* grad_out = grad_map\[fwd_n->id\];)'
    content, n1 = re.subn(pat_loop, repl_loop, content, count=1)
    print(f"  loop-trace inserted: {n1}")

    def repl_mmb(m):
        return (m.group(1) +
                '\n            fprintf(stderr, "[GDB] *** MATMUL_BATCHED BACKWARD ENTERED ***\\n");')
    pat_mmb = r'(else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{)'
    content, n2 = re.subn(pat_mmb, repl_mmb, content, count=1)
    print(f"  MATMUL_BATCHED-entry trace inserted: {n2}")

    open(AUTODIFF, "w").write(content)

    hr("[4] REBUILD (clean)")
    subprocess.run(["make", "clean"], cwd=REPO, capture_output=True)
    r = subprocess.run(["make", "audit_fault_injection"], cwd=REPO, capture_output=True, text=True)
    print(f"  make exit={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        shutil.copy2(BACKUP, AUTODIFF); os.remove(BACKUP); return

    hr("[5] RUN INSTRUMENTED TEST")
    r = subprocess.run([TESTBIN], cwd=REPO, capture_output=True, text=True)
    print(f"  exit={r.returncode}")
    hr("STDOUT"); print(r.stdout)
    hr("STDERR (GDB trace)"); print(r.stderr)

    hr("[6] RESTORE + clean rebuild")
    shutil.copy2(BACKUP, AUTODIFF); os.remove(BACKUP)
    subprocess.run(["make", "clean"], cwd=REPO, capture_output=True)
    subprocess.run(["make", "audit_fault_injection"], cwd=REPO, capture_output=True)
    print("  restored original + rebuilt clean")

def main():
    hr("PHASE 203 GDB v2")
    print(f"  REPO    : {REPO}")
    print(f"  AUTODIFF: exists={os.path.exists(AUTODIFF)}")
    print(f"  LIB     : exists={os.path.exists(LIB)}")
    print(f"  TESTBIN : exists={os.path.exists(TESTBIN)}")

    hr("[1] SOURCE CHECK")
    src = open(AUTODIFF).read()
    print(f"  marker v12R1-203 in source : {'v12R1-203' in src}")
    print(f"  FATAL string in source     : {FATAL in src}")
    m = re.search(r'LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{(.*?)\}\s*else', src, re.DOTALL)
    print(f"  block has 'return NULL'    : {('return NULL' in m.group(1)) if m else 'block not found'}")

    hr("[2] DECISIVE — is the fix actually in the compiled binaries?")
    lib_hits = grep_binary(LIB, FATAL)
    bin_hits = grep_binary(TESTBIN, FATAL)
    print(f"  FATAL string in liblancius.a         : {lib_hits}")
    print(f"  FATAL string in audit_fault_injection: {bin_hits}")

    if lib_hits == 0:
        hr("DIAGNOSIS")
        print("  The fix is in the SOURCE but NOT in liblancius.a.")
        print("  The library was not rebuilt from the edited source.")
        print("  FIX: make clean && make audit_fault_injection")
        return
    if bin_hits == 0:
        hr("DIAGNOSIS")
        print("  Fix is in the library but NOT in the test binary (stale binary).")
        print("  FIX: rm -f audit_fault_injection && make audit_fault_injection")
        return

    hr("DIAGNOSIS")
    print("  Fix IS compiled into both the library and the test binary.")
    print("  Therefore the MATMUL_BATCHED branch is NOT being reached at runtime.")
    print("  Running instrumentation to find out why...")
    runtime_trace()

if __name__ == "__main__":
    main()

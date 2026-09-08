#!/usr/bin/env python3
"""Orchestrator: runs phase scripts in numeric order, gating each on a build."""
import os, glob, re, subprocess, sys

HERE        = os.path.dirname(os.path.abspath(__file__))
REPAIR_ROOT = os.path.dirname(HERE)
REPO_ROOT   = os.path.dirname(REPAIR_ROOT)
PHASES      = os.path.join(REPAIR_ROOT, "phases")
VERIFY      = os.path.join(HERE, "verify_build.sh")
LOG         = os.path.join(REPAIR_ROOT, "repair.log")

def log(msg):
    print(msg)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def main():
    scripts = sorted(glob.glob(os.path.join(PHASES, "*.py")))
    scripts = [s for s in scripts if re.match(r"^\d+_", os.path.basename(s))]
    if not scripts:
        log("[run_repair] no phase scripts found")
        return 1
    log(f"[run_repair] {len(scripts)} phase script(s) queued")

    for s in scripts:
        name = os.path.basename(s)
        log(f"\n=== RUN {name} ===")
        r = subprocess.run([sys.executable, s], cwd=REPO_ROOT)
        if r.returncode != 0:
            log(f"[run_repair] HALT: {name} failed (exit {r.returncode})")
            return r.returncode
        log(f"--- VERIFY after {name} ---")
        v = subprocess.run(["bash", VERIFY], cwd=REPO_ROOT)
        if v.returncode != 0:
            log(f"[run_repair] HALT: build failed after {name}")
            return v.returncode
        log(f"[run_repair] {name} OK + build green")

    log("\n[run_repair] ALL PHASES GREEN")
    return 0

if __name__ == "__main__":
    sys.exit(main())

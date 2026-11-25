#!/usr/bin/env python3
"""
I6 Training Monitor

Shows real-time status of all 4 family-expert training processes.
Run this in a separate terminal while training is in progress.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(HERE, "models_verified")
CORPUS_DIR = os.path.join(HERE, "corpus_verified")
FAMILIES = ["algebra", "calculus", "linear_algebra", "number_theory"]


def find_pids(name):
    out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [int(p) for p in out.stdout.strip().split() if p]


def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def last_lines(path, n=5):
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:]) if lines else "(empty)"
    except FileNotFoundError:
        return "(no log yet)"


def main():
    print("=" * 70)
    print("I6 Multi-Expert Training Monitor")
    print("=" * 60)
    print()

    all_done = True
    for fam in FAMILIES:
        pids = find_pids(f"train_i6 {CORPUS_DIR}/{fam}")
        log_path = os.path.join(MODELS_DIR, fam, "train.log")

        if pids:
            status = "RUNNING"
            pids_str = ", ".join(str(p) for p in pids)
        else:
            # Check if log indicates completion
            try:
                with open(log_path) as f:
                    content = f.read()
                if "Targets overall MAE/MSE < 1e-3: PASS/PASS; per-step MAE/MSE < 1e-3: PASS/PASS" in content:
                    status = "PASS"
                elif "Targets overall MAE/MSE < 1e-3:" in content or "FATAL" in content:
                    status = "FAILED"
                elif "epoch" in content:
                    status = "STARTED (no progress yet)"
                else:
                    status = "UNKNOWN"
            except FileNotFoundError:
                status = "NOT STARTED"
            pids_str = "—"
            all_done = False

        print(f"[{fam.upper():15}] {status}")
        if pids_str != "—":
            print(f"                   PIDs: {pids_str}")
        print(f"                   Log:  {log_path}")

        # Show last 3 lines of log
        tail = last_lines(log_path, 3)
        for line in tail.rstrip("\n").split("\n"):
            if line.strip():
                print(f"                   > {line.strip()[:100]}")
        print()

    if all_done:
        print("All trainers finished. Run evaluation:")
        print(f"  python3 {HERE}/run_big_i6.py 1000 \\")
        print(f"    {MODELS_DIR}/alu_algebra.lancius 20261026")
        print(f"    --experts {MODELS_DIR} --strict-targets")
    else:
        print("Training in progress. Re-run this script to check again.")


if __name__ == "__main__":
    main()

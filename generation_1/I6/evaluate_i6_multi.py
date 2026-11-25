#!/usr/bin/env python3
"""
I6 Multi-Expert Evaluation Runner

Waits for all 4 family-expert trainers to finish, then runs head-to-head
evaluation against the I3 ALU pool using multi-expert dispatch.

Usage:
    python3 generation_1/I6/evaluate_i6_multi.py [--per-algo 1000]
"""

import os
import sys
import subprocess
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(HERE, "models_verified")
RESULTS_DIR = os.path.join(HERE, "results_verified")
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


def wait_for_trainers(poll_interval=30):
    """Wait for the sequential (or parallel) family trainers to complete.

    Any live train_i6 process counts as running.  The orchestrator may be
    between families for a moment, so also treat a live orchestrate_i6.py as
    running.
    """
    while True:
        running = []
        for fam in FAMILIES:
            pids = find_pids(f"train_i6 {CORPUS_DIR}/{fam}")
            if pids:
                running.append((fam, pids))
        if find_pids("orchestrate_i6.py"):
            running.append(("orchestrator", find_pids("orchestrate_i6.py")))

        if not running:
            print("[evaluate] All trainers finished.")
            return True

        print(f"[evaluate] {len(running)} trainer(s) still running: "
              f"{', '.join(f for f, _ in running)}")
        time.sleep(poll_interval)


def run_multi_expert_eval(per_algo=1000, seed=20261026):
    """Run head-to-head evaluation with multi-expert dispatch."""
    outdir = os.path.join(RESULTS_DIR, f"eval_multi_expert_{int(time.time())}")
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "python3", os.path.join(HERE, "run_big_i6.py"),
        str(per_algo),
        # Placeholder for the loader; routing uses --experts.
        os.path.join(MODELS_DIR, "alu_algebra.lancius"),
        str(seed),
        outdir,
        "--alu-pool", os.path.join(REPO, "generation_1", "I3", "alu_pool"),
        "--musd", os.path.join(CORPUS_DIR, "mu_sd.txt"),
        "--i3-musd", os.path.join(REPO, "generation_1", "I3", "gen1_mu_sd.txt"),
        "--experts", MODELS_DIR,
        "--strict-targets",
    ]

    print(f"[evaluate] Running multi-expert evaluation:")
    print(f"[evaluate]   {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=REPO)
    return result.returncode == 0, outdir


def main():
    parser = argparse.ArgumentParser(description="Wait for training, then evaluate multi-expert I6")
    parser.add_argument("--per-algo", type=int, default=1000, help="Runs per algorithm")
    parser.add_argument("--seed", type=int, default=20261026, help="Master seed")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting, just evaluate")
    args = parser.parse_args()

    if not args.no_wait:
        print("[evaluate] Waiting for all 4 family trainers to complete...")
        if not wait_for_trainers():
            print("[evaluate] ERROR: One or more trainers failed.")
            return 1

    print("\n" + "=" * 60)
    print("Running multi-expert evaluation...")
    print("=" * 60)

    success, outdir = run_multi_expert_eval(args.per_algo, args.seed)

    if success:
        print(f"\n[evaluate] Evaluation complete. Results: {outdir}")
        print(f"\nTo view results:")
        print(f"  cat {outdir}/big_summary.txt")
        return 0
    else:
        print(f"\n[evaluate] Evaluation failed. Check {outdir} for logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

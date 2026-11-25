#!/usr/bin/env python3
"""
I6 Training & Evaluation Orchestrator

Runs all 4 family-expert trainers (algebra, calculus, linear_algebra, number_theory)
sequentially by default (one trainer at a time, 2 OpenMP threads), waits for
completion, then runs head-to-head evaluation against the I3 ALU pool using
multi-expert dispatch.  Pass --parallel to run all four at once on a machine
that can take the load.

Usage:
    python3 generation_1/I6/orchestrate_i6.py [--cores N] [--epochs N] [--lr L]
                                               [--parallel] [--skip-eval]

Defaults: 2 cores each, 80 epochs, lr 0.003, compact 128x64 trunk.
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

FAMILIES = ["algebra", "calculus", "linear_algebra", "number_theory"]

ARCH = "128 64 80 256"  # H1 H2 EPOCHS BATCH
SEED = 20261026
LR = 0.003
MUSD = os.path.join(HERE, "corpus_verified", "mu_sd.txt")

MODELS_DIR = os.path.join(HERE, "models_verified")
RESULTS_DIR = os.path.join(HERE, "results_verified")
CORPUS_DIR = os.path.join(HERE, "corpus_verified")


def find_process(name):
    """Return list of PIDs matching a process name fragment."""
    out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [int(p) for p in out.stdout.strip().split() if p]


def is_running(pid):
    """Check if PID is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def train_family(family, cores):
    """Launch one trainer subprocess. Returns Popen object."""
    corpus_dir = os.path.join(CORPUS_DIR, family)
    model_out = os.path.join(MODELS_DIR, f"alu_{family}.lancius")
    save_dir = os.path.join(MODELS_DIR, family)
    log_path = os.path.join(MODELS_DIR, family, "train.log")

    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "env", f"OMP_NUM_THREADS={cores}",
        os.path.join(REPO, "build", "bin", "train_i6"),
        corpus_dir,
        *ARCH.split(),
        "--seed", str(SEED),
        "--lr", str(LR),
        "--mu-sd", MUSD,
        "--overall-repeat", "2",
        "--pos-repeat", "1",
        "--hard-negative-frac", "0",
        "--hard-negative-start", "2",
        "--lancius-rows", "128",
        "--export-lancius", model_out,
        "--verify-lancius",
        "--strict-targets",
        "--save-dir", save_dir,
    ]

    print(f"[orchestrator] Starting {family} trainer (PID will be assigned)...")
    log_f = open(log_path, "w")
    p = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    return p, log_f


def wait_for_all(procs, poll_interval=30):
    """Wait for all training processes to finish."""
    procs = list(procs)  # copy
    failures = []
    while procs:
        done = []
        for p, log_f in procs:
            ret = p.poll()
            if ret is not None:
                done.append((p, log_f, ret))
        for p, log_f, ret in done:
            procs.remove((p, log_f))
            log_f.close()
            status = "SUCCESS" if ret == 0 else f"FAILED (exit {ret})"
            print(f"[orchestrator] {p.pid} {status}", flush=True)
            if ret != 0:
                failures.append(p.pid)
        if procs:
            print(f"[orchestrator] {len(procs)} trainer(s) still running... "
                  f"next check in {poll_interval}s", flush=True)
            time.sleep(poll_interval)
    return failures


def run_evaluation(models_dir, alu_pool, musd, per_algo=1000):
    """Run multi-expert head-to-head evaluation."""
    outdir = os.path.join(RESULTS_DIR, f"eval_multi_expert_{int(time.time())}")
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "python3", os.path.join(HERE, "run_big_i6.py"),
        str(per_algo),
        # Multi-expert dispatch does the real work; this positional model is
        # only the loader's placeholder, so point it at an expert that exists.
        os.path.join(MODELS_DIR, "alu_algebra.lancius"),
        str(SEED),
        outdir,
        "--alu-pool", alu_pool,
        "--musd", musd,
        "--i3-musd", os.path.join(REPO, "generation_1", "I3", "gen1_mu_sd.txt"),
        "--experts", MODELS_DIR,
        "--strict-targets",
    ]

    print(f"[orchestrator] Running multi-expert evaluation...")
    print(f"[orchestrator] {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Save a copy of output
    out_log = os.path.join(outdir, "orchestrator_log.txt")
    with open(out_log, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n" + result.stderr)

    return result.returncode == 0, outdir


def main():
    parser = argparse.ArgumentParser(description="Orchestrate I6 multi-expert training + evaluation")
    parser.add_argument("--cores", type=int, default=2, help="OpenMP threads per trainer")
    parser.add_argument("--epochs", type=int, default=80, help="Epochs per trainer")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--per-algo", type=int, default=1000, help="Runs per algo in eval")
    parser.add_argument("--parallel", action="store_true",
                        help="run all four trainers at once (default: sequential)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="train only; do not run the head-to-head evaluation")
    args = parser.parse_args()

    global ARCH, SEED, LR
    ARCH = f"128 64 {args.epochs} 256"
    LR = args.lr

    print("=" * 60)
    print("I6 Multi-Expert Training & Evaluation Orchestrator")
    print("=" * 60)
    print(f"Families: {', '.join(FAMILIES)}")
    print(f"Cores per trainer: {args.cores}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Seed: {SEED}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 60)

    # Verify prerequisites
    trainer_bin = os.path.join(REPO, "build", "bin", "train_i6")
    if not os.path.exists(trainer_bin):
        print(f"[ERROR] Trainer binary not found: {trainer_bin}")
        print("Run: make build/bin/train_i6")
        return 1

    for fam in FAMILIES:
        corpus_hdr = os.path.join(CORPUS_DIR, fam, "corpus.hdr")
        if not os.path.exists(corpus_hdr):
            print(f"[ERROR] Corpus header missing: {corpus_hdr}")
            print("Run: python3 generation_1/I6/split_corpus_by_family.py")
            return 1

    # Launch trainers.  Sequential is the default: the laptop is
    # thermal-sensitive, and four 2-thread trainers at once is 8 threads.
    if args.parallel:
        procs = []
        for fam in FAMILIES:
            p, log_f = train_family(fam, args.cores)
            procs.append((p, log_f))
            print(f"[orchestrator] {fam} trainer PID {p.pid} started", flush=True)
        failed_pids = wait_for_all(procs, poll_interval=30)
    else:
        failed_pids = []
        for fam in FAMILIES:
            p, log_f = train_family(fam, args.cores)
            print(f"[orchestrator] {fam} trainer PID {p.pid} started", flush=True)
            failed_pids.extend(wait_for_all([(p, log_f)], poll_interval=30))

    # Check for failures
    failed = []
    for fam in FAMILIES:
        log_path = os.path.join(MODELS_DIR, fam, "train.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if "FATAL" in content or "Traceback" in content:
                failed.append(fam)

    if failed or failed_pids:
        print(f"[ERROR] Training or required targets failed for: "
              f"{', '.join(failed)} (failed PIDs: {failed_pids})")
        return 1

    if args.skip_eval:
        print("\n[orchestrator] Training complete; --skip-eval set, not evaluating.")
        return 0

    print("\n" + "=" * 60)
    print("All 4 trainers completed. Running multi-expert evaluation...")
    print("=" * 60)

    success, outdir = run_evaluation(
        MODELS_DIR,
        os.path.join(REPO, "generation_1", "I3", "alu_pool"),
        MUSD,
        per_algo=args.per_algo,
    )

    if success:
        print(f"\n[orchestrator] Evaluation complete. Results in: {outdir}")
        print("[orchestrator] I6 multi-experiment pipeline complete.")
        return 0
    else:
        print(f"\n[orchestrator] Evaluation failed. See {outdir}/orchestrator_log.txt")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[orchestrator] Interrupted by user. Cleaning up...")
        # Kill any remaining trainers
        for p in find_process("train_i6"):
            try:
                os.kill(p, signal.SIGTERM)
                print(f"[orchestrator] Killed PID {p}")
            except OSError:
                pass
        sys.exit(1)

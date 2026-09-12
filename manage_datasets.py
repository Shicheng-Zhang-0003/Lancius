#!/usr/bin/env python3
"""Lancius dataset pullset.

Vision (C-trainable, unchanged):
  - MNIST, CIFAR-10 — consumed directly by examples/train_mnist.c,
    examples/train_cifar10.c and the parity runners.

Mathematical / proof / derivational-logic sets (Python-side analysis):
  These are TEXT datasets. The Lancius C11 core trains tiny CNNs/MLPs, so
  these sets are NOT fed to the C trainer directly. Use them on the Python
  side for derivational-chain analysis, proof-step auditing, and parity
  checks against Lancius kernels (matmul/softmax/attention), then distill
  fixed-size numeric vectors into .lancius models if needed.

  - gsm8k      : 8.5k multi-step grade-school word problems with
                 calculator-annotated derivations. Best for derivational
                 chain analysis (2-8 steps, + - * /).
  - math       : 12.5k competition problems (AMC/AIME) with full step-by-step
                 solutions across 7 subjects + difficulty 1-5. Best for proof
                 analysis at increasing depth.
  - minif2f    : 488 formal Olympiad statements (Lean/Metamath, v1 frozen).
                 Best for formal proof derivations.
  - prm800k    : 800k human step-level correctness labels (-1/0/+1) on
                 MATH solutions. THE verifier set: each derivation step is
                 already judged false / check-it-out / true.
  - proofwriter: ~500k NL rulebase questions + answers + proofs (depths
                 D0-D5, closed/open world). Best for derivational logic proofs.
  - ruletaker  : synthetic NL theories generator (AllenAI). Best for
                 deductive depth-generalization experiments.
  - svamp      : 1k perturbed word problems + MAWPS/ASDiv-A CV splits. Best
                 for robustness of derivations under variation.
"""

import os
import sys
import shutil
import urllib.request
import urllib.error
import tarfile
import gzip
import zipfile
import json

DATA_DIR = os.environ.get("LANCIUS_DATA_DIR", "data_text")

# Files and directories that are safe to nuke
CLEANUP_TARGETS = [
    "cifar-10-batches-bin",
    "cifar-10-batches-py",
    "data",  # Torchvision cache
    "cifar.tar.gz",
    "cifar-10-binary.tar.gz",
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    "test_batch.bin",
    "parity_input.bin",
    "baseline_out.bin",
    "lancius_out.bin",
    "lancius_preds.bin",
    "fuzz.lancius",
    "garbage.lancius",
    "trunc.lancius",
    "malicious.lancius",
    DATA_DIR,
]


def _fetch(url, dest, optional=False):
    """Download url -> dest. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  SKIP (exists): {dest}")
        return True
    print(f"  Fetching {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {dest} ({os.path.getsize(dest)} bytes)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  !! HTTP {e.code} for {url}")
    except Exception as e:  # network, disk, ...
        print(f"  !! Failed {url}: {e}")
    if os.path.exists(dest):
        try:
            os.remove(dest)
        except OSError:
            pass
    if not optional:
        print("  !! Required file missing — see manual fallback in help.")
    return False


def clean_datasets():
    print("🧹 Cleaning up raw datasets and intermediate binaries...")
    freed = 0
    for target in CLEANUP_TARGETS:
        if os.path.exists(target):
            if os.path.isdir(target):
                size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(target) for f in fn)
                shutil.rmtree(target)
                freed += size
                print(f"  🗑️  Deleted directory: {target}")
            else:
                size = os.path.getsize(target)
                os.remove(target)
                freed += size
                print(f"  🗑️  Deleted file: {target}")

    print(f"\n✅ Cleanup complete. Freed ~{freed / (1024*1024):.2f} MB.")
    print("💡 Note: Your trained .lancius, .onnx, and .gguf models are SAFE and untouched.")


def download_mnist():
    print("📥 Downloading MNIST...")
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    for f in files:
        out_name = f.replace(".gz", "")
        if not os.path.exists(out_name):
            print(f"  Fetching {f}...")
            urllib.request.urlretrieve(base_url + f, f)
            with gzip.open(f, 'rb') as f_in:
                with open(out_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f)
    print("✅ MNIST ready.")


def download_cifar10():
    print("📥 Downloading CIFAR-10 (Binary version for C)...")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    tar_name = "cifar-10-binary.tar.gz"
    if not os.path.exists("cifar-10-batches-bin/data_batch_1.bin"):
        print(f"  Fetching {tar_name}...")
        urllib.request.urlretrieve(url, tar_name)
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()
        os.remove(tar_name)
    print("✅ CIFAR-10 ready.")


# ---------------------------------------------------------------- math sets

def download_gsm8k():
    """GSM8K via OpenAI's public GitHub raw JSONL (no auth needed)."""
    print("📥 Downloading GSM8K (grade-school multi-step, derivational)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    base = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
    ok = True
    ok &= _fetch(base + "train.jsonl", os.path.join(DATA_DIR, "gsm8k_train.jsonl"))
    ok &= _fetch(base + "test.jsonl", os.path.join(DATA_DIR, "gsm8k_test.jsonl"))
    # Socratic subquestions are optional (may 404 on some mirrors)
    _fetch(base + "train.socratic.jsonl",
           os.path.join(DATA_DIR, "gsm8k_train.socratic.jsonl"), optional=True)
    print("✅ GSM8K ready." if ok else "⚠️ GSM8K partial — train/test are required.")
    return ok


MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def download_math():
    """Hendrycks MATH via EleutherAI HF parquet mirror (per-subject)."""
    print("📥 Downloading MATH (competition proofs, 7 subjects)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    base = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/"
    got = 0
    for subj in MATH_SUBJECTS:
        for split in ("train", "test"):
            dest = os.path.join(DATA_DIR, f"math_{subj}_{split}.parquet")
            if _fetch(f"{base}{subj}/{split}.parquet", dest, optional=True):
                got += 1
    if got == 0:
        print("⚠️ MATH parquet mirror unreachable. Manual fallback:")
        print("   pip install datasets && python3 -c "
              "\"from datasets import load_dataset; "
              "load_dataset('EleutherAI/hendrycks_math')\"")
        print("   or clone https://github.com/hendrycks/math (see README for data link).")
        return False
    print(f"✅ MATH ready ({got}/14 subject/split files). Needs pyarrow/pandas to read parquet.")
    return True


def download_prm800k():
    """PRM800K phase1: human step labels (-1/0/+1) on MATH derivations.

    Raw GitHub JSONL (no auth). Label semantics map directly onto the
    verifier head: -1 = false, 0 = check-it-out, +1 = true.
    """
    print("📥 Downloading PRM800K (step-level verifier labels)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    # NOTE: data files are Git-LFS objects — raw.githubusercontent.com only
    # returns the pointer; media.githubusercontent.com resolves the blob.
    base = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/"
    ok = True
    ok &= _fetch(base + "phase1_train.jsonl",
                 os.path.join(DATA_DIR, "prm800k_phase1_train.jsonl"))
    ok &= _fetch(base + "phase1_test.jsonl",
                 os.path.join(DATA_DIR, "prm800k_phase1_test.jsonl"))
    print("✅ PRM800K ready." if ok else "⚠️ PRM800K failed.")
    return ok


def download_svamp():
    """SVAMP challenge set + MAWPS/ASDiv-A CV splits (MIT, GitHub raw)."""
    print("📥 Downloading SVAMP + MAWPS/ASDiv-A (robustness derivations)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    ok = _fetch("https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
                os.path.join(DATA_DIR, "svamp.json"))
    # Combined train pool used by the SVAMP paper (graceful if layout changes)
    _fetch("https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/train.csv",
           os.path.join(DATA_DIR, "mawps_asdiva_train.csv"), optional=True)
    _fetch("https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/test.csv",
           os.path.join(DATA_DIR, "mawps_asdiva_test.csv"), optional=True)
    print("✅ SVAMP ready." if ok else "⚠️ SVAMP failed.")
    return ok


# --------------------------------------------------------------- logic sets

def download_minif2f():
    """miniF2F v1 frozen branch (Lean + Metamath statements)."""
    print("📥 Downloading miniF2F v1 (formal Olympiad statements)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "miniF2F-v1.zip")
    out_dir = os.path.join(DATA_DIR, "miniF2F-v1")
    if not os.path.isdir(out_dir):
        if _fetch("https://github.com/openai/miniF2F/archive/refs/heads/v1.zip",
                  zip_path, optional=True):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(DATA_DIR)
            # Normalize top-level dir name
            for name in os.listdir(DATA_DIR):
                full = os.path.join(DATA_DIR, name)
                if name.startswith("miniF2F") and os.path.isdir(full) and full != out_dir:
                    if os.path.isdir(out_dir):
                        shutil.rmtree(out_dir)
                    shutil.move(full, out_dir)
            os.remove(zip_path)
    # HF parquet mirror as a light alternative (optional)
    _fetch("https://huggingface.co/datasets/Tonic/MiniF2F/resolve/main/default/train.parquet",
           os.path.join(DATA_DIR, "minif2f.parquet"), optional=True)
    print("✅ miniF2F ready." if os.path.isdir(out_dir) else "⚠️ miniF2F zip failed; parquet mirror may still have landed.")
    return os.path.isdir(out_dir)


def download_proofwriter(sample_depths=("d3", "d5"), max_files=4):
    """ProofWriter NL proofs.

    Primary: HuggingFace `datasets` mirror (tasksource/proofwriter).
    The full 3.37GB release lives on Kaggle (needs auth), so here we pull
    a small depth-D3/D5 sample for local derivational-logic work.
    """
    print("📥 Downloading ProofWriter sample (NL proofs, depths D0-D5)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("  !! `datasets` lib missing: pip install datasets pyarrow")
        print("  Manual: https://huggingface.co/datasets/tasksource/proofwriter")
        print("  Full 3.37GB: https://www.kaggle.com/datasets/mathurinache/proofwriter (login)")
        return False
    try:
        ds = load_dataset("tasksource/proofwriter")
        print(f"  Splits: {list(ds.keys())}")
        n = 0
        for split, d in ds.items():
            dest = os.path.join(DATA_DIR, f"proofwriter_{split}.jsonl")
            if os.path.exists(dest):
                continue
            with open(dest, "w") as f:
                for i, row in enumerate(d):
                    if i >= 2000:
                        break
                    f.write(json.dumps({k: str(v)[:2000] for k, v in row.items()}) + "\n")
            n += 1
            if n >= max_files:
                break
        print("✅ ProofWriter sample ready (2k rows/split cap).")
        return True
    except Exception as e:
        print(f"  !! ProofWriter pull failed: {e}")
        return False


def download_ruletaker():
    """RuleTaker: clone AllenAI's theory generator (synthetic logic)."""
    print("📥 Fetching RuleTaker generator (synthetic NL theories)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "ruletaker-gen")
    if os.path.isdir(dest):
        print(f"  SKIP (exists): {dest}")
        return True
    if not shutil.which("git"):
        print("  !! git not found. Manual: git clone https://github.com/allenai/ruletaker")
        return False
    rc = os.system(f"git clone --depth 1 https://github.com/allenai/ruletaker {dest}")
    if rc == 0:
        print("✅ RuleTaker generator ready. Generate theories per repo README")
        print("   (needs problog: pip install problog).")
        return True
    print("  !! Clone failed.")
    return False


MATH_TARGETS = ["gsm8k", "math", "prm800k", "svamp"]
LOGIC_TARGETS = ["minif2f", "proofwriter", "ruletaker"]

DOWNLOADERS = {
    "mnist": download_mnist,
    "cifar10": download_cifar10,
    "gsm8k": download_gsm8k,
    "math": download_math,
    "prm800k": download_prm800k,
    "svamp": download_svamp,
    "minif2f": download_minif2f,
    "proofwriter": download_proofwriter,
    "ruletaker": download_ruletaker,
}


def _usage():
    print("Lancius Dataset Manager")
    print("-----------------------")
    print("Vision (C-trainable): MNIST, CIFAR-10")
    print("Math/proof/logic (Python-side derivational analysis):")
    print("  gsm8k, math, svamp, minif2f, proofwriter, ruletaker")
    print("Usage:")
    print("  python3 manage_datasets.py clean")
    print("  python3 manage_datasets.py download all       # everything")
    print("  python3 manage_datasets.py download vision    # mnist + cifar10")
    print("  python3 manage_datasets.py download math      # gsm8k + math + svamp")
    print("  python3 manage_datasets.py download logic     # minif2f + proofwriter + ruletaker")
    print("  python3 manage_datasets.py download mnist|cifar10|gsm8k|math|svamp|minif2f|proofwriter|ruletaker")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    action = sys.argv[1]
    if action == "clean":
        clean_datasets()
    elif action == "download":
        target = sys.argv[2] if len(sys.argv) > 2 else "all"
        if target == "all":
            targets = ["mnist", "cifar10"] + MATH_TARGETS + LOGIC_TARGETS
        elif target == "vision":
            targets = ["mnist", "cifar10"]
        elif target == "math":
            targets = MATH_TARGETS
        elif target == "logic":
            targets = LOGIC_TARGETS
        elif target in DOWNLOADERS:
            targets = [target]
        else:
            print(f"Unknown target: {target}")
            _usage()
            sys.exit(1)
        results = {}
        for t in targets:
            results[t] = DOWNLOADERS[t]()
        print("\nSummary:")
        for t, ok in results.items():
            print(f"  {'✅' if ok else '❌'} {t}")
        sys.exit(0 if all(results.values()) else 2)
    else:
        print("Unknown action.")
        _usage()

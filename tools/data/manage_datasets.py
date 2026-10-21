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


MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB cap per file


# ---------------------------------------------------------------- LDFD path
#
# Dataset acquisition goes through the Lancius Live Data Feeding Framework
# (3463-LDFD) when its shared library is available: one native fetch with
# streaming gzip inflate and tar extraction, landed atomically. When the
# library is absent (no libcurl headers, not built yet) every fetch falls
# back to the hardened urllib path below, so `download` always works.
#
# Deliberately not required: a data pipeline must not gain a hard build
# dependency just because a faster path exists.

try:
    import ldfd_bridge as _ldfd
except Exception as _exc:  # pragma: no cover - import guard
    _ldfd = None
    _LDFD_IMPORT_ERROR = repr(_exc)
else:
    _LDFD_IMPORT_ERROR = None

_LDFD_SESSION = None
_LDFD_CHECKED = False


def _ldfd_session():
    """Return a usable LDFD session, or None. Cached; never raises."""
    global _LDFD_SESSION, _LDFD_CHECKED
    if _LDFD_CHECKED:
        return _LDFD_SESSION
    _LDFD_CHECKED = True
    if _ldfd is None:
        _LDFD_SESSION = None
        return None
    if not _ldfd.available():
        _LDFD_SESSION = None
        return None
    try:
        _LDFD_SESSION = _ldfd.session(max_bytes=MAX_DOWNLOAD_BYTES)
    except Exception:                          # noqa: BLE001
        _LDFD_SESSION = None
    return _LDFD_SESSION


def ldfd_status() -> str:
    """One line describing which acquisition path is active."""
    if _ldfd is None:
        return f"unavailable (bridge import failed: {_LDFD_IMPORT_ERROR})"
    if not _ldfd.available():
        return f"unavailable ({_ldfd.unavailable_reason()})"
    return "active"


def _ldfd_fetch(url, dest, *, gzip_payload=False, tar=False):
    """Fetch via LDFD. Returns True on success, False to fall back."""
    sess = _ldfd_session()
    if sess is None:
        return False
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  SKIP (exists): {dest}")
        return True
    print(f"  Fetching {url} [ldfd] ...")
    try:
        res = sess.fetch_to_file(
            url, dest,
            decompress="gzip" if (gzip_payload or tar) else "auto",
            extract_tar=tar,
        )
    except _ldfd.LDFDError as exc:
        print(f"  !! LDFD fetch failed: {exc}")
        return False
    except Exception as exc:                   # noqa: BLE001
        print(f"  !! LDFD fetch error: {exc!r}")
        return False

    note = ""
    if res.inflated:
        note += ", inflated"
    if tar:
        note += (f", extracted {res.tar_files} file(s) "
                 f"(skipped {res.tar_skipped}, rejected {res.tar_rejected})")
    elif res.bytes_written:
        note += f" ({res.bytes_written} bytes{note})"
    print(f"  Saved: {dest}{note}")
    return True


def _fetch(url, dest, optional=False):
    """Download url -> dest with size cap. Returns True on success.

    Tries LDFD first; falls back to urllib if the native path is missing
    or reports an error, so behaviour is identical either way.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  SKIP (exists): {dest}")
        return True

    # .gz payloads and .tar.gz archives are unpacked by LDFD itself.
    lower = dest.lower()
    is_gz = lower.endswith(".gz")
    is_tar = lower.endswith(".tar.gz") or lower.endswith(".tgz")
    if _ldfd_fetch(url, dest, gzip_payload=is_gz, tar=is_tar):
        return True

    print(f"  Fetching {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(dest, 'wb') as out:
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise IOError(f"Download exceeds {MAX_DOWNLOAD_BYTES} byte cap")
                out.write(chunk)
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
    # Despot truth: DATA_DIR is env-controlled; refuse to nuke outside cwd
    # (was: LANCIUS_DATA_DIR=/ wiped arbitrary trees).
    cwd = os.path.realpath(os.getcwd())
    for target in CLEANUP_TARGETS:
        real = os.path.realpath(target)
        if real != cwd and not real.startswith(cwd + os.sep):
            print(f"  SKIP (outside cwd): {target}")
            continue
        if os.path.exists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(target) for f in fn)
                shutil.rmtree(target)
                freed += size
                print(f"  🗑️  Deleted directory: {target}")
            elif os.path.isfile(target):
                size = os.path.getsize(target)
                os.remove(target)
                freed += size
                print(f"  🗑️  Deleted file: {target}")

    print(f"\n✅ Cleanup complete. Freed ~{freed / (1024*1024):.2f} MB.")
    print("💡 Note: Your trained .lancius, .onnx, and .gguf models are SAFE and untouched.")


def _safe_members_tar(tar):
    """Despot truth: tar.extractall is tar-slip (was: malicious member wrote
    outside cwd on MITM/mirror).
    Despot V9: total decompressed size capped at 8GB (was unbounded inflate
    after a capped 2GB download -> disk-fill). CIFAR-10 raw is ~170MB."""
    total = 0
    for m in tar.getmembers():
        if m.name.startswith('/') or '..' in m.name.split('/'):
            raise ValueError(f"refusing unsafe tar member: {m.name!r}")
        if m.issym() or m.islnk():
            raise ValueError(f"refusing tar link: {m.name!r}")
        total += m.size
        if total > 8 * 1024 * 1024 * 1024:
            raise ValueError("tar payload exceeds 8GB decompressed cap; refusing (zip-bomb guard)")
    return tar.getmembers()


def _safe_extract_zip(z, dest_dir):
    """Despot truth: ZipFile.extractall is zip-slip (was: unsanitized)."""
    base = os.path.realpath(dest_dir)
    for name in z.namelist():
        if name.startswith('/') or '..' in name.split('/'):
            raise ValueError(f"refusing unsafe zip member: {name!r}")
        target = os.path.realpath(os.path.join(dest_dir, name))
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(f"refusing zip escape: {name!r}")
    z.extractall(dest_dir)


def download_mnist():
    """MNIST idx files. LFD inflates the .gz in one pass so only the raw
    bytes train_mnist.c reads are ever written to disk."""
    print("📥 Downloading MNIST...")
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    ok = True
    for f in files:
        raw_name = f[:-3]                     # strip .gz
        url = base_url + f
        # LDFD path: fetch + inflate, landing the decompressed file.
        if _ldfd_fetch(url, raw_name, gzip_payload=True):
            continue
        # urllib path: fetch the .gz, then expand it here.
        try:
            print(f"  Fetching {f}...")
            with urllib.request.urlopen(url, timeout=60) as response, open(f, 'wb') as out:
                total = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise IOError(f"Download exceeds {MAX_DOWNLOAD_BYTES} byte cap")
                    out.write(chunk)
            # Despot V9: decompressed size was uncapped (zip-bomb fills disk:
            # compressed capped at 2GB, inflate via copyfileobj unbounded).
            # Cap inflate at 8GB (MNIST raw is ~50MB; 8GB is 160x headroom).
            with gzip.open(f, 'rb') as f_in:
                with open(raw_name, 'wb') as f_out:
                    _total_out = 0
                    while True:
                        _chunk = f_in.read(65536)
                        if not _chunk:
                            break
                        _total_out += len(_chunk)
                        if _total_out > 8 * 1024 * 1024 * 1024:
                            raise IOError("Decompressed MNIST exceeds 8GB cap; refusing (zip-bomb guard)")
                        f_out.write(_chunk)
            os.remove(f)
        except Exception as e:  # noqa: BLE001
            print(f"  !! MNIST download failed: {e}")
            for p in (f, raw_name):
                try:
                    if os.path.exists(p) and os.path.getsize(p) == 0:
                        os.remove(p)
                except OSError:
                    pass
            ok = False
    print("✅ MNIST ready." if ok else "❌ MNIST failed.")
    return ok


def download_cifar10():
    """CIFAR-10 binary. LDFD extracts the .tar.gz in place, so the archive
    itself never lands and train_cifar10.c finds cifar-10-batches-bin/."""
    print("📥 Downloading CIFAR-10 (Binary version for C)...")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    marker = "cifar-10-batches-bin/data_batch_1.bin"

    if os.path.exists(marker):
        print("✅ CIFAR-10 ready.")
        return True

    # LDFD path: fetch + extract, atomically landing the batch directory.
    if _ldfd_fetch(url, "cifar-10-batches-bin", tar=True):
        print("✅ CIFAR-10 ready.")
        return True

    # urllib path: fetch, then extract with the existing safe extractor.
    tar_name = "cifar-10-binary.tar.gz"
    try:
        print(f"  Fetching {tar_name}...")
        with urllib.request.urlopen(url, timeout=60) as response, open(tar_name, 'wb') as out:
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise IOError(f"Download exceeds {MAX_DOWNLOAD_BYTES} byte cap")
                out.write(chunk)
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall(members=_safe_members_tar(tar))
        os.remove(tar_name)
    except Exception as e:  # noqa: BLE001
        print(f"  !! CIFAR-10 download failed: {e}")
        try:
            if os.path.exists(tar_name):
                os.remove(tar_name)
        except OSError:
            pass
        return False
    print("✅ CIFAR-10 ready.")
    return True


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
                _safe_extract_zip(z, DATA_DIR)
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
                    f.write(json.dumps({k: str(v) for k, v in row.items()}) + "\n")
            n += 1
            if n >= max_files:
                break
        print("✅ ProofWriter sample ready (2k rows/split cap).")
        return True
    except Exception as e:
        print(f"  !! ProofWriter pull failed: {e}")
        return False


def _safe_dest(sub):
    """Despot truth: DATA_DIR is env-controlled; resolve dest and refuse
    escapes (was: unquoted os.system + rmtree targets from env)."""
    base = os.path.realpath(DATA_DIR)
    dest = os.path.realpath(os.path.join(DATA_DIR, sub))
    if dest != base and not dest.startswith(base + os.sep):
        raise ValueError(f"refusing path escape: {sub!r} -> {dest!r}")
    return dest


def download_ruletaker():
    """RuleTaker: clone AllenAI's theory generator (synthetic logic)."""
    print("📥 Fetching RuleTaker generator (synthetic NL theories)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = _safe_dest("ruletaker-gen")
    if os.path.isdir(dest):
        print(f"  SKIP (exists): {dest}")
        return True
    if not shutil.which("git"):
        print("  !! git not found. Manual: git clone https://github.com/allenai/ruletaker")
        return False
    # Despot truth: unquoted os.system with env-controlled dest was command
    # injection (LANCIUS_DATA_DIR='x;rm -rf ~'). No shell.
    import subprocess
    try:
        r = subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/allenai/ruletaker", dest],
                           capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"  !! Clone failed: {e}")
        return False
    if r.returncode == 0:
        print("✅ RuleTaker generator ready. Generate theories per repo README")
        print("   (needs problog: pip install problog).")
        return True
    print(f"  !! Clone failed: {r.stderr.strip()[:200]}")
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


def show_status():
    """Report which acquisition path is active (no network access)."""
    print("Lancius dataset acquisition")
    print(f"  data dir      : {DATA_DIR}")
    print(f"  LDFD (3463)   : {ldfd_status()}")
    print(f"  fallback      : urllib (always available)")
    print(f"  size cap      : {MAX_DOWNLOAD_BYTES} bytes")
    installed = []
    for name, probe in (("mnist", "train-images-idx3-ubyte"),
                        ("cifar10", "cifar-10-batches-bin/data_batch_1.bin"),
                        ("prm800k", os.path.join(DATA_DIR, "prm800k_phase1_train.jsonl")),
                        ("gsm8k", os.path.join(DATA_DIR, "gsm8k_train.jsonl")),
                        ("svamp", os.path.join(DATA_DIR, "svamp.json"))):
        installed.append((name, os.path.exists(probe)))
    print("  datasets present: " +
          ", ".join(f"{n}{'' if p else ' (missing)'}" for n, p in installed))
    return 0


def _usage():
    print("Lancius Dataset Manager")
    print("-----------------------")
    print("Vision (C-trainable): MNIST, CIFAR-10")
    print("Math/proof/logic (Python-side derivational analysis):")
    print("  gsm8k, math, svamp, minif2f, proofwriter, ruletaker")
    print("Usage:")
    print("  python3 manage_datasets.py status               # which fetch path is live")
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
    if action == "status":
        sys.exit(show_status())
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

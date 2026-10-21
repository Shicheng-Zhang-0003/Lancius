#!/usr/bin/env python3
"""Gen1 gauntlet: the bench_complexity gauntlet, scaled to the 10M model.

Same two questions as tools/prm/bench_complexity.py, asked in the model's
native 65-dim measurement space instead of the 16-dim byte-count space, and
across all 20 algorithms in tools/prm/algos.py (2x the theorems of bench's
10 hand-built levels) x 3 complexity tiers x 5 veracity levels:

  A. ACCURACY vs COMPLEXITY. Overall rows (step_idx=-1) from
     verifier5.make_candidate + check + row_features, standardised with the
     model's own fit-only MU/SD, scored by argmax over the 5 heads.
     Reported per algorithm, per tier, per level, with confusion.
  B. MINIMAL-PAIR INVARIANCE in 65-dim. Pairs whose 65-dim feature vectors
     are BIT-IDENTICAL but whose labels differ. bench_complexity proved the
     16-dim model MUST collide (1150/1150). The 65-dim space carries checker
     measurements (exactness, rel error, structure, coefficients), so
     collisions should be ~0 -- the anti-bench proof that the representation
     carries signal.
  C. DILUTION + WEAK-SPOT. Accuracy vs derivation length (short 2-4 steps
     vs long 7-12: euclid_gcd, mod_pow_fast) and the +0.5 detection rate
     (the known weakest level: right answer, sloppy working).

Weights come from the frozen .lancius (bound INPUT nodes), never from ASCII:
parse via export_lancius_onnx.parse_lancius. MU/SD are recomputed EXACTLY as
examples/train_gen1.c does (xorshift64* shuffle with SEED, VALFRAC split,
fit-only standardise), so the gauntlet scores what the model saw.

Read-only. Usage:
    python3 tools/prm/bench_gen1.py temp/gen1-runs/gen1-10M.lancius [samples_per_cell]
"""
import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from algos import ALGORITHMS
import verifier5 as V
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "onnx"))
from export_lancius_onnx import parse_lancius

LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
LVLNAME = {-1.0: "-1.0 plain false", -0.5: "-0.5 probably false",
           0.0: "0.0 undecidable", 0.5: "+0.5 mostly true", 1.0: "+1.0 plain true"}
_FALLBACKS = 0
SEED = 20261007
VALFRAC = 0.15
FEAT = 65
NCLASS = 5

# ---- C RNG mirror (train_gen1.c xorshift64*) ----
MASK = (1 << 64) - 1


class CRng:
    def __init__(self, s):
        self.s = ((s * 6364136223846793005 + 1442695040888963407) & MASK)

    def u01(self):
        self.s = ((self.s * 6364136223846793007 + 1442695040888963407) & MASK)
        z = self.s
        z ^= z >> 12
        z ^= z << 25
        z ^= z >> 27
        z = (z * 0x2545F4914F6CDD1D) & MASK
        self.s = z
        return float(z >> 11) / 9007199254740992.0


class PyRng:
    """adaptor: verifier5 uses rng.randrange/randint/choice."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def randrange(self, a, b=None):
        return self.r.randrange(a, b) if b is not None else self.r.randrange(a)

    def randint(self, a, b):
        return self.r.randint(a, b)

    def choice(self, seq):
        return self.r.choice(seq)

    def random(self):
        return self.r.random()


# ---- model ----
def load_gen1(path):
    import numpy as np
    d = parse_lancius(path)
    nodes = d["nodes"] if isinstance(d, dict) and "nodes" in d else d
    Ws = {}
    for n in nodes:
        if not n.get("has_w"):
            continue
        sh = tuple(n["shape"][:n["ndim"]])
        w = np.asarray(n["weights"], dtype=np.float64).reshape(sh)
        Ws[sh] = w
    def need(sh):
        if sh not in Ws:
            raise SystemExit(f"missing weights for shape {sh} in {path}")
        return Ws[sh]
    W1 = need((65, H1G))
    b1 = need((1, H1G)).reshape(-1)
    W2 = need((H1G, H2G))
    b2 = need((1, H2G)).reshape(-1)
    W3 = need((H2G, 5))
    b3 = need((1, 5)).reshape(-1)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


def compute_musd(data_dir):
    """Mirror train_gen1.c: load train bins, C-shuffle with SEED, VALFRAC split,
    fit-only MU/SD."""
    import numpy as np
    import struct as _st
    with open(os.path.join(data_dir, "train.X.bin"), "rb") as f:
        blob = f.read()
    n = len(blob) // (FEAT * 8)
    X = np.frombuffer(blob, dtype=np.float64).reshape(n, FEAT).copy()
    rng = CRng(SEED)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.u01() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    nval = int(VALFRAC * n)
    nfit = n - nval
    fit = X[idx[:nfit]]
    mu = fit.mean(axis=0)
    sd = fit.std(axis=0)
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd


def fwd_batch(m, X):
    import numpy as np
    h1 = np.tanh(X @ m["W1"] + m["b1"])
    h2 = np.tanh(h1 @ m["W2"] + m["b2"])
    return h2 @ m["W3"] + m["b3"]


# ---- gauntlet ----
def gen_cell(algo, tier, level, k, seed_base):
    """One overall row: (features_raw65, label_idx)."""
    rng = PyRng(seed_base + k)
    params0, _steps0 = algo.build(rng)
    params = dict(params0)
    if tier == "big":
        params = {kk: (vv * 7 if hasattr(vv, "__mul__") else vv)
                  for kk, vv in params.items()}
    elif tier == "neg":
        # sign-flipped parameters: exercises the sign-agreement channel with
        # disjoint inputs (not a relabel of base).
        params = {kk: (-vv if hasattr(vv, "__neg__") else vv)
                  for kk, vv in params.items()}
    # steps are a deterministic function of params (algo._steps), so tier
    # modifications flow through honestly instead of being re-derived.
    # Some tier transforms are invalid for some algorithms (negated loop
    # counts yield zero steps; x7 breaks 0**negative in definite_integral);
    # fall back to base params for that cell and count it, rather than
    # scoring a broken derivation.
    global _FALLBACKS
    try:
        steps = algo._steps(params)
        if not steps:
            raise ValueError("empty steps")
    except Exception:
        _FALLBACKS += 1
        params = dict(params0)
        steps = algo._steps(params)
    rep, struct, kind = V.make_candidate(algo, params, steps, rng, level)
    chk = V.check(algo, params, steps, rep, struct)
    f = V.row_features(algo, chk, -1)
    return f, LEVELS.index(level), kind, algo.name, tier


def main():
    import numpy as np
    REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(REPO, "temp", "gen1-runs", "gen1-10M.lancius"))
    per_cell = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    data_dir = (sys.argv[3] if len(sys.argv) > 3
                else os.path.join(REPO, "data_vec", "v5"))
    print(f"model: {model_path}")
    global H1G, H2G
    # discover hidden sizes from the file itself
    d0 = parse_lancius(model_path)
    _nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
    _shapes = {tuple(n["shape"][:n["ndim"]]) for n in _nodes if n.get("has_w")}
    H1G = [s[1] for s in _shapes if len(s) == 2 and s[0] == 65][0]
    H2G = [s[0] for s in _shapes if len(s) == 2 and s[1] == 5][0]
    assert (H1G, H2G) in _shapes, f"W2 shape {(H1G,H2G)} not in file"
    print(f"arch: 65 -> {H1G} -> {H2G} -> 5 "
          f"({(65*H1G+H1G+H1G*H2G+H2G+H2G*5+5)/1e6:.2f}M params)")
    m = load_gen1(model_path)
    mu, sd = compute_musd(data_dir)
    print(f"standardise: fit-only MU/SD recomputed (C-mirror, SEED={SEED}), "
          f"mu[0]={mu[0]:.4f} sd[0]={sd[0]:.4f}")

    tiers = ["base", "big", "neg"]
    rows, labels, meta = [], [], []
    seed_base = 9000
    for ai, algo in enumerate(ALGORITHMS):
        for ti, tier in enumerate(tiers):
            for li, lv in enumerate(LEVELS):
                for k in range(per_cell):
                    f, li2, kind, an, tr = gen_cell(
                        algo, tier, lv, k, seed_base + ai * 100000 + ti * 20000 + li * 3000)
                    rows.append(f)
                    labels.append(li2)
                    meta.append((an, tier, lv, kind))
    X = (np.asarray(rows) - mu) / sd
    Z = fwd_batch(m, X)
    pred = Z.argmax(axis=1)
    Y = np.asarray(labels)
    n = len(Y)
    print(f"\nrows: {n} = {len(ALGORITHMS)} algos x {len(tiers)} tiers x "
          f"{len(LEVELS)} levels x {per_cell}")
    if _FALLBACKS:
        print(f"note: {_FALLBACKS} tier cells fell back to base params "
              f"(transform invalid for that algorithm: empty steps)")

    print("\n==== A. OVERALL ====")
    acc = (pred == Y).mean()
    w1 = (np.abs(pred - Y) <= 1).mean()
    print(f"exact {acc:.4f}  within1 {w1:.4f}  (chance 0.2000)")

    print("\nper-level accuracy")
    for li, lv in enumerate(LEVELS):
        msk = Y == li
        p = pred[msk]
        print(f"  {LVLNAME[lv]:22s} {(p == li).mean():.4f} ({(p==li).sum()}/{msk.sum()})")

    print("\nconfusion (rows=true, cols=pred)")
    print("        " + "".join(f"{v:+9.1f}" for v in LEVELS))
    for r in range(5):
        print(f"  {LEVELS[r]:+5.1f}  " + "".join(f"{((pred[Y==r])==c).sum():9d}" for c in range(5)))

    print("\nper-tier accuracy")
    for tier in tiers:
        msk = np.array([mm[1] == tier for mm in meta])
        print(f"  {tier:6s} {(pred[msk]==Y[msk]).mean():.4f} ({msk.sum()} rows)")

    print("\nper-algorithm accuracy (worst 5 / best 5)")
    algacc = []
    for an in sorted({mm[0] for mm in meta}):
        msk = np.array([mm[0] == an for mm in meta])
        algacc.append((an, (pred[msk] == Y[msk]).mean(), msk.sum()))
    algacc.sort(key=lambda t: t[1])
    for an, a, cnt in algacc[:5] + [("...", 0, 0)] + algacc[-5:]:
        if an == "...":
            print("  ...")
        else:
            print(f"  {an:26s} {a:.4f} ({cnt})")

    print("\n==== B. MINIMAL-PAIR INVARIANCE (65-dim) ====")
    # exact float compare on raw (pre-standardised) vectors
    byvec = {}
    collisions = 0
    total_pairs = 0
    for i, f in enumerate(rows):
        key = tuple(f)
        byvec.setdefault(key, []).append(i)
    for key, idxs in byvec.items():
        labs = {labels[i] for i in idxs}
        if len(labs) > 1:
            collisions += 1
            total_pairs += len(idxs)
    print(f"distinct 65-dim vectors : {len(byvec)} / {n}")
    print(f"vectors shared across different labels (collisions): {collisions}")
    print(f"rows involved in collisions: {total_pairs}")
    if collisions == 0:
        print("No two rows with different labels share a bit-identical 65-dim "
              "vector: the representation separates what the labels separate.")

    print("\n==== C. DILUTION (length) + WEAK SPOT (+0.5) ====")
    short = np.array([mm[0] in ("linear_solve", "quadratic_formula",
                                 "difference_of_squares", "slope_from_points")
                      for mm in meta])
    long = np.array([mm[0] in ("euclid_gcd", "mod_pow_fast") for mm in meta])
    print(f"  short derivations (2-4 steps): {(pred[short]==Y[short]).mean():.4f}")
    print(f"  long derivations (7-12 steps): {(pred[long]==Y[long]).mean():.4f}")
    p05 = (Y == 3)
    print(f"  +0.5 recall (said +0.5 | truth +0.5): "
          f"{((pred[p05])==3).mean():.4f}")
    print(f"  +0.5 -> +1 rate (dangerous direction): "
          f"{((pred[p05])==4).mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generation-1 expanded runtime harness: 32 algorithms x N parameter sets.

run_tests.py grown up: the same two sections (DETAIL for the first parameter
set of each algorithm, SWEEP over N sets cycling veracity levels), but

  * 32 algorithms: tools/prm/algos.ALGORITHMS (20) + algos_gen1_ext.EXT_ALGOS (12)
  * 3 tiers per cell (base / big x7 / negated) instead of one bare build
  * overall rows AND per-step rows scored (S==0 vs S>0, like training)
  * the Gen1 10M model (65->4096->2560->5) as the scorer, NumPy forward over
    weights read from the frozen .lancius (never ASCII), MU/SD recomputed as
    the exact C-mirror of examples/train_gen1.c

USAGE
    python3 generation_1/run_tests_gen1.py [params_per_algo] [model] [outdir]
      defaults: 25, temp/gen1-runs/gen1-10M.lancius, generation_1/results
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # generation_1/ lives at repo root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
sys.path.insert(0, REPO)

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
import verifier5 as V
from bench_gen1 import PyRng, load_gen1, compute_musd, fwd_batch, LEVELS, LVLNAME

ALL = list(ALGORITHMS) + list(EXT_ALGOS)
TIERS = ["base", "big", "neg"]
FALLBACKS = 0


def build_cell(algo, tier, seed):
    """(params, steps) with tier transform; base fallback counted."""
    global FALLBACKS
    rng = PyRng(seed)
    params0, _ = algo.build(rng)
    params = dict(params0)
    if tier == "big":
        params = {k: (v * 7 if hasattr(v, "__mul__") else v) for k, v in params.items()}
    elif tier == "neg":
        params = {k: (-v if hasattr(v, "__neg__") else v) for k, v in params.items()}
    try:
        steps = algo._steps(params)
        if not steps:
            raise ValueError("empty")
    except Exception:
        FALLBACKS += 1
        params = dict(params0)
        steps = algo._steps(params)
    return algo, params, steps


def main():
    import numpy as np
    per_algo = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    model_path = (sys.argv[2] if len(sys.argv) > 2
                  else os.path.join(REPO, "temp", "gen1-runs", "gen1-10M.lancius"))
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "results")
    data_dir = (sys.argv[4] if len(sys.argv) > 4
                else os.path.join(REPO, "data_vec", "v5"))
    os.makedirs(outdir, exist_ok=True)

    import bench_gen1 as B
    d0 = B.parse_lancius(model_path)
    _nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
    _shapes = {tuple(n["shape"][:n["ndim"]]) for n in _nodes if n.get("has_w")}
    B.H1G = [s[1] for s in _shapes if len(s) == 2 and s[0] == 65][0]
    B.H2G = [s[0] for s in _shapes if len(s) == 2 and s[1] == 5][0]
    m = load_gen1(model_path)
    mu, sd = compute_musd(data_dir)
    print(f"model: {model_path}  65 -> {B.H1G} -> {B.H2G} -> 5")
    print(f"algos: {len(ALL)} (20 base + 12 gen1-ext), tiers={TIERS}, "
          f"params/algo={per_algo}")

    # ---- SECTION 1: DETAIL, first set of each algorithm ----
    print("\n###### SECTION 1 -- DETAIL (first parameter set per algorithm) ######")
    for algo in ALL:
        a, params, steps = build_cell(algo, "base", 1)
        rng = PyRng(1)
        rep, struct, kind = V.make_candidate(a, params, steps, rng, 1.0)
        chk = V.check(a, params, steps, rep, struct)
        print(f"\n### {a.name} [{a.family}]  steps={len(steps)} kind={kind}")
        print(f"    math: {a.math_desc}")
        for i, s in enumerate(steps):
            got = rep[i] if i < len(rep) else None
            flag = "ok" if got is not None and got == s["value"] else "DIFF"
            print(f"    step{i} {s['name']:16s} exp={s['value']} got={got} {flag}")
        f = V.row_features(a, chk, -1)
        xs = (np.asarray([f]) - mu) / sd
        z = fwd_batch(m, xs)[0]
        p = int(np.argmax(z))
        print(f"    model overall -> {V.LEVELS[p] if hasattr(V, 'LEVELS') else p} "
              f"(logits {[f'{v:.3f}' for v in z]})")

    # ---- SECTION 2: SWEEP ----
    print("\n###### SECTION 2 -- SWEEP ######")
    orows, olabs, ometa = [], [], []
    srows, slabs, smeta = [], [], []
    sb = 7000
    for ai, algo in enumerate(ALL):
        for ti, tier in enumerate(TIERS):
            for k in range(per_algo):
                lv = LEVELS[(k + ai + ti) % len(LEVELS)]
                a, params, steps = build_cell(algo, tier, sb + ai * 100000 + ti * 20000 + k)
                rng = PyRng(sb + ai * 100000 + ti * 20000 + k + 1)
                rep, struct, kind = V.make_candidate(a, params, steps, rng, lv)
                chk = V.check(a, params, steps, rep, struct)
                orows.append(V.row_features(a, chk, -1))
                olabs.append(LEVELS.index(lv))
                ometa.append((a.name, tier, lv))
                for si in range(len(steps)):
                    srows.append(V.row_features(a, chk, si))
                    slabs.append(V.step_label(lv, kind, chk, si))
                    smeta.append((a.name, tier, lv, si))
    Xo = (np.asarray(orows) - mu) / sd
    Zo = fwd_batch(m, Xo)
    po = Zo.argmax(axis=1)
    Yo = np.asarray(olabs)
    Xs = (np.asarray(srows) - mu) / sd
    # step rows in batches (memory: 60k x 4096 f32-ish)
    ps = np.empty(len(srows), dtype=int)
    CH = 8000
    for i in range(0, len(srows), CH):
        ps[i:i + CH] = fwd_batch(m, Xs[i:i + CH]).argmax(axis=1)
    Ys = np.asarray(slabs)

    print(f"\noverall verdict correct : {(po==Yo).sum()}/{len(Yo)} ({(po==Yo).mean():.4f})")
    print(f"per-step verdict correct: {(ps==Ys).sum()}/{len(Ys)} ({(ps==Ys).mean():.4f})")
    print(f"tier fallbacks to base: {FALLBACKS}")
    print("\nper-level accuracy (overall rows)")
    for li, lv in enumerate(LEVELS):
        msk = Yo == li
        print(f"  {LVLNAME[lv]:22s} {(po[msk]==li).mean():.4f} ({(po[msk]==li).sum()}/{msk.sum()})")
    print("\nper-tier accuracy (overall rows)")
    for tier in TIERS:
        msk = np.array([mm[1] == tier for mm in ometa])
        print(f"  {tier:5s} {(po[msk]==Yo[msk]).mean():.4f} ({msk.sum()})")
    print("\nper-algorithm overall accuracy")
    for algo in ALL:
        msk = np.array([mm[0] == algo.name for mm in ometa])
        print(f"  {algo.name:26s} {(po[msk]==Yo[msk]).mean():.4f} "
              f"({(po[msk]==Yo[msk]).sum()}/{msk.sum()})")
    print("\nconfusion (overall rows, rows=true)")
    print("        " + "".join(f"{v:+9.1f}" for v in LEVELS))
    for r in range(5):
        print(f"  {LEVELS[r]:+5.1f}  " + "".join(f"{((po[Yo==r])==c).sum():9d}" for c in range(5)))
    print("\nfirst 40 overall mismatches")
    shown = 0
    for i in range(len(Yo)):
        if po[i] != Yo[i] and shown < 40:
            an, tier, lv = ometa[i]
            print(f"  {an:26s} {tier:4s} truth {lv:+.1f} said {LEVELS[po[i]]:+.1f}")
            shown += 1
    with open(os.path.join(outdir, "sweep_summary.txt"), "w") as f:
        f.write(f"overall {(po==Yo).mean():.4f} per-step {(ps==Ys).mean():.4f} "
                f"n_overall={len(Yo)} n_step={len(Ys)} fallbacks={FALLBACKS}\n")
    print(f"\nsummary -> {outdir}/sweep_summary.txt\nDone.")


if __name__ == "__main__":
    main()

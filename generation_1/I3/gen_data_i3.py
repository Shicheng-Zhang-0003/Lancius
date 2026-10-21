#!/usr/bin/env python3
"""Generation-3 (I3) corpus: the verdict harness over 150 algorithms.

Fork of generation_1/I1/gen_data_gen1.py with exactly three deltas:
  * ALL  = 20 base + 12 gen1-ext + 68 big-ext + 50 i3-ext = 150 algorithms
    (the 50 new I3 families: rational trig geometry, Newton/Euler/Simpson
    numerics, statistics/probability, 3x3 linear algebra, base/bit
    arithmetic, combinatorics)
  * LONG  = gen1 set (euclid_gcd, mod_pow_fast, fibonacci_iter,
    telescoping_series, harmonic_partial, factorial_recurrence) plus the
    long I3 derivations (median_five 12 steps, popcount_halve 11,
    prime_factor_small 11, digit_sum_base 6, log2_pow2 10)
  * meta/edge notes updated for I3

Everything else is byte-compatible with the gen1 pipeline: randomized
regimes (base/wide/frac/neg) applied to built params with validity retry,
steps recomputed from transformed params, edge-case rows by construction,
same .bin format (X/T/S/L/A + algos.txt + meta.json + corpus.hdr), same
0.8/0.2 split. train_i3.c reads this corpus unchanged.

USAGE
    python3 generation_1/I3/gen_data_i3.py [n_per_level] [seed] [out_dir]
      defaults: 3500, 20261019, generation_1/I3/corpus
"""
import os
import sys
import json
import random
import struct as _bin
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
I1DIR = os.path.join(os.path.dirname(HERE), "I1")
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, I1DIR)
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
from algos_i3 import EXT3_ALGOS
import verifier5 as V

ALL = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS) + list(EXT3_ALGOS)
REGIMES = ["base", "wide", "frac", "neg"]
LONG = {"euclid_gcd", "mod_pow_fast", "fibonacci_iter", "telescoping_series",
        "harmonic_partial", "factorial_recurrence",
        "median_five", "popcount_halve", "prime_factor_small",
        "digit_sum_base", "log2_pow2"}


def transform(params, regime, rng):
    """Regime transform on built params (exactness preserved: Fractions only)."""
    if regime == "base":
        return dict(params)
    if regime == "wide":
        k = rng.randint(2, 9)
        return {kk: (vv * k if isinstance(vv, F) else vv) for kk, vv in params.items()}
    if regime == "frac":
        out = {}
        for kk, vv in params.items():
            if isinstance(vv, F) and vv.denominator == 1 and abs(vv) < 100:
                out[kk] = vv + F(rng.choice([1, 2, 3]), rng.choice([2, 3, 4, 5]))
            else:
                out[kk] = vv
        return out
    if regime == "neg":
        return {kk: (-vv if isinstance(vv, F) else vv) for kk, vv in params.items()}
    return dict(params)


def build_regime(algo, regime, rng):
    """Build params under a regime with validity retry; (params, steps, regime_used)."""
    params0, _ = algo.build(rng)
    for _ in range(8):
        try:
            p = transform(params0, regime, rng) if regime != "base" else dict(params0)
            steps = algo._steps(p)
            if steps and len(steps) <= V.MAX_STEPS:
                return p, steps, regime
        except Exception:
            continue
        # transform broke validity: fresh base params, keep trying the regime
        try:
            params0, _ = algo.build(rng)
        except Exception:
            continue
    p, steps = algo.build(rng)
    return p, steps, "base"


def edge_rows(rng):
    """Explicit edge cases -> (algo, params, steps, level). Labeled by construction."""
    out = []
    # E-EMPTY: vacuous derivation (all steps omitted) -> 0.0
    algo = next(a for a in ALL if a.name == "linear_solve")
    params, steps = algo.build(rng)
    rep = [None] * len(steps)
    struct = {k: params[k] for k in algo.structure_keys}
    out.append((algo, params, steps, rep, struct, "empty", 0.0))
    # E-IDENT: horizontal line (zero slope, still valid math)
    algo = next(a for a in ALL if a.name == "slope_from_points")
    for _ in range(4):
        try:
            params, steps = algo.build(rng)
            if len(steps) >= 2:
                out.append((algo, params, steps, None, None, "ident", None))
                break
        except Exception:
            continue
    # E-SIGN: force answers near zero via small params (sign-channel stress)
    algo = next(a for a in ALL if a.name == "difference_of_squares")
    params = {"a": F(1), "b": F(1)}
    try:
        steps = algo._steps(params)  # 0 = a^2-b^2, boundary zero answer
        out.append((algo, params, steps, None, None, "sign", None))
    except Exception:
        pass
    # E-BOUND: large exact magnitudes (near rel-cap regime, still exact)
    algo = next(a for a in ALL if a.name == "arithmetic_series")
    params = {"a": F(99991), "d": F(99989), "n": F(13)}
    try:
        steps = algo._steps(params)
        out.append((algo, params, steps, None, None, "bound", None))
    except Exception:
        pass
    return out


def emit(algo, params, steps, rng, level, kind, acc, rep=None, struct=None):
    """One derivation -> overall row + one row per canonical step.

    kind=None: normal draw, step_label gets the candidate kind from
    make_candidate. kind set: edge row semantics (composed with the
    candidate kind when the edge params go through a normal draw).
    """
    if rep is None:
        rep, struct, kd = V.make_candidate(algo, params, steps, rng, level)
        if kind is not None:
            kind = f"{kind}/{kd}"
        else:
            kind = kd
    chk = V.check(algo, params, steps, rep, struct)
    X, T, S, L, A = acc
    ai = ALL.index(algo)
    li = int(round((level + 1) * 2))
    X.append(V.row_features(algo, chk, -1)); T.append(li)
    S.append(0); L.append(li); A.append(ai)
    for i in range(len(steps)):
        X.append(V.row_features(algo, chk, i)); T.append(-1)
        S.append(i + 1); L.append(V.step_label(level, kind, chk, i)); A.append(ai)


def generate(n_per_level, seed, out_dir):
    rng = random.Random(seed)
    acc = (X, T, S, L, A) = ([], [], [], [], [])
    per_level = {lv: 0 for lv in V.LEVELS}
    per_algo = {a.name: [0, 0] for a in ALL}
    per_regime = {r: 0 for r in REGIMES}
    n_edge = 0
    n_skipped = 0

    # canonical sweeps: for each level, one explicit sweep of every LONG algo
    for level in V.LEVELS:
        for name in sorted(LONG):
            algo = next(a for a in ALL if a.name == name)
            try:
                params, steps, used = build_regime(algo, rng.choice(REGIMES), rng)
                emit(algo, params, steps, rng, level, None, acc)
                per_level[level] += 1
                per_regime[used] += 1
                per_algo[name][0] += 1
            except Exception:
                n_skipped += 1
        # mixed draws
        for _ in range(n_per_level):
            ok = False
            for _outer in range(6):
                algo = rng.choice(ALL)
                if rng.random() < 0.25:
                    algo = rng.choice([a for a in ALL if a.name in LONG])
                regime = rng.choice(REGIMES)
                for _try in range(12):
                    try:
                        params, steps, used = build_regime(algo, regime, rng)
                        break
                    except Exception:
                        continue
                else:
                    continue
                try:
                    emit(algo, params, steps, rng, level, None, acc)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                n_skipped += 1
                continue
            per_level[level] += 1
            per_regime[used] += 1
            per_algo[algo.name][0] += 1

    # edge rows: same pipeline, levels by construction or normal draw
    for _ in range(n_per_level // 4):
        for algo, params, steps, rep, struct, kind, lv in edge_rows(rng):
            try:
                emit(algo, params, steps, rng, lv, kind, acc, rep, struct)
            except Exception:
                n_skipped += 1
                continue
            per_level[lv] += 1
            per_algo[algo.name][0] += 1
            n_edge += 1

    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))

    def dump(name, ids):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.X.bin"), "wb") as f:
            for i in ids:
                f.write(_bin.pack("<%dd" % V.FEAT_DIM, *X[i]))
        for suf, arr in (("T", T), ("S", S), ("L", L), ("A", A)):
            with open(os.path.join(out_dir, f"{name}.{suf}.bin"), "wb") as f:
                for i in ids:
                    f.write(_bin.pack("<i", arr[i]))

    dump("train", idx[:cut])
    dump("test", idx[cut:])
    with open(os.path.join(out_dir, "corpus.hdr"), "w") as f:
        f.write(f"FEAT {V.FEAT_DIM}\nNCLASS {V.NLEV}\nNTRAIN {cut}\n"
                f"NTEST {len(X) - cut}\n")
    with open(os.path.join(out_dir, "algos.txt"), "w") as f:
        for a in ALL:
            f.write(a.name + "\n")
    fam_names = {}
    for a in ALL:
        fam_names[a.family] = fam_names.get(a.family, 0) + 1
    zero_algos = sorted(n for n, c in per_algo.items() if c[0] == 0)
    meta = {
        "n_algorithms": len(ALL),
        "algorithm_names": [a.name for a in ALL],
        "families": fam_names,
        "feat_dim": V.FEAT_DIM, "n_levels": V.NLEV, "max_steps": V.MAX_STEPS,
        "levels": V.LEVELS, "n_rows": len(X),
        "n_train": cut, "n_test": len(X) - cut,
        "rows_per_level": {str(k): v for k, v in per_level.items()},
        "rows_per_regime": per_regime,
        "n_edge_derivations": n_edge,
        "n_skipped_invalid": n_skipped,
        "zero_coverage_algorithms": zero_algos,
        "regimes": REGIMES,
        "i3_new_algorithms": [a.name for a in EXT3_ALGOS],
        "note": "I3 corpus: 150 algos (20 base + 12 ext + 68 big + 50 i3), "
                "randomized regimes (base/wide/frac/neg), explicit edges "
                "(empty/ident/sign/bound), LONG derivations swept explicitly "
                "then 25% boosted in mixed draws. Malformed inputs excluded by "
                "design (they belong to the -2 gate).",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"rows={len(X)} train={cut} test={len(X)-cut} edge_derivs={n_edge} "
          f"skipped_invalid={n_skipped}")
    print(f"per_level={per_level}")
    print(f"per_regime={per_regime}")
    print(f"families={fam_names}")
    if zero_algos:
        print(f"WARNING zero_coverage={zero_algos}")
    return meta


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261019
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "corpus")
    generate(n, seed, out)

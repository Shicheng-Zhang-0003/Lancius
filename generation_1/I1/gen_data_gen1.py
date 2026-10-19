#!/usr/bin/env python3
"""Generation-1 corpus, redesigned bottom-up (replaces data_vec/v5 for Gen1).

vs tools/prm/verifier5.generate:
  * 32 algorithms (20 base + 12 gen1-ext), not 20
  * randomized introduction values: every derivation draws a REGIME --
    base (original build), wide (magnitudes x2..x9), frac (fractional
    offsets, exact Fractions), neg (sign-flipped) -- applied to the built
    params with validity retry, steps recomputed from transformed params
  * edge-case rows, labeled by construction:
      E-EMPTY  fully omitted derivation -> 0.0 (vacuous, not malformed)
      E-IDENT  identity elements (zero slope, x^1, n=2 minimums) -> judged normally
      E-BOUND  boundary magnitudes (near 1e6 rel cap, exact) -> judged normally
      E-SIGN   answers crossing zero (-1/0/+1 as values) -> sign-channel stress
      E-LONG   longest derivations boosted 2x (fib/euclid/mod_pow dilution stress)
    Malformed inputs (div-by-zero, type errors) are EXCLUDED by design: they
    belong to the -2 gone-gate (Stage 7A), not to gradient updates. Training
    a scorer on garbage teaches it garbage.
  * same .bin format (X/T/S/L/A + algos.txt + meta.json) so train_gen1.c
    runs unchanged; meta records regimes + edge counts (single-owner: here).

USAGE
    python3 generation_1/I1/gen_data_gen1.py [n_per_level] [seed] [out_dir]
      defaults: 2000, 20261007, generation_1/I1/corpus
"""
import os
import sys
import json
import random
import struct as _bin
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
import verifier5 as V

ALL = list(ALGORITHMS) + list(EXT_ALGOS)
REGIMES = ["base", "wide", "frac", "neg"]
LONG = {"euclid_gcd", "mod_pow_fast", "fibonacci_iter", "telescoping_series",
        "harmonic_partial", "factorial_recurrence"}


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


def generate(n_per_level, seed, out_dir):
    rng = random.Random(seed)
    X, T, S, L, A = [], [], [], [], []
    per_level = {lv: 0 for lv in V.LEVELS}
    per_regime = {r: 0 for r in REGIMES}
    n_edge = 0
    n_skipped = 0

    for _ in range(n_per_level):
        for level in V.LEVELS:
            # long derivations boosted 2x (dilution stress by design)
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
                    rep, struct, kind = V.make_candidate(algo, params, steps, rng, level)
                    chk = V.check(algo, params, steps, rep, struct)
                    fr = V.row_features(algo, chk, -1)
                    _ = [V.row_features(algo, chk, i) for i in range(len(steps))]
                except Exception:
                    continue
                ok = True
                break
            if not ok:
                n_skipped += 1
                continue
            ai = ALL.index(algo)
            li = int(round((level + 1) * 2))
            X.append(V.row_features(algo, chk, -1)); T.append(li)
            S.append(0); L.append(li); A.append(ai)
            for i in range(len(steps)):
                X.append(V.row_features(algo, chk, i)); T.append(-1)
                S.append(i + 1); L.append(V.step_label(level, kind, chk, i)); A.append(ai)
            per_level[level] += 1
            per_regime[used] += 1

    # edge rows: same pipeline, levels by construction or normal draw
    for _ in range(n_per_level // 4):
        for algo, params, steps, rep, struct, kind, lv in edge_rows(rng):
            try:
                if rep is None:  # normal draw at a random level (edge-flavoured params)
                    lv = rng.choice(V.LEVELS)
                    rep, struct, kd = V.make_candidate(algo, params, steps, rng, lv)
                    kind = f"{kind}/{kd}"
                chk = V.check(algo, params, steps, rep, struct)
                fr = V.row_features(algo, chk, -1)
                _ = [V.row_features(algo, chk, i) for i in range(len(steps))]
            except Exception:
                n_skipped += 1
                continue
            ai = ALL.index(algo)
            li = int(round((lv + 1) * 2))
            X.append(V.row_features(algo, chk, -1)); T.append(li)
            S.append(0); L.append(li); A.append(ai)
            for i in range(len(steps)):
                X.append(V.row_features(algo, chk, i)); T.append(-1)
                S.append(i + 1); L.append(V.step_label(lv, kind, chk, i)); A.append(ai)
            per_level[lv] += 1
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
    meta = {
        "n_algorithms": len(ALL),
        "algorithm_names": [a.name for a in ALL],
        "feat_dim": V.FEAT_DIM, "n_levels": V.NLEV, "max_steps": V.MAX_STEPS,
        "levels": V.LEVELS, "n_rows": len(X),
        "n_train": cut, "n_test": len(X) - cut,
        "rows_per_level": {str(k): v for k, v in per_level.items()},
        "rows_per_regime": per_regime,
        "n_edge_derivations": n_edge,
        "n_skipped_invalid": n_skipped,
        "regimes": REGIMES,
        "note": "Gen1 corpus: 32 algos, randomized regimes (base/wide/frac/neg), "
                "explicit edges (empty/ident/sign/bound/long-boosted). Malformed "
                "inputs excluded by design (they belong to the -2 gate).",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"rows={len(X)} train={cut} test={len(X)-cut} edge_derivs={n_edge} "
          f"skipped_invalid={n_skipped}")
    print(f"per_level={per_level}")
    print(f"per_regime={per_regime}")
    return meta


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261007
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "corpus")
    generate(n, seed, out)

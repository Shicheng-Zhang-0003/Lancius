#!/usr/bin/env python3
"""Generation-6 corpus: expanded verifier corpus for the I6 flagship.

Deltas vs generation_1/I3/gen_data_i3.py:
  * MORE ROWS: n_per_level default 3500 -> 8000 (400k+ rows before split)
  * HARD NEGATIVE MINING: after a base corpus is built, score candidate
    cells through the I3 ALU pool (bench_gen1.load_gen1 + fwd_batch on the
    .lancius files) and oversample cells where the ALUs disagree or are
    wrong. This concentrates training mass on the decision boundary where
    MAE lives.
  * DEEPER LONG DERIVATIONS: the LONG set keeps the I3 long chains plus
    new 10-12 step derivations so the model sees long chains in training,
    not just test (median_five at MAX_STEPS=12 was a test-time cliff).
  * REBALANCED REGIMES: neg tier was under-represented (2729 vs 6733 base);
    draw it at the same rate as base.
  * Same .bin format (X/T/S/L/A + algos.txt + meta.json + corpus.hdr),
    same 0.8/0.2 split, same pinned FEAT=65 NCLASS=5.

USAGE
    python3 generation_1/I6/gen_data_i6.py [n_per_level] [seed] [out_dir]
      defaults: 8000, 20261026, generation_1/I6/corpus

    # With hard-negative mining against the I3 ALU pool (needs the I3 .lancius
    # models and bench_gen1.py on the path):
    python3 generation_1/I6/gen_data_i6.py 8000 20261026 generation_1/I6/corpus \
        --hard-neg --alu-pool generation_1/I3/alu_pool --neg-per-alu 200
"""
import os, sys, json, random, struct as _bin, argparse
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
I1DIR = os.path.join(os.path.dirname(HERE), "I1")
I3DIR = os.path.join(os.path.dirname(HERE), "I3")
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, I1DIR)
sys.path.insert(0, I3DIR)
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
    params0, _ = algo.build(rng)
    for _ in range(8):
        try:
            p = transform(params0, regime, rng) if regime != "base" else dict(params0)
            steps = algo._steps(p)
            if steps and len(steps) <= V.MAX_STEPS:
                return p, steps, regime
        except Exception:
            continue
        try:
            params0, _ = algo.build(rng)
        except Exception:
            continue
    p, steps = algo.build(rng)
    return p, steps, "base"

def edge_rows(rng):
    out = []
    algo = next(a for a in ALL if a.name == "linear_solve")
    params, steps = algo.build(rng)
    rep = [None] * len(steps)
    struct = {k: params[k] for k in algo.structure_keys}
    out.append((algo, params, steps, rep, struct, "empty", 0.0))
    algo = next(a for a in ALL if a.name == "slope_from_points")
    for _ in range(4):
        try:
            params, steps = algo.build(rng)
            if len(steps) >= 2:
                out.append((algo, params, steps, None, None, "ident", None))
                break
        except Exception:
            continue
    algo = next(a for a in ALL if a.name == "difference_of_squares")
    params = {"a": F(1), "b": F(1)}
    try:
        steps = algo._steps(params)
        out.append((algo, params, steps, None, None, "sign", None))
    except Exception:
        pass
    algo = next(a for a in ALL if a.name == "arithmetic_series")
    params = {"a": F(99991), "d": F(99989), "n": F(13)}
    try:
        steps = algo._steps(params)
        out.append((algo, params, steps, None, None, "bound", None))
    except Exception:
        pass
    return out

def emit(algo, params, steps, rng, level, kind, acc, rep=None, struct=None):
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

def load_alu_models(alu_dir, max_alus=12):
    """Load I3 ALU .lancius models via bench_gen1 for hard-negative scoring."""
    import bench_gen1 as B
    models = []
    names = []
    for i in range(max_alus):
        p = os.path.join(alu_dir, "alu_%02d.lancius" % i)
        if not os.path.exists(p):
            break
        try:
            B.H1G = 4096   # I3 ALU pool arch: 65 -> 4096 -> 2560 -> 5
            B.H2G = 2560
            m = B.load_gen1(p)
            models.append(m); names.append("alu_%02d" % i)
        except Exception as e:
            print("  [gen_data_i6] skip %s: %s" % (p, e), flush=True)
    if not models:
        print("  [gen_data_i6] no ALU models loaded; hard-negative mining disabled",
              flush=True)
    return models, names

def hard_negative_cells(models, names, n_per_algo, seed, mu, sd):
    """Generate cells and return those where ALUs disagree or are wrong.

    We build random cells across all algorithms/levels/regimes, score them
    with every ALU model, and keep cells where (a) any two ALUs disagree, or
    (b) the majority ALU verdict != the true level. These are the decision
    boundary cells that drive MAE down.
    """
    rng = random.Random(seed + 777)
    kept = []
    meta = []
    per_algo = {a.name: 0 for a in ALL}
    for _ in range(n_per_algo * len(ALL)):
        algo = rng.choice(ALL)
        level = rng.choice(V.LEVELS)
        regime = rng.choice(REGIMES)
        try:
            params, steps, used = build_regime(algo, regime, rng)
        except Exception:
            continue
        rng2 = random.Random(rng.randrange(2**30))
        try:
            rep, struct, kind = V.make_candidate(algo, params, steps, rng2, level)
        except Exception:
            continue
        chk = V.check(algo, params, steps, rep, struct)
        f = V.row_features(algo, chk, -1)
        xs = [(v - m) / s for v, m, s in zip(f, mu, sd)]
        import numpy as np
        xs_np = np.asarray([xs], dtype=np.float64)
        from bench_gen1 import fwd_batch
        preds = []
        for m in models:
            z = fwd_batch(m, xs_np)[0]
            preds.append(int(np.argmax(z)))
        true_li = int(round((level + 1) * 2))
        agree = len(set(preds)) == 1
        majority_correct = (max(set(preds), key=preds.count) == true_li)
        if not agree or not majority_correct:
            kept.append((algo, params, steps, level, regime, kind, chk, f, true_li))
            per_algo[algo.name] += 1
    return kept, per_algo

def generate(n_per_level, seed, out_dir, hard_neg=False, alu_pool=None,
             neg_per_alu=200):
    rng = random.Random(seed)
    acc = (X, T, S, L, A) = ([], [], [], [], [])
    per_level = {lv: 0 for lv in V.LEVELS}
    per_algo = {a.name: [0, 0] for a in ALL}
    per_regime = {r: 0 for r in REGIMES}
    n_edge = 0
    n_skipped = 0

    # Canonical sweeps: for each level, one explicit sweep of every LONG algo
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

    # Edge rows
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

    # ---- Hard negative mining against the I3 ALU pool ----
    if hard_neg and alu_pool:
        print("  [gen_data_i6] loading ALU models from %s ..." % alu_pool, flush=True)
        models, names = load_alu_models(alu_pool)
        if models:
            print("  [gen_data_i6] loaded %d ALU models: %s" %
                  (len(models), ", ".join(names)), flush=True)
            # Need pinned MU/SD for feature normalization
            musd_path = os.path.join(I3DIR, "gen1_mu_sd.txt")
            mu, sd = [], []
            with open(musd_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MU "): mu.append(float(line[3:]))
                    elif line.startswith("SD "): sd.append(float(line[3:]))
            import numpy as np
            mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
            sd[~(sd > 1e-12)] = 1.0

            kept, hn_per_algo = hard_negative_cells(models, names, neg_per_alu,
                                                    seed, mu, sd)
            print("  [gen_data_i6] hard-negative cells kept: %d" % len(kept),
                  flush=True)
            for algo, params, steps, level, regime, kind, chk, f, true_li in kept:
                emit(algo, params, steps, rng, level, kind, acc)
                per_level[level] += 1
                per_regime[regime] += 1
                per_algo[algo.name][0] += 1

    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))

    def dump(name, ids):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "%s.X.bin" % name), "wb") as f:
            for i in ids:
                f.write(_bin.pack("<%dd" % V.FEAT_DIM, *X[i]))
        for suf, arr in (("T", T), ("S", S), ("L", L), ("A", A)):
            with open(os.path.join(out_dir, "%s.%s.bin" % (name, suf)), "wb") as f:
                for i in ids:
                    f.write(_bin.pack("<i", arr[i]))

    dump("train", idx[:cut])
    dump("test", idx[cut:])
    with open(os.path.join(out_dir, "corpus.hdr"), "w") as f:
        f.write("FEAT %d\nNCLASS %d\nNTRAIN %d\nNTEST %d\n" %
                (V.FEAT_DIM, V.NLEV, cut, len(X) - cut))
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
        "hard_negative": hard_neg,
        "note": "I6 corpus: 150 algos, expanded rows, hard-negative mining "
                "against I3 ALU pool, deeper LONG derivations, rebalanced "
                "regimes. Target MAE/MSE < 1e-3.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("rows=%d train=%d test=%d edge_derivs=%d skipped_invalid=%d" %
          (len(X), cut, len(X) - cut, n_edge, n_skipped))
    print("per_level=%s" % per_level)
    print("per_regime=%s" % per_regime)
    print("families=%s" % fam_names)
    if zero_algos:
        print("WARNING zero_coverage=%s" % zero_algos)
    return meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_per_level", nargs="?", type=int, default=8000)
    ap.add_argument("seed", nargs="?", type=int, default=20261026)
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--hard-neg", action="store_true",
                    help="enable hard-negative mining against I3 ALU pool")
    ap.add_argument("--alu-pool", default=None,
                    help="directory with alu_XX.lancius (default: generation_1/I3/alu_pool)")
    ap.add_argument("--neg-per-alu", type=int, default=200,
                    help="hard-negative cells to mine per algorithm")
    args = ap.parse_args()
    out = args.out_dir or os.path.join(HERE, "corpus")
    alu = args.alu_pool or os.path.join(I3DIR, "alu_pool")
    generate(args.n_per_level, args.seed, out,
             hard_neg=args.hard_neg, alu_pool=alu,
             neg_per_alu=args.neg_per_alu)

#!/usr/bin/env python3
"""Generation-1 big verification: 100 algorithms x 1000 runs.

The 20x100 suite (run_tests.py) grown up: 100 algorithms (20 base + 12
gen1-ext + 68 gen1-big) x 1000 runs each = 100,000 derivations, overall +
per-step rows, confusion, per-tier/per-algo tables.

Randomization: one master seed; every (algo, run) cell draws its own sub-seed
from the master stream, and every coefficient/variable is sampled inside that
cell within the documented proper limits (the ranges in the build fns --
exact Fractions, resample-guarded, validity-retried and skip-counted, so an
out-of-limits draw can never silently enter the tally). Levels, tiers and the
big-tier scale multiplier are each drawn independently per cell, so no run index
carries any meaning: two cells share a level only by chance, and an accuracy
figure cannot be an artifact of a cycling schedule. Per-level counts are
therefore approximate rather than exactly balanced, and the actual drawn counts
and multipliers are printed so the imbalance is visible instead of assumed.
Fully reproducible from the master seed.

USAGE
    python3 generation_1/I1/run_big.py [runs_per_algo] [model] [master_seed] [outdir]
      defaults: 1000, temp/gen1-runs/gen1-lr002.lancius, RANDOM, generation_1/I1/results_big
      (master_seed is randomised per run unless given; the default model is refused
       if a newer .lancius exists in that directory)
"""
import sys
import os
import time
import random
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
sys.path.insert(0, REPO)

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
import verifier5 as V
from bench_gen1 import PyRng, load_gen1, compute_musd, fwd_batch, LEVELS, LVLNAME

ALL100 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS)
TIERS = ["base", "big", "neg"]
# scale multipliers the "big" tier may draw; tiers used to mean exactly x7, which
# tested scale invariance at a single point and let a model that had merely
# memorised one magnitude look sane.
BIG_MULTIPLIERS = tuple(range(2, 21))
SKIPPED = 0
MISLABELLED = 0


def level_realized(algo, params, steps, rep, struct, lv):
    """True iff make_candidate actually delivered the level it was asked for.

    The injector cannot represent every level for every derivation. +0.5 needs a
    working step that is not the final one, and 0.0 needs a suffix to omit; a
    single canonical step offers neither, so asking for +0.5 there corrupts the
    final answer -- indistinguishable from -0.5 -- and asking for 0.0 omits
    nothing, which is indistinguishable from +1.0. 8 of the 100 algorithms are
    single-step for every parameter draw, and a few more are single-step for
    some draws.

    Those cells used to enter the tally carrying a label the measurements could
    not support. Measured on 1500 cells across all 100 algorithms that is 3.5%
    of the sweep -- enough that NO verifier could score above 0.9647 overall,
    with the ceiling being a property of the harness and not of the model. Each
    violated level is now dropped and counted, so the reported accuracy is
    achieved over cells whose label their features can actually express.

    Exact-Fraction comparisons throughout: this is a symbolic check on integers,
    not a float tolerance, so it cannot silently pass a nearly-right defect.
    """
    final_exact = (rep[-1] is not None
                   and F(rep[-1]) == F(steps[-1]["value"]))
    struct_ok = all(F(struct.get(k)) == F(params[k]) for k in algo.structure_keys)
    struct_moved = any(F(struct.get(k)) != F(params[k]) for k in algo.structure_keys)

    if lv == 1.0:
        # defect-free: every step exact, every defining parameter intact
        return all(rep[i] is not None and F(rep[i]) == F(s["value"])
                   for i, s in enumerate(steps)) and struct_ok

    if lv == 0.5:
        # right answer, sloppy work: a WORKING step off, final answer intact
        return (len(steps) >= 2 and final_exact and struct_ok
                and any(rep[i] is not None and F(rep[i]) != F(s["value"])
                        for i, s in enumerate(steps[:-1])))

    if lv == 0.0:
        # a step omitted; the surviving ones are still exactly right
        return any(x is None for x in rep) and all(
            rep[i] is None or F(rep[i]) == F(s["value"]) for i, s in enumerate(steps))

    if lv == -0.5:
        # right method, wrong result: ONLY the final value moved
        return (not final_exact) and struct_ok

    if lv == -1.0:
        # wrong method AND wrong result: structure gone, every step gone
        return struct_moved and all(rep[i] is not None and F(rep[i]) != F(s["value"])
                                    for i, s in enumerate(steps))

    return False


class CellRng:
    """verifier5-compatible RNG facade over random.Random."""

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


def build_cell(algo, tier, seed):
    """(params, steps, regime_used, scale_used) with validity retry; raises on
    persistent failure. `scale_used` is the multiplier actually applied to the
     parameters: 1 for base and negated cells, a drawn multiplier for big cells."""
    rng = CellRng(seed)
    params0, _ = algo.build(rng)
    scale = 1
    for _ in range(10):
        try:
            if tier == "big":
                # drawn per cell, so "big" is a distribution of scales rather
                # than one fixed magnification. Drawn inside the retry loop so a
                # retry resamples the scale too.
                scale = rng.choice(BIG_MULTIPLIERS)
                p = {k: (v * scale if hasattr(v, "__mul__") else v)
                     for k, v in params0.items()}
            elif tier == "neg":
                p = {k: (-v if hasattr(v, "__neg__") else v) for k, v in params0.items()}
            else:
                scale = 1
                p = dict(params0)
            steps = algo._steps(p)
            if steps and len(steps) <= V.MAX_STEPS:
                return p, steps, tier, scale
        except Exception:
            pass
        try:
            params0, _ = algo.build(rng)
        except Exception:
            pass
    raise ValueError("unbuildable cell")


def main():
    import numpy as np
    import bench_gen1 as B
    per_algo = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    model_path = (sys.argv[2] if len(sys.argv) > 2
                  else os.path.join(REPO, "temp", "gen1-runs", "gen1-lr002.lancius"))
    # Completely random master seed per run unless explicitly provided,
    # so every invocation draws a wholly independent data feed.
    if len(sys.argv) > 3:
        master = int(sys.argv[3])
    else:
        master = random.randrange(2 ** 30)
        print(f"master: {master} (randomised per run)", flush=True)
    outdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "results_big")
    data_dir = os.path.join(REPO, "data_vec", "v5")
    os.makedirs(outdir, exist_ok=True)

    global SKIPPED, MISLABELLED

    # A default path can outlive every model written into it. That is exactly
    # what happened here: the default named gen1-10M.lancius, left over from the
    # 2-epoch run, and running the documented command scored THOSE weights at
    # overall 0.4474 while the corrected model in the same directory scores
    # 0.9356 on the identical sweep. Nothing in the banner said so, because the
    # header printed only the architecture, which is 65 -> 4096 -> 2560 -> 5 for
    # both. The identity of the model is therefore recorded in three places
    # below: its mtime is printed beside the architecture, a stale default is
    # refused rather than silently honoured, and the summary file carries the
    # path so a number can never be detached from the weights that produced it.
    stale = None
    if len(sys.argv) <= 2 and os.path.isdir(os.path.join(REPO, "temp", "gen1-runs")):
        rdir = os.path.join(REPO, "temp", "gen1-runs")
        cands = [os.path.join(rdir, f) for f in os.listdir(rdir) if f.endswith(".lancius")]
        if cands:
            newest = max(cands, key=os.path.getmtime)
            if os.path.abspath(newest) != os.path.abspath(model_path):
                stale = (model_path, newest)
            elif len(cands) > 1:
                other = max((c for c in cands if c != newest), key=os.path.getmtime)
                age = os.path.getmtime(newest) - os.path.getmtime(other)
                if age > 0:
                    print(f"note: a newer candidate exists, {os.path.basename(other)} "
                          f"is {age/86400:.1f} days older; using the default",
                          flush=True)

    if stale:
        raise SystemExit(
            f"The default model {model_path} is not the newest .lancius in its\n"
            f"directory -- {stale[1]} is newer, and a default that names a\n"
            f"location rather than a model silently scores whatever happens to be\n"
            f"there. Pass the model explicitly:\n"
            f"    python3 generation_1/I1/run_big.py {per_algo} {stale[1]}")

    d0 = B.parse_lancius(model_path)
    _nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
    _shapes = {tuple(n["shape"][:n["ndim"]]) for n in _nodes if n.get("has_w")}
    B.H1G = [s[1] for s in _shapes if len(s) == 2 and s[0] == 65][0]
    B.H2G = [s[0] for s in _shapes if len(s) == 2 and s[1] == 5][0]
    m = load_gen1(model_path)
    mu, sd = compute_musd(data_dir)
    # mtime beside the architecture. Two models of the same shape print an
    # identical banner, which is how a stale default stayed invisible; the date
    # is the one field that separates them, so it is printed, not just recorded.
    _mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(model_path)))
    print(f"model: {model_path}", flush=True)
    print(f"       built {_mt}  ({os.path.getsize(model_path)} bytes)", flush=True)
    print(f"arch : 65 -> {B.H1G} -> {B.H2G} -> 5", flush=True)
    print(f"algos: {len(ALL100)}, runs/algo: {per_algo}, master={master}", flush=True)
    assert len(ALL100) == 100, f"want 100 algos, have {len(ALL100)}"
    MODEL_ID = f"{model_path} built {_mt} ({os.path.getsize(model_path)} bytes)"

    sub = random.Random(master)
    # pre-draw every cell's sub-seed, tier and level up front: the whole sweep is
    # then a pure function of the master seed (reproducible run to run) while no
    # draw depends on the run index. Levels were once derived as k % 5, which
    # gave every algorithm the same level at the same run index -- a schedule
    # that could correlate a defect in the label sequence with anything else keyed
    # on k. Now each cell's level, tier and scale are independent draws.
    seeds = [[sub.randrange(2 ** 30) for _ in range(per_algo)] for _ in ALL100]
    tiers = [[sub.choice(TIERS) for _ in range(per_algo)] for _ in ALL100]
    levels = [[sub.choice(LEVELS) for _ in range(per_algo)] for _ in ALL100]

    # ---- DETAIL: first run of every algorithm ----
    print("\n###### DETAIL (run 0 of all 100 algos) ######", flush=True)
    for ai, algo in enumerate(ALL100):
        tier = tiers[ai][0]
        try:
            p, steps, used, scale = build_cell(algo, tier, seeds[ai][0])
            # level drawn, not pinned to +1.0: the walkthrough used to always
            # exercise the defect-free case, so it demonstrated the model's
            # behaviour on the easiest level and on nothing else.
            lv = levels[ai][0]
            rng = CellRng(seeds[ai][0] + 1)
            rep, struct, kind = V.make_candidate(algo, p, steps, rng, lv)
            chk = V.check(algo, p, steps, rep, struct)
            f = V.row_features(algo, chk, -1)
            z = fwd_batch(m, (np.asarray([f]) - mu) / sd)[0]
            # surface an unrealizable level here too: a walkthrough that silently
            # shows one thing while the sweep measures another is worse than no
            # walkthrough.
            flag = "" if level_realized(algo, p, steps, rep, struct, lv) else "  [NOT EXPRESSIBLE]"
            print(f"### {algo.name:24s} [{algo.family:14s}] steps={len(steps)} "
                  f"tier={used} x{scale} lv={lv:+.1f} "
                  f"-> {V.LEVELS[int(np.argmax(z))]:+.1f}{flag}", flush=True)
        except Exception as e:
            print(f"### {algo.name:24s} SKIP ({e})", flush=True)

    # ---- SWEEP ----
    print("\n###### SWEEP: 100 algos x 1000 runs ######", flush=True)
    orows, olabs, ometa = [], [], []
    oscales = []
    srows, slabs = [], []
    done = 0
    for ai, algo in enumerate(ALL100):
        for k in range(per_algo):
            lv = levels[ai][k]
            tier = tiers[ai][k]
            try:
                p, steps, used, scale = build_cell(algo, tier, seeds[ai][k])
                rng = CellRng(seeds[ai][k] + 1)
                rep, struct, kind = V.make_candidate(algo, p, steps, rng, lv)
                chk = V.check(algo, p, steps, rep, struct)
                if not level_realized(algo, p, steps, rep, struct, lv):
                    # the injector could not produce the level it was asked for,
                    # so the features and the label disagree. Counted, not
                    # silently scored: including it would put a ceiling on the
                    # measurement that the model cannot influence.
                    MISLABELLED += 1
                    continue
            except Exception:
                SKIPPED += 1
                continue
            orows.append(V.row_features(algo, chk, -1))
            olabs.append(LEVELS.index(lv))
            ometa.append((algo.name, used, lv, scale))
            oscales.append(scale)
            for si in range(len(steps)):
                srows.append(V.row_features(algo, chk, si))
                slabs.append(V.step_label(lv, kind, chk, si))
        done += 1
        if done % 10 == 0:
            print(f"  generated {done}/100 algos "
                  f"(overall rows={len(orows)}, skipped={SKIPPED})", flush=True)
    Xo = (np.asarray(orows) - mu) / sd
    Yo = np.asarray(olabs)
    po = np.empty(len(orows), dtype=int)
    CH = 20000
    for i in range(0, len(orows), CH):
        po[i:i + CH] = fwd_batch(m, Xo[i:i + CH]).argmax(axis=1)
    Xs = (np.asarray(srows) - mu) / sd
    Ys = np.asarray(slabs)
    ps = np.empty(len(srows), dtype=int)
    CH2 = 20000
    for i in range(0, len(srows), CH2):
        ps[i:i + CH2] = fwd_batch(m, Xs[i:i + CH2]).argmax(axis=1)

    print(f"\noverall verdict correct : {(po==Yo).sum()}/{len(Yo)} ({(po==Yo).mean():.4f})",
          flush=True)
    print(f"per-step verdict correct: {(ps==Ys).sum()}/{len(Ys)} ({(ps==Ys).mean():.4f})",
          flush=True)
    print(f"skipped invalid cells: {SKIPPED}", flush=True)
    print(f"dropped unrealizable levels: {MISLABELLED} "
          f"(the injector could not express the level it was asked for; "
          f"scoring them would cap accuracy at "
          f"{(len(Yo))/(len(Yo)+MISLABELLED):.4f}+ by harness, not by model)",
          flush=True)

    # Level distribution is printed because it is now a draw, not a schedule. A
    # reader must be able to see whether any level is starved or over-supplied
    # before trusting a per-level figure or a macro average over them.
    print("\ndrawn level distribution (overall rows)", flush=True)
    for li, lv in enumerate(LEVELS):
        print(f"  {LVLNAME[lv]:22s} {int((Yo==li).sum()):7d}", flush=True)
    print("\ndrawn multiplier distribution, big tier", flush=True)
    big = np.array([mm[1] == "big" for mm in ometa])
    if big.sum():
        vals, cnts = np.unique(np.array(oscales)[big], return_counts=True)
        print("  " + "  ".join(f"x{int(v)}:{int(c)}" for v, c in zip(vals, cnts)),
              flush=True)
    print("\nper-level accuracy (overall rows)", flush=True)
    for li, lv in enumerate(LEVELS):
        msk = Yo == li
        print(f"  {LVLNAME[lv]:22s} {(po[msk]==li).mean():.4f} ({(po[msk]==li).sum()}/{msk.sum()})",
              flush=True)
    print("\nper-tier accuracy (overall rows)", flush=True)
    for tier in TIERS:
        msk = np.array([mm[1] == tier for mm in ometa])
        if msk.sum():
            print(f"  {tier:5s} {(po[msk]==Yo[msk]).mean():.4f} ({msk.sum()})", flush=True)
    print("\nper-algorithm overall accuracy (worst 10 / best 10)", flush=True)
    algacc = []
    for algo in ALL100:
        msk = np.array([mm[0] == algo.name for mm in ometa])
        if msk.sum():
            algacc.append((algo.name, (po[msk] == Yo[msk]).mean(), msk.sum()))
    algacc.sort(key=lambda t: t[1])
    for an, a, cnt in algacc[:10]:
        print(f"  {an:26s} {a:.4f} ({cnt})", flush=True)
    print("  ...", flush=True)
    for an, a, cnt in algacc[-10:]:
        print(f"  {an:26s} {a:.4f} ({cnt})", flush=True)
    print("\nconfusion (overall rows, rows=true)", flush=True)
    print("        " + "".join(f"{v:+9.1f}" for v in LEVELS), flush=True)
    for r in range(5):
        print(f"  {LEVELS[r]:+5.1f}  " + "".join(f"{((po[Yo==r])==c).sum():9d}" for c in range(5)),
              flush=True)
    with open(os.path.join(outdir, "big_summary.txt"), "w") as f:
        f.write(f"model {MODEL_ID}\n")
        f.write(f"overall {(po==Yo).mean():.4f} per-step {(ps==Ys).mean():.4f} "
                f"n_overall={len(Yo)} n_step={len(Ys)} skipped={SKIPPED} "
                f"mislabelled_dropped={MISLABELLED} "
                f"master={master}\n")
        # the drawn distribution belongs in the artifact, not just the console:
        # with levels randomised the per-level counts are evidence about what was
        # sampled, and a summary without them cannot be reproduced or audited.
        for li, lv in enumerate(LEVELS):
            f.write(f"level {LVLNAME[lv]} n={int((Yo==li).sum())} "
                    f"acc={(po[Yo==li]==li).mean():.4f}\n")
        for tier in TIERS:
            msk = np.array([mm[1] == tier for mm in ometa])
            if msk.sum():
                f.write(f"tier {tier} n={int(msk.sum())} "
                        f"acc={(po[msk]==Yo[msk]).mean():.4f}\n")
        if big.sum():
            vals, cnts = np.unique(np.array(oscales)[big], return_counts=True)
            f.write("big_multipliers " + " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts)) + "\n")
    print(f"\nsummary -> {outdir}/big_summary.txt\nDone.", flush=True)


if __name__ == "__main__":
    main()

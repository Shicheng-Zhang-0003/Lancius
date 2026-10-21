#!/usr/bin/env python3
"""Generation-3 big verification: 150 algorithms, TWO models on identical cells.

Fork of generation_1/I1/run_big.py with exactly:

  * ALL150 = 20 base + 12 gen1-ext + 68 gen1-big + 50 i3-ext = 150 algorithms,
    with assert 150.
  * Standardization is PINNED to generation_1/I3/gen1_mu_sd.txt -- the same
    MU/SD train_gen1.c fit on its own corpus. The I2 basis (alu_00) and the
    I3 model share that scaling, so the two score the same feature vectors on
    equal terms. compute_musd() is deliberately NOT called here: recomputing
    from some data_dir would silently drift away from what the warm trunk
    actually consumed.
  * HEAD-TO-HEAD instead of a single-model sweep: both models score the
    identical rows (same cells, same labels), and the output carries per-model
    accuracies, level/tier/per-algorithm tables for each, a paired agreement
    table (both right / i3 only / alu_00 only / neither), and the ten
    algorithms where the delta is largest. A delta number that is not tied to
    identical cells is not comparable; these are the same cells.

USAGE
    python3 generation_1/I3/run_big_i3.py [runs_per_algo] [i3_model] [master_seed] [outdir]
      defaults: 1000, temp/i3-runs/i3_alu_00.lancius, RANDOM, generation_1/I3/results_big
      (the alu_00 basis is generation_1/I2/alu_pool/alu_00.lancius by default)
      (the default i3 model is refused if a newer .lancius exists in its dir)
"""
import sys
import os
import time
import random
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "I1"))
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
sys.path.insert(0, REPO)

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
from algos_i3 import EXT3_ALGOS
import verifier5 as V
from bench_gen1 import PyRng, load_gen1, fwd_batch, LEVELS, LVLNAME

ALL150 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS) + list(EXT3_ALGOS)
DEFAULT_I3 = os.path.join(REPO, "temp", "i3-runs", "i3_alu_00.lancius")
DEFAULT_ALU = os.path.join(REPO, "generation_1", "I2", "alu_pool", "alu_00.lancius")
DEFAULT_MUSD = os.path.join(HERE, "gen1_mu_sd.txt")
TIERS = ["base", "big", "neg"]
BIG_MULTIPLIERS = tuple(range(2, 21))
SKIPPED = 0
MISLABELLED = 0


def read_musd(path):
    """The pinned normalizer: MU/SD lines exactly as the trainers write them."""
    mu, sd = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MU "):
                mu.append(float(line[3:]))
            elif line.startswith("SD "):
                sd.append(float(line[3:]))
    if len(mu) < 65 or len(sd) < 65:
        raise SystemExit(f"{path}: only {len(mu)} MU / {len(sd)} SD lines, want 65")
    import numpy as np
    mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd


def level_realized(algo, params, steps, rep, struct, lv):
    """True iff make_candidate actually delivered the level it was asked for.

    Same contract as generation_1/I1/run_big.py: the injector cannot express
    every level for every derivation (+0.5 needs a non-final working step,
    0.0 needs a suffix to omit), and asking for such a level would put a
    harness cap on the score. Unrealizable cells are dropped and counted.
    Exact-Fraction comparisons throughout.
    """
    final_exact = (rep[-1] is not None
                   and F(rep[-1]) == F(steps[-1]["value"]))
    struct_ok = all(F(struct.get(k)) == F(params[k]) for k in algo.structure_keys)
    struct_moved = any(F(struct.get(k)) != F(params[k]) for k in algo.structure_keys)

    if lv == 1.0:
        return all(rep[i] is not None and F(rep[i]) == F(s["value"])
                   for i, s in enumerate(steps)) and struct_ok

    if lv == 0.5:
        return (len(steps) >= 2 and final_exact and struct_ok
                and any(rep[i] is not None and F(rep[i]) != F(s["value"])
                        for i, s in enumerate(steps[:-1])))

    if lv == 0.0:
        return any(x is None for x in rep) and all(
            rep[i] is None or F(rep[i]) == F(s["value"]) for i, s in enumerate(steps))

    if lv == -0.5:
        return (not final_exact) and struct_ok

    if lv == -1.0:
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
    """(params, steps, regime_used, scale_used) with validity retry."""
    rng = CellRng(seed)
    params0, _ = algo.build(rng)
    scale = 1
    for _ in range(10):
        try:
            if tier == "big":
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


def model_banner(model_path):
    import bench_gen1 as B
    d0 = B.parse_lancius(model_path)
    _nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
    _shapes = {tuple(n["shape"][:n["ndim"]]) for n in _nodes if n.get("has_w")}
    h1 = [s[1] for s in _shapes if len(s) == 2 and s[0] == 65]
    h2 = [s[0] for s in _shapes if len(s) == 2 and s[1] == 5]
    if not h1 or not h2:
        raise SystemExit(f"{model_path}: not a 65-in / 5-out verifier shape")
    _mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(model_path)))
    return h1[0], h2[0], f"{model_path} built {_mt} ({os.path.getsize(model_path)} bytes)"


def main():
    import numpy as np
    per_algo = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_I3
    if len(sys.argv) > 3:
        master = int(sys.argv[3])
    else:
        master = random.randrange(2 ** 30)
        print(f"master: {master} (randomised per run)", flush=True)
    outdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "results_big")
    os.makedirs(outdir, exist_ok=True)

    global SKIPPED, MISLABELLED

    # stale-default guard, same reason as generation_1/I1/run_big.py: a
    # location default silently scores whatever weights happen to sit there.
    stale = None
    idir = os.path.dirname(os.path.abspath(model_path))
    if len(sys.argv) <= 2 and os.path.isdir(idir):
        cands = [os.path.join(idir, f) for f in os.listdir(idir) if f.endswith(".lancius")]
        if cands:
            newest = max(cands, key=os.path.getmtime)
            if os.path.abspath(newest) != os.path.abspath(model_path):
                stale = (model_path, newest)
    if stale:
        raise SystemExit(
            f"The default model {model_path} is not the newest .lancius in its\n"
            f"directory -- {stale[1]} is newer. Pass the model explicitly:\n"
            f"    python3 generation_1/I3/run_big_i3.py {per_algo} {stale[1]}")
    if not os.path.exists(model_path):
        raise SystemExit(f"i3 model not found: {model_path}")
    if not os.path.exists(DEFAULT_ALU):
        raise SystemExit(f"alu_00 basis not found: {DEFAULT_ALU}")
    if not os.path.exists(DEFAULT_MUSD):
        raise SystemExit(f"pinned MU/SD not found: {DEFAULT_MUSD}")

    h1a, h2a, ida = model_banner(model_path)
    h1b, h2b, idb = model_banner(DEFAULT_ALU)
    if (h1a, h2a) != (h1b, h2b):
        raise SystemExit(f"architectures differ: i3 65->{h1a}->{h2a} vs "
                         f"alu_00 65->{h1b}->{h2b}; identical cells need "
                         f"identical trunks (bench_gen1 mirrors one shape)")
    import bench_gen1 as B
    B.H1G = h1a  # load_gen1 reads module-level hidden sizes
    B.H2G = h2a
    m_i3 = load_gen1(model_path)
    m_alu = load_gen1(DEFAULT_ALU)
    models = {"i3": m_i3, "alu00": m_alu}
    mu, sd = read_musd(DEFAULT_MUSD)

    print(f"i3    : {ida}   arch: 65 -> {h1a} -> {h2a} -> 5", flush=True)
    print(f"alu00 : {idb}   arch: 65 -> {h1b} -> {h2b} -> 5", flush=True)
    print(f"mu/sd : pinned from {DEFAULT_MUSD} (gen1 fit; shared by both models)",
          flush=True)
    print(f"algos : {len(ALL150)} (20 base + 12 ext + 68 big + 50 i3), "
          f"runs/algo: {per_algo}, master={master}", flush=True)
    assert len(ALL150) == 150, f"want 150 algos, have {len(ALL150)}"
    MODEL_ID = f"i3={ida} | alu00={idb} | musd={DEFAULT_MUSD}"

    sub = random.Random(master)
    seeds = [[sub.randrange(2 ** 30) for _ in range(per_algo)] for _ in ALL150]
    tiers = [[sub.choice(TIERS) for _ in range(per_algo)] for _ in ALL150]
    levels = [[sub.choice(LEVELS) for _ in range(per_algo)] for _ in ALL150]

    # ---- DETAIL: first draw of every algorithm, both verdicts side by side ----
    print("\n###### DETAIL (run 0 of all 150 algos: i3 vs alu_00) ######", flush=True)
    diff0 = 0
    for ai, algo in enumerate(ALL150):
        tier = tiers[ai][0]
        try:
            p, steps, used, scale = build_cell(algo, tier, seeds[ai][0])
            lv = levels[ai][0]
            rng = CellRng(seeds[ai][0] + 1)
            rep, struct, kind = V.make_candidate(algo, p, steps, rng, lv)
            chk = V.check(algo, p, steps, rep, struct)
            f = V.row_features(algo, chk, -1)
            xs = (np.asarray([f]) - mu) / sd
            zi_i3 = int(np.argmax(fwd_batch(m_i3, xs)[0]))
            zi_al = int(np.argmax(fwd_batch(m_alu, xs)[0]))
            lv_i3, lv_al = LEVELS[zi_i3], LEVELS[zi_al]
            flag = "" if level_realized(algo, p, steps, rep, struct, lv) \
                else "  [NOT EXPRESSIBLE]"
            agree = " " if zi_i3 == zi_al else " DIFF"
            diff0 += 0 if zi_i3 == zi_al else 1
            print(f"### {algo.name:24s} [{algo.family:14s}] steps={len(steps)} "
                  f"tier={used} x{scale} lv={lv:+.1f} "
                  f"i3->{lv_i3:+.1f} alu->{lv_al:+.1f}{agree}{flag}", flush=True)
        except Exception as e:
            print(f"### {algo.name:24s} SKIP ({e})", flush=True)
    print(f"(detail agreement: {len(ALL150)-diff0}/{len(ALL150)})", flush=True)

    # ---- SWEEP ----
    print("\n###### SWEEP: 150 algos x runs ######", flush=True)
    orows, olabs, ometa, oscales = [], [], [], []
    srows, slabs = [], []
    done = 0
    for ai, algo in enumerate(ALL150):
        for k in range(per_algo):
            lv = levels[ai][k]
            tier = tiers[ai][k]
            try:
                p, steps, used, scale = build_cell(algo, tier, seeds[ai][k])
                rng = CellRng(seeds[ai][k] + 1)
                rep, struct, kind = V.make_candidate(algo, p, steps, rng, lv)
                chk = V.check(algo, p, steps, rep, struct)
                if not level_realized(algo, p, steps, rep, struct, lv):
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
        if done % 15 == 0:
            print(f"  generated {done}/150 algos "
                  f"(overall rows={len(orows)}, skipped={SKIPPED})", flush=True)

    Yo = np.asarray(olabs)
    Ys = np.asarray(slabs)
    R_o = (np.asarray(orows) - mu) / sd
    R_s = (np.asarray(srows) - mu) / sd

    def preds(m, X):
        out = np.empty(len(X), dtype=int)
        CH = 20000
        for i in range(0, len(X), CH):
            out[i:i + CH] = fwd_batch(m, X[i:i + CH]).argmax(axis=1)
        return out

    p_i3_o = preds(m_i3, R_o); p_al_o = preds(m_alu, R_o)
    p_i3_s = preds(m_i3, R_s); p_al_s = preds(m_alu, R_s)

    for name, po, ps in (("i3", p_i3_o, p_i3_s), ("alu00", p_al_o, p_al_s)):
        print(f"\n[{name}] overall verdict correct : {(po==Yo).sum()}/{len(Yo)} "
              f"({(po==Yo).mean():.4f})", flush=True)
        print(f"[{name}] per-step verdict correct: {(ps==Ys).sum()}/{len(Ys)} "
              f"({(ps==Ys).mean():.4f})", flush=True)
        for li, lv in enumerate(LEVELS):
            msk = Yo == li
            print(f"[{name}]   level {LVLNAME[lv]:22s} {(po[msk]==li).mean():.4f} "
                  f"({(po[msk]==li).sum()}/{msk.sum()})", flush=True)
        for tier in TIERS:
            msk = np.array([mm[1] == tier for mm in ometa])
            if msk.sum():
                print(f"[{name}]   tier {tier:5s} {(po[msk]==Yo[msk]).mean():.4f} "
                      f"({msk.sum()})", flush=True)

    # paired table on identical cells
    both = int(((p_i3_o == Yo) & (p_al_o == Yo)).sum())
    only_i3 = int(((p_i3_o == Yo) & (p_al_o != Yo)).sum())
    only_al = int(((p_i3_o != Yo) & (p_al_o == Yo)).sum())
    neither = int(((p_i3_o != Yo) & (p_al_o != Yo)).sum())
    print(f"\npaired on identical cells (overall rows):", flush=True)
    print(f"  both right={both}  i3 only={only_i3}  alu_00 only={only_al}  "
          f"neither={neither}", flush=True)

    # per-algorithm delta table: the ten biggest moves either way
    print("\nper-algorithm deltas (i3 minus alu_00, identical cells)", flush=True)
    deltas = []
    for algo in ALL150:
        msk = np.array([mm[0] == algo.name for mm in ometa])
        if msk.sum():
            ai3 = (p_i3_o[msk] == Yo[msk]).mean()
            aal = (p_al_o[msk] == Yo[msk]).mean()
            deltas.append((algo.name, ai3, aal, msk.sum()))
    deltas.sort(key=lambda t: t[1] - t[2])
    print("  i3 loses ground (worst 8):", flush=True)
    for an, ai3, aal, cnt in deltas[:8]:
        print(f"    {an:26s} i3 {ai3:.4f} alu {aal:.4f} ({cnt})", flush=True)
    print("  i3 gains ground (best 8):", flush=True)
    for an, ai3, aal, cnt in deltas[-8:]:
        print(f"    {an:26s} i3 {ai3:.4f} alu {aal:.4f} ({cnt})", flush=True)

    new_algos = [a.name for a in EXT3_ALGOS]
    for name, po in (("i3", p_i3_o), ("alu00", p_al_o)):
        ins = np.array([mm[0] in new_algos for mm in ometa])
        old = ~ins
        print(f"[{name}] on the 50 NEW i3 algorithms : ins-new={(po[ins]==Yo[ins]).mean():.4f}"
              f" (n={ins.sum()})  pre-existing-100={(po[old]==Yo[old]).mean():.4f}"
              f" (n={old.sum()})", flush=True)

    print(f"\nskipped invalid cells: {SKIPPED}", flush=True)
    print(f"dropped unrealizable levels: {MISLABELLED} "
          f"(scoring them would cap accuracy at "
          f"{(len(Yo))/(len(Yo)+MISLABELLED):.4f}+ by harness, not by model)",
          flush=True)

    print("\ndrawn level distribution (overall rows)", flush=True)
    for li, lv in enumerate(LEVELS):
        print(f"  {LVLNAME[lv]:22s} {int((Yo==li).sum()):7d}", flush=True)
    big = np.array([mm[1] == "big" for mm in ometa])
    if big.sum():
        vals, cnts = np.unique(np.array(oscales)[big], return_counts=True)
        print("drawn multiplier distribution, big tier", flush=True)
        print("  " + "  ".join(f"x{int(v)}:{int(c)}" for v, c in zip(vals, cnts)),
              flush=True)

    with open(os.path.join(outdir, "big_summary.txt"), "w") as f:
        f.write(f"models {MODEL_ID}\n")
        f.write(f"overall_i3 {(p_i3_o==Yo).mean():.4f} per_step_i3 {(p_i3_s==Ys).mean():.4f}\n")
        f.write(f"overall_alu00 {(p_al_o==Yo).mean():.4f} per_step_alu00 {(p_al_s==Ys).mean():.4f}\n")
        f.write(f"paired both={both} only_i3={only_i3} only_alu00={only_al} "
                f"neither={neither}\n")
        for name, po, ps in (("i3", p_i3_o, p_i3_s), ("alu00", p_al_o, p_al_s)):
            for li, lv in enumerate(LEVELS):
                msk = Yo == li
                f.write(f"{name} level {LVLNAME[lv]} n={int(msk.sum())} "
                        f"acc={(po[msk]==li).mean():.4f}\n")
            ins = np.array([mm[0] in new_algos for mm in ometa])
            old = ~ins
            f.write(f"{name} new50 {(po[ins]==Yo[ins]).mean():.4f} n={int(ins.sum())} "
                    f"old100 {(po[old]==Yo[old]).mean():.4f} n={int(old.sum())}\n")
        for tier in TIERS:
            msk = np.array([mm[1] == tier for mm in ometa])
            if msk.sum():
                f.write(f"tier {tier} n={int(msk.sum())} "
                        f"i3 {(p_i3_o[msk]==Yo[msk]).mean():.4f} "
                        f"alu00 {(p_al_o[msk]==Yo[msk]).mean():.4f}\n")
        if big.sum():
            vals, cnts = np.unique(np.array(oscales)[big], return_counts=True)
            f.write("big_multipliers " + " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts)) + "\n")
        f.write(f"skipped={SKIPPED} mislabelled_dropped={MISLABELLED} "
                f"n_overall={len(Yo)} n_step={len(Ys)} master={master}\n")
        f.write("delta per algorithm (i3 - alu_00):\n")
        for an, ai3, aal, cnt in deltas:
            f.write(f"  {an:26s} i3 {ai3:.4f} alu00 {aal:.4f} delta {ai3-aal:+.4f} (n={cnt})\n")
    print(f"\nsummary -> {outdir}/big_summary.txt\nDone.", flush=True)


if __name__ == "__main__":
    main()

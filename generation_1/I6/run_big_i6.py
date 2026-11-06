#!/usr/bin/env python3
"""Generation-6 big verification: I6 vs the I3 ALU pool on identical cells.

Head-to-head between the trained I6 model and every I3 ALU (alu_00..alu_11)
on the same 150-algorithm x runs_per_algo randomized cells. Reports per-model
overall/per-step exact, MAE, MSE, and the paired agreement table.

Target: I6 MAE(level) < 1e-3 and MSE(level) < 1e-3 on held-out cells.

USAGE
    python3 generation_1/I6/run_big_i6.py [runs_per_algo] [i6_model] [master_seed] [outdir]
      defaults: 1000, generation_1/I6/models/i6_alu_00.lancius, RANDOM,
                generation_1/I6/results_big
      (the I3 ALU pool is generation_1/I3/alu_pool/ by default)
      (the default i6 model is refused if a newer .lancius exists in its dir)
"""
import sys, os, time, random, argparse
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "I1"))
sys.path.insert(0, os.path.join(HERE, "..", "I3"))
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
sys.path.insert(0, REPO)

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
from algos_i3 import EXT3_ALGOS
import verifier5 as V
from bench_gen1 import PyRng, load_gen1, fwd_batch, LEVELS, LVLNAME

ALL150 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS) + list(EXT3_ALGOS)
DEFAULT_I6 = os.path.join(HERE, "models", "i6_alu_00.lancius")
DEFAULT_ALU_POOL = os.path.join(REPO, "generation_1", "I3", "alu_pool")
DEFAULT_MUSD = os.path.join(HERE, "gen1_mu_sd.txt")
TIERS = ["base", "big", "neg"]
BIG_MULTIPLIERS = tuple(range(2, 21))
SKIPPED = 0
MISLABELLED = 0

def read_musd(path):
    mu, sd = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MU "):
                mu.append(float(line[3:]))
            elif line.startswith("SD "):
                sd.append(float(line[3:]))
    if len(mu) < 65 or len(sd) < 65:
        raise SystemExit("%s: only %d MU / %d SD lines, want 65" % (path, len(mu), len(sd)))
    import numpy as np
    mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd

def level_realized(algo, params, steps, rep, struct, lv):
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
        raise SystemExit("%s: not a 65-in / 5-out verifier shape" % model_path)
    _mt = time.strftime("%Y-%m-%d %H:%M:%S",
                        time.localtime(os.path.getmtime(model_path)))
    return h1[0], h2[0], "%s built %s (%d bytes)" % (model_path, _mt,
                                                      os.path.getsize(model_path))

def load_alu_pool(alu_dir, max_alus=12):
    models = {}
    for i in range(max_alus):
        p = os.path.join(alu_dir, "alu_%02d.lancius" % i)
        if not os.path.exists(p):
            break
        try:
            models["alu_%02d" % i] = load_gen1(p)
        except Exception as e:
            print("  [run_big_i6] skip %s: %s" % (p, e), flush=True)
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_per_algo", nargs="?", type=int, default=1000)
    ap.add_argument("i6_model", nargs="?", default=None)
    ap.add_argument("master_seed", nargs="?", type=int, default=None)
    ap.add_argument("outdir", nargs="?", default=None)
    ap.add_argument("--alu-pool", default=DEFAULT_ALU_POOL)
    ap.add_argument("--musd", default=DEFAULT_MUSD)
    args = ap.parse_args()

    per_algo = args.runs_per_algo
    model_path = args.i6_model or DEFAULT_I6
    master = args.master_seed
    if master is None:
        master = random.randrange(2 ** 30)
        print("master: %d (randomised per run)" % master, flush=True)
    outdir = args.outdir or os.path.join(HERE, "results_big")
    os.makedirs(outdir, exist_ok=True)

    global SKIPPED, MISLABELLED

    # stale-default guard: only applies when using the default model path.
    # If the user passed a model explicitly, respect their choice.
    stale = None
    if model_path == DEFAULT_I6:
        idir = os.path.dirname(os.path.abspath(model_path))
        if os.path.isdir(idir):
            cands = [os.path.join(idir, f) for f in os.listdir(idir)
                     if f.endswith(".lancius")]
            if cands:
                newest = max(cands, key=os.path.getmtime)
                if os.path.abspath(newest) != os.path.abspath(model_path):
                    stale = (model_path, newest)
    if stale:
        raise SystemExit(
            "The default model %s is not the newest .lancius in its\n"
            "directory -- %s is newer. Pass the model explicitly:\n"
            "    python3 generation_1/I6/run_big_i6.py %d %s"
            % (stale[0], stale[1], per_algo, stale[1]))
    if not os.path.exists(model_path):
        raise SystemExit("i6 model not found: %s" % model_path)
    if not os.path.exists(args.musd):
        raise SystemExit("pinned MU/SD not found: %s" % args.musd)
    if not os.path.isdir(args.alu_pool):
        raise SystemExit("alu pool dir not found: %s" % args.alu_pool)

    h1a, h2a, ida = model_banner(model_path)
    if (h1a, h2a) != (4096, 2560) and (h1a, h2a) != (5120, 3072):
        print("WARN: i6 arch 65->%d->%d->5 (expected 65->5120->3072->5 or "
              "65->4096->2560->5)" % (h1a, h2a), flush=True)
    import bench_gen1 as B
    B.H1G = h1a
    B.H2G = h2a
    m_i6 = load_gen1(model_path)
    mu, sd = read_musd(args.musd)

    print("i6    : %s   arch: 65 -> %d -> %d -> 5" % (ida, h1a, h2a), flush=True)
    print("mu/sd : %s (pinned; shared with I3 ALUs)" % args.musd, flush=True)
    print("algos : %d (20 base + 12 ext + 68 big + 50 i3), "
          "runs/algo: %d, master=%d" % (len(ALL150), per_algo, master), flush=True)
    assert len(ALL150) == 150, "want 150 algos, have %d" % len(ALL150)

    alu_models = load_alu_pool(args.alu_pool)
    print("alu pool: %d models loaded: %s" %
          (len(alu_models), ", ".join(sorted(alu_models))), flush=True)

    MODEL_ID = "i6=%s | musd=%s" % (ida, args.musd)

    sub = random.Random(master)
    seeds = [[sub.randrange(2 ** 30) for _ in range(per_algo)] for _ in ALL150]
    tiers = [[sub.choice(TIERS) for _ in range(per_algo)] for _ in ALL150]
    levels = [[sub.choice(LEVELS) for _ in range(per_algo)] for _ in ALL150]

    # ---- DETAIL ----
    print("\n###### DETAIL (run 0 of all 150 algos) ######", flush=True)
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
            xs = ((np.asarray([f]) - mu) / sd)
            zi_i6 = int(np.argmax(fwd_batch(m_i6, xs)[0]))
            flag = "" if level_realized(algo, p, steps, rep, struct, lv) \
                else "  [NOT EXPRESSIBLE]"
            print("### %s [%s] steps=%d tier=%s x%s lv=%+.1f i6->%s%s" %
                  (algo.name, algo.family, len(steps), used, scale, lv,
                   LEVELS[zi_i6], flag), flush=True)
        except Exception as e:
            print("### %s SKIP (%s)" % (algo.name, e), flush=True)

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
            print("  generated %d/150 algos (overall rows=%d, skipped=%d)"
                  % (done, len(orows), SKIPPED), flush=True)

    import numpy as np
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

    p_i6_o = preds(m_i6, R_o); p_i6_s = preds(m_i6, R_s)

    print("\n[i6] overall verdict correct : %d/%d (%.4f)"
          % ((p_i6_o == Yo).sum(), len(Yo), (p_i6_o == Yo).mean()), flush=True)
    print("[i6] per-step verdict correct: %d/%d (%.4f)"
          % ((p_i6_s == Ys).sum(), len(Ys), (p_i6_s == Ys).mean()), flush=True)
    for li, lv in enumerate(LEVELS):
        msk = Yo == li
        print("[i6]   level %-22s %.4f (%d/%d)"
              % (LVLNAME[lv], (p_i6_o[msk] == li).mean(),
                 (p_i6_o[msk] == li).sum(), msk.sum()), flush=True)
    for tier in TIERS:
        msk = np.array([mm[1] == tier for mm in ometa])
        if msk.sum():
            print("[i6]   tier %-5s %.4f (%d)"
                  % (tier, (p_i6_o[msk] == Yo[msk]).mean(), msk.sum()), flush=True)

    # Per-model results for every ALU
    alu_results = {}
    for aname in sorted(alu_models):
        am = alu_models[aname]
        pa_o = preds(am, R_o); pa_s = preds(am, R_s)
        alu_results[aname] = (pa_o, pa_s)
        print("[%s] overall correct : %d/%d (%.4f)"
              % (aname, (pa_o == Yo).sum(), len(Yo), (pa_o == Yo).mean()),
              flush=True)
        print("[%s] per-step correct : %d/%d (%.4f)"
              % (aname, (pa_s == Ys).sum(), len(Ys), (pa_s == Ys).mean()),
              flush=True)

    # MAE/MSE for i6 vs each ALU (level scale)
    def mae_mse(preds_arr, true_arr):
        pred_levels = np.array([LEVELS[int(p)] for p in preds_arr])
        true_levels = np.array([LEVELS[int(t)] for t in true_arr])
        d = pred_levels - true_levels
        mae = float(np.mean(np.abs(d)))
        mse = float(np.mean(d * d))
        return mae, mse

    i6_mae, i6_mse = mae_mse(p_i6_o, Yo)
    print("\n[i6] MAE(level)=%.6f  MSE(level)=%.6f" % (i6_mae, i6_mse), flush=True)
    print("Target MAE < 1e-3: %s   Target MSE < 1e-3: %s"
          % ("PASS" if i6_mae < 1e-3 else "FAIL",
             "PASS" if i6_mse < 1e-3 else "FAIL"), flush=True)

    for aname in sorted(alu_results):
        pa_o, _ = alu_results[aname]
        amae, amse = mae_mse(pa_o, Yo)
        print("[%s] MAE=%.6f  MSE=%.6f  delta_vs_i6 MAE=%+.6f"
              % (aname, amae, amse, amae - i6_mae), flush=True)

    # Paired table: i6 vs each ALU on identical cells
    print("\npaired on identical cells (overall rows):", flush=True)
    for aname in sorted(alu_results):
        pa_o, _ = alu_results[aname]
        both = int(((p_i6_o == Yo) & (pa_o == Yo)).sum())
        only_i6 = int(((p_i6_o == Yo) & (pa_o != Yo)).sum())
        only_al = int(((p_i6_o != Yo) & (pa_o == Yo)).sum())
        neither = int(((p_i6_o != Yo) & (pa_o != Yo)).sum())
        print("  vs %-6s both=%d i6_only=%d alu_only=%d neither=%d"
              % (aname, both, only_i6, only_al, neither), flush=True)

    # Per-algorithm delta table
    print("\nper-algorithm deltas (i6 minus best alu, identical cells)", flush=True)
    deltas = []
    for algo in ALL150:
        msk = np.array([mm[0] == algo.name for mm in ometa])
        if msk.sum():
            ai6 = (p_i6_o[msk] == Yo[msk]).mean()
            best_al = max((alu_results[a][0][msk] == Yo[msk]).mean()
                          for a in alu_results)
            deltas.append((algo.name, ai6, best_al, msk.sum()))
    deltas.sort(key=lambda t: t[1] - t[2])
    print("  i6 loses ground (worst 8):", flush=True)
    for an, ai6, abal, cnt in deltas[:8]:
        print("    %s i6 %.4f best_alu %.4f (%d)" % (an, ai6, abal, cnt),
              flush=True)
    print("  i6 gains ground (best 8):", flush=True)
    for an, ai6, abal, cnt in deltas[-8:]:
        print("    %s i6 %.4f best_alu %.4f (%d)" % (an, ai6, abal, cnt),
              flush=True)

    new_algos = [a.name for a in EXT3_ALGOS]
    ins = np.array([mm[0] in new_algos for mm in ometa])
    old = ~ins
    print("\n[i6] on 50 NEW i3 algorithms : %.4f (n=%d)"
          % ((p_i6_o[ins] == Yo[ins]).mean(), ins.sum()), flush=True)
    print("[i6] on pre-existing 100   : %.4f (n=%d)"
          % ((p_i6_o[old] == Yo[old]).mean(), old.sum()), flush=True)

    print("\nskipped invalid cells: %d" % SKIPPED, flush=True)
    print("dropped unrealizable levels: %d"
          % MISLABELLED, flush=True)
    print("\ndrawn level distribution (overall rows)", flush=True)
    for li, lv in enumerate(LEVELS):
        print("  %-22s %d" % (LVLNAME[lv], int((Yo == li).sum())), flush=True)

    with open(os.path.join(outdir, "big_summary.txt"), "w") as f:
        f.write("models %s\n" % MODEL_ID)
        f.write("overall_i6 %.4f per_step_i6 %.4f\n"
                % ((p_i6_o == Yo).mean(), (p_i6_s == Ys).mean()))
        f.write("i6_MAE %.6f i6_MSE %.6f\n" % (i6_mae, i6_mse))
        f.write("target_MAE_1e-3 %s target_MSE_1e-3 %s\n"
                % ("PASS" if i6_mae < 1e-3 else "FAIL",
                   "PASS" if i6_mse < 1e-3 else "FAIL"))
        for aname in sorted(alu_results):
            pa_o, pa_s = alu_results[aname]
            amae, amse = mae_mse(pa_o, Yo)
            f.write("%s overall %.4f per_step %.4f MAE %.6f MSE %.6f\n"
                    % (aname, (pa_o == Yo).mean(), (pa_s == Ys).mean(), amae, amse))
        for name, po in (("i6", p_i6_o),):
            for li, lv in enumerate(LEVELS):
                msk = Yo == li
                f.write("%s level %s n=%d acc=%.4f\n"
                        % (name, lv, int(msk.sum()),
                           (po[msk] == li).mean()))
        f.write("skipped=%d mislabelled_dropped=%d n_overall=%d n_step=%d "
                "master=%d\n"
                % (SKIPPED, MISLABELLED, len(Yo), len(Ys), master))
        f.write("delta per algorithm (i6 - best_alu):\n")
        for an, ai6, abal, cnt in deltas:
            f.write("  %s i6 %.4f best_alu %.4f delta %+.4f (n=%d)\n"
                    % (an, ai6, abal, ai6 - abal, cnt))
    print("\nsummary -> %s/big_summary.txt\nDone." % outdir, flush=True)

if __name__ == "__main__":
    import numpy as np
    main()

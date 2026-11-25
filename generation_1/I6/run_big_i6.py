#!/usr/bin/env python3
"""Generation-6 big verification: I6 vs the I3 ALU pool on identical cells.

Head-to-head between the trained I6 model and every I3 ALU (alu_00..alu_11)
on the same 150-algorithm x runs_per_algo randomized cells. Reports per-model
overall/per-step exact, MAE, MSE, and the paired agreement table.

Target: I6 MAE(level) < 1e-3 and MSE(level) < 1e-3 on held-out cells.

USAGE
    python3 generation_1/I6/run_big_i6.py [runs_per_algo] [i6_model] [master_seed] [outdir]
    defaults: 1000, generation_1/I6/models_verified/alu_algebra.lancius, RANDOM,
                generation_1/I6/results_verified
      (the I3 ALU pool is generation_1/I3/alu_pool/ by default)
"""
import sys, os, time, random, argparse
from fractions import Fraction as F
import numpy as np

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
FAMILIES = list(V.FAMILIES)
DEFAULT_I6 = os.path.join(HERE, "models_verified", "alu_algebra.lancius")
DEFAULT_ALU_POOL = os.path.join(REPO, "generation_1", "I3", "alu_pool")
DEFAULT_MUSD = os.path.join(HERE, "corpus_verified", "mu_sd.txt")
DEFAULT_I3_MUSD = os.path.join(REPO, "generation_1", "I3", "gen1_mu_sd.txt")
DEFAULT_EXPERTS = os.path.join(HERE, "models_verified")
TIERS = ["base", "big", "neg"]
BIG_MULTIPLIERS = tuple(range(2, 21))
SKIPPED = 0
MISLABELLED = 0
# Softmax can saturate to exactly 0/1; keep the reported verdict strictly
# inside the open interval (-1, 1).  The 1e-6 bias is far below the 1e-3
# target and never leaves the logic scale.
VERDICT_EPS = 1e-6

# Algorithm -> family mapping for multi-expert evaluation
ALGO_FAMILY = {}
for a in ALL150:
    ALGO_FAMILY[a.name] = a.family

FAMILY_MODEL_MAP = {
    "algebra": "alu_algebra.lancius",
    "calculus": "alu_calculus.lancius",
    "linear_algebra": "alu_linear_algebra.lancius",
    "number_theory": "alu_number_theory.lancius",
}

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
    mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd


def softmax(z):
    """Row-wise softmax for a batch of logits."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def continuous_verdict(prob, fault_code=None):
    """Continuous verdict in (-1, 1), or -2 for a non-logic fault/error.

    The five level logits are turned into a probability distribution and the
    verdict is the expected level.  The result is strictly inside (-1, 1)
    because it is a convex combination of LEVELS.  A caller that detects a
    non-logic fault (bad input, numerical failure, missing model) returns -2
    instead, which is deliberately outside the logic scale.
    """
    if fault_code is not None:
        return -2.0
    v = float(np.dot(np.asarray(prob, dtype=np.float64), LEVELS))
    return min(1.0 - VERDICT_EPS, max(-1.0 + VERDICT_EPS, v))


def normalize_i6(rows, mu, sd):
    """Match train_i6.c: pinned z-score followed by the training clamp."""
    return np.clip((np.asarray(rows, dtype=np.float64) - mu) / sd, -8.0, 8.0)


def normalize_i3(rows, mu, sd):
    """I1-I3 checkpoints use the legacy feature schema and no input clamp."""
    return (np.asarray(rows, dtype=np.float64) - mu) / sd

# Single source of truth lives in verifier5 so corpus generation and
# evaluation cannot drift apart.
level_realized = V.level_realized

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
    """Load each I3 model with its stored architecture, then restore I6 dims."""
    import bench_gen1 as B
    models = {}
    previous_arch = (B.H1G, B.H2G)
    try:
        for i in range(max_alus):
            p = os.path.join(alu_dir, "alu_%02d.lancius" % i)
            if not os.path.exists(p):
                break
            try:
                h1, h2, _ = model_banner(p)
                B.H1G, B.H2G = h1, h2
                models["alu_%02d" % i] = load_gen1(p)
            except Exception as e:
                print("  [run_big_i6] skip %s: %s" % (p, e), flush=True)
    finally:
        B.H1G, B.H2G = previous_arch
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_per_algo", nargs="?", type=int, default=1000)
    ap.add_argument("i6_model", nargs="?", default=None)
    ap.add_argument("master_seed", nargs="?", type=int, default=None)
    ap.add_argument("outdir", nargs="?", default=None)
    ap.add_argument("--alu-pool", default=DEFAULT_ALU_POOL)
    ap.add_argument("--musd", default=DEFAULT_MUSD)
    ap.add_argument("--i3-musd", default=DEFAULT_I3_MUSD,
                    help="legacy I3 normalizer for comparison models")
    ap.add_argument("--experts", default=DEFAULT_EXPERTS,
                    help="directory containing per-family .lancius models "
                         "(alu_algebra.lancius, alu_calculus.lancius, etc.)")
    ap.add_argument("--strict-targets", action="store_true",
                    help="fail unless aggregate and condition MAE/MSE gates pass")
    ap.add_argument("--each-expert", action="store_true",
                    help="evaluate every I6 expert on every generated scenario")
    ap.add_argument("--skip-alu-baselines", action="store_true",
                    help="skip the legacy I3 comparison models")
    args = ap.parse_args()

    per_algo = args.runs_per_algo
    model_path = args.i6_model or DEFAULT_I6
    master = args.master_seed
    if master is None:
        master = random.randrange(2 ** 30)
        print("master: %d (randomised per run)" % master, flush=True)
    outdir = args.outdir or os.path.join(HERE, "results_verified")
    os.makedirs(outdir, exist_ok=True)

    global SKIPPED, MISLABELLED

    if not os.path.exists(model_path):
        raise SystemExit("i6 model not found: %s" % model_path)
    if not os.path.exists(args.musd):
        raise SystemExit("pinned MU/SD not found: %s" % args.musd)
    if not os.path.exists(args.i3_musd):
        raise SystemExit("I3 MU/SD not found: %s" % args.i3_musd)
    if not args.skip_alu_baselines and not os.path.isdir(args.alu_pool):
        raise SystemExit("alu pool dir not found: %s" % args.alu_pool)

    h1a, h2a, ida = model_banner(model_path)
    print("i6 architecture: 65 -> %d -> %d -> 5" % (h1a, h2a), flush=True)
    import bench_gen1 as B
    B.H1G = h1a
    B.H2G = h2a
    m_i6 = load_gen1(model_path)
    mu, sd = read_musd(args.musd)
    i3_mu, i3_sd = read_musd(args.i3_musd)

    print("i6    : %s   arch: 65 -> %d -> %d -> 5" % (ida, h1a, h2a), flush=True)
    print("mu/sd : %s (I6 sparse semantic features + clamp)"
          % args.musd, flush=True)
    print("I3 mu/sd: %s (legacy mean_rel schema)" % args.i3_musd, flush=True)
    print("algos : %d (20 base + 12 ext + 68 big + 50 i3), "
          "runs/algo: %d, master=%d" % (len(ALL150), per_algo, master), flush=True)
    assert len(ALL150) == 150, "want 150 algos, have %d" % len(ALL150)

    if args.skip_alu_baselines:
        alu_models = {}
        print("I3 ALU baseline comparison skipped", flush=True)
    else:
        alu_models = load_alu_pool(args.alu_pool)
        print("alu pool: %d models loaded: %s" %
              (len(alu_models), ", ".join(sorted(alu_models))), flush=True)

    # Load per-family expert models
    expert_models = {}
    for fam_name, fname in FAMILY_MODEL_MAP.items():
        fpath = os.path.join(args.experts, fname)
        if os.path.exists(fpath):
            expert_models[fam_name] = load_gen1(fpath)
            print("expert %s: loaded %s" % (fam_name, fname), flush=True)
        else:
            print("expert %s: NOT FOUND %s" % (fam_name, fpath), flush=True)

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
            f = V.row_features_i6(algo, chk, -1)
            xs = normalize_i6([f], mu, sd)
            m_exp = expert_models.get(ALGO_FAMILY.get(algo.name, "algebra"))
            if m_exp is None:
                m_exp = m_i6
            zi_i6 = int(np.argmax(fwd_batch(m_exp, xs)[0]))
            flag = "" if level_realized(algo, p, steps, rep, struct, lv) \
                else "  [NOT EXPRESSIBLE]"
            print("### %s [%s] steps=%d tier=%s x%s lv=%+.1f i6->%s%s" %
                  (algo.name, algo.family, len(steps), used, scale, lv,
                   LEVELS[zi_i6], flag), flush=True)
        except Exception as e:
            print("### %s SKIP (%s)" % (algo.name, e), flush=True)

    # ---- SWEEP ----
    print("\n###### SWEEP: 150 algos x runs ######", flush=True)
    orows, i3_orows, olabs, ometa, oscales = [], [], [], [], []
    srows, i3_srows, slabs, sfams, smeta = [], [], [], [], []
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
            orows.append(V.row_features_i6(algo, chk, -1))
            i3_orows.append(V.row_features(algo, chk, -1))
            olabs.append(LEVELS.index(lv))
            ometa.append((algo.name, used, lv, scale))
            oscales.append(scale)
            for si in range(len(steps)):
                srows.append(V.row_features_i6(algo, chk, si))
                i3_srows.append(V.row_features(algo, chk, si))
                slabs.append(V.step_label(lv, kind, chk, si))
                sfams.append(ALGO_FAMILY[algo.name])
                smeta.append((algo.name, used, lv, scale))
        done += 1
        if done % 15 == 0:
            print("  generated %d/150 algos (overall rows=%d, skipped=%d)"
                  % (done, len(orows), SKIPPED), flush=True)

    Yo = np.asarray(olabs)
    Ys = np.asarray(slabs)
    R_o_i6 = normalize_i6(orows, mu, sd)
    R_s_i6 = normalize_i6(srows, mu, sd)
    R_o_i3 = normalize_i3(i3_orows, i3_mu, i3_sd)
    R_s_i3 = normalize_i3(i3_srows, i3_mu, i3_sd)

    def preds(m, X):
        out = np.empty(len(X), dtype=int)
        CH = 20000
        for i in range(0, len(X), CH):
            out[i:i + CH] = fwd_batch(m, X[i:i + CH]).argmax(axis=1)
        return out

    # Multi-expert prediction: route each row to its family expert
    fam_of_row = np.array([ALGO_FAMILY[mm[0]] for mm in ometa])

    def preds_multi(models_dict, X, fam_of_row):
        out = np.empty(len(X), dtype=int)
        covered = np.zeros(len(X), dtype=bool)
        CH = 20000
        for fam_name in sorted(models_dict.keys()):
            m = models_dict[fam_name]
            fam_mask = fam_of_row == fam_name
            if not fam_mask.any():
                continue
            fam_X = X[fam_mask]
            fam_out = np.empty(len(fam_X), dtype=int)
            for i in range(0, len(fam_X), CH):
                fam_out[i:i + CH] = fwd_batch(m, fam_X[i:i + CH]).argmax(axis=1)
            out[fam_mask] = fam_out
            covered[fam_mask] = True
        rest = ~covered
        if rest.any():
            # Missing expert: fall back to the single trunk instead of
            # leaving rows uninitialised.
            out[rest] = preds(m_i6, X[rest])
        return out

    def verdict_multi(models_dict, X, fam_of_row):
        """Continuous verdict = expected level under the 5-class softmax."""
        out = np.empty(len(X), dtype=np.float64)
        covered = np.zeros(len(X), dtype=bool)
        CH = 20000
        lv = np.asarray(LEVELS, dtype=np.float64)
        for fam_name in sorted(models_dict.keys()):
            m = models_dict[fam_name]
            fam_mask = fam_of_row == fam_name
            if not fam_mask.any():
                continue
            fam_X = X[fam_mask]
            fam_out = np.empty(len(fam_X), dtype=np.float64)
            for i in range(0, len(fam_X), CH):
                p = softmax(fwd_batch(m, fam_X[i:i + CH]))
                fam_out[i:i + CH] = np.clip(p @ lv,
                                            -1.0 + VERDICT_EPS,
                                            1.0 - VERDICT_EPS)
            out[fam_mask] = fam_out
            covered[fam_mask] = True
        rest = ~covered
        if rest.any():
            p = softmax(fwd_batch(m_i6, X[rest]))
            out[rest] = np.clip(p @ lv, -1.0 + VERDICT_EPS, 1.0 - VERDICT_EPS)
        return out

    p_i6_o = preds_multi(expert_models, R_o_i6, fam_of_row)
    p_i6_s = preds_multi(expert_models, R_s_i6, np.array(sfams))
    v_i6_o = verdict_multi(expert_models, R_o_i6, fam_of_row)
    v_i6_s = verdict_multi(expert_models, R_s_i6, np.array(sfams))

    def continuous_predictions(m, X):
        out = np.empty(len(X), dtype=np.float64)
        lv = np.asarray(LEVELS, dtype=np.float64)
        CH = 20000
        for i in range(0, len(X), CH):
            prob = softmax(fwd_batch(m, X[i:i + CH]))
            out[i:i + CH] = np.clip(prob @ lv,
                                    -1.0 + VERDICT_EPS,
                                    1.0 - VERDICT_EPS)
        return out

    def continuous_metrics(verdict, true_arr):
        true_levels = np.asarray(LEVELS, dtype=np.float64)[true_arr]
        delta = np.asarray(verdict, dtype=np.float64) - true_levels
        return float(np.mean(np.abs(delta))), float(np.mean(delta * delta))

    each_expert_predictions = {}
    each_expert_summaries = {}
    if args.each_expert:
        print("\n###### EACH I6 EXPERT ON THE FULL SCENARIO SET ######",
              flush=True)
        for expert_name, model in sorted(expert_models.items()):
            pred_o = preds(model, R_o_i6)
            pred_s = preds(model, R_s_i6)
            verdict_o = continuous_predictions(model, R_o_i6)
            verdict_s = continuous_predictions(model, R_s_i6)
            each_expert_predictions[expert_name] = (
                pred_o, pred_s, verdict_o, verdict_s)
            mae_o, mse_o = continuous_metrics(verdict_o, Yo)
            mae_s, mse_s = continuous_metrics(verdict_s, Ys)
            each_expert_summaries[expert_name] = (
                (pred_o == Yo).mean(), (pred_s == Ys).mean(),
                mae_o, mse_o, mae_s, mse_s)
            print("[expert/%s] overall n=%d exact=%.6f MAE=%.9g MSE=%.9g"
                  % (expert_name, len(Yo), (pred_o == Yo).mean(),
                     mae_o, mse_o), flush=True)
            print("[expert/%s] local-step n=%d exact=%.6f MAE=%.9g MSE=%.9g"
                  % (expert_name, len(Ys), (pred_s == Ys).mean(),
                     mae_s, mse_s), flush=True)

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
        pa_o = preds(am, R_o_i3); pa_s = preds(am, R_s_i3)
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

    def mae_mse_cont(verdict, true_arr):
        true_levels = np.array([LEVELS[int(t)] for t in true_arr],
                               dtype=np.float64)
        d = np.asarray(verdict, dtype=np.float64) - true_levels
        return float(np.mean(np.abs(d))), float(np.mean(d * d))

    i6_mae, i6_mse = mae_mse(p_i6_o, Yo)
    i6_mae_c, i6_mse_c = mae_mse_cont(v_i6_o, Yo)
    i6_mae_step_c, i6_mse_step_c = mae_mse_cont(v_i6_s, Ys)
    print("\n[i6] discrete  MAE(level)=%.9g  MSE(level)=%.9g"
          % (i6_mae, i6_mse), flush=True)
    print("[i6] continuous MAE(level)=%.9g  MSE(level)=%.9g"
          % (i6_mae_c, i6_mse_c), flush=True)
    print("[i6] per-step continuous MAE(level)=%.9g  MSE(level)=%.9g"
          % (i6_mae_step_c, i6_mse_step_c), flush=True)
    target_results = (i6_mae_c < 1e-3, i6_mse_c < 1e-3,
                      i6_mae_step_c < 1e-3, i6_mse_step_c < 1e-3)
    print("Continuous targets overall MAE/MSE and per-step MAE/MSE < 1e-3: "
          "%s/%s %s/%s"
          % ("PASS" if i6_mae_c < 1e-3 else "FAIL",
             "PASS" if i6_mse_c < 1e-3 else "FAIL",
             "PASS" if i6_mae_step_c < 1e-3 else "FAIL",
             "PASS" if i6_mse_step_c < 1e-3 else "FAIL"), flush=True)

    # Keep a compact condition matrix: every family expert, stress tier,
    # algorithm, and target level gets independent continuous MAE and MSE.
    condition_rows = []
    def report_condition(label, mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            print("[i6] condition %-32s EMPTY" % label, flush=True)
            return
        mae, mse = mae_mse_cont(v_i6_o[mask], Yo[mask])
        print("[i6] condition %-32s n=%6d MAE=%.7g MSE=%.7g" %
              (label, int(mask.sum()), mae, mse), flush=True)
        condition_rows.append((label, int(mask.sum()), mae, mse))

    for family in FAMILIES:
        fam_mask = fam_of_row == family
        report_condition("family/" + family, fam_mask)
        for tier in TIERS:
            report_condition("family/%s/tier/%s" % (family, tier),
                             fam_mask & np.array([m[1] == tier for m in ometa]))
    for li, level in enumerate(LEVELS):
        report_condition("level/%s" % level, Yo == li)
    for tier in TIERS:
        report_condition("tier/" + tier,
                         np.array([m[1] == tier for m in ometa]))
    for algo in ALL150:
        report_condition("algorithm/" + algo.name,
                         np.array([m[0] == algo.name for m in ometa]))

    print("\n[i6] per-step condition matrix", flush=True)
    def report_step_condition(label, mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            print("[i6] step-condition %-30s EMPTY" % label, flush=True)
            return
        mae, mse = mae_mse_cont(v_i6_s[mask], Ys[mask])
        print("[i6] step-condition %-30s n=%6d MAE=%.7g MSE=%.7g" %
              (label, int(mask.sum()), mae, mse), flush=True)
        condition_rows.append(("step/" + label, int(mask.sum()), mae, mse))

    step_fams = np.asarray(sfams)
    step_tiers = np.asarray([m[1] for m in smeta])
    step_algos = np.asarray([m[0] for m in smeta])
    for family in FAMILIES:
        fam_mask = step_fams == family
        report_step_condition("family/" + family, fam_mask)
        for tier in TIERS:
            report_step_condition("family/%s/tier/%s" % (family, tier),
                                  fam_mask & (step_tiers == tier))
    for li, level in enumerate(LEVELS):
        report_step_condition("level/%s" % level, Ys == li)
    for tier in TIERS:
        report_step_condition("tier/" + tier, step_tiers == tier)
    for algo in ALL150:
        report_step_condition("algorithm/" + algo.name,
                              step_algos == algo.name)

    conditions_per_row_type = (len(FAMILIES) * (1 + len(TIERS))
                               + len(LEVELS) + len(TIERS) + len(ALL150))
    expected_conditions = 2 * conditions_per_row_type
    condition_coverage_pass = len(condition_rows) == expected_conditions
    condition_targets_pass = all(mae < 1e-3 and mse < 1e-3
                                 for _, _, mae, mse in condition_rows)
    print("Condition targets MAE/MSE < 1e-3: %s; coverage %d/%d %s"
          % ("PASS" if condition_targets_pass else "FAIL",
             len(condition_rows), expected_conditions,
             "PASS" if condition_coverage_pass else "FAIL"), flush=True)

    each_expert_condition_rows = []
    each_expert_condition_counts = {}
    each_expert_targets_pass = True
    each_expert_coverage_pass = True
    if args.each_expert:
        overall_masks = []
        for family in FAMILIES:
            fam_mask = fam_of_row == family
            overall_masks.append(("family/" + family, fam_mask))
            for tier in TIERS:
                overall_masks.append(("family/%s/tier/%s" % (family, tier),
                                      fam_mask & np.array([m[1] == tier for m in ometa])))
        for li, level in enumerate(LEVELS):
            overall_masks.append(("level/%s" % level, Yo == li))
        for tier in TIERS:
            overall_masks.append(("tier/" + tier,
                                  np.array([m[1] == tier for m in ometa])))
        for algo in ALL150:
            overall_masks.append(("algorithm/" + algo.name,
                                  np.array([m[0] == algo.name for m in ometa])))

        step_masks = []
        for family in FAMILIES:
            fam_mask = step_fams == family
            step_masks.append(("family/" + family, fam_mask))
            for tier in TIERS:
                step_masks.append(("family/%s/tier/%s" % (family, tier),
                                   fam_mask & (step_tiers == tier)))
        for li, level in enumerate(LEVELS):
            step_masks.append(("level/%s" % level, Ys == li))
        for tier in TIERS:
            step_masks.append(("tier/" + tier, step_tiers == tier))
        for algo in ALL150:
            step_masks.append(("algorithm/" + algo.name,
                               step_algos == algo.name))

        for expert_name in FAMILY_MODEL_MAP:
            if expert_name not in each_expert_predictions:
                each_expert_condition_counts[expert_name] = 0
                each_expert_targets_pass = False
                each_expert_coverage_pass = False
                continue
            pred_o, pred_s, verdict_o, verdict_s = each_expert_predictions[expert_name]
            count = 0
            for row_type, masks, verdict, labels, predictions in (
                    ("overall", overall_masks, verdict_o, Yo, pred_o),
                    ("local-step", step_masks, verdict_s, Ys, pred_s)):
                for condition, mask in masks:
                    mask = np.asarray(mask, dtype=bool)
                    if not mask.any():
                        print("[expert/%s] condition %s/%s EMPTY"
                              % (expert_name, row_type, condition), flush=True)
                        continue
                    mae, mse = continuous_metrics(verdict[mask], labels[mask])
                    exact = float((predictions[mask] == labels[mask]).mean())
                    full_label = "expert/%s/%s/%s" % (
                        expert_name, row_type, condition)
                    each_expert_condition_rows.append(
                        (full_label, int(mask.sum()), exact, mae, mse))
                    count += 1
                    if mae >= 1e-3 or mse >= 1e-3:
                        each_expert_targets_pass = False
                    print("[expert-condition] %-64s n=%6d exact=%.6f MAE=%.7g MSE=%.7g"
                          % (full_label, int(mask.sum()), exact, mae, mse),
                          flush=True)
            each_expert_condition_counts[expert_name] = count
            if count != expected_conditions:
                each_expert_coverage_pass = False

        for expert_name, metrics in each_expert_summaries.items():
            _, _, mae_o, mse_o, mae_s, mse_s = metrics
            if any(value >= 1e-3 for value in (mae_o, mse_o, mae_s, mse_s)):
                each_expert_targets_pass = False
        each_expert_coverage_pass = (
            each_expert_coverage_pass
            and len(each_expert_summaries) == len(FAMILY_MODEL_MAP))
        print("Each-expert MAE/MSE targets: %s; condition coverage: %s"
              % ("PASS" if each_expert_targets_pass else "FAIL",
                 "PASS" if each_expert_coverage_pass else "FAIL"), flush=True)

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
    if alu_results:
        for algo in ALL150:
            msk = np.array([mm[0] == algo.name for mm in ometa])
            if msk.sum():
                ai6 = (p_i6_o[msk] == Yo[msk]).mean()
                best_al = max((alu_results[a][0][msk] == Yo[msk]).mean()
                              for a in alu_results)
                deltas.append((algo.name, ai6, best_al, msk.sum()))
    else:
        print("  skipped (I3 ALU baselines not loaded)", flush=True)
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
        f.write("i6_MAE %.9g i6_MSE %.9g\n" % (i6_mae, i6_mse))
        f.write("i6_continuous_MAE %.9g i6_continuous_MSE %.9g\n"
                % (i6_mae_c, i6_mse_c))
        f.write("i6_per_step_continuous_MAE %.9g i6_per_step_continuous_MSE %.9g\n"
                % (i6_mae_step_c, i6_mse_step_c))
        f.write("target_MAE_1e-3 %s target_MSE_1e-3 %s\n"
                % ("PASS" if i6_mae < 1e-3 else "FAIL",
                   "PASS" if i6_mse < 1e-3 else "FAIL"))
        f.write("target_continuous_MAE_1e-3 %s target_continuous_MSE_1e-3 %s\n"
                % ("PASS" if i6_mae_c < 1e-3 else "FAIL",
                   "PASS" if i6_mse_c < 1e-3 else "FAIL"))
        f.write("target_per_step_continuous_MAE_1e-3 %s "
                "target_per_step_continuous_MSE_1e-3 %s\n"
                % ("PASS" if i6_mae_step_c < 1e-3 else "FAIL",
                   "PASS" if i6_mse_step_c < 1e-3 else "FAIL"))
        f.write("condition_slices %d/%d metrics_%s coverage_%s\n"
                % (len(condition_rows), expected_conditions,
                   "PASS" if condition_targets_pass else "FAIL",
                   "PASS" if condition_coverage_pass else "FAIL"))
        f.write("condition_matrix label n MAE MSE\n")
        for label, n, mae, mse in condition_rows:
            f.write("condition %s n=%d MAE=%.9g MSE=%.9g\n"
                    % (label, n, mae, mse))
        if args.each_expert:
            f.write("each_expert_summary name overall_exact step_exact "
                    "overall_MAE overall_MSE step_MAE step_MSE\n")
            for expert_name, metrics in sorted(each_expert_summaries.items()):
                acc_o, acc_s, mae_o, mse_o, mae_s, mse_s = metrics
                f.write("each_expert %s overall_exact=%.9g step_exact=%.9g "
                        "overall_MAE=%.9g overall_MSE=%.9g "
                        "step_MAE=%.9g step_MSE=%.9g\n"
                        % (expert_name, acc_o, acc_s, mae_o, mse_o,
                           mae_s, mse_s))
            f.write("each_expert_condition_coverage expected_each=%d "
                    "counts=%s target_gates_%s coverage_%s\n"
                    % (expected_conditions, each_expert_condition_counts,
                       "PASS" if each_expert_targets_pass else "FAIL",
                       "PASS" if each_expert_coverage_pass else "FAIL"))
            f.write("each_expert_condition_matrix label n exact MAE MSE\n")
            for label, n, exact, mae, mse in each_expert_condition_rows:
                f.write("each_expert_condition %s n=%d exact=%.9g "
                        "MAE=%.9g MSE=%.9g\n"
                        % (label, n, exact, mae, mse))
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
    if args.strict_targets and (not all(target_results)
                                or not condition_targets_pass
                                or not condition_coverage_pass
                                or (args.each_expert
                                    and (not each_expert_targets_pass
                                         or not each_expert_coverage_pass))):
        raise SystemExit(3)

if __name__ == "__main__":
    main()

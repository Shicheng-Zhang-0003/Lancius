#!/usr/bin/env python3
"""
RUNTIME TEST HARNESS -- 20 equations/theorems/algorithms x N parameter sets.

Two sections:

  SECTION 1  DETAIL. For the first parameter set of each algorithm, print the
             mathematical description, the algorithmic description, the problem,
             the reference derivation, a per-STEP table (expected / reported /
             rel. error / verdict), the four metrics, and the overall 5-level
             verdict next to the injected ground truth.

  SECTION 2  SWEEP. Run every algorithm over N parameter sets (default 100),
             each at a cycling veracity level, and report agreement, per-level
             accuracy, the confusion matrix, per-algorithm and per-step
             breakdowns, plus every mismatch for debugging.

Edge cases are deliberately NOT targeted here; this measures ordinary
parameter variation. Explicit edge-case training is a separate exercise.

USAGE
    python3 tools/prm/run_tests.py                     # 100 params/algorithm
    python3 tools/prm/run_tests.py --params 20         # quicker pass
    python3 tools/prm/run_tests.py --clean
    python3 tools/prm/run_tests.py --defect -0.5
    python3 tools/prm/run_tests.py --only power_rule
    python3 tools/prm/run_tests.py --no-detail          # sweep only
    python3 tools/prm/run_tests.py --seed 7
"""
import sys, os, argparse, math, time
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algos import ALGORITHMS, _fmt
import verifier5 as V

LV = {-1.0: "-1.0", -0.5: "-0.5", 0.0: " 0.0", 0.5: "+0.5", 1.0: "+1.0"}
MEAN = {-1.0: "plain false",
        -0.5: "probably false, something right in there",
        0.0: "50/50 / undecidable",
        0.5: "mostly true but something wrong in there",
        1.0: "plain true"}
LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
MAGNITUDE_CAP = 10 ** 7      # skip absurd instances; edge cases are out of scope

# one illustrative parameter set per algorithm, all different on purpose
TESTS = [
    ("linear_solve",           {"a": F(7), "b": F(-5), "x": F(-9), "c": F(-68)}),
    ("quadratic_formula",      {"a": F(1), "b": F(-5), "c": F(6), "r1": F(3), "r2": F(2)}),
    ("binomial_expand",        {"a": F(2), "b": F(3), "n": 4}),
    ("difference_of_squares",  {"a": F(12), "b": F(5)}),
    ("perfect_square_trinomial", {"a": F(7), "b": F(4)}),
    ("sum_of_cubes",           {"a": F(5), "b": F(3)}),
    ("geometric_series",       {"a": F(3), "r": F(2), "n": 6}),
    ("arithmetic_series",      {"a": F(4), "d": F(3), "n": 8}),
    ("slope_from_points",      {"x1": F(2), "y1": F(3), "x2": F(6), "y2": F(11)}),
    ("substitution_system",    {"S": F(14), "D": F(6)}),
    ("power_rule",             {"a": F(7), "n": F(4)}),
    ("chain_rule",             {"a": F(6), "n": F(5), "b": F(9)}),
    ("product_rule",           {"a": F(3), "m": F(4), "b": F(5), "n": F(2)}),
    ("quotient_rule",          {"a": F(4), "b": F(3), "c": F(5), "d": F(2)}),
    ("definite_integral",      {"a": F(3), "n": F(2), "lo": F(-1), "hi": F(4)}),
    ("gauss_2x2",              {"a": F(2), "b": F(1), "c": F(3), "d": F(4),
                               "x1": F(5), "x2": F(-1), "e": F(9), "f": F(11)}),
    ("dot_product_3d",         {"a1": F(2), "a2": F(-3), "a3": F(4),
                               "b1": F(5), "b2": F(6), "b3": F(-1)}),
    ("matrix_det_2x2",         {"a": F(5), "b": F(3), "c": F(2), "d": F(7)}),
    ("euclid_gcd",             {"x": F(1071), "y": F(462)}),
    ("mod_pow_fast",           {"base": F(5), "exp": 13, "mod": 11}),
]


# ---------------------------------------------------------------- model ----
def load_model(path):
    M = {k: [] for k in ("MU", "SD", "W1", "B1", "W2", "B2")}
    H = FEAT = NC = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition(" ")
        if k == "H": H = int(v)
        elif k == "FEAT": FEAT = int(v)
        elif k == "NCLASS": NC = int(v)
        elif k in M: M[k].append(float(v))
    return dict(H=H, FEAT=FEAT, NC=NC, **{k.lower(): v for k, v in M.items()})


def _tanh(x):
    if x > 30: return 1.0
    if x < -30: return -1.0
    e = math.exp(2 * x)
    return (e - 1) / (e + 1)


class Net:
    """Batched-ish forward. Rows are scored one at a time; the matrices are
    flattened once so the inner loop is a flat dot product."""
    def __init__(self, d):
        self.H, self.FEAT, self.NC = d["H"], d["FEAT"], d["NC"]
        self.W1, self.b1 = d["w1"], d["b1"]
        self.W2, self.b2 = d["w2"], d["b2"]
        self.mu, self.sd = d["mu"], d["sd"]

    def _logits(self, f):
        H, FEAT, NC = self.H, self.FEAT, self.NC
        mu, sd, W1, W2 = self.mu, self.sd, self.W1, self.W2
        xs = [(f[k] - mu[k]) / sd[k] for k in range(FEAT)]
        hid = [0.0] * H
        for j in range(H):
            s = self.b1[j]; base = j
            for k in range(FEAT):
                s += xs[k] * W1[base + k * H]
            hid[j] = _tanh(s)
        out = [0.0] * NC
        for c in range(NC):
            s = self.b2[c]
            for j in range(H):
                s += hid[j] * W2[j * NC + c]
            out[c] = s
        return out

    def level(self, f):
        z = self._logits(f)
        p = 0
        for c in range(1, self.NC):
            if z[c] > z[p]: p = c
        return V.LEVELS[p]

    def verdict(self, algo, chk, step_idx):
        return self.level(V.row_features(algo, chk, step_idx))


# ------------------------------------------------------------- utilities ---
def sane(params, steps):
    """Reject absurd instances. Edge cases are explicitly out of scope for this
    sweep, so we only guard against values that make exact arithmetic or the
    relative-error metric meaningless."""
    for v in params.values():
        try:
            if abs(v) > MAGNITUDE_CAP: return False
        except Exception:
            return False
    for s in steps:
        try:
            if abs(s["value"]) > MAGNITUDE_CAP: return False
        except Exception:
            return False
    return True


def show_detail(net, algo, params, steps, rep, struct, truth, kind):
    chk = V.check(algo, params, steps, rep, struct)
    W = 78
    print("=" * W)
    print(f"TEST  {algo.name}")
    print("=" * W)
    print(f"  MATH : {algo.math_desc}")
    print(f"  ALGO : {algo.algo_desc}")
    print("  PARAMS: " + ", ".join(f"{k}={_fmt(params[k])}"
                                  for k in algo.structure_keys))
    print(f"  CANDIDATE: {kind}   GROUND TRUTH {LV[truth]} ({MEAN[truth]})")
    print()
    print(f"  {'#':>2}  {'step':<16} {'expected':>12} {'reported':>12} "
          f"{'rel.err':>9}  ok  verdict")
    print("  " + "-" * (W - 4))
    agree = 0
    for i, s in enumerate(steps):
        p = chk["per_step"][i]
        got = rep[i]
        if not p["missing"] and p["exact"]: agree += 1
        if p["missing"] or got is None:
            print(f"  {i:>2}  {s['name']:<16} {_fmt(s['value']):>12} "
                  f"{'(omitted)':>12} {'-':>9}   ?  "
                  f"{LV[net.verdict(algo, chk, i)]} (undecidable)")
            continue
        print(f"  {i:>2}  {s['name']:<16} {_fmt(s['value']):>12} "
              f"{_fmt(got):>12} {p['rel']:>9.4f}  "
              f"{'OK' if p['exact'] else 'XX'}  {LV[net.verdict(algo, chk, i)]}")
    print("  " + "-" * (W - 4))
    fin = net.verdict(algo, chk, -1)
    print()
    print("  METRICS")
    print(f"    structure          : {'MATCH' if chk['struct_ok'] else 'MISMATCH'}")
    print(f"    coefficients       : mean relative error = {chk['coef_err']:.4f}")
    print(f"    per-step agreement : {agree}/{len(steps)} exact "
          f"({agree/max(1,len(steps))*100:.0f}%)")
    print(f"    final result       : expected {_fmt(steps[-1]['value'])}, "
          f"reported {'(omitted)' if rep[-1] is None else _fmt(rep[-1])} -> "
          f"{'EXACT' if chk['final_exact'] else 'WRONG'}")
    ok = "MATCH" if abs(fin - truth) < 1e-9 else "*** WRONG VERDICT ***"
    print(f"    OVERALL VERDICT    : {LV[fin]}  ({MEAN[fin]})")
    print(f"                         ground truth {LV[truth]}  ({MEAN[truth]})  {ok}")
    print()
    return chk, agree, fin


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data_vec/v5/model5.txt")
    ap.add_argument("--params", type=int, default=100,
                    help="parameter sets per algorithm")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--defect", type=float, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("missing %s -- train it first:\n"
                 "  python3 tools/prm/verifier5.py data_vec/v5 1600\n"
                 "  ./temp/train_v5 data_vec/v5 96 60 128" % args.model)

    net = Net(load_model(args.model))
    print("network: %s   %d -> %d -> %d   scale {-1,-.5,0,+.5,+1}"
          % (args.model, net.FEAT, net.H, net.NC))
    print("sweep   : %d algorithms x %d parameter sets"
          % (len(TESTS), args.params))
    print("note    : edge cases are out of scope for this sweep\n")

    import random
    by = {a.name: a for a in ALGORITHMS}
    t0 = time.time()

    # ---------------- SECTION 1: detail on the first parameter set --------
    if not args.no_detail and not args.clean and args.defect is None:
        print("#" * 78)
        print("# SECTION 1 -- DETAIL, first parameter set of each algorithm")
        print("#" * 78)
        for name, params in TESTS:
            if args.only and args.only != name: continue
            algo = by[name]
            steps = algo._steps(params)
            rep, struct, kind = V.make_candidate(algo, params, steps, random.Random(9), 1.0)
            show_detail(net, algo, params, steps, rep, struct, 1.0, kind)

    # ---------------- SECTION 2: the sweep -------------------------------
    print("#" * 78)
    print(f"# SECTION 2 -- SWEEP: {len(TESTS)} algorithms x {args.params} "
          f"parameter sets")
    print("#" * 78)

    cm = [[0] * 5 for _ in range(5)]
    per_algo = {n: [0, 0, 0] for n, _ in TESTS}     # ok, total, exact_steps
    per_algo_steps = {n: [0, 0] for n, _ in TESTS}  # step ok, step total
    per_level = {lv: [0, 0] for lv in LEVELS}
    mismatches = []
    skipped = 0
    total = 0

    for ai, (name, base_params) in enumerate(TESTS):
        if args.only and args.only != name: continue
        algo = by[name]
        prng = random.Random(args.seed * 1000 + ai)
        made = 0
        tries = 0
        while made < args.params and tries < args.params * 12:
            tries += 1
            if made == 0 and not args.no_detail and args.clean is False \
               and args.defect is None:
                params = base_params          # section 1 already used this one
            else:
                try:
                    params, _ = algo.build(prng)
                except Exception:
                    skipped += 1
                    continue
            try:
                steps = algo._steps(params)
            except Exception:
                skipped += 1
                continue
            if not sane(params, steps):
                skipped += 1
                continue

            if args.clean:
                lv = 1.0
            elif args.defect is not None:
                lv = args.defect
            else:
                lv = LEVELS[made % 5]

            rep, struct, kind = V.make_candidate(algo, params, steps, prng, lv)
            chk = V.check(algo, params, steps, rep, struct)
            fin = net.verdict(algo, chk, -1)
            li = LEVELS.index(lv); fi = LEVELS.index(fin)
            cm[li][fi] += 1
            total += 1
            per_algo[name][1] += 1
            per_algo_steps[name][1] += len(steps)
            agree = 0
            for i in range(len(steps)):
                p = chk["per_step"][i]
                if not p["missing"] and p["exact"]: agree += 1
                sv = net.verdict(algo, chk, i)
                sl = V.step_label(lv, kind, chk, i)
                per_algo_steps[name][0] += (1 if abs(sv - V.LEVELS[sl]) < 1e-9 else 0)
            per_algo[name][2] += agree
            per_level[lv][1] += 1
            ok = abs(fin - lv) < 1e-9
            if ok:
                per_algo[name][0] += 1
                per_level[lv][0] += 1
            elif len(mismatches) < 40:
                mismatches.append((name, lv, fin, kind, agree, len(steps),
                                   {k: _fmt(params[k]) for k in algo.structure_keys}))
            made += 1

    # ---------------- report ----------------------------------------------
    n = total
    okc = sum(v[0] for v in per_algo.values())
    print()
    print(f"overall verdict correct : {okc}/{n}  ({okc/max(1,n)*100:.2f}%)")
    stepok = sum(v[0] for v in per_algo_steps.values())
    stepn = sum(v[1] for v in per_algo_steps.values())
    print(f"per-step verdict correct: {stepok}/{stepn}  "
          f"({stepok/max(1,stepn)*100:.2f}%)")
    if skipped:
        print(f"skipped instances        : {skipped} (magnitude cap / build failure)")
    print(f"elapsed                  : {time.time()-t0:.1f}s")

    print("\nper-level accuracy")
    print(f"  {'level':>6} {'meaning':<38} {'acc':>8}")
    for lv in LEVELS:
        o, t = per_level[lv]
        print(f"  {LV[lv]:>6} {MEAN[lv]:<38} {o/max(1,t):>8.4f}  ({o}/{t})")

    print("\nper-algorithm overall accuracy")
    print(f"  {'algorithm':<24} {'acc':>8} {'ok/total':>12} {'steps exact':>14}")
    for name, _ in TESTS:
        if args.only and args.only != name: continue
        ok, t, ex = per_algo[name]
        ns = per_algo_steps[name][1]
        print(f"  {name:<24} {ok/max(1,t):>8.4f} {str(ok)+'/'+str(t):>12} "
              f"{str(ex)+'/'+str(ns):>14}")

    print("\nconfusion (rows = ground truth, cols = predicted)")
    print("        " + "".join(f"{LV[l]:>8}" for l in LEVELS))
    for i, lv in enumerate(LEVELS):
        print(f"  {LV[lv]:>5} " + "".join(f"{cm[i][j]:>8}" for j in range(5)))

    if mismatches:
        print(f"\nfirst {len(mismatches)} verdict mismatches")
        for name, lv, fin, kind, agree, ns, pr in mismatches:
            print(f"  {name:<24} truth {LV[lv]}  said {LV[fin]}  "
                  f"defect={kind}  steps {agree}/{ns}  {pr}")


if __name__ == "__main__":
    main()

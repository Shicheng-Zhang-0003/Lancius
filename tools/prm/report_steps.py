#!/usr/bin/env python3
"""
Per-step veracity report.

Given a known algorithm and a candidate answer, print:
  - the problem
  - the canonical derivation the algorithm requires (the reference)
  - for EVERY step: expected value, reported value, exact/relative error, verdict
  - the final result compared against the reference
  - the overall 5-level verdict from the trained network
  - the four metrics: structure, coefficients/constants, per-step agreement, final

Uses the trained network in data_vec/v5/model5.txt. The network supplies
veracity; the symbolic checker supplies the measured values and errors.

    python3 tools/prm/report_steps.py                # demo cases
    python3 tools/prm/report_steps.py --algo power_rule --a 7 --n 4
"""
import sys, os, math
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algos import ALGORITHMS, _fmt
import verifier5 as V

LEVEL = {-1.0: "-1.0", -0.5: "-0.5", 0.0: " 0.0", 0.5: "+0.5", 1.0: "+1.0"}
MEAN = {-1.0: "plain false",
        -0.5: "probably false, something right",
        0.0: "50/50 / undecidable",
        0.5: "mostly true, something wrong",
        1.0: "plain true"}


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


def fwd(m, x):
    H, FEAT, NC = m["H"], m["FEAT"], m["NC"]
    h = []
    for j in range(H):
        s = m["b1"][j]
        for k in range(FEAT):
            s += x[k] * m["w1"][k * H + j]
        h.append(math.tanh(s))
    out = []
    for c in range(NC):
        s = m["b2"][c]
        for j in range(H):
            s += h[j] * m["w2"][j * NC + c]
        out.append(s)
    return out


def verdict(m, algo, chk, step_idx):
    f = V.row_features(algo, chk, step_idx)
    z = fwd(m, [(f[i] - m["mu"][i]) / m["sd"][i] for i in range(m["FEAT"])])
    p = max(range(len(z)), key=lambda c: z[c])
    return V.LEVELS[p]


def report(m, algo, params, steps, rep, struct, title):
    chk = V.check(algo, params, steps, rep, struct)
    w = 78
    print("=" * w)
    print(title)
    print("=" * w)
    print("problem  : %s   params %s" % (
        algo.name, ", ".join(f"{k}={_fmt(v)}" for k, v in params.items()
                             if k in algo.structure_keys)))
    print("reference: the derivation '%s' requires" % algo.name)
    print()
    print(f"  {'#':>2}  {'step':<14} {'expected':>12} {'reported':>12} "
          f"{'rel.err':>9}  {'ok':>3}  verdict")
    print("  " + "-" * (w - 4))
    for i, s in enumerate(steps):
        got = rep[i]
        p = chk["per_step"][i]
        if p["missing"] or got is None:
            print(f"  {i:>2}  {s['name']:<14} {_fmt(s['value']):>12} "
                  f"{'(omitted)':>12} {'-':>9}  {'?':>3}  "
                  f"{LEVEL[verdict(m, algo, chk, i)]}  (undecidable)")
            continue
        v = verdict(m, algo, chk, i)
        print(f"  {i:>2}  {s['name']:<14} {_fmt(s['value']):>12} "
              f"{_fmt(got):>12} {p['rel']:>9.4f}  "
              f"{'OK' if p['exact'] else 'XX':>3}  {LEVEL[v]}")
    print("  " + "-" * (w - 4))

    fin = verdict(m, algo, chk, -1)
    print()
    print("METRICS")
    coef = [k for k in algo.structure_keys if k not in ("c", "e", "f")]
    print(f"  structure        : {'MATCH' if chk['struct_ok'] else 'MISMATCH'}"
          f"   (mean |delta| = {chk['struct_rel']:.4f})")
    print(f"  coefficients     : " + ", ".join(
        f"{k}: reported {_fmt(struct[k])} vs {_fmt(params[k])}"
        for k in coef))
    print(f"                     mean relative error = {chk['coef_err']:.4f}")
    print(f"  per-step agreement: {chk['frac_exact']*100:.1f}% of "
          f"{chk['n_steps']} steps exact  "
          f"(mean rel err {chk['mean_rel']:.4f}, worst {chk['max_rel']:.4f})")
    print(f"  final result     : expected {_fmt(steps[-1]['value'])}, "
          f"reported "
          f"{'(omitted)' if rep[-1] is None else _fmt(rep[-1])}"
          f"   -> {'EXACT' if chk['final_exact'] else 'WRONG'}"
          f" (rel err {chk['final_rel']:.4f})")
    print()
    print(f"  OVERALL VERDICT  : {LEVEL[fin]}   ({MEAN[fin]})")
    print("=" * w)
    print()
    return chk


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else "data_vec/v5/model5.txt"
    m = load_model(model_path)
    print("network: %s  (%d -> %d -> %d)\n" % (model_path, m["FEAT"], m["H"], m["NC"]))

    by_name = {a.name: a for a in ALGORITHMS}
    import random
    rng = random.Random(4)
    demos = [
        ("gauss_2x2", "+1.0  plain true", 1.0),
        ("power_rule", "+0.5  right answer, sloppy working", 0.5),
        ("quadratic_formula", "0.0  undecidable (truncated)", 0.0),
        ("linear_solve", "-0.5  right method, wrong answer", -0.5),
        ("binomial_expand", "-1.0  wrong method and wrong answer", -1.0),
    ]
    for name, title, lvl in demos:
        algo = by_name[name]
        for _ in range(40):
            try:
                params, steps = algo.build(rng)
                break
            except Exception:
                continue
        else:
            continue
        rep, struct, kind = V.make_candidate(algo, params, steps, rng, lvl)
        report(m, algo, params, steps, rep, struct,
               f"DEMO  algorithm={algo.name}   ground truth {title}   "
               f"(injected defect: {kind})")


if __name__ == "__main__":
    main()

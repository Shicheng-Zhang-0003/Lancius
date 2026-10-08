#!/usr/bin/env python3
"""
5-level veracity verifier: defect injection, symbolic checking, features.

THE SCALE (extended boolean, ordinal -1 < -0.5 < 0 < +0.5 < +1)
    +1.0  plain true          every checked property correct
    +0.5  mostly true, one thing wrong   no fatal defect, >=1 minor defect
     0.0  50/50              genuinely undecidable by construction
    -0.5  probably false, something right  fatal defect present, >=1 correct property
    -1.0  plain false        fatal defect and nothing correct

LABEL PROVENANCE -- important and deliberate:
    The ground-truth label comes from the injected defect TYPE, decided at
    generation time. It is never re-derived by calling the checker. Otherwise
    label and features would both be functions of the checker and the network
    would only be learning to re-encode the checker's own output, which would
    look like success while validating nothing. The checker supplies
    MEASUREMENTS (features); the defect supplies TRUTH (label).

FEATURES are therefore measurements of a candidate against the known algorithm:
per-step exactness, per-step relative error, sign agreement, structural match,
coefficient error, final-result agreement, decidability.

Output: data_vec/v5/{train,test}.X.bin, .T.bin, .S.bin, .L.bin, .meta
    X  float64 feature rows
    T  int32   overall level index 0..4  (=-1,-0.5,0,0.5,1)
    S  int32   step-level index for each row (0 = overall row, else step+1)
    L  int32   level index for THAT row
"""
import os, sys, json, random
import struct as _bin   # aliased: `struct` is used as a variable name below
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algos import ALGORITHMS, _fmt

LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
NLEV = 5
MAX_STEPS = 12
FAMILIES = ["algebra", "calculus", "linear_algebra", "number_theory"]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}


# ---------------------------------------------------------------------------
# defect injection: build a candidate answer for a target veracity level
# ---------------------------------------------------------------------------

def _bump(x, rng, rel=None):
    """Perturb a rational, GUARANTEED to change it.

    An earlier version computed x + int(x*scale), which is a no-op whenever
    |x*scale| < 1 -- so any answer of magnitude below ~3 was never actually
    corrupted. That silently relabelled whole -0.5 and -1 samples as +1 and was
    a direct source of the model calling "probably false" "plain true". Enforce
    a minimum absolute change."""
    if x == 0:
        return F(rng.choice([1, -1, 2, 3]), 1)
    # minimum RELATIVE change of 5%. Without a floor, "a step is slightly off"
    # ranged over a continuum from 1% to 30% and the +0.5/+1 boundary became an
    # arbitrary judgement call rather than a defined one, which capped +0.5
    # recall. The scale is now separated by a stated, checkable margin.
    rel = F(rng.randint(5, 25), 100)
    delta = F(max(1, int(float(abs(x)) * float(rel))), 1)
    if delta == 0:
        delta = F(1, 1)
    if rng.random() < 0.5:
        delta = -delta
    y = x + delta
    if y == x:                      # belt and braces
        y = x + F(1, 1)
    return y


def make_candidate(algo, params, steps, rng, level):
    """Return (reported_steps, reported_struct, defect_kind).

    reported_steps[i] is the value the candidate CLAIMS at canonical step i, or
    None if the step is omitted (undecidable). reported_struct is a dict of the
    structural fields the candidate asserts."""
    kind = "none"
    rep = [F(s["value"]) for s in steps]
    struct = {k: params[k] for k in algo.structure_keys}

    # The five levels are defined DISJOINTLY on (structure, final answer):
    #
    #   +1.0  structure ok, final ok, every step exact
    #   +0.5  structure ok, final ok, but a working step is off
    #          -> "mostly true but something in there wrong" (right answer, sloppy work)
    #    0.0  a step is missing: the derivation cannot be judged either way
    #   -0.5  structure ok, but the FINAL ANSWER is wrong
    #          -> "probably false but something right in there" (right method, wrong result)
    #   -1.0  structure wrong AND final wrong
    #          -> "just plain false" (wrong method, wrong result)
    #
    # An earlier version produced -0.5 by corrupting an intermediate step while
    # leaving structure and final intact, which is indistinguishable from +0.5 by
    # construction -- the model then sent 74% of true -0.5 to +1, i.e. it called
    # "probably false" "plain true". Disjointness is what makes the middle of the
    # scale learnable.

    if level == 1.0:
        kind = "none"

    elif level == 0.5:
        kind = "minor_step"
        # perturb a WORKING step, never the final one
        last = len(rep) - 1
        if last >= 1:
            k = rng.randrange(0, last)
        else:
            k = 0
        rep[k] = _bump(rep[k], rng, None)

    elif level == 0.0:
        kind = "undecidable"
        cut = rng.randint(1, max(1, len(steps) - 1))
        for i in range(cut, len(steps)):
            rep[i] = None

    elif level == -0.5:
        kind = "wrong_answer"
        # method right, result wrong: corrupt ONLY the final value
        rep[-1] = _bump(rep[-1], rng, None)

    elif level == -1.0:
        kind = "fatal"
        for k in algo.structure_keys:
            struct[k] = struct[k] + F(rng.choice([3, -3, 7]))
        for i in range(len(rep)):
            rep[i] = _bump(rep[i], rng, None)

    return rep, struct, kind


# ---------------------------------------------------------------------------
# the checker: measure a candidate against the known algorithm
# ---------------------------------------------------------------------------

def check(algo, params, steps, rep, struct):
    """Return measurements. No labels here -- this is instrumentation only."""
    n = len(steps)
    per_step = []
    for i, s in enumerate(steps):
        exp = F(s["value"])
        got = rep[i] if i < len(rep) else None
        if got is None:
            per_step.append({"missing": True, "exact": 0, "rel": 1.0,
                             "logrel": 0.0, "sign": 0.0})
            continue
        exact = 1 if F(got) == exp else 0
        denom = abs(exp) if exp != 0 else F(1)
        rel = float(abs(F(got) - exp) / denom)
        rel = min(rel, 1e6)
        sign = 1.0 if (F(got) >= 0) == (exp >= 0) else -1.0
        per_step.append({"missing": False, "exact": exact, "rel": rel,
                         "logrel": _log1p(rel), "sign": sign})

    # structural agreement
    struct_ok = 1
    struct_diff = []
    for k in algo.structure_keys:
        want = params[k]
        got = struct.get(k)
        if got is None or F(got) != want:
            struct_ok = 0
            d = float(abs(F(got) - want)) if got is not None else 1.0
            struct_diff.append(min(d, 1e6))
    struct_rel = (sum(struct_diff) / len(struct_diff)) if struct_diff else 0.0

    # coefficient accuracy: the parameters that define the equation itself
    coef_keys = [k for k in algo.structure_keys if k not in ("c", "e", "f")]
    coef_errs = []
    for k in coef_keys:
        want = params[k]
        got = struct.get(k)
        if got is None:
            coef_errs.append(1.0); continue
        denom = abs(want) if want != 0 else F(1)
        coef_errs.append(min(float(abs(F(got) - want) / denom), 1e6))

    # final result
    final_exp = F(steps[-1]["value"])
    final_got = rep[-1] if rep and rep[-1] is not None else None
    if final_got is None:
        final_exact, final_rel = 0, 1.0
    else:
        final_exact = 1 if F(final_got) == final_exp else 0
        denom = abs(final_exp) if final_exp != 0 else F(1)
        final_rel = min(float(abs(F(final_got) - final_exp) / denom), 1e6)

    present = [p for p in per_step if not p["missing"]]
    frac_exact = sum(p["exact"] for p in present) / n if n else 0.0
    n_missing = sum(1 for p in per_step if p["missing"])
    undecidable = 1 if n_missing > 0 else 0
    max_rel = max([p["rel"] for p in present], default=1.0)
    mean_rel = (sum(p["rel"] for p in present) / len(present)) if present else 1.0

    return {
        "per_step": per_step, "n_steps": n,
        "struct_ok": struct_ok, "struct_rel": min(struct_rel, 1e6),
        "coef_err": (sum(coef_errs) / len(coef_errs)) if coef_errs else 0.0,
        "final_exact": final_exact, "final_rel": final_rel,
        "frac_exact": frac_exact, "undecidable": undecidable,
        "max_rel": max_rel, "mean_rel": mean_rel,
        "n_missing": n_missing,
    }


def _log1p(x):
    import math
    return math.log1p(min(x, 1e12))


# ---------------------------------------------------------------------------
# feature layout (fixed length)
# ---------------------------------------------------------------------------

FEAT_DIM = 4 * MAX_STEPS + 17


def row_features(algo, chk, step_idx):
    """step_idx = -1 for the overall row, else the canonical step index."""
    f = [0.0] * FEAT_DIM
    o = 0
    for k in range(MAX_STEPS):
        if k < len(chk["per_step"]):
            p = chk["per_step"][k]
            f[o + 0] = float(p["exact"]); f[o + 1] = p["rel"]
            f[o + 2] = p["logrel"];    f[o + 3] = p["sign"]
        o += 4
    # step-local view: the exactness of the step being judged, plus its index.
    # Without these the per-step rows were indistinguishable from each other
    # (they all carried the same padded vector) and the net collapsed to +1.
    if 0 <= step_idx < len(chk["per_step"]):
        q = chk["per_step"][step_idx]
        loc_exact, loc_rel, loc_logrel, loc_sign = q["exact"], q["rel"], q["logrel"], q["sign"]
    else:
        loc_exact = 1.0; loc_rel = 0.0; loc_logrel = 0.0; loc_sign = 0.0
    g = [
        float(chk["n_steps"]),
        chk["frac_exact"],
        float(chk["struct_ok"]),
        chk["struct_rel"],
        chk["coef_err"],
        float(chk["final_exact"]),
        chk["final_rel"],
        float(chk["undecidable"]),
        chk["max_rel"],
        chk["mean_rel"],
        float(FAM_ID.get(algo.family, 0)),
        1.0 if step_idx < 0 else 0.0,
        float(step_idx),
        loc_exact, loc_rel, loc_logrel, loc_sign,
    ]
    f[o:o + len(g)] = g
    return f


def step_label(level, kind, chk, step_idx):
    """Veracity of ONE step, or of the whole derivation when step_idx < 0."""
    if step_idx < 0:
        return int(round((level + 1.0) * 2))      # overall -> index 0..4

    p = chk["per_step"][step_idx] if step_idx < len(chk["per_step"]) else None
    if p is None or p["missing"]:
        return 2                                   # 0.0: not judgeable

    ok = p["exact"] == 1
    if level == 1.0:
        return 4 if ok else 0                      # every step right
    if level == 0.5:
        return 4 if ok else 3                      # 3 = +0.5, the sloppy step
    if level == 0.0:
        return 2                                   # incomplete derivation
    if level == -0.5:
        return 1 if not ok else 4                  # the bad final step is -0.5
    return 0                                       # -1.0: everything wrong


# ---------------------------------------------------------------------------
# dataset generation
# ---------------------------------------------------------------------------

def generate(n_per_level, seed, out_dir):
    rng = random.Random(seed)
    X, T, S, L, A = [], [], [], [], []
    per_level = {lv: 0 for lv in LEVELS}

    for _ in range(n_per_level):
        for level in LEVELS:
            algo = rng.choice(ALGORITHMS)
            for _try in range(20):
                try:
                    params, steps = algo.build(rng)
                    break
                except Exception:
                    continue
            else:
                continue
            rep, struct, kind = make_candidate(algo, params, steps, rng, level)
            chk = check(algo, params, steps, rep, struct)
            ai = ALGORITHMS.index(algo)
            X.append(row_features(algo, chk, -1)); T.append(int(round((level + 1) * 2)))
            S.append(0); L.append(int(round((level + 1) * 2))); A.append(ai)
            for i in range(len(steps)):
                X.append(row_features(algo, chk, i)); T.append(-1)
                S.append(i + 1); L.append(step_label(level, kind, chk, i)); A.append(ai)
            per_level[level] += 1

    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    def dump(name, ids):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.X.bin"), "wb") as f:
            for i in ids:
                f.write(_bin.pack("<%dd" % FEAT_DIM, *X[i]))
        for suf, arr in (("T", T), ("S", S), ("L", L), ("A", A)):
            with open(os.path.join(out_dir, f"{name}.{suf}.bin"), "wb") as f:
                for i in ids: f.write(_bin.pack("<i", arr[i]))
    dump("train", idx[:cut])
    dump("test",  idx[cut:])
    with open(os.path.join(out_dir, "algos.txt"), "w") as f:
        for a in ALGORITHMS:
            f.write(a.name + "\n")
    meta = {
        "n_algorithms": len(ALGORITHMS),
        "algorithm_names": [a.name for a in ALGORITHMS],
        "feat_dim": FEAT_DIM, "n_levels": NLEV, "max_steps": MAX_STEPS,
        "levels": LEVELS, "n_rows": len(X),
        "n_train": cut, "n_test": len(X) - cut,
        "rows_per_level": {str(k): v for k, v in per_level.items()},
        "algorithms": [a.name for a in ALGORITHMS],
        "families": FAMILIES,
        "feature_layout": "4 per padded step (exact,rel,logrel,sign) x12, then 12 global, then step-local index+4. Raw error magnitudes are kept deliberately: log-transforming them scored 0.9320 and clipping at 10 scored 0.9346, both worse than the 0.9377 baseline, because the SIZE of the error is what separates +0.5 from -1.0.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data_vec/v5"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261007
    m = generate(n, seed, out)
    print(json.dumps(m, indent=2))

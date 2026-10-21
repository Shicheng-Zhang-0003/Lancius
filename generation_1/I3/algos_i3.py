#!/usr/bin/env python3
"""I3 extension pack: 50 more algorithms (150 total with 20+12+68).

Same contract as tools/prm/algos.py / algos_gen1_ext.py / algos_gen1_big.py:
exact Fractions, integer-friendly, <=12 canonical steps, the four existing
families (the 65-dim feature vector encodes family as FAM_ID, so new families
are not representable), valid-by-construction builds with resample guards.

The coverage rule behind this set: the existing 100 dominate school algebra,
derivative/integral mechanics and modular arithmetic. I3-00 must instead show
measurable competence on families that DID NOT EXIST in I1/I2 -- rational
trig geometry (Pythagorean settings), Newton/Euler/Simpson numerics,
statistics and probability, 3x3 linear algebra, base/bit arithmetic,
combinatorics. Every algorithm is exactly executable: a candidate is checked
against Fraction ground truth, never against a float tolerance.

Structure/scale guards matter here as much as the math: several of these
prime- or range-dependent families raise ValueError when a scaled regime
breaks their contract, so the harness retries rather than shipping a lied
"exact" row (the sigma2 semantics would silently rot if a scaled prime
stopped being prime and the closed form kept being applied).

Exposes EXT3_ALGOS. Full set = ALGORITHMS (20) + EXT_ALGOS (12)
+ EXT2_ALGOS (68) + EXT3 (50) = 150.
"""
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "prm"))
from algos import Algorithm, _isqrt, _sqrtq  # noqa: F401


def _mk(*a):
    return Algorithm(*a)


def _nz(rng, lo=-9, hi=9):
    v = F(0)
    while v == 0:
        v = F(rng.randint(lo, hi))
    return v


EXT3_ALGOS = []

# ================= ALGEBRA (14) ============================


def _b_amgm(rng):
    a = F(rng.randint(1, 6))
    return {"a": a}


def _s_amgm(p):
    a = p["a"]
    if a <= 0:
        raise ValueError("AM-GM needs a positive pair")
    b = 4 * a
    t1 = a + b
    am = t1 / 2
    gm = 2 * a
    return [{"name": "pair", "expr": "(a, 4a)", "value": a},
            {"name": "sum", "expr": "a+4a", "value": t1},
            {"name": "AM", "expr": "(a+b)/2", "value": am},
            {"name": "GM", "expr": "sqrt(a*b)", "value": gm},
            {"name": "gap", "expr": "AM-GM", "value": am - gm}]


_add = None
EXT3_ALGOS.append(_mk("am_gm_two", "algebra",
    "AM-GM on the pair (a, 4a): (a+4a)/2 >= sqrt(4a^2) = 2a, gap a >= 0.",
    "build GM as an exact square root by construction, then compare to AM",
    _b_amgm, _s_amgm, ["a"]))


def _b_abslin(rng):
    a = F(rng.randint(2, 9))
    t = F(rng.randint(-4, 4))
    x0 = F(rng.randint(-9, 9))
    b = a * t
    c = a * x0 + b
    if c <= 0:
        raise ValueError("no solutions without c>0")
    return {"a": a, "b": b, "c": c, "t": t}


def _s_abslin(p):
    a, b, c = p["a"], p["b"], p["c"]
    if a == 0 or c == 0:
        raise ValueError("no solution")
    x1 = (c - b) / a
    x2 = (-c - b) / a
    return [{"name": "inside", "expr": "|ax+b|=c", "value": c},
            {"name": "branch_plus", "expr": "x=(c-b)/a", "value": x1},
            {"name": "branch_minus", "expr": "x=(-c-b)/a", "value": x2},
            {"name": "solution_count", "expr": "2 (c>0, a!=0)", "value": F(2)}]


EXT3_ALGOS.append(_mk("abs_linear", "algebra",
    "abs(a*x + b) = c has exactly the two solutions x=(c-b)/a and x=(-c-b)/a for c>0, a!=0.",
    "solve both branches of the absolute value, symmetric about -b/a",
    _b_abslin, _s_abslin, ["a", "b", "c"]))


def _b_expol(rng):
    a = F(rng.randint(2, 5))
    if a <= 0:
        raise ValueError
    m = F(rng.randint(1, 4))
    n = F(rng.randint(1, 4))
    if a ** int(m + n) > 10 ** 9:
        raise ValueError("too big")
    return {"a": a, "m": m, "n": n}


def _s_expol(p):
    a, m, n = p["a"], int(p["m"]), int(p["n"])
    if any(v.denominator != 1 for v in (p["a"], p["m"], p["n"])):
        raise ValueError("expo laws keep the integer domain")
    if a <= 0:
        raise ValueError("a^m requires a>0")
    if a ** (m + n) > 10 ** 9:
        raise ValueError("power too large for the rel cap")
    return [{"name": "left", "expr": "a^m", "value": a ** m},
            {"name": "right", "expr": "a^n", "value": a ** n},
            {"name": "product", "expr": "a^m * a^n", "value": a ** m * a ** n},
            {"name": "merged", "expr": "a^(m+n)", "value": a ** (m + n)}]


EXT3_ALGOS.append(_mk("expo_laws", "algebra",
    "a^m * a^n = a^(m+n); verified exactly on small integer powers.",
    "compute both factors, their product, and the merged power",
    _b_expol, _s_expol, ["a", "m", "n"]))


def _b_log2(rng):
    k = F(rng.randint(1, 8))
    return {"N": F(2) ** int(k)}


def _s_log2(p):
    N = p["N"]
    if N <= 0 or N % 2 != 0:
        raise ValueError("not a power of two")
    steps = [{"name": "N", "expr": "N", "value": N}]
    x = N
    nsplit = 0
    while x > 1:
        x = x / 2
        nsplit += 1
        steps.append({"name": f"half{nsplit}", "expr": "/2", "value": x})
        if nsplit > 9:
            raise ValueError("too many halvings")
    if x != 1:
        raise ValueError("N is not a power of two")
    steps.append({"name": "k", "expr": "log2(N)", "value": F(nsplit)})
    return steps


EXT3_ALGOS.append(_mk("log2_pow2", "algebra",
    "if N is a power of two, N = 2^k and k is the halving count.",
    "halve N until 1; the count is the exponent",
    _b_log2, _s_log2, ["N"]))


def _b_gmean(rng):
    s = F(rng.randint(1, 4))
    q = F(rng.choice([2, 3]))          # Q such that Q^2 is a perfect square
    return {"a": s * q * q, "b": s}


def _s_gmean(p):
    a, b = p["a"], p["b"]
    if a <= 0 or b <= 0:
        raise ValueError("GM needs positive entries")
    am = (a + b) / 2
    gm = _sqrtq(a * b)
    return [{"name": "sum", "expr": "a+b", "value": a + b},
            {"name": "AM", "expr": "(a+b)/2", "value": am},
            {"name": "ab", "expr": "a*b", "value": a * b},
            {"name": "GM", "expr": "sqrt(ab)", "value": gm},
            {"name": "gap", "expr": "AM-GM", "value": am - gm}]


EXT3_ALGOS.append(_mk("geometric_mean_pair", "algebra",
    "GM of (sQ^2, s) equals sQ exactly; AM >= GM with gap (s(Q-1)^2 ... )/2 >= 0.",
    "build a pair whose product is a perfect square, so the GM is exact",
    _b_gmean, _s_gmean, ["a", "b"]))


def _b_gauss(rng):
    n = F(rng.randint(10, 90))
    return {"n": n}


def _s_gauss(p):
    n = p["n"]
    if n <= 0:
        raise ValueError("n>0")
    t1 = n * (n + 1)
    half = t1 / 2
    sum_v = n * (n + 1) / 2
    return [{"name": "pairs", "expr": "n/2", "value": n / 2},
            {"name": "pair_sum", "expr": "n+1", "value": n + 1},
            {"name": "product", "expr": "(n/2)(n+1)", "value": sum_v},
            {"name": "sum", "expr": "n(n+1)/2", "value": half},
            {"name": "consistency", "expr": "sum", "value": sum_v}]


EXT3_ALGOS.append(_mk("gauss_pairing", "algebra",
    "Gauss pairing: sum_{1..n} = (n/2)(n+1); both factor forms agree exactly.",
    "pair argument in two equivalent exact forms",
    _b_gauss, _s_gauss, ["n"]))


def _b_rshare(rng):
    p = F(rng.randint(1, 5))
    q = F(rng.randint(1, 5))
    r = F(rng.randint(1, 5))
    m = F(rng.randint(1, 20))
    tot = (p + q + r) * m
    return {"N": tot, "p": p, "q": q, "r": r}


def _s_rshare(p):
    N, P, Q, R = p["N"], p["p"], p["q"], p["r"]
    parts = P + Q + R
    if N <= 0 or parts <= 0:
        raise ValueError("share needs positive parts")
    unit = N / parts
    return [{"name": "parts", "expr": "p+q+r", "value": parts},
            {"name": "unit", "expr": "N/(p+q+r)", "value": unit},
            {"name": "share_p", "expr": "unit*p", "value": unit * P},
            {"name": "share_q", "expr": "unit*q", "value": unit * Q},
            {"name": "share_r", "expr": "unit*r", "value": unit * R}]


EXT3_ALGOS.append(_mk("ratio_share", "algebra",
    "N split in ratio p:q:r gives unit N/(p+q+r) and shares unit*p, unit*q, unit*r.",
    "divide by the ratio sum, then scale each share",
    _b_rshare, _s_rshare, ["N", "p", "q", "r"]))


def _b_pct(rng):
    q = F(rng.choice([5, 10, 20, 25, 50, 75]))
    n = F(rng.randint(2, 40)) * 4
    return {"n": n, "q": q}


def _s_pct(p):
    n, q = p["n"], p["q"]
    if q == 0:
        raise ValueError("no percentage")
    a = q * n / 100
    back = a * 100 / q
    if back != n:
        raise ValueError("reverse check failed")
    return [{"name": "part", "expr": "q% of n", "value": a},
            {"name": "reverse", "expr": "part*100/q", "value": back},
            {"name": "n", "expr": "n", "value": n}]


EXT3_ALGOS.append(_mk("percentage_pair", "algebra",
    "q% of n is qn/100 and reversing recovers n exactly.",
    "percentage and its exact reverse",
    _b_pct, _s_pct, ["n", "q"]))


def _b_fcomp(rng):
    b1, d1 = F(1), F(1)
    a = F(rng.randint(2, 6))
    b = F(rng.randint(1, 9))
    c = F(rng.randint(2, 6))
    d = F(rng.randint(1, 9))
    x0 = F(rng.randint(-5, 5))
    return {"a": a, "b": b, "c": c, "d": d, "x0": x0}


def _s_fcomp(p):
    a, b, c, d, x0 = p["a"], p["b"], p["c"], p["d"], p["x0"]
    gx = c * x0 + d
    fg = a * gx + b
    ac = a * c
    const = a * d + b
    return [{"name": "g(x0)", "expr": "cx+d at x0", "value": gx},
            {"name": "f(g(x0))", "expr": "a(g(x0))+b", "value": fg},
            {"name": "ac", "expr": "a*c", "value": ac},
            {"name": "const", "expr": "ad+b", "value": const},
            {"name": "composite", "expr": "ac*x+const at x0", "value": ac * x0 + const}]


EXT3_ALGOS.append(_mk("composition_affine", "algebra",
    "f(x)=ax+b composed with g(x)=cx+d is (ac)x + (ad+b).",
    "compose two affine maps in coefficient form and evaluate",
    _b_fcomp, _s_fcomp, ["a", "b", "c", "d"]))


def _b_wavg(rng):
    v1 = F(rng.randint(1, 30))
    v2 = F(rng.randint(1, 30))
    k1 = F(rng.randint(1, 9))
    k2 = F(rng.randint(1, 9))
    return {"v1": v1, "v2": v2, "k1": k1, "k2": k2}


def _s_wavg(p):
    v1, v2, k1, k2 = p["v1"], p["v2"], p["k1"], p["k2"]
    tot = k1 + k2
    if tot == 0:
        raise ValueError("no weights")
    wsum = k1 * v1 + k2 * v2
    avg = wsum / tot
    return [{"name": "weighted_sum", "expr": "k1v1+k2v2", "value": wsum},
            {"name": "count", "expr": "k1+k2", "value": tot},
            {"name": "weighted_mean", "expr": "wsum/tot", "value": avg},
            {"name": "gap_from_v1", "expr": "mean-v1", "value": avg - v1}]


EXT3_ALGOS.append(_mk("weighted_average", "algebra",
    "weighted mean (k1 v1 + k2 v2)/(k1 + k2) lies between v1 and v2, computed exactly.",
    "weight, sum, then divide by the total weight",
    _b_wavg, _s_wavg, ["v1", "v2", "k1", "k2"]))


def _b_inveval(rng):
    a = F(rng.randint(1, 6)) or F(3)
    if a == 0:
        a = F(3)
    b = _nz(rng, 1, 6)
    x0 = F(rng.randint(1, 9))
    if x0 + b == 0:
        x0 = F(1)
    return {"a": a, "b": b, "x0": x0}


def _s_inveval(p):
    a, b, x0 = p["a"], p["b"], p["x0"]
    if a == 0 or b == 0:
        raise ValueError("degenerate f")
    y0 = a * x0 / (x0 + b)
    if a - y0 == 0:
        raise ValueError("inverse undefined")
    inv = y0 * b / (a - y0)
    if inv != x0:
        raise ValueError("inverse mismatch")
    return [{"name": "denom", "expr": "x0+b", "value": x0 + b},
            {"name": "f(x0)", "expr": "a x0/(x0+b)", "value": y0},
            {"name": "inv", "expr": "yb/(a-y)", "value": inv},
            {"name": "x0", "expr": "x0", "value": x0}]


EXT3_ALGOS.append(_mk("inv_function_eval", "algebra",
    "f(x)=ax/(x+b) inverts to x = yb/(a-y), and the round trip is exact.",
    "evaluate an affine-coefficient rational map and invert it",
    _b_inveval, _s_inveval, ["a", "b", "x0"]))


def _b_dsum(rng):
    b = rng.choice([2, 3, 4, 5, 6, 7, 8])
    n = F(rng.randint(b, b ** 5 - 1))
    return {"N": n, "b": F(b)}


def _s_dsum(p):
    N, b = p["N"], p["b"]
    if b.denominator != 1 or not (F(2) <= b <= F(8)) or N.denominator != 1:
        raise ValueError("digit_sum keeps the integer base-2..8 domain")
    if N < b ** 5:
        pass
    else:
        raise ValueError("digit count exceeds canonical budget")
    steps = []
    x = N
    total = F(0)
    for i in range(5):
        d = x % b
        steps.append({"name": f"d{i}", "expr": f"N mod b^{i+1} over b^i", "value": d})
        total += d
        x = F(x // b)
    steps.append({"name": "digit_sum", "expr": "sum digits", "value": total})
    return steps


EXT3_ALGOS.append(_mk("digit_sum_base", "algebra",
    "the base-b digit sum of N via the mod/div chain, exact for any base 2..8.",
    "extract five digits by successive mod and division, then sum them",
    _b_dsum, _s_dsum, ["N", "b"]))


def _b_median(rng):
    vals = [F(rng.randint(-30, 30)) for _ in range(5)]
    if len(set(vals)) < 5:
        raise ValueError("want distinct")
    return {"x1": vals[0], "x2": vals[1], "x3": vals[2], "x4": vals[3], "x5": vals[4]}


def _s_median(p):
    xs = [p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]]
    if len(set(xs)) < 5:
        raise ValueError("want distinct")
    lo12, hi12 = min(xs[0], xs[1]), max(xs[0], xs[1])
    lo34, hi34 = min(xs[2], xs[3]), max(xs[2], xs[3])
    c1 = max(lo12, lo34)
    c2 = min(hi12, hi34)
    tot = c1 + c2 + xs[4]
    lo3 = min(c1, min(c2, xs[4]))
    hi3 = max(c1, max(c2, xs[4]))
    med = tot - lo3 - hi3
    return [{"name": "low12", "expr": "min(x1,x2)", "value": lo12},
            {"name": "hi12", "expr": "max(x1,x2)", "value": hi12},
            {"name": "low34", "expr": "min(x3,x4)", "value": lo34},
            {"name": "hi34", "expr": "max(x3,x4)", "value": hi34},
            {"name": "cand_low", "expr": "min of lows", "value": lo12},
            {"name": "cand1", "expr": "max(low12,low34)", "value": c1},
            {"name": "cand2", "expr": "min(hi12,hi34)", "value": c2},
            {"name": "cand_sum", "expr": "c1+c2+x5", "value": tot},
            {"name": "surrounded_lo", "expr": "min3", "value": lo3},
            {"name": "surrounded_hi", "expr": "max3", "value": hi3},
            {"name": "median", "expr": "sum - min - max", "value": med},
            {"name": "consistency", "expr": "median", "value": med}]


EXT3_ALGOS.append(_mk("median_five", "algebra",
    "median of five distinct numbers: two pair partitions leave three candidates; sum-min-max is the middle.",
    "selection network on five values with exact arithmetic",
    _b_median, _s_median, ["x1", "x2", "x3", "x4", "x5"]))


def _b_odds(rng):
    n = F(rng.randint(1, 12))
    return {"n": n}


def _s_odds(p):
    n = p["n"]
    if n <= 0:
        raise ValueError("n>0")
    last = 2 * n - 1
    avg = (1 + last) / 2
    s = n * n
    return [{"name": "count", "expr": "n", "value": n},
            {"name": "first", "expr": "1", "value": F(1)},
            {"name": "last_term", "expr": "2n-1", "value": last},
            {"name": "average", "expr": "(first+last)/2", "value": avg},
            {"name": "sum", "expr": "avg * n", "value": avg * n},
            {"name": "square_check", "expr": "n^2", "value": s}]


EXT3_ALGOS.append(_mk("sum_first_odds", "algebra",
    "the sum of the first n odd numbers is n^2: average (2n-1+1)/2 = n, times n terms.",
    "arithmetic-series average-times-count with a square target",
    _b_odds, _s_odds, ["n"]))


# ================= CALCULUS (12) ============================


def _b_tanq(rng):
    a = F(rng.randint(1, 4))
    b = F(rng.randint(-9, 9))
    c = F(rng.randint(-9, 9))
    x0 = F(rng.randint(-6, 6))
    return {"a": a, "b": b, "c": c, "x0": x0}


def _s_tanq(p):
    a, b, c, x0 = p["a"], p["b"], p["c"], p["x0"]
    fx = a * x0 * x0 + b * x0 + c
    m = 2 * a * x0 + b
    t = fx - m * x0
    return [{"name": "f(x0)", "expr": "ax0^2+bx0+c", "value": fx},
            {"name": "slope", "expr": "2ax0+b", "value": m},
            {"name": "intercept", "expr": "f(x0)-m*x0", "value": t},
            {"name": "tangent_at_x0", "expr": "m*x0+t", "value": m * x0 + t}]


EXT3_ALGOS.append(_mk("tangent_line_quad", "calculus",
    "tangent to ax^2+bx+c at x0: slope 2ax0+b, intercept c - a x0^2.",
    "point-slope construction on a parabola, exact",
    _b_tanq, _s_tanq, ["a", "b", "c", "x0"]))


def _b_serversant(rng):
    al = F(rng.randint(1, 6))
    be = F(rng.randint(-20, 20))
    x1 = F(rng.randint(-8, 8))
    x2 = F(rng.randint(-8, 8))
    if x1 == x2:
        x2 = x1 + 1
    return {"A": al, "B": be, "x1": x1, "x2": x2}


def _s_secant(p):
    A, B, x1, x2 = p["A"], p["B"], p["x1"], p["x2"]
    f1 = A * x1 * x1 + B
    f2 = A * x2 * x2 + B
    den = x2 - x1
    if den == 0:
        raise ValueError("same point")
    slope = (f2 - f1) / den
    pred = A * (x1 + x2)
    return [{"name": "f(x1)", "expr": "Ax1^2+B", "value": f1},
            {"name": "f(x2)", "expr": "Ax2^2+B", "value": f2},
            {"name": "slope", "expr": "(f2-f1)/(x2-x1)", "value": slope},
            {"name": "closed_form", "expr": "A(x1+x2)", "value": pred}]


EXT3_ALGOS.append(_mk("secant_poly_slope", "calculus",
    "secant slope of Ax^2+B between x1, x2 equals A(x1+x2) exactly.",
    "difference quotient and its closed form agree",
    _b_serversant, _s_secant, ["A", "B", "x1", "x2"]))


def _b_printint(rng):
    c = F(rng.randint(1, 6))
    n = rng.randint(2, 5)
    B = rng.randint(1, 4)
    if c * B ** (n + 1) > 10 ** 8:
        raise ValueError("integral too large")
    return {"c": F(c), "n": F(n), "B": F(B)}


def _s_printint(p):
    c, n, B = p["c"], int(p["n"]), int(p["B"])
    if p["n"].denominator != 1 or p["B"].denominator != 1:
        raise ValueError("power rule keeps the integer domain")
    coef = c / (n + 1)
    bp = F(B) ** (n + 1)
    integral = coef * bp
    return [{"name": "antiderivative_coef", "expr": "c/(n+1)", "value": coef},
            {"name": "B^(n+1)", "expr": "B^(n+1)", "value": bp},
            {"name": "integral", "expr": "c/(n+1) * B^(n+1)", "value": integral},
            {"name": "check", "expr": "integral", "value": integral}]


EXT3_ALGOS.append(_mk("power_rule_integrate", "calculus",
    "integral of c x^n over [0,B] is c B^(n+1)/(n+1), exact in rationals.",
    "antiderivative power rule evaluated at the upper bound",
    _b_printint, _s_printint, ["c", "n", "B"]))


def _b_newton(rng):
    x0 = F(rng.randint(4, 20))
    m = F(rng.randint(1, 30))
    N = x0 * x0 + x0 * m
    return {"N": N, "x0": x0}


def _s_newton(p):
    N, x0 = p["N"], p["x0"]
    if x0 <= 0 or N <= 0:
        raise ValueError("Newton sqrt needs positives")
    q = N / x0
    if N != q * x0:
        raise ValueError("division not exact")
    s = x0 + q
    x1 = s / 2
    return [{"name": "N/x0", "expr": "N/x0", "value": q},
            {"name": "x0+N/x0", "expr": "x0+N/x0", "value": s},
            {"name": "x1", "expr": "(x0+N/x0)/2", "value": x1},
            {"name": "residual", "expr": "x1^2-N (squared improvement)", "value": x1 * x1 - N}]


EXT3_ALGOS.append(_mk("newton_sqrt_step", "calculus",
    "one Newton step for sqrt(N): x1=(x0+N/x0)/2, kept exact because x0 | N.",
    "square-root iteration with an exactly divisible trial",
    _b_newton, _s_newton, ["N", "x0"]))


def _b_euler(rng):
    k = F(rng.randint(2, 6))
    x0 = F(rng.randint(1, 5))
    y0 = F(rng.randint(1, 9))
    d = rng.choice([2, 3, 4])
    h = F(1, d)
    return {"k": k, "x0": x0, "y0": y0, "h": h}


def _s_euler(p):
    k, x0, y0, h = p["k"], p["x0"], p["y0"], p["h"]
    y1 = y0 + h * k * x0
    x1 = x0 + h
    y2 = y1 + h * k * x1
    return [{"name": "step_h", "expr": "h", "value": h},
            {"name": "y1", "expr": "y0+h*k*x0", "value": y1},
            {"name": "x1", "expr": "x0+h", "value": x1},
            {"name": "y2", "expr": "y1+h*k*x1", "value": y2}]


EXT3_ALGOS.append(_mk("euler_method_linear", "calculus",
    "Euler method for y'=kx, y(x0)=y0: two steps with rational h are exact.",
    "explicit Euler on a linear slope field",
    _b_euler, _s_euler, ["k", "x0", "y0", "h"]))


def _b_simpson(rng):
    a = rng.randint(1, 4)
    b = rng.randint(2, 6)
    if b <= a:
        b = a + 1
    A = F(rng.randint(1, 3))
    Bc = F(rng.randint(-9, 9))
    C = F(rng.randint(-9, 9))
    return {"A": A, "b": Bc, "c": C, "u": F(a), "v": F(b)}


def _s_simpson(p):
    A, B, C, u, v = p["A"], p["b"], p["c"], p["u"], p["v"]
    if v == u:
        raise ValueError("degenerate interval")
    mid = (u + v) / 2

    def f(x):
        return A * x * x + B * x + C
    fu = f(u)
    fmid = f(mid)
    fv = f(v)
    S = (v - u) / 6 * (fu + 4 * fmid + fv)
    I = A * (v ** 3 - u ** 3) / 3 + B * (v * v - u * u) / 2 + C * (v - u)
    return [{"name": "mid", "expr": "(u+v)/2", "value": mid},
            {"name": "f(u)", "expr": "f(u)", "value": fu},
            {"name": "4f(mid)", "expr": "4f(mid)", "value": 4 * fmid},
            {"name": "f(v)", "expr": "f(v)", "value": fv},
            {"name": "S", "expr": "(v-u)/6*(f(u)+4f(m)+f(v))", "value": S},
            {"name": "I", "expr": "exact integral", "value": I},
            {"name": "defect", "expr": "S-I (0 for quadratics)", "value": S - I}]


EXT3_ALGOS.append(_mk("simpson_quadratic", "calculus",
    "Simpson's rule is exact for quadratics: S - I = 0, verified exactly.",
    "midpoint-weighted quadrature compared to the antiderivative",
    _b_simpson, _s_simpson, ["A", "b", "c", "u", "v"]))


def _b_mvt(rng):
    A = F(rng.randint(1, 5))
    Bc = F(rng.randint(-9, 9))
    C = F(rng.randint(-9, 9))
    pp = F(rng.randint(-5, 3))
    qq = pp + F(rng.randint(1, 5))
    return {"A": A, "B": Bc, "C": C, "p": pp, "q": qq}


def _s_mvt(p):
    A, B, C, pp, qq = p["A"], p["B"], p["C"], p["p"], p["q"]
    if qq <= pp:
        raise ValueError("need p<q")
    fp = A * pp * pp + B * pp + C
    fq = A * qq * qq + B * qq + C
    sec = (fq - fp) / (qq - pp)
    xi = (pp + qq) / 2
    fpi = 2 * A * xi + B
    return [{"name": "f(p)", "expr": "f(p)", "value": fp},
            {"name": "f(q)", "expr": "f(q)", "value": fq},
            {"name": "secant", "expr": "(f(q)-f(p))/(q-p)", "value": sec},
            {"name": "xi", "expr": "(p+q)/2", "value": xi},
            {"name": "fprime_xi", "expr": "2Axi+B", "value": fpi},
            {"name": "defect", "expr": "f'(xi)-secant (0)", "value": fpi - sec}]


EXT3_ALGOS.append(_mk("mvt_quadratic", "calculus",
    "Mean value theorem for a quadratic: the tangent-slope point is exactly (p+q)/2.",
    "secant slope, midpoint, derivative agreement",
    _b_mvt, _s_mvt, ["A", "B", "C", "p", "q"]))


def _b_critc(rng):
    r1 = rng.randint(1, 4)
    r2 = rng.randint(1, 4)
    if (r1 + r2) % 2 != 0:
        r2 += 1
    a = F(-3 * (r1 + r2) // 2) if (r1 + r2) % 2 == 0 else F(-3, 2) * (r1 + r2)
    if a.denominator != 1:
        raise ValueError("want integer quartic coeff")
    b = F(3 * r1 * r2)
    c = F(rng.randint(-9, 9))
    return {"a": a, "b": b, "c": c}


def _s_critc(p):
    a, b, c = p["a"], p["b"], p["c"]
    f2 = 3 * b - 2 * a * a          # f'(0) reduced coefficient set
    D = 4 * a * a - 12 * b
    isq = _isqrt(D.numerator) if D.denominator == 1 else None
    if isq is None or isq * isq != D.numerator:
        raise ValueError("discriminant not a perfect square")
    s = F(isq)
    rplus = (-2 * a + s) / 6
    rminus = (-2 * a - s) / 6
    xcrit = -a / 3
    return [{"name": "f'(0) pair", "expr": "3b-2a^2", "value": f2},
            {"name": "D", "expr": "4a^2-12b", "value": D},
            {"name": "sqrtD", "expr": "3|r1-r2|", "value": s},
            {"name": "crit_plus", "expr": "(-2a+sqrtD)/6", "value": rplus},
            {"name": "crit_minus", "expr": "(-2a-sqrtD)/6", "value": rminus},
            {"name": "vertex", "expr": "-a/3", "value": xcrit},
            {"name": "concavity", "expr": "f''(0)=2a at origin", "value": 2 * a}]


EXT3_ALGOS.append(_mk("critical_cubic", "calculus",
    "critical points of x^3+ax^2+bx+c via the perfect-square discriminant of f'.",
    "quadratic criticals with an exact sqrt built in",
    _b_critc, _s_critc, ["a", "b", "c"]))


def _b_split(rng):
    A = F(rng.randint(1, 4))
    B = F(rng.randint(-9, 9))
    C = F(rng.randint(-20, 20))
    d = F(rng.randint(-4, 4))
    return {"A": A, "B": B, "C": C, "d": d}


def _s_split(p):
    A, B, C, d = p["A"], p["B"], p["C"], p["d"]
    if A == 0 or d == 0:
        raise ValueError("no oblique split")
    qA = A
    qB = B + A * d
    rem = C + d * (B + A * d)
    return [{"name": "quotient_slope", "expr": "A", "value": qA},
            {"name": "quotient_const", "expr": "B+Ad", "value": qB},
            {"name": "remainder", "expr": "C+d(B+Ad)", "value": rem},
            {"name": "vertical", "expr": "x=d", "value": d}]


EXT3_ALGOS.append(_mk("asymptote_oblique", "calculus",
    "divide (Ax^2+Bx+C) by (x-d): quotient A x + (B+Ad), remainder C+d(B+Ad), vertical at d.",
    "synthetic division against a linear denominator",
    _b_split, _s_split, ["A", "B", "C", "d"]))


def _b_aul(rng):
    m = F(rng.randint(1, 6))
    c = F(rng.randint(-9, 9))
    a = F(rng.randint(-5, 2))
    b = a + rng.randint(1, 5)
    return {"m": m, "c": c, "a": a, "b": b}


def _s_aul(p):
    m, c, a, b = p["m"], p["c"], p["a"], p["b"]
    if b <= a:
        raise ValueError("need a<b")
    t1 = m * (b * b - a * a) / 2
    t2 = c * (b - a)
    tot = t1 + t2
    return [{"name": "quadratic_part", "expr": "m(b^2-a^2)/2", "value": t1},
            {"name": "constant_part", "expr": "c(b-a)", "value": t2},
            {"name": "area", "expr": "total", "value": tot}]


EXT3_ALGOS.append(_mk("area_under_linear", "calculus",
    "area under mx+c on [a,b]: m(b^2-a^2)/2 + c(b-a), exact.",
    "trapezoid integral of a line in two exact parts",
    _b_aul, _s_aul, ["m", "c", "a", "b"]))


def _b_central(rng):
    A = F(rng.randint(1, 4))
    x = F(rng.randint(1, 5))
    d = rng.choice([2, 3, 4])
    h = F(1, d)
    return {"A": A, "x": x, "h": h}


def _s_central(p):
    A, x, h = p["A"], p["x"], p["h"]
    if h == 0:
        raise ValueError("h=0")

    def f(t):
        return A * t * t * t
    fp = f(x + h)
    fm = f(x - h)
    d2 = (fp - 2 * A * x ** 3 + fm) / (h * h)
    exact = 6 * A * x
    return [{"name": "f(x+h)", "expr": "A(x+h)^3", "value": fp},
            {"name": "f(x-h)", "expr": "A(x-h)^3", "value": fm},
            {"name": "central_diff2", "expr": "(f(x+h)-2f(x)+f(x-h))/h^2", "value": d2},
            {"name": "exact_second", "expr": "6Ax", "value": exact},
            {"name": "error", "expr": "d2-6Ax (h^2 term)", "value": d2 - exact}]


EXT3_ALGOS.append(_mk("second_diff_central", "calculus",
    "central second difference of Ax^3 equals 6Ax exactly, the h^2 error term vanishing for cubics.",
    "finite difference on a cubic with the vanishing error term visible",
    _b_central, _s_central, ["A", "x", "h"]))


def _b_rolle(rng):
    pp = F(rng.randint(-6, 0))
    qq = pp + rng.randint(1, 6)
    A = F(rng.randint(1, 5))
    return {"A": A, "p": pp, "q": qq}


def _s_rolle(p):
    A, pp, qq = p["A"], p["p"], p["q"]
    if qq <= pp:
        raise ValueError("need p<q")

    def f(t):
        return A * (t - pp) * (t - qq)
    fp = f(pp)
    fq = f(qq)
    xi = (pp + qq) / 2
    fpi = 2 * A * xi - A * (pp + qq)
    return [{"name": "f(p)", "expr": "zero factor", "value": fp},
            {"name": "f(q)", "expr": "zero factor", "value": fq},
            {"name": "xi", "expr": "(p+q)/2", "value": xi},
            {"name": "fprime_xi", "expr": "f'(xi)", "value": fpi},
            {"name": "between", "expr": "xi strictly inside", "value": (xi - pp) * (qq - xi)}]


EXT3_ALGOS.append(_mk("rolle_check", "calculus",
    "f=A(x-p)(x-q): f vanishes at p,q and f' vanishes exactly at (p+q)/2, strictly between.",
    "Rolle's theorem executed on a factored quadratic",
    _b_rolle, _s_rolle, ["A", "p", "q"]))


# ================= LINEAR_ALGEBRA (12) ============================


def _b_det3(rng):
    e = [F(rng.randint(-4, 4)) for _ in range(9)]
    if all(v == 0 for v in e):
        e[0] = F(1)
    return {k: e[i] for i, k in enumerate("abcdefghi")}


def _s_det3(p):
    a, b, c = p["a"], p["b"], p["c"]
    d, e2, f = p["d"], p["e"], p["f"]
    g, h, i = p["g"], p["h"], p["i"]
    t1 = a * e2 * i
    t2 = b * f * g
    t3 = c * d * h
    t4 = a * f * h
    t5 = b * d * i
    t6 = c * e2 * g
    det = t1 + t2 + t3 - t4 - t5 - t6
    return [{"name": "pos1", "expr": "aei", "value": t1},
            {"name": "pos2", "expr": "bfg", "value": t2},
            {"name": "pos3", "expr": "cdh", "value": t3},
            {"name": "neg1", "expr": "afh", "value": t4},
            {"name": "neg2", "expr": "bdi", "value": t5},
            {"name": "neg3", "expr": "ceg", "value": t6},
            {"name": "det", "expr": "sum", "value": det}]


EXT3_ALGOS.append(_mk("det_3x3_cofactor", "linear_algebra",
    "det of a 3x3 by the six Sarrus triple products, exact integers.",
    "rule of Sarrus in explicit triple-product steps",
    _b_det3, _s_det3, ["a", "b", "c", "d", "e", "f", "g", "h", "i"]))


def _b_matvec(rng):
    e = [F(rng.randint(-5, 5)) for _ in range(9)]
    u = [F(rng.randint(-5, 5)) for _ in range(3)]
    if u == [F(0), F(0), F(0)]:
        u[0] = F(1)
    out = {"r" + str(r) + "c" + str(cc): e[3 * r + cc]
           for r in range(3) for cc in range(3)}
    out["u1"], out["u2"], out["u3"] = u
    return out


def _s_matvec(p):
    u = [p["u1"], p["u2"], p["u3"]]
    rows = []
    for r in range(3):
        row = [p["r" + str(r) + "c" + str(cc)] for cc in range(3)]
        rows.append(sum(row[cc] * u[cc] for cc in range(3)))
    return [{"name": "y1", "expr": "row1.u", "value": rows[0]},
            {"name": "y2", "expr": "row2.u", "value": rows[1]},
            {"name": "y3", "expr": "row3.u", "value": rows[2]}]


EXT3_ALGOS.append(_mk("matvec_3", "linear_algebra",
    "3x3 matrix times a 3-vector, three exact dot products.",
    "apply a linear map to a vector row by row",
    _b_matvec, _s_matvec, ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2",
                            "r2c0", "r2c1", "r2c2", "u1", "u2", "u3"]))


def _b_rank2(rng):
    m11 = F(rng.choice([1, 2, 3, -1, -2]))
    m12 = F(rng.randint(-6, 6))
    m13 = F(rng.randint(-6, 6))
    m21 = F(rng.randint(-6, 6))
    m22 = F(rng.randint(-6, 6))
    m23 = F(rng.randint(-6, 6))
    if m11 == 0:
        m11 = F(2)
    out = {"m11": m11, "m12": m12, "m13": m13, "m21": m21, "m22": m22, "m23": m23}
    return out


def _s_rank2(p):
    m11, m12, m13 = p["m11"], p["m12"], p["m13"]
    m21, m22, m23 = p["m21"], p["m22"], p["m23"]
    if m11 == 0:
        raise ValueError("no pivot")
    t = m21 / m11
    n21 = m21 - t * m11
    n22 = m22 - t * m12
    n23 = m23 - t * m13
    if n21 != 0:
        raise ValueError("elimination failed")
    nonzero2 = any(v != 0 for v in (n22, n23))
    if not (m12 or m13) and not nonzero2:
        raise ValueError("zero rows everywhere")
    rank = F(2) if nonzero2 else F(1)
    return [{"name": "multiplier", "expr": "m21/m11", "value": t},
            {"name": "new_pivot_row_col1", "expr": "m21-t*m11", "value": n21},
            {"name": "new22", "expr": "m22-t*m12", "value": n22},
            {"name": "new23", "expr": "m23-t*m13", "value": n23},
            {"name": "rank", "expr": "1 or 2 free columns", "value": rank}]


EXT3_ALGOS.append(_mk("row_reduce_rank2", "linear_algebra",
    "rank of a 2x3: one exact elimination row op, then count the free columns.",
    "gaussian elimination to row echelon form, rank verdict",
    _b_rank2, _s_rank2, ["m11", "m12", "m13", "m21", "m22", "m23"]))


def _b_frobenius(rng):
    a, b, c, d = [F(rng.randint(-5, 5)) for _ in range(4)]
    if a == b == c == d == 0:
        a = F(1)
    return {"a": a, "b": b, "c": c, "d": d}


def _s_frobenius(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    s = a * a + b * b + c * c + d * d
    if s == 0:
        raise ValueError("zero matrix")
    return [{"name": "a2", "expr": "a^2", "value": a * a},
            {"name": "b2", "expr": "b^2", "value": b * b},
            {"name": "c2", "expr": "c^2", "value": c * c},
            {"name": "d2", "expr": "d^2", "value": d * d},
            {"name": "frobenius_sq", "expr": "sum", "value": s}]


EXT3_ALGOS.append(_mk("trace_frobenius2", "linear_algebra",
    "Frobenius norm squared of a 2x2: four entry squares summed, exact.",
    "entrywise quadratic form",
    _b_frobenius, _s_frobenius, ["a", "b", "c", "d"]))


def _b_invmult(rng):
    a = F(rng.randint(-5, 5))
    b = F(rng.randint(-6, 6))
    c = F(rng.randint(-6, 6))
    d = F(rng.randint(-5, 5))
    det = a * d - b * c
    if det == 0:
        a = F(1); b = F(0); c = F(0); d = F(3)
        det = F(3)
    return {"a": a, "b": b, "c": c, "d": d}


def _s_invmult(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular")
    e11 = (d * a + (-b) * c) / det
    e12 = (d * b + (-b) * d) / det
    e21 = ((-c) * a + a * c) / det
    e22 = ((-c) * b + a * d) / det
    return [{"name": "det", "expr": "ad-bc", "value": det},
            {"name": "Ainv_A_11", "expr": "(d*a-b*c)/det", "value": e11},
            {"name": "Ainv_A_12", "expr": "(d*b-b*d)/det", "value": e12},
            {"name": "Ainv_A_21", "expr": "(c*a-a*c)/det", "value": e21},
            {"name": "Ainv_A_22", "expr": "(a*d-c*b)/det", "value": e22}]


EXT3_ALGOS.append(_mk("inverse_mult_identity2", "linear_algebra",
    "A^{-1} times A is the identity: all four product entries are det/det, 0, 0, det/det.",
    "inverse product entries collapse exactly",
    _b_invmult, _s_invmult, ["a", "b", "c", "d"]))


def _b_deppair(rng):
    u1 = F(rng.randint(-4, 4))
    u2 = F(rng.randint(-4, 4))
    if u1 == 0 and u2 == 0:
        u1 = F(2)
    if u1 == 0:
        u1 = F(2)
    if u2 == 0:
        u2 = F(2)
    p_ = rng.randint(-4, 4)
    q_ = rng.choice([1, 2, 3])
    k = F(p_, q_)
    if k == 0:
        k = F(2, 3)
    v1 = k * u1
    v2 = k * u2
    out = {"u1": u1, "u2": u2, "v1": v1, "v2": v2}
    return out


def _s_deppair(p):
    u1, u2, v1, v2 = p["u1"], p["u2"], p["v1"], p["v2"]
    cross = u1 * v2 - u2 * v1
    if cross != 0:
        raise ValueError("vectors independent (bad build)")
    if u1 == 0 or u2 == 0:
        raise ValueError("zero component ratio")
    k1 = v1 / u1
    k2 = v2 / u2
    if k1 != k2:
        raise ValueError("inconsistent scaling")
    return [{"name": "cross2", "expr": "u1v2-u2v1", "value": cross},
            {"name": "ratio1", "expr": "v1/u1", "value": k1},
            {"name": "ratio2", "expr": "v2/u2", "value": k2},
            {"name": "dependent", "expr": "cross==0 yes", "value": F(1)}]


EXT3_ALGOS.append(_mk("dep_pair_cross", "linear_algebra",
    "two vectors are parallel exactly when their 2D cross product is zero; the ratio is the scale.",
    "collinearity verdict by exact cross product",
    _b_deppair, _s_deppair, ["u1", "u2", "v1", "v2"]))


def _b_diageig(rng):
    d1 = F(rng.randint(-6, 6))
    d2 = F(rng.randint(-6, 6))
    if d1 == 0 and d2 == 0:
        d1 = F(2)
    return {"d1": d1, "d2": d2}


def _s_diageig(p):
    d1, d2 = p["d1"], p["d2"]
    det = d1 * d2
    tr = d1 + d2
    if det == 0 and tr == 0:
        raise ValueError("zero matrix")
    return [{"name": "det", "expr": "d1*d2", "value": det},
            {"name": "trace", "expr": "d1+d2", "value": tr},
            {"name": "eig1", "expr": "Dv1=v1*d1", "value": d1},
            {"name": "eig2", "expr": "Dv2=v2*d2", "value": d2}]


EXT3_ALGOS.append(_mk("diagonal_eigencheck", "linear_algebra",
    "a diagonal matrix has its eigenvalues exactly on the diagonal; det and trace confirm.",
    "eigenpairs of diag(d1,d2) verified by action on basis vectors",
    _b_diageig, _s_diageig, ["d1", "d2"]))


PYTHS = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (20, 21, 29)]


def _b_projpyth(rng):
    m, n, s = rng.choice(PYTHS)
    u1 = F(rng.randint(-9, 9))
    u2 = F(rng.randint(-9, 9))
    if s == 0:
        raise ValueError
    return {"u1": u1, "u2": u2, "m": F(m), "n": F(n), "s": F(s)}


def _s_projpyth(p):
    u1, u2, m, n, s = p["u1"], p["u2"], p["m"], p["n"], p["s"]
    normsq = m * m + n * n
    if normsq != s * s or normsq == 0:
        raise ValueError("not a valid pythagorean vector")
    dot = u1 * m + u2 * n
    coef = dot / normsq
    return [{"name": "dot", "expr": "u.v", "value": dot},
            {"name": "normsq", "expr": "m^2+n^2", "value": normsq},
            {"name": "coef", "expr": "u.v / (v.v)", "value": coef},
            {"name": "projx", "expr": "coef*m", "value": coef * m},
            {"name": "projy", "expr": "coef*n", "value": coef * n}]


EXT3_ALGOS.append(_mk("proj_pyth", "linear_algebra",
    "orthogonal projection of u onto a Pythagorean vector (m,n,|v|=s) with exact coef uv/|v|^2.",
    "projection onto a vector with exactly known length",
    _b_projpyth, _s_projpyth, ["u1", "u2", "m", "n", "s"]))


def _b_paraarea(rng):
    a1 = _nz(rng, -6, 6)
    a2 = _nz(rng, -6, 6)
    b1 = _nz(rng, -6, 6)
    b2 = _nz(rng, -6, 6)
    if a1 * b2 - a2 * b1 == 0:
        b2 = b2 + 1
        if a1 * b2 - a2 * b1 == 0:
            a2 = a2 + 1
    if a1 * b2 - a2 * b1 == 0:
        raise ValueError("degenerate parallelogram")
    return {"a1": a1, "a2": a2, "b1": b1, "b2": b2}


def _s_paraarea(p):
    a1, a2, b1, b2 = p["a1"], p["a2"], p["b1"], p["b2"]
    cross = a1 * b2 - a2 * b1
    if cross == 0:
        raise ValueError("degenerate parallelogram")
    area = F(abs(cross.numerator), cross.denominator)
    return [{"name": "cross2", "expr": "a1b2-a2b1", "value": cross},
            {"name": "area", "expr": "|cross2|", "value": area},
            {"name": "orientation", "expr": "sign(cross)", "value": F(1) if cross > 0 else F(-1)}]


EXT3_ALGOS.append(_mk("area_parallelogram2", "linear_algebra",
    "parallelogram area from column vectors is |a1 b2 - a2 b1|, exact.",
    "cross product and its absolute value",
    _b_paraarea, _s_paraarea, ["a1", "a2", "b1", "b2"]))


def _b_affinearn(rng):
    p1 = _nz(rng, -6, 6)
    p2 = _nz(rng, -6, 6)
    q1 = _nz(rng, -6, 6)
    q2 = _nz(rng, -6, 6)
    al = F(rng.choice([1, 2, 3]), rng.choice([2, 3, 4]))
    if (p1 == q1 and p2 == q2):
        q1 = q1 + 2
    return {"p1": p1, "p2": p2, "q1": q1, "q2": q2, "alpha": al}


def _s_affinearn(p):
    p1, p2, q1, q2, al = p["p1"], p["p2"], p["q1"], p["q2"], p["alpha"]
    if (p1 == q1 and p2 == q2):
        raise ValueError("degenerate segment")
    d1 = q1 - p1
    d2 = q2 - p2
    t1 = al * d1
    t2 = al * d2
    r1 = p1 + t1
    r2 = p2 + t2
    return [{"name": "dq1", "expr": "q1-p1", "value": d1},
            {"name": "dq2", "expr": "q2-p2", "value": d2},
            {"name": "scaled1", "expr": "alpha*dq1", "value": t1},
            {"name": "scaled2", "expr": "alpha*dq2", "value": t2},
            {"name": "r1", "expr": "p1+alpha(q1-p1)", "value": r1},
            {"name": "r2", "expr": "p2+alpha(q2-p2)", "value": r2}]


EXT3_ALGOS.append(_mk("affine_point", "linear_algebra",
    "r = p + alpha(q-p) is the affine interpolation point on segment pq, exact in rationals.",
    "affine combination of two points",
    _b_affinearn, _s_affinearn, ["p1", "p2", "q1", "q2", "alpha"]))


def _b_sympart(rng):
    a = F(rng.randint(-5, 5))
    b = F(rng.randint(-6, 6))
    c = F(rng.randint(-6, 6))
    d = F(rng.randint(-5, 5))
    return {"a": a, "b": b, "c": c, "d": d}


def _s_sympart(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    off = (b + c) / 2
    return [{"name": "S11", "expr": "(a+a)/2", "value": a},
            {"name": "S12", "expr": "(b+c)/2", "value": off},
            {"name": "S21", "expr": "(c+b)/2", "value": off},
            {"name": "S22", "expr": "(d+d)/2", "value": d},
            {"name": "skew_trace", "expr": "b-S12", "value": b - off}]


EXT3_ALGOS.append(_mk("matrix_sym_part", "linear_algebra",
    "the symmetric part (A+A^T)/2 of a 2x2 averages each off-diagonal pair exactly.",
    "symmetric decomposition entries",
    _b_sympart, _s_sympart, ["a", "b", "c", "d"]))


def _b_cossq(rng):
    u = rng.choice([(3, 4), (6, 8), (9, 12), (5, 12), (8, 15)])
    v = rng.choice([(3, 4), (5, 12), (8, 15), (9, 12)])
    u1, u2 = F(u[0]), F(u[1])
    v1, v2 = F(v[0]), F(v[1])
    normu = u1 * u1 + u2 * u2
    normv = v1 * v1 + v2 * v2
    if normu == 0 or normv == 0:
        raise ValueError("zero vector")
    return {"u1": u1, "u2": u2, "v1": v1, "v2": v2}


def _s_cossq(p):
    u1, u2, v1, v2 = p["u1"], p["u2"], p["v1"], p["v2"]
    dot = u1 * v1 + u2 * v2
    nu = u1 * u1 + u2 * u2
    nv = v1 * v1 + v2 * v2
    if nu == 0 or nv == 0:
        raise ValueError("zero vector")
    c2 = dot * dot / (nu * nv)
    if c2 > 1:
        raise ValueError("cosine impossible")
    return [{"name": "dot", "expr": "u.v", "value": dot},
            {"name": "normu", "expr": "|u|^2", "value": nu},
            {"name": "normv", "expr": "|v|^2", "value": nv},
            {"name": "cos_squared", "expr": "(u.v)^2/(|u|^2|v|^2)", "value": c2}]


EXT3_ALGOS.append(_mk("vector_angle_cossq", "linear_algebra",
    "cos^2 of the angle between Pythagorean vectors is (u.v)^2/(|u|^2|v|^2), fully rational.",
    "angle measure without transcendentals",
    _b_cossq, _s_cossq, ["u1", "u2", "v1", "v2"]))


# ================= NUMBER_THEORY (12) ============================


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime_small(v):
    v = int(v)
    if v < 2:
        return False
    if v in (2, 3):
        return True
    if v % 2 == 0 or v % 3 == 0:
        return False
    i = 5
    while i * i <= v:
        if v % i == 0 or v % (i + 2) == 0:
            return False
        i += 6
    return True


def _b_primefac(rng):
    p = F(rng.choice([2, 3, 5, 7, 11]))
    q = F(rng.choice([2, 3, 5, 7, 11]))
    if p == q:
        raise ValueError("want distinct primes")
    e1 = F(rng.randint(1, 2))
    e2 = F(rng.randint(1, 2))
    N = int(p) ** int(e1) * int(q) ** int(e2)
    return {"N": F(N)}


def _s_primefac(p):
    N = p["N"]
    if N <= 1 or N > 10 ** 5:
        raise ValueError("out of range")
    if N.denominator != 1:
        raise ValueError("N left the integer domain")
    steps = [{"name": "N", "expr": "N", "value": N}]
    x = N
    pairs = 0
    for pr in _PRIMES:
        while x != 1 and int(x) % pr == 0:
            pairs += 1
            if pairs > 5:
                raise ValueError("exceeds the canonical division budget")
            steps.append({"name": f"div{pr}", "expr": f"{pr} | x", "value": F(1)})
            x = x / pr
            steps.append({"name": f"quot{pr}", "expr": f"x/{pr}", "value": x})
    if x != 1:
        raise ValueError("leftover factor outside prime list")
    return steps


EXT3_ALGOS.append(_mk("prime_factor_small", "number_theory",
    "factor N=p^a q^b by trial division through the small primes, exact.",
    "trial-division factorization with running quotients",
    _b_primefac, _s_primefac, ["N"]))


def _b_psqcheck(rng):
    r = rng.randint(2, 200)
    return {"N": F(r * r)}


def _s_psqcheck(p):
    N = p["N"]
    if N <= 0:
        raise ValueError("bad N")
    r = F(_isqrt(N.numerator), N.denominator)
    rsq = r * r
    verdict = F(1) if rsq == N else F(-1)
    return [{"name": "isqrt", "expr": "floor(sqrt(N))", "value": r},
            {"name": "r_squared", "expr": "r^2", "value": rsq},
            {"name": "verdict", "expr": "N==r^2 ? 1 : -1", "value": verdict}]


EXT3_ALGOS.append(_mk("perfect_square_check", "number_theory",
    "N is a perfect square exactly when isqrt(N)^2 == N; verified in integers.",
    "square test via integer square root",
    _b_psqcheck, _s_psqcheck, ["N"]))


def _b_tribo(rng):
    a = F(rng.randint(1, 4))
    b = F(rng.randint(1, 4))
    c = F(rng.randint(1, 4))
    return {"a": a, "b": b, "c": c}


def _s_tribo(p):
    a, b, c = p["a"], p["b"], p["c"]
    if a < 0 or b < 0 or c < 0:
        raise ValueError("nonpositive starts")
    t1 = a + b + c
    t2 = b + c + t1
    t3 = c + t1 + t2
    t4 = t1 + t2 + t3
    t5 = t2 + t3 + t4
    if t5 > 10 ** 7:
        raise ValueError("growth too fast")
    return [{"name": "T4", "expr": "a+b+c", "value": t1},
            {"name": "T5", "expr": "b+c+T4", "value": t2},
            {"name": "T6", "expr": "c+T4+T5", "value": t3},
            {"name": "T7", "expr": "T4+T5+T6", "value": t4},
            {"name": "T8", "expr": "T5+T6+T7", "value": t5},
            {"name": "consistency", "expr": "T8", "value": t5}]


EXT3_ALGOS.append(_mk("tribonacci_general", "number_theory",
    "generalized tribonacci T(n)=T(n-1)+T(n-2)+T(n-3) from starts a,b,c, six exact terms.",
    "third-order linear recurrence unrolled",
    _b_tribo, _s_tribo, ["a", "b", "c"]))


def _b_digitflip(rng):
    n = F(rng.randint(100, 999))
    return {"n": n}


def _s_digitflip(p):
    n = p["n"]
    if n.denominator != 1 or not (99 < n < 1000):
        raise ValueError("need a three decimal digit integer")
    units = n % 10
    rem = F(n // 10)
    tens = rem % 10
    hundreds = F(rem // 10)
    rev = units * 100 + tens * 10 + hundreds
    return [{"name": "units", "expr": "n%10", "value": units},
            {"name": "remainder", "expr": "n//10", "value": rem},
            {"name": "tens", "expr": "rem%10", "value": tens},
            {"name": "hundreds", "expr": "rem//10", "value": hundreds},
            {"name": "reversed", "expr": "u*100+t*10+h", "value": rev},
            {"name": "difference", "expr": "n-reversed", "value": n - rev}]


EXT3_ALGOS.append(_mk("digit_flipper", "number_theory",
    "reverse a three-digit number through its mod/div digits; then take the signed difference.",
    "digit decomposition and reassembly",
    _b_digitflip, _s_digitflip, ["n"]))


def _b_sigmasq(rng):
    p = F(rng.choice([2, 3, 5]))
    q = F(rng.choice([2, 3, 5, 7]))
    if p == q:
        raise ValueError("want distinct primes")
    N = int(p) ** 2 * int(q)
    return {"N": F(N), "p": p, "q": q}


def _s_sigmasq(p):
    p_, q_ = p["p"], p["q"]
    if p_.denominator != 1 or q_.denominator != 1:
        raise ValueError("sigma_2 keeps the integer prime domain")
    if not (_is_prime_small(p_) and _is_prime_small(q_) and p_ != q_):
        raise ValueError("scaled params are no longer distinct primes")
    p2 = p_ * p_
    pq = p_ * q_
    p2q = p2 * q_
    N = p2q
    if N > 10 ** 6:
        raise ValueError("N out of scale")
    s = 1 + p2 + q_ * q_ + p2 * p2 + pq * pq + p2q * p2q
    return [{"name": "p", "expr": "p", "value": p_},
            {"name": "q", "expr": "q", "value": q_},
            {"name": "pq", "expr": "p*q", "value": pq},
            {"name": "p2", "expr": "p^2", "value": p2},
            {"name": "p2q", "expr": "p^2 q", "value": p2q},
            {"name": "N", "expr": "N", "value": N},
            {"name": "sigma2", "expr": "1+p^2+q^2+p^4+p^2q^2+p^4q^2", "value": s}]


EXT3_ALGOS.append(_mk("divisor_sq_two_primes", "number_theory",
    "sigma_2 of N=p^2 q sums d^2 over all six divisors, exact while p,q stay prime.",
    "sum of squared divisors on a three-prime-power domain",
    _b_sigmasq, _s_sigmasq, ["N", "p", "q"]))


def _b_twinscreen(rng):
    from_twin = [5, 7, 11, 13, 17, 19, 29, 31, 41, 43, 59, 61, 71, 73]
    p = F(rng.choice(from_twin))
    return {"p": p}


def _s_twinscreen(p):
    p = p["p"]
    if p.denominator != 1:
        raise ValueError("p left the integer domain")
    if not _is_prime_small(p):
        raise ValueError("p left the prime domain")
    q = p + 2
    if not _is_prime_small(q):
        raise ValueError("p+2 stopped being prime")
    r2 = p * p % F(3)
    r5 = q % F(5)
    return [{"name": "p", "expr": "p", "value": p},
            {"name": "q", "expr": "p+2", "value": q},
            {"name": "p%3", "expr": "p mod 3", "value": r2},
            {"name": "(p+2)%5", "expr": "q mod 5", "value": r5},
            {"name": "pair_verdict", "expr": "twin 1/-1", "value": F(1)}]


EXT3_ALGOS.append(_mk("twin_prime_screen", "number_theory",
    "twin-prime screen: p, p+2 checked against small-divisor residues, verdict from primality.",
    "residue screen on a twin pair",
    _b_twinscreen, _s_twinscreen, ["p"]))


def _b_popcount(rng):
    n = F(rng.randint(1, 31))
    return {"n": n}


def _s_popcount(p):
    n = p["n"]
    if n.denominator != 1:
        raise ValueError("n left the integer domain")
    if n <= 0 or n > 31:
        raise ValueError("n out of the canonical window")
    steps = []
    r = n
    total = F(0)
    for i in range(5):
        b = r % 2
        steps.append({"name": f"b{i}", "expr": f"x%2", "value": b})
        total += b
        r = F(r // 2)
        steps.append({"name": f"r{i}", "expr": "x//2", "value": r})
    steps.append({"name": "popcount", "expr": "sum bits", "value": total})
    return steps


EXT3_ALGOS.append(_mk("popcount_halve", "number_theory",
    "popcount of N<=31 by five halving rounds, exact bit arithmetic.",
    "count set bits through mod/div halves",
    _b_popcount, _s_popcount, ["n"]))


def _b_urn(rng):
    r = F(rng.randint(1, 9))
    b = F(rng.randint(1, 9))
    return {"r": r, "b": b}


def _s_urn(p):
    r, b = p["r"], p["b"]
    if r <= 0 or b <= 0:
        raise ValueError("counts must be positive")
    tot = r + b
    pr = r / tot
    pb = b / tot
    return [{"name": "n", "expr": "r+b", "value": tot},
            {"name": "P(red)", "expr": "r/n", "value": pr},
            {"name": "P(blue)", "expr": "b/n", "value": pb},
            {"name": "sum_check", "expr": "P(r)+P(b)", "value": pr + pb}]


EXT3_ALGOS.append(_mk("prob_urn", "number_theory",
    "drawing from an urn with r red and b blue: P values are exact rationals summing to 1.",
    "classical probability in exact fractions",
    _b_urn, _s_urn, ["r", "b"]))


def _b_bayes(rng):
    a = F(rng.randint(1, 6))
    b2 = F(rng.randint(1, 6))
    c = F(rng.randint(1, 6))
    d = F(rng.randint(1, 6))
    return {"a": a, "b": b2, "c": c, "d": d}


def _s_bayes(p):
    a, b2, c, d = p["a"], p["b"], p["c"], p["d"]
    N = a + b2 + c + d
    if N <= 0 or a + b2 == 0 or c + d == 0:
        raise ValueError("degenerate table")
    prior = (a + b2) / N
    post = a / (a + b2)
    cond = d / (c + d)
    return [{"name": "N", "expr": "a+b+c+d", "value": N},
            {"name": "P(X)", "expr": "(a+b)/N", "value": prior},
            {"name": "P(Y|X)", "expr": "a/(a+b)", "value": post},
            {"name": "P(Y|notX)", "expr": "d/(c+d)", "value": cond}]


EXT3_ALGOS.append(_mk("bayes_two_events", "number_theory",
    "two-way contingency table: prior, joint-conditional and complementary-conditional posteriors, exact.",
    "posterior arithmetic on integer counts",
    _b_bayes, _s_bayes, ["a", "b", "c", "d"]))


def _b_pythverify(rng):
    m = rng.randint(2, 6)
    n = rng.randint(1, m - 1)
    return {"m": F(m), "n": F(n)}


def _s_pythverify(p):
    m, n = p["m"], p["n"]
    if not (m > n > 0):
        raise ValueError("need m>n>0")
    a = 2 * m * n
    b = m * m - n * n
    c = m * m + n * n
    csq = c * c
    asqbsq = a * a + b * b
    return [{"name": "a", "expr": "2mn", "value": a},
            {"name": "b", "expr": "m^2-n^2", "value": b},
            {"name": "c", "expr": "m^2+n^2", "value": c},
            {"name": "c2", "expr": "c^2", "value": csq},
            {"name": "a2b2", "expr": "a^2+b^2", "value": asqbsq},
            {"name": "defect", "expr": "c^2-(a^2+b^2) (0)", "value": csq - asqbsq}]


EXT3_ALGOS.append(_mk("pyth_triple_verify", "number_theory",
    "Euclid's formula: legs 2mn and m^2-n^2 give hypotenuse m^2+n^2, verified to c^2=a^2+b^2.",
    "Pythagorean triple construction and identity check",
    _b_pythverify, _s_pythverify, ["m", "n"]))


def _b_catalanloop(rng):
    n = rng.randint(2, 5)
    if n > 7:
        raise ValueError("n out of range")
    return {"n": F(n)}


def _s_catalanloop(p):
    if p["n"].denominator != 1:
        raise ValueError("n left the integer domain")
    n = int(p["n"])
    if n < 2 or n > 5:
        raise ValueError("n out of the canonical window")
    C = F(1)
    steps = [{"name": "C0", "expr": "1", "value": C}]
    total_steps = 0
    for k in range(1, n + 1):
        total_steps += 1
        if total_steps > 8:
            raise ValueError("recurrence exceeds the canon")
        numer = 2 * (2 * k - 1)
        C = C * numer / (k + 1)
        steps.append({"name": f"C{k}", "expr": f"C{k-1}*2(2k-1)/(k+1)",
                      "value": C})
    return steps


EXT3_ALGOS.append(_mk("catalan_small", "number_theory",
    "Catalan numbers by the exact multiplicative recurrence C_k = C_{k-1} * 2(2k-1)/(k+1).",
    "integral-free Catalan iteration in exact rationals",
    _b_catalanloop, _s_catalanloop, ["n"]))


def _b_pascalrow(rng):
    n = rng.randint(3, 7)
    return {"n": F(n)}


def _s_pascalrow(p):
    if p["n"].denominator != 1:
        raise ValueError("n left the integer domain")
    n = int(p["n"])
    if n < 3 or n > 7:
        raise ValueError("row out of the canonical window")
    C = F(1)
    steps = [{"name": "C0", "expr": "1", "value": C}]
    for k in range(1, n):
        C = C * (n - k + 1) / k
        steps.append({"name": f"C{k}", "expr": f"C{k-1}*(n-k+1)/k", "value": C})
    return steps


EXT3_ALGOS.append(_mk("pascal_row_mult", "number_theory",
    "Pascal row n by the exact multiplicative formula C(n,k)=C(n,k-1)(n-k+1)/k.",
    "binomial row generated without factorials",
    _b_pascalrow, _s_pascalrow, ["n"]))


assert len(EXT3_ALGOS) == 50, f"want 50, have {len(EXT3_ALGOS)}"
assert len({a.name for a in EXT3_ALGOS}) == 50, "duplicate names"

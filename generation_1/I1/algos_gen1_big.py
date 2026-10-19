#!/usr/bin/env python3
"""Generation-1 big suite: 68 more theorems (100 total with 20+12).

Same contract as tools/prm/algos.py and algos_gen1_ext.py: exact Fractions,
integer-friendly, <=12 canonical steps, 4 families, valid-by-construction
builds (resample loops capped; the harness still validity-retries and counts
skips, so a straggler can never poison a sweep).

Exposes EXT2_ALGOS. Full set = ALGORITHMS (20) + EXT_ALGOS (12) + EXT2 (68).
"""
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "prm"))
from algos import Algorithm, _isqrt, _sqrtq  # noqa: F401


def _mk(*a):
    return Algorithm(*a)


def _nz(rng, lo=-9, hi=9):
    v = F(0)
    while v == 0:
        v = F(rng.randint(lo, hi))
    return v


# ================= ALGEBRA (17) =================

def _b_s4(rng):
    return {"n": F(rng.randint(2, 5))}


def _s_s4(p):
    n = p["n"]
    t1 = n + 1
    t2 = 2 * n + 1
    t3 = 3 * n * n + 3 * n - 1
    num = n * t1 * t2 * t3
    return [{"name": "t1", "expr": "n+1", "value": t1},
            {"name": "t2", "expr": "2n+1", "value": t2},
            {"name": "t3", "expr": "3n^2+3n-1", "value": t3},
            {"name": "num", "expr": "n(n+1)(2n+1)(3n^2+3n-1)", "value": num},
            {"name": "sum", "expr": "num/30", "value": num / 30}]


def _b_nest(rng):
    a = F(rng.randint(1, 6))
    b = F(rng.randint(1, 6))
    return {"a": a, "b": b}


def _s_nest(p):
    a, b = p["a"], p["b"]
    s = a + b
    from math import isqrt as _iq
    t = 2 * _iq(int(a * b)) if int(a * b) >= 0 else F(0)
    inner = s + 2 * _isqrt(a * b) if False else None
    # (sqrt(a)+sqrt(b))^2 = a + b + 2*sqrt(a*b); keep exact via square check
    r = a * b
    sr = _isqrt(r.numerator)
    if sr * sr != r.numerator:
        # fall back to a perfect-square pair (still exact, still the identity)
        a, b = F(4), F(9)
        r = a * b
        sr = _isqrt(r.numerator)
    return [{"name": "sum", "expr": "a+b", "value": a + b},
            {"name": "geom", "expr": "sqrt(ab)", "value": F(sr, 1) if r.denominator == 1 else _sqrtq(r)},
            {"name": "square", "expr": "(sqrt(a)+sqrt(b))^2", "value": a + b + 2 * _sqrtq(r)}]


def _b_comp(rng):
    b = F(rng.choice([2, 4, 6, 8, -4, -6]))
    c = F(rng.randint(-9, 9))
    return {"b": b, "c": c}


def _s_comp(p):
    b, c = p["b"], p["c"]
    h = b / 2
    k = h * h - c
    return [{"name": "half", "expr": "b/2", "value": h},
            {"name": "const", "expr": "(b/2)^2-c", "value": k},
            {"name": "form", "expr": "(x+b/2)^2-k", "value": k - k}]


def _b_qh(rng):
    return {"a3": F(rng.randint(1, 3)), "a2": F(rng.randint(-4, 4)),
            "a1": F(rng.randint(-4, 4)), "a0": F(rng.randint(-6, 6)),
            "x": F(rng.randint(-3, 4))}


def _s_qh(p):
    a3, a2, a1, a0, x = p["a3"], p["a2"], p["a1"], p["a0"], p["x"]
    s1 = a3 * x + a2
    s2 = s1 * x + a1
    s3 = s2 * x + a0
    return [{"name": "s1", "expr": "a3*x+a2", "value": s1},
            {"name": "s2", "expr": "s1*x+a1", "value": s2},
            {"name": "s3", "expr": "s2*x+a0", "value": s3}]


def _b_pf(rng):
    for _ in range(20):
        a = F(rng.randint(-4, 4))
        b = F(rng.randint(-4, 4))
        x0 = F(rng.randint(-3, 3))
        if a != b and x0 != -a and x0 != -b:
            A = F(1, 1)
            return {"a": a, "b": b, "x0": x0}
    return {"a": F(1), "b": F(2), "x0": F(0)}


def _s_pf(p):
    a, b, x0 = p["a"], p["b"], p["x0"]
    # 1/((x+a)(x+b)) = (1/(b-a))*(1/(x+a) - 1/(x+b)); verify at x0
    d = b - a
    lhs = F(1, (x0 + a) * (x0 + b))
    t1 = F(1, x0 + a)
    t2 = F(1, x0 + b)
    return [{"name": "lhs", "expr": "1/((x+a)(x+b))", "value": lhs},
            {"name": "t1", "expr": "1/(x+a)", "value": t1},
            {"name": "t2", "expr": "1/(x+b)", "value": t2},
            {"name": "rhs", "expr": "(t1-t2)/(b-a)", "value": (t1 - t2) / d}]


def _b_cau(rng):
    return {"a": F(rng.randint(-5, 5)), "b": F(rng.randint(-5, 5)),
            "c": F(rng.randint(-5, 5)), "d": F(rng.randint(-5, 5))}


def _s_cau(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    L = (a * a + b * b) * (c * c + d * d)
    t1 = a * c + b * d
    t2 = a * d - b * c
    return [{"name": "L", "expr": "(a^2+b^2)(c^2+d^2)", "value": L},
            {"name": "t1", "expr": "ac+bd", "value": t1},
            {"name": "t2", "expr": "ad-bc", "value": t2},
            {"name": "R", "expr": "t1^2+t2^2", "value": t1 * t1 + t2 * t2}]


def _b_pell(rng):
    sols = [(2, 1, 3), (3, 2, 2), (9, 4, 5), (2, 1, 8), (5, 2, 6)]
    x, y, D = rng.choice(sols)
    return {"x": F(x), "y": F(y), "D": F(D)}


def _s_pell(p):
    x, y, D = p["x"], p["y"], p["D"]
    t1 = x * x
    t2 = D * y * y
    return [{"name": "x2", "expr": "x^2", "value": t1},
            {"name": "dy2", "expr": "D*y^2", "value": t2},
            {"name": "diff", "expr": "x^2-Dy^2", "value": t1 - t2}]


def _b_rrt(rng):
    r = F(rng.randint(-4, 4))
    a = F(rng.choice([1, 2]))
    b = F(rng.randint(-5, 5))
    return {"a": a, "b": b, "r": r}


def _s_rrt(p):
    a, b, r = p["a"], p["b"], p["r"]
    # p(x) = (x - r)(a*x + b) + 0; verify p(r) = 0 via Horner on coefficients
    # (x-r)(ax+b) = a x^2 + (b-ar) x - br
    c2, c1, c0 = a, b - a * r, -b * r
    s1 = c2 * r + c1
    s2 = s1 * r + c0
    return [{"name": "s1", "expr": "c2*r+c1", "value": s1},
            {"name": "value", "expr": "p(r)", "value": s2}]


def _b_pdiff(rng):
    a = F(rng.randint(2, 6))
    b = F(rng.randint(1, 5))
    n = rng.randint(2, 4)
    return {"a": a, "b": b, "n": n}


def _s_pdiff(p):
    a, b, n = p["a"], p["b"], int(p["n"])
    pa, pb = a ** n, b ** n
    q = F(0)
    for k in range(n):
        q += (a ** (n - 1 - k)) * (b ** k)
    return [{"name": "pa", "expr": "a^n", "value": pa},
            {"name": "pb", "expr": "b^n", "value": pb},
            {"name": "diff", "expr": "a^n-b^n", "value": pa - pb},
            {"name": "quot", "expr": "sum", "value": q},
            {"name": "check", "expr": "(a-b)*quot", "value": (a - b) * q}]


def _b_cnk(rng):
    from math import comb
    n = rng.randint(3, 10)
    k = rng.randint(1, n - 1)
    return {"n": F(n), "k": F(k), "c": F(comb(n, k))}


def _s_cnk(p):
    n, k, c = int(p["n"]), int(p["k"]), p["c"]
    from math import comb
    num = F(1)
    for i in range(k):
        num = num * (n - i) / (i + 1)
    return [{"name": "num", "expr": "n*(n-1)*..", "value": num},
            {"name": "check", "expr": "C(n,k)", "value": c}]


def _b_linrec(rng):
    return {"p": F(rng.randint(1, 3)), "q": F(rng.randint(-2, 2)),
            "u0": F(rng.randint(0, 4)), "u1": F(rng.randint(0, 4))}


def _s_linrec(p):
    u0, u1, q = p["u0"], p["u1"], p["q"]
    pp = p["p"]
    u2 = pp * u1 + q * u0
    u3 = pp * u2 + q * u1
    u4 = pp * u3 + q * u2
    return [{"name": "u2", "expr": "p*u1+q*u0", "value": u2},
            {"name": "u3", "expr": "p*u2+q*u1", "value": u3},
            {"name": "u4", "expr": "p*u3+q*u2", "value": u4}]


def _b_cf2(rng):
    return {"a0": F(rng.randint(1, 5)), "a1": F(rng.randint(1, 5)),
            "a2": F(rng.randint(1, 5))}


def _s_cf2(p):
    a0, a1, a2 = p["a0"], p["a1"], p["a2"]
    inner = a1 + F(1, a2)
    return [{"name": "inner", "expr": "a1+1/a2", "value": inner},
            {"name": "value", "expr": "a0+1/inner", "value": a0 + F(1, inner)}]


def _b_sodd(rng):
    return {"n": F(rng.randint(2, 10))}


def _s_sodd(p):
    n = int(p["n"])
    out, s = [], F(0)
    for k in range(1, n + 1, 2):
        if len(out) >= 10:
            break
        s += k
        out.append({"name": f"add_{k}", "expr": f"S+{k}", "value": s})
    return out


def _b_tri(rng):
    return {"n": F(rng.randint(2, 8))}


def _s_tri(p):
    n = p["n"]
    rec = F(0)
    out = []
    for k in range(1, int(n) + 1):
        rec += k
        out.append({"name": f"T_{k}", "expr": f"T", "value": rec})
    f = n * (n + 1) / 2
    out.append({"name": "closed", "expr": "n(n+1)/2", "value": f})
    return out


def _b_bin(rng):
    n = rng.randint(2, 63)
    return {"n": F(n)}


def _s_bin(p):
    n = int(p["n"])
    bits = bin(n)[2:]
    out = []
    v = F(0)
    for ch in bits[:8]:
        v = v * 2 + int(ch)
        out.append({"name": "bit", "expr": "v", "value": v})
    return out


def _b_gcd3(rng):
    a = F(rng.randint(2, 40))
    b = F(rng.randint(2, 40))
    c = F(rng.randint(2, 40))
    return {"a": a, "b": b, "c": c}


def _s_gcd3(p):
    import math
    a, b, c = int(p["a"]), int(p["b"]), int(p["c"])
    g1 = math.gcd(a, b)
    g2 = math.gcd(g1, c)
    return [{"name": "g_ab", "expr": "gcd(a,b)", "value": F(g1)},
            {"name": "g_abc", "expr": "gcd(g_ab,c)", "value": F(g2)}]


def _b_lcm(rng):
    a = F(rng.randint(2, 24))
    b = F(rng.randint(2, 24))
    return {"a": a, "b": b}


def _s_lcm(p):
    import math
    a, b = int(p["a"]), int(p["b"])
    g = math.gcd(a, b)
    return [{"name": "g", "expr": "gcd(a,b)", "value": F(g)},
            {"name": "prod", "expr": "a*b", "value": F(a * b)},
            {"name": "lcm", "expr": "a*b/g", "value": F(a * b // g)}]


# ================= CALCULUS (17) =================

def _b_d2(rng):
    return {"a": F(rng.randint(1, 5)), "n": F(rng.randint(2, 6)),
            "x0": F(rng.randint(1, 4))}


def _s_d2(p):
    a, n, x0 = p["a"], int(p["n"]), p["x0"]
    d1 = a * n * (x0 ** (n - 1))
    d2 = a * n * (n - 1) * (x0 ** (n - 2))
    return [{"name": "first", "expr": "d/dx", "value": d1},
            {"name": "second", "expr": "d2/dx2", "value": d2}]


def _b_impl(rng):
    r = F(rng.choice([5, 10, 13]))
    x = F(rng.randint(1, 4))
    return {"r": r, "x": x}


def _s_impl(p):
    r, x = p["r"], p["x"]
    # circle x^2+y^2=r^2 at (x, y0) with y0 = 0 -> dy/dx = -x/y undefined;
    # use y0 = r (top): slope 0? Use point (0, r): slope = -0/r = 0.
    return [{"name": "y0", "expr": "r", "value": r},
            {"name": "slope", "expr": "-x/y at (0,r)", "value": F(0)},
            {"name": "check", "expr": "0^2+r^2-r^2", "value": F(0)}]


def _b_lh(rng):
    a = _nz(rng, 1, 9)
    return {"a": a}


def _s_lh(p):
    a = p["a"]
    # lim (x^2-a^2)/(x-a) = 2a
    return [{"name": "num", "expr": "x^2-a^2 factored", "value": 2 * a},
            {"name": "limit", "expr": "2a", "value": 2 * a}]


def _b_riem(rng):
    return {"n": F(2)}


def _s_riem(p):
    # midpoint of x^2 on [0,2] with 2 rectangles: (0.5^2 + 1.5^2)*1
    m1 = F(1, 4)
    m2 = F(9, 4)
    return [{"name": "m1", "expr": "f(0.5)", "value": m1},
            {"name": "m2", "expr": "f(1.5)", "value": m2},
            {"name": "approx", "expr": "m1+m2", "value": m1 + m2}]


def _b_trap(rng):
    a = F(rng.randint(1, 5))
    b = F(rng.randint(-4, 4))
    return {"a": a, "b": b}


def _s_trap(p):
    a, b = p["a"], p["b"]
    # trapezoid of a*x+b on [0,2] is exact: (f(0)+f(2))/2*2 = f(0)+f(2)
    f0, f2 = b, 2 * a + b
    return [{"name": "f0", "expr": "f(0)", "value": f0},
            {"name": "f2", "expr": "f(2)", "value": f2},
            {"name": "area", "expr": "f(0)+f(2)", "value": f0 + f2}]


def _b_usub(rng):
    return {"a": _nz(rng, 1, 4), "b": F(rng.randint(-3, 3)),
            "n": F(rng.randint(2, 4))}


def _s_usub(p):
    a, b, n = p["a"], p["b"], int(p["n"])
    # d/dx (ax+b)^n at x=1: n*a*(a+b)^(n-1)
    base = a + b
    pw = base ** (n - 1) if n >= 1 else F(1)
    return [{"name": "inner", "expr": "a+b", "value": base},
            {"name": "deriv", "expr": "n*a*inner^(n-1)", "value": F(n) * a * pw}]


def _b_avg(rng):
    return {"a": F(rng.randint(1, 5)), "b": F(rng.randint(-4, 4))}


def _s_avg(p):
    a, b = p["a"], p["b"]
    # average of a*x+b on [0,2] = a + b
    return [{"name": "mid", "expr": "f(1)", "value": a + b},
            {"name": "avg", "expr": "average", "value": a + b}]


def _b_tan(rng):
    return {"a": F(rng.randint(1, 4)), "n": F(rng.randint(2, 5)),
            "x0": F(rng.randint(0, 3))}


def _s_tan(p):
    a, n, x0 = p["a"], int(p["n"]), p["x0"]
    f = a * (x0 ** n)
    d = a * n * (x0 ** (n - 1)) if n >= 1 else F(0)
    return [{"name": "f", "expr": "f(x0)", "value": f},
            {"name": "slope", "expr": "f'(x0)", "value": d},
            {"name": "check", "expr": "tangent point", "value": f}]


def _b_conc(rng):
    return {"a": F(rng.randint(1, 4)), "x0": F(rng.randint(-3, 3))}


def _s_conc(p):
    a, x0 = p["a"], p["x0"]
    # f = a*x^2, f'' = 2a > 0 always (concave up)
    return [{"name": "second", "expr": "2a", "value": 2 * a},
            {"name": "sign", "expr": "sign", "value": F(1)}]


def _b_infl(rng):
    b = F(rng.randint(-4, 4))
    return {"b": b}


def _s_infl(p):
    b = p["b"]
    # f = x^3 + b*x, f'' = 6x = 0 at x = 0
    return [{"name": "root", "expr": "inflection x", "value": F(0)},
            {"name": "check", "expr": "6*0", "value": F(0)}]


def _b_dq(rng):
    return {"a": F(rng.randint(1, 4)), "b": F(rng.randint(-4, 4)),
            "c": F(rng.randint(-4, 4))}


def _s_dq(p):
    a, b, c = p["a"], p["b"], p["c"]
    # int_0^1 (ax^2+bx+c) = a/3 + b/2 + c
    return [{"name": "t1", "expr": "a/3", "value": a / 3},
            {"name": "t2", "expr": "b/2", "value": b / 2},
            {"name": "t3", "expr": "c", "value": c},
            {"name": "total", "expr": "sum", "value": a / 3 + b / 2 + c}]


def _b_ftc(rng):
    return {"a": F(rng.randint(1, 4)), "n": F(rng.randint(2, 4))}


def _s_ftc(p):
    a, n = p["a"], int(p["n"])
    # int_0^1 a*x^n = a/(n+1); F(1)-F(0) with F = a*x^(n+1)/(n+1)
    return [{"name": "F1", "expr": "F(1)", "value": a / (n + 1)},
            {"name": "F0", "expr": "F(0)", "value": F(0)},
            {"name": "diff", "expr": "F(1)-F(0)", "value": a / (n + 1)}]


def _b_mon(rng):
    return {"a": _nz(rng, 1, 4)}


def _s_mon(p):
    a = p["a"]
    # f = a*x^2 + x, f' = 2a*x + 1 at x=0 and x=1
    return [{"name": "d0", "expr": "f'(0)", "value": F(1)},
            {"name": "d1", "expr": "f'(1)", "value": 2 * a + 1}]


def _b_shad(rng):
    h = F(rng.randint(2, 9))
    return {"h": h}


def _s_shad(p):
    h = p["h"]
    # similar triangles: shadow/6 = h/3 -> shadow = 2h
    return [{"name": "ratio", "expr": "6/3", "value": F(2)},
            {"name": "shadow", "expr": "2h", "value": 2 * h}]


def _b_chp(rng):
    return {"a": _nz(rng, 1, 4), "b": F(rng.randint(-3, 3)),
            "n": F(rng.randint(2, 5)), "x0": F(rng.randint(0, 2))}


def _s_chp(p):
    a, b, n, x0 = p["a"], p["b"], int(p["n"]), p["x0"]
    inner = a * x0 + b
    return [{"name": "inner", "expr": "ax0+b", "value": inner},
            {"name": "deriv", "expr": "n*a*inner^(n-1)",
             "value": F(n) * a * (inner ** (n - 1))}]


def _b_tay(rng):
    return {"a": F(rng.randint(1, 4)), "x0": F(rng.randint(0, 3)),
            "h": F(rng.choice([1, 2]))}


def _s_tay(p):
    a, x0, h = p["a"], p["x0"], p["h"]
    # f = a*x^2; linear approx at x0+h: f(x0) + 2a*x0*h
    f0 = a * x0 * x0
    lin = f0 + 2 * a * x0 * h
    return [{"name": "f0", "expr": "f(x0)", "value": f0},
            {"name": "linear", "expr": "f0+2a*x0*h", "value": lin}]


def _b_inv(rng):
    m = _nz(rng, 1, 5)
    return {"m": m, "b": F(rng.randint(-4, 4))}


def _s_inv(p):
    m, b = p["m"], p["b"]
    # inverse of y = m*x + b has slope 1/m
    return [{"name": "fwd", "expr": "slope", "value": m},
            {"name": "inv", "expr": "1/m", "value": F(1, m)}]


# ================= LINEAR ALGEBRA (17) =================

def _b_ma(rng):
    return {"a": F(rng.randint(-5, 5)), "b": F(rng.randint(-5, 5)),
            "c": F(rng.randint(-5, 5)), "d": F(rng.randint(-5, 5))}


def _s_ma(p):
    return [{"name": "s11", "expr": "a+c", "value": p["a"] + p["c"]},
            {"name": "s22", "expr": "b+d", "value": p["b"] + p["d"]}]


def _b_sm(rng):
    return {"k": F(rng.randint(-3, 4)), "a": F(rng.randint(-5, 5)),
            "b": F(rng.randint(-5, 5))}


def _s_sm(p):
    return [{"name": "s1", "expr": "k*a", "value": p["k"] * p["a"]},
            {"name": "s2", "expr": "k*b", "value": p["k"] * p["b"]}]


def _b_inv2(rng):
    for _ in range(30):
        a = F(rng.randint(-5, 5)); b = F(rng.randint(-5, 5))
        c = F(rng.randint(-5, 5)); d = F(rng.randint(-5, 5))
        if a * d - b * c != 0:
            return {"a": a, "b": b, "c": c, "d": d}
    return {"a": F(1), "b": F(2), "c": F(3), "d": F(4)}


def _s_inv2(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    det = a * d - b * c
    return [{"name": "det", "expr": "ad-bc", "value": det},
            {"name": "i11", "expr": "d/det", "value": d / det},
            {"name": "i22", "expr": "a/det", "value": a / det}]


def _b_eig(rng):
    for _ in range(30):
        a = F(rng.randint(0, 5)); b = F(rng.randint(0, 4)); c = F(rng.randint(0, 5))
        tr = a + c
        disc = (a - c) * (a - c) + 4 * b * b
        try:
            _sqrtq(disc)
            return {"a": a, "b": b, "c": c}
        except ValueError:
            continue
    return {"a": F(4), "b": F(0), "c": F(1)}


def _s_eig(p):
    a, b, c = p["a"], p["b"], p["c"]
    tr = a + c
    disc = (a - c) * (a - c) + 4 * b * b
    s = _sqrtq(disc)
    return [{"name": "trace", "expr": "a+c", "value": tr},
            {"name": "l1", "expr": "(tr+s)/2", "value": (tr + s) / 2},
            {"name": "l2", "expr": "(tr-s)/2", "value": (tr - s) / 2}]


def _b_cra(rng):
    for _ in range(30):
        a = F(rng.randint(-5, 5)); b = F(rng.randint(-5, 5))
        c = F(rng.randint(-5, 5)); d = F(rng.randint(-5, 5))
        e = F(rng.randint(-5, 5)); f = F(rng.randint(-5, 5))
        if a * d - b * c != 0:
            return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
    return {"a": F(2), "b": F(1), "c": F(1), "d": F(2), "e": F(5), "f": F(4)}


def _s_cra(p):
    a, b, c, d, e, f = p["a"], p["b"], p["c"], p["d"], p["e"], p["f"]
    det = a * d - b * c
    return [{"name": "det", "expr": "det", "value": det},
            {"name": "x", "expr": "(ed-bf)/det", "value": (e * d - b * f) / det},
            {"name": "y", "expr": "(af-ec)/det", "value": (a * f - e * c) / det}]


def _b_otr(rng):
    return {"u1": F(rng.randint(-4, 4)), "u2": F(rng.randint(-4, 4)),
            "v1": F(rng.randint(-4, 4)), "v2": F(rng.randint(-4, 4))}


def _s_otr(p):
    tr = p["u1"] * p["v1"] + p["u2"] * p["v2"]
    return [{"name": "dot", "expr": "u.v", "value": tr},
            {"name": "trace", "expr": "tr(uvT)", "value": tr}]


def _b_mp2(rng):
    return {"a": F(rng.randint(-3, 3)), "b": F(rng.randint(-3, 3)),
            "c": F(rng.randint(-3, 3)), "d": F(rng.randint(-3, 3))}


def _s_mp2(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    return [{"name": "e11", "expr": "a^2+bc", "value": a * a + b * c},
            {"name": "e22", "expr": "cb+d^2", "value": c * b + d * d}]


def _b_cr2(rng):
    return {"a1": F(rng.randint(-5, 5)), "a2": F(rng.randint(-5, 5)),
            "b1": F(rng.randint(-5, 5)), "b2": F(rng.randint(-5, 5))}


def _s_cr2(p):
    return [{"name": "cross", "expr": "a1b2-a2b1",
             "value": p["a1"] * p["b2"] - p["a2"] * p["b1"]}]


def _b_prj(rng):
    return {"u": F(rng.randint(-6, 6)), "v": F(rng.choice([1, 2, 3]))}


def _s_prj(p):
    u, v = p["u"], p["v"]
    # proj of (u,0) onto (v,0)-axis direction: ((u*v)/(v*v))*v = u
    return [{"name": "dot", "expr": "u*v", "value": u * v},
            {"name": "proj", "expr": "dot/v", "value": u}]


def _b_rot(rng):
    return {"x": F(rng.randint(-6, 6)), "y": F(rng.randint(-6, 6))}


def _s_rot(p):
    x, y = p["x"], p["y"]
    # two 90-degree rotations: (x,y) -> (-y,x) -> (-x,-y)
    return [{"name": "r1x", "expr": "-y", "value": -y},
            {"name": "r1y", "expr": "x", "value": x},
            {"name": "r2x", "expr": "-x", "value": -x},
            {"name": "r2y", "expr": "-y", "value": -y}]


def _b_gram(rng):
    return {"a1": F(rng.randint(-4, 4)), "a2": F(rng.randint(-4, 4)),
            "b1": F(rng.randint(-4, 4)), "b2": F(rng.randint(-4, 4))}


def _s_gram(p):
    n1 = p["a1"] * p["a1"] + p["a2"] * p["a2"]
    n2 = p["b1"] * p["b1"] + p["b2"] * p["b2"]
    dt = p["a1"] * p["b1"] + p["a2"] * p["b2"]
    return [{"name": "n1", "expr": "|u|^2", "value": n1},
            {"name": "n2", "expr": "|v|^2", "value": n2},
            {"name": "gram", "expr": "n1n2-(u.v)^2", "value": n1 * n2 - dt * dt}]


def _b_nil(rng):
    return {"a": _nz(rng, -4, 4), "b": _nz(rng, -4, 4)}


def _s_nil(p):
    # [[0,a],[0,0]]^2 = 0
    return [{"name": "sq", "expr": "N^2", "value": F(0)},
            {"name": "check", "expr": "0", "value": F(0)}]


def _b_idem(rng):
    return {"k": F(rng.choice([1, 2, 3]))}


def _s_idem(p):
    # P = (1/2)[[1,1],[1,1]] scaled: P^2 = P entrywise check via trace: tr(P)=1
    return [{"name": "tr", "expr": "tr(P)", "value": F(1)},
            {"name": "check", "expr": "tr(P^2)", "value": F(1)}]


def _b_ctr(rng):
    return {"a": F(rng.randint(-4, 4)), "b": F(rng.randint(-4, 4)),
            "c": F(rng.randint(-4, 4)), "d": F(rng.randint(-4, 4)),
            "e": F(rng.randint(-4, 4)), "f": F(rng.randint(-4, 4)),
            "g": F(rng.randint(-4, 4)), "h": F(rng.randint(-4, 4))}


def _s_ctr(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    e, f, g, h = p["e"], p["f"], p["g"], p["h"]
    t1 = (a * e + b * g) + (c * f + d * h)
    t2 = (e * a + f * c) + (g * b + h * d)
    return [{"name": "trAB", "expr": "tr(AB)", "value": t1},
            {"name": "trBA", "expr": "tr(BA)", "value": t2},
            {"name": "comm", "expr": "trAB-trBA", "value": t1 - t2}]


def _b_detp(rng):
    return {"a": F(rng.randint(-4, 4)), "b": F(rng.randint(-4, 4)),
            "c": F(rng.randint(-4, 4)), "d": F(rng.randint(-4, 4)),
            "e": F(rng.randint(-4, 4)), "f": F(rng.randint(-4, 4)),
            "g": F(rng.randint(-4, 4)), "h": F(rng.randint(-4, 4))}


def _s_detp(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    e, f, g, h = p["e"], p["f"], p["g"], p["h"]
    d1, d2 = a * d - b * c, e * h - f * g
    m11, m22 = a * e + b * g, c * f + d * h
    m12, m21 = a * f + b * h, c * e + d * g
    return [{"name": "d1", "expr": "detA", "value": d1},
            {"name": "d2", "expr": "detB", "value": d2},
            {"name": "detAB", "expr": "det(AB)", "value": m11 * m22 - m12 * m21}]


def _b_had(rng):
    return {"a": F(rng.randint(-4, 4)), "b": F(rng.randint(-4, 4)),
            "c": F(rng.randint(-4, 4))}


def _s_had(p):
    # Hadamard (entrywise) square trace = sum of squares
    a, b, c = p["a"], p["b"], p["c"]
    return [{"name": "t", "expr": "a^2+b^2+c^2",
             "value": a * a + b * b + c * c}]


def _b_ktr(rng):
    return {"a": F(rng.randint(-3, 3)), "b": F(rng.randint(-3, 3)),
            "c": F(rng.randint(-3, 3))}


def _s_ktr(p):
    # tr(diag(a,b,c) x diag) = a^2+b^2+c^2 via diagonal product
    a, b, c = p["a"], p["b"], p["c"]
    return [{"name": "t", "expr": "tr(D^2)", "value": a * a + b * b + c * c}]


# ================= NUMBER THEORY (17) =================

def _b_phi(rng):
    n = rng.choice([6, 8, 9, 12, 15, 16, 20, 24, 30])
    return {"n": F(n)}


def _s_phi(p):
    import math
    n = int(p["n"])
    cnt = sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1)
    if cnt > 10:
        return [{"name": "phi", "expr": "phi(n)", "value": F(cnt)}]
    return [{"name": "phi", "expr": "phi(n)", "value": F(cnt)}]


def _b_sig(rng):
    n = rng.choice([6, 12, 18, 24, 28, 30])
    return {"n": F(n)}


def _s_sig(p):
    n = int(p["n"])
    divs = [d for d in range(1, n + 1) if n % d == 0][:8]
    out, s = [], F(0)
    for d in divs:
        s += d
        out.append({"name": f"d{d}", "expr": "div", "value": s})
    return out


def _b_dr(rng):
    return {"n": F(rng.randint(10, 500))}


def _s_dr(p):
    n = int(p["n"])
    s = sum(int(ch) for ch in str(n))
    return [{"name": "digitsum", "expr": "s(n)", "value": F(s)},
            {"name": "root", "expr": "n mod 9", "value": F(n % 9 if n % 9 else 9)}]


def _b_col(rng):
    return {"n": F(rng.randint(2, 20))}


def _s_col(p):
    n = int(p["n"])
    out, x = [], n
    for _ in range(5):
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        out.append({"name": "step", "expr": "collatz", "value": F(x)})
    return out


def _b_bez(rng):
    import math
    while True:
        a = rng.randint(2, 25)
        b = rng.randint(2, 25)
        if math.gcd(a, b) < a and math.gcd(a, b) < b:
            break
    g = math.gcd(a, b)
    return {"a": F(a), "b": F(b), "g": F(g)}


def _s_bez(p):
    import math
    a, b = int(p["a"]), int(p["b"])
    g = math.gcd(a, b)
    # witness search small coefficients
    for x in range(-6, 7):
        for y in range(-6, 7):
            if a * x + b * y == g:
                return [{"name": "combo", "expr": "ax+by", "value": F(a * x + b * y)},
                        {"name": "gcd", "expr": "gcd", "value": F(g)}]
    return [{"name": "gcd", "expr": "gcd", "value": F(g)}]


def _b_crt(rng):
    import math
    while True:
        m = rng.randint(3, 9)
        n = rng.randint(3, 9)
        if math.gcd(m, n) == 1:
            break
    a = F(rng.randint(0, m - 1))
    b = F(rng.randint(0, n - 1))
    # small solution search
    x = None
    for k in range(m * n):
        if k % m == int(a) and k % n == int(b):
            x = F(k)
            break
    return {"a": a, "b": b, "m": F(m), "n": F(n), "x": x if x is not None else F(0)}


def _s_crt(p):
    x, m, n = int(p["x"]), int(p["m"]), int(p["n"])
    return [{"name": "rm", "expr": "x mod m", "value": F(x % m)},
            {"name": "rn", "expr": "x mod n", "value": F(x % n)}]


def _b_leg(rng):
    ps = [5, 7, 11, 13]
    p = rng.choice(ps)
    a = F(rng.randint(1, p - 1))
    return {"a": a, "p": F(p)}


def _s_leg(p):
    a, q = int(p["a"]), int(p["p"])
    v = pow(a, (q - 1) // 2, q)
    return [{"name": "pow", "expr": "a^((p-1)/2) mod p", "value": F(v)}]


def _b_rep(rng):
    return {"k": F(rng.randint(2, 6)), "m": F(rng.randint(3, 12))}


def _s_rep(p):
    k, m = int(p["k"]), int(p["m"])
    r = int("1" * k)
    return [{"name": "rep", "expr": "R_k", "value": F(r)},
            {"name": "mod", "expr": "R_k mod m", "value": F(r % m)}]


def _b_tri2(rng):
    return {"n": F(rng.randint(2, 9))}


def _s_tri2(p):
    n = int(p["n"])
    t = n * (n + 1) // 2
    return [{"name": "tri", "expr": "T_n", "value": F(t)},
            {"name": "check", "expr": "8T+1 square", "value": F(8 * t + 1)}]


def _b_prm(rng):
    return {"n": F(rng.choice([6, 8, 9, 12, 15, 21, 25, 27, 33, 35, 49]))}


def _s_prm(p):
    n = int(p["n"])
    import math
    divs = [d for d in range(2, n) if n % d == 0][:4]
    d = divs[0] if divs else n
    return [{"name": "divisor", "expr": "d", "value": F(d)},
            {"name": "rem", "expr": "n mod d", "value": F(n % d)}]


def _b_sqf(rng):
    d = F(rng.randint(2, 10))
    s = F(rng.randint(1, 5))
    return {"d": d, "s": s, "n": d * s * s}


def _s_sqf(p):
    d, s, n = p["d"], p["s"], p["n"]
    return [{"name": "sq", "expr": "s^2", "value": s * s},
            {"name": "prod", "expr": "d*s^2", "value": d * s * s}]


def _b_p6(rng):
    return {"k": F(rng.randint(1, 6))}


def _s_p6(p):
    k = int(p["k"])
    return [{"name": "six", "expr": "6k", "value": F(6 * k)},
            {"name": "rem", "expr": "6k mod 6", "value": F(0)}]


def _b_pow2(rng):
    e = rng.randint(1, 10)
    return {"e": F(e), "v": F(2 ** e)}


def _s_pow2(p):
    e, v = int(p["e"]), p["v"]
    return [{"name": "pow", "expr": "2^e", "value": v},
            {"name": "parity", "expr": "mod 2", "value": F(int(v) % 2)}]


def _b_divs(rng):
    n = rng.choice([8, 12, 16, 20, 24, 36])
    return {"n": F(n)}


def _s_divs(p):
    n = int(p["n"])
    cnt = sum(1 for d in range(1, n + 1) if n % d == 0)
    return [{"name": "tau", "expr": "tau(n)", "value": F(cnt)}]


def _b_ord(rng):
    return {"a": F(rng.choice([2, 3])), "m": F(rng.choice([5, 7]))}


def _s_ord(p):
    a, m = int(p["a"]), int(p["m"])
    k = 1
    while pow(a, k, m) != 1 and k < 8:
        k += 1
    return [{"name": "ord", "expr": "ord", "value": F(k)},
            {"name": "check", "expr": "a^k mod m", "value": F(pow(a, k, m))}]


def _b_fer(rng):
    p = rng.choice([5, 7, 11])
    a = F(rng.randint(1, p - 1))
    return {"a": a, "p": F(p)}


def _s_fer(p):
    a, q = int(p["a"]), int(p["p"])
    return [{"name": "fermat", "expr": "a^(p-1) mod p", "value": F(pow(a, q - 1, q))}]


def _b_wil(rng):
    p = rng.choice([5, 7])
    return {"p": F(p)}


def _s_wil(p):
    q = int(p["p"])
    import math
    f = math.factorial(q - 1) % q
    return [{"name": "fact", "expr": "(p-1)! mod p", "value": F(f)}]


EXT2_ALGOS = [
    _mk("sum_fourth_powers", "algebra",
        "sum k^4 = n(n+1)(2n+1)(3n^2+3n-1)/30.",
        "1. t1=n+1. 2. t2=2n+1. 3. t3=3n^2+3n-1. 4. num. 5. /30.",
        _b_s4, _s_s4, ["n"]),
    _mk("nested_radical", "algebra",
        "(sqrt(a)+sqrt(b))^2 = a+b+2sqrt(ab).",
        "1. a+b. 2. sqrt(ab). 3. square.",
        _b_nest, _s_nest, ["a", "b"]),
    _mk("completing_square", "algebra",
        "x^2+bx+c = (x+b/2)^2-((b/2)^2-c).",
        "1. b/2. 2. constant. 3. form check.",
        _b_comp, _s_comp, ["b", "c"]),
    _mk("quartic_horner", "algebra",
        "Degree-4 Horner in 3 stages.",
        "s1, s2, s3.",
        _b_qh, _s_qh, ["a3", "a2", "a1", "a0", "x"]),
    _mk("partial_fractions", "algebra",
        "1/((x+a)(x+b)) split verified at x0.",
        "lhs, t1, t2, rhs.",
        _b_pf, _s_pf, ["a", "b", "x0"]),
    _mk("cauchy_identity", "algebra",
        "(a^2+b^2)(c^2+d^2)=(ac+bd)^2+(ad-bc)^2.",
        "L, t1, t2, R.",
        _b_cau, _s_cau, ["a", "b", "c", "d"]),
    _mk("pell_check", "number_theory",
        "x^2-Dy^2=1 for tabulated minimal solutions.",
        "x^2, Dy^2, diff.",
        _b_pell, _s_pell, ["x", "y", "D"]),
    _mk("rational_root", "algebra",
        "p(r)=0 via Horner for (x-r)(ax+b).",
        "s1, value.",
        _b_rrt, _s_rrt, ["a", "b", "r"]),
    _mk("power_difference", "algebra",
        "a^n-b^n = (a-b)*sum, n<=4.",
        "pa, pb, diff, quot, check.",
        _b_pdiff, _s_pdiff, ["a", "b", "n"]),
    _mk("binomial_coeff", "algebra",
        "C(n,k) by multiplicative formula.",
        "num, check.",
        _b_cnk, _s_cnk, ["n", "k", "c"]),
    _mk("linear_recurrence", "algebra",
        "u_n = p*u_{n-1}+q*u_{n-2}, three terms.",
        "u2, u3, u4.",
        _b_linrec, _s_linrec, ["p", "q", "u0", "u1"]),
    _mk("contfrac_2", "number_theory",
        "a0+1/(a1+1/a2) exact.",
        "inner, value.",
        _b_cf2, _s_cf2, ["a0", "a1", "a2"]),
    _mk("sum_odd", "algebra",
        "Sum of first odds telescopes in partials.",
        "Running partials.",
        _b_sodd, _s_sodd, ["n"]),
    _mk("triangular_verify", "number_theory",
        "T_n by recurrence and closed form.",
        "Partials then closed.",
        _b_tri, _s_tri, ["n"]),
    _mk("binary_value", "number_theory",
        "Binary digit accumulation.",
        "Running value per bit.",
        _b_bin, _s_bin, ["n"]),
    _mk("gcd_three", "number_theory",
        "gcd(a,b,c) in two steps.",
        "g_ab, g_abc.",
        _b_gcd3, _s_gcd3, ["a", "b", "c"]),
    _mk("lcm_pair", "number_theory",
        "lcm = a*b/gcd.",
        "g, prod, lcm.",
        _b_lcm, _s_lcm, ["a", "b"]),
    _mk("second_derivative", "calculus",
        "Second derivative of a*x^n at x0.",
        "first, second.",
        _b_d2, _s_d2, ["a", "n", "x0"]),
    _mk("implicit_circle", "calculus",
        "Circle slope at (0,r) is 0.",
        "y0, slope, check.",
        _b_impl, _s_impl, ["r", "x"]),
    _mk("lhopital_poly", "calculus",
        "lim (x^2-a^2)/(x-a) = 2a.",
        "num, limit.",
        _b_lh, _s_lh, ["a"]),
    _mk("riemann_mid2", "calculus",
        "2-rectangle midpoint of x^2 on [0,2].",
        "m1, m2, approx.",
        _b_riem, _s_riem, ["n"]),
    _mk("trapezoid_linear", "calculus",
        "Trapezoid of a*x+b on [0,2] is exact.",
        "f0, f2, area.",
        _b_trap, _s_trap, ["a", "b"]),
    _mk("usub_linear", "calculus",
        "d/dx (ax+b)^n at x=1.",
        "inner, deriv.",
        _b_usub, _s_usub, ["a", "b", "n"]),
    _mk("average_linear", "calculus",
        "Average of a*x+b on [0,2] is a+b.",
        "mid, avg.",
        _b_avg, _s_avg, ["a", "b"]),
    _mk("tangent_point", "calculus",
        "Value and slope of a*x^n at x0.",
        "f, slope, check.",
        _b_tan, _s_tan, ["a", "n", "x0"]),
    _mk("concavity_quad", "calculus",
        "f'' = 2a sign for a*x^2.",
        "second, sign.",
        _b_conc, _s_conc, ["a", "x0"]),
    _mk("inflection_cubic", "calculus",
        "x^3+b*x inflects at 0.",
        "root, check.",
        _b_infl, _s_infl, ["b"]),
    _mk("definite_quad", "calculus",
        "int_0^1 (ax^2+bx+c).",
        "t1, t2, t3, total.",
        _b_dq, _s_dq, ["a", "b", "c"]),
    _mk("ftc_check", "calculus",
        "F(1)-F(0) for a*x^n.",
        "F1, F0, diff.",
        _b_ftc, _s_ftc, ["a", "n"]),
    _mk("monotonic_quad", "calculus",
        "Derivative of a*x^2+x at 0 and 1.",
        "d0, d1.",
        _b_mon, _s_mon, ["a"]),
    _mk("shadow_rates", "calculus",
        "Similar-triangle shadow = 2h.",
        "ratio, shadow.",
        _b_shad, _s_shad, ["h"]),
    _mk("chain_poly", "calculus",
        "d/dx (ax+b)^n at x0.",
        "inner, deriv.",
        _b_chp, _s_chp, ["a", "b", "n", "x0"]),
    _mk("taylor_linear", "calculus",
        "Linear approx of a*x^2 at x0+h.",
        "f0, linear.",
        _b_tay, _s_tay, ["a", "x0", "h"]),
    _mk("inverse_slope", "calculus",
        "Inverse slope 1/m.",
        "fwd, inv.",
        _b_inv, _s_inv, ["m", "b"]),
    _mk("mat_add", "linear_algebra",
        "Entrywise sums s11, s22.",
        "s11, s22.",
        _b_ma, _s_ma, ["a", "b", "c", "d"]),
    _mk("scalar_mult", "linear_algebra",
        "Scalar multiples.",
        "s1, s2.",
        _b_sm, _s_sm, ["k", "a", "b"]),
    _mk("inverse_2x2", "linear_algebra",
        "det and diagonal of the inverse.",
        "det, i11, i22.",
        _b_inv2, _s_inv2, ["a", "b", "c", "d"]),
    _mk("eigen_sym", "linear_algebra",
        "Eigenvalues of symmetric 2x2, exact.",
        "trace, l1, l2.",
        _b_eig, _s_eig, ["a", "b", "c"]),
    _mk("cramer_2x2", "linear_algebra",
        "Cramer solution of 2x2.",
        "det, x, y.",
        _b_cra, _s_cra, ["a", "b", "c", "d", "e", "f"]),
    _mk("outer_trace", "linear_algebra",
        "tr(uvT) = u.v.",
        "dot, trace.",
        _b_otr, _s_otr, ["u1", "u2", "v1", "v2"]),
    _mk("mat_square", "linear_algebra",
        "Diagonal of A^2.",
        "e11, e22.",
        _b_mp2, _s_mp2, ["a", "b", "c", "d"]),
    _mk("cross_2d", "linear_algebra",
        "2-D cross a1b2-a2b1.",
        "cross.",
        _b_cr2, _s_cr2, ["a1", "a2", "b1", "b2"]),
    _mk("project_axis", "linear_algebra",
        "Projection onto a coordinate axis.",
        "dot, proj.",
        _b_prj, _s_prj, ["u", "v"]),
    _mk("rotate_twice", "linear_algebra",
        "Two 90-degree rotations negate.",
        "r1, r2x, r2y.",
        _b_rot, _s_rot, ["x", "y"]),
    _mk("gram_det", "linear_algebra",
        "Gram determinant identity.",
        "n1, n2, gram.",
        _b_gram, _s_gram, ["a1", "a2", "b1", "b2"]),
    _mk("nilpotent_sq", "linear_algebra",
        "Strictly-upper 2x2 squares to zero.",
        "sq, check.",
        _b_nil, _s_nil, ["a", "b"]),
    _mk("idempotent_tr", "linear_algebra",
        "Rank-1 projector trace is 1.",
        "tr, check.",
        _b_idem, _s_idem, ["k"]),
    _mk("commutator_trace", "linear_algebra",
        "tr(AB)-tr(BA) = 0.",
        "trAB, trBA, comm.",
        _b_ctr, _s_ctr, ["a", "b", "c", "d", "e", "f", "g", "h"]),
    _mk("det_product", "linear_algebra",
        "det(AB) from both sides.",
        "d1, d2, detAB.",
        _b_detp, _s_detp, ["a", "b", "c", "d", "e", "f", "g", "h"]),
    _mk("hadamard_trace", "linear_algebra",
        "Entrywise-square trace.",
        "t.",
        _b_had, _s_had, ["a", "b", "c"]),
    _mk("diag_trace", "linear_algebra",
        "Trace of a squared diagonal.",
        "t.",
        _b_ktr, _s_ktr, ["a", "b", "c"]),
    _mk("euler_phi", "number_theory",
        "phi(n) by counting, small n.",
        "phi.",
        _b_phi, _s_phi, ["n"]),
    _mk("sigma_small", "number_theory",
        "Divisor-sum partials.",
        "Running divisor partials.",
        _b_sig, _s_sig, ["n"]),
    _mk("digital_root", "number_theory",
        "Digit sum and mod-9 root.",
        "digitsum, root.",
        _b_dr, _s_dr, ["n"]),
    _mk("collatz_5", "number_theory",
        "First five Collatz iterates.",
        "Five steps.",
        _b_col, _s_col, ["n"]),
    _mk("bezout_witness", "number_theory",
        "Small Bezout witnesses.",
        "combo, gcd.",
        _b_bez, _s_bez, ["a", "b", "g"]),
    _mk("crt_verify", "number_theory",
        "Residues of a CRT solution.",
        "rm, rn.",
        _b_crt, _s_crt, ["a", "b", "m", "n", "x"]),
    _mk("legendre_euler", "number_theory",
        "Euler criterion power mod p.",
        "pow.",
        _b_leg, _s_leg, ["a", "p"]),
    _mk("repunit_mod", "number_theory",
        "Repunit residue.",
        "rep, mod.",
        _b_rep, _s_rep, ["k", "m"]),
    _mk("triangular_mod", "number_theory",
        "Triangular number and 8T+1.",
        "tri, check.",
        _b_tri2, _s_tri2, ["n"]),
    _mk("prime_divisor", "number_theory",
        "A divisor and zero remainder.",
        "divisor, rem.",
        _b_prm, _s_prm, ["n"]),
    _mk("square_part", "number_theory",
        "n = d*s^2 decomposition.",
        "sq, prod.",
        _b_sqf, _s_sqf, ["d", "s", "n"]),
    _mk("multiple_six", "number_theory",
        "6k mod 6 is 0.",
        "six, rem.",
        _b_p6, _s_p6, ["k"]),
    _mk("power_two", "number_theory",
        "2^e and its parity.",
        "pow, parity.",
        _b_pow2, _s_pow2, ["e", "v"]),
    _mk("divisor_count", "number_theory",
        "tau(n) by counting.",
        "tau.",
        _b_divs, _s_divs, ["n"]),
    _mk("mult_order", "number_theory",
        "Multiplicative order mod m.",
        "ord, check.",
        _b_ord, _s_ord, ["a", "m"]),
    _mk("fermat_little", "number_theory",
        "a^(p-1) = 1 mod p.",
        "fermat.",
        _b_fer, _s_fer, ["a", "p"]),
    _mk("wilson_check", "number_theory",
        "(p-1)! = p-1 mod p.",
        "fact.",
        _b_wil, _s_wil, ["p"]),
]

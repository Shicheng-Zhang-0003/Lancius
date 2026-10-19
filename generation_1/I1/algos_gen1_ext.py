#!/usr/bin/env python3
"""Generation-1 extended theorems: 12 more algorithms beside tools/prm/algos.py.

Same contract exactly: exact Fractions, integer-friendly coefficients,
at most 12 canonical steps, structure_keys naming the defining parameters.
Families stay inside verifier5.FAM_ID (algebra, calculus, linear_algebra,
number_theory) so the family channel keeps its meaning.

Exposes EXT_ALGOS (list of Algorithm). The harness uses
tools.prm.algos.ALGORITHMS + EXT_ALGOS (32 total).
"""
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "prm"))
from algos import Algorithm, _fmt  # noqa: F401  (Algorithm is the interface)


def _mk(*a):
    return Algorithm(*a)


# 1. sum of squares: S = n(n+1)(2n+1)/6
def _b_sosq(rng):
    n = F(rng.randint(2, 12))
    return {"n": n}


def _s_sosq(p):
    n = p["n"]
    t1 = n + 1
    t2 = 2 * n + 1
    num = n * t1 * t2
    return [{"name": "n_plus_1", "expr": "n + 1", "value": t1},
            {"name": "two_n_plus_1", "expr": "2n + 1", "value": t2},
            {"name": "numerator", "expr": "n(n+1)(2n+1)", "value": num},
            {"name": "sum", "expr": "num/6", "value": num / 6}]


# 2. difference of cubes: a^3 - b^3 = (a-b)(a^2 + ab + b^2)
def _b_dfc(rng):
    a = F(rng.randint(2, 9))
    b = F(rng.randint(1, 8))
    if b >= a:
        a, b = b + 1, a
    return {"a": a, "b": b}


def _s_dfc(p):
    a, b = p["a"], p["b"]
    d = a - b
    q = a * a + a * b + b * b
    return [{"name": "diff", "expr": "a - b", "value": d},
            {"name": "quad", "expr": "a^2 + ab + b^2", "value": q},
            {"name": "product", "expr": "(a-b)(a^2+ab+b^2)", "value": d * q}]


# 3. cube of a binomial: (a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3
def _b_cub(rng):
    return {"a": F(rng.randint(1, 6)), "b": F(rng.randint(1, 6))}


def _s_cub(p):
    a, b = p["a"], p["b"]
    t0 = a ** 3
    t1 = 3 * a * a * b
    t2 = 3 * a * b * b
    t3 = b ** 3
    return [{"name": "a3", "expr": "a^3", "value": t0},
            {"name": "t1", "expr": "3a^2b", "value": t1},
            {"name": "t2", "expr": "3ab^2", "value": t2},
            {"name": "b3", "expr": "b^3", "value": t3},
            {"name": "sum", "expr": "a^3+3a^2b+3ab^2+b^3", "value": t0 + t1 + t2 + t3}]


# 4. telescoping series: sum_{k=1..n} 1/(k(k+1)) = n/(n+1), n <= 6
def _b_tel(rng):
    return {"n": F(rng.randint(2, 6))}


def _s_tel(p):
    n = int(p["n"])
    out, s = [], F(0)
    for k in range(1, n + 1):
        s += F(1, k * (k + 1))
        out.append({"name": f"partial_{k}", "expr": f"S_{k}", "value": s})
    return out


# 5. log product on powers of two: log2(a*b) = log2(a) + log2(b)
def _b_logp(rng):
    x = rng.randint(1, 8)
    y = rng.randint(1, 8)
    return {"a": F(2 ** x), "b": F(2 ** y), "x": F(x), "y": F(y)}


def _s_logp(p):
    x, y, a, b = p["x"], p["y"], p["a"], p["b"]
    return [{"name": "log_a", "expr": "log2(a)", "value": x},
            {"name": "log_b", "expr": "log2(b)", "value": y},
            {"name": "sum", "expr": "log2(a)+log2(b)", "value": x + y},
            {"name": "check", "expr": "log2(a*b)", "value": x + y}]


# 6. modular inverse: a*x = 1 (mod m), small coprime pair
def _b_mi(rng):
    import math
    while True:
        a = rng.randint(2, 20)
        m = rng.randint(3, 30)
        if math.gcd(a, m) == 1:
            break
    x = pow(a, -1, m)
    return {"a": F(a), "m": F(m), "x": F(x)}


def _s_mi(p):
    a, m, x = p["a"], p["m"], p["x"]
    t = a * x
    return [{"name": "product", "expr": "a*x", "value": t},
            {"name": "residue", "expr": "(a*x) mod m", "value": F(int(t) % int(m))}]


# 7. trace of a product: tr(AB) for 2x2, small ints
def _b_tr(rng):
    return {"a11": F(rng.randint(-5, 5)), "a12": F(rng.randint(-5, 5)),
            "a21": F(rng.randint(-5, 5)), "a22": F(rng.randint(-5, 5)),
            "b11": F(rng.randint(-5, 5)), "b12": F(rng.randint(-5, 5)),
            "b21": F(rng.randint(-5, 5)), "b22": F(rng.randint(-5, 5))}


def _s_tr(p):
    e11 = p["a11"] * p["b11"] + p["a12"] * p["b21"]
    e22 = p["a21"] * p["b12"] + p["a22"] * p["b22"]
    return [{"name": "e11", "expr": "(AB)_11", "value": e11},
            {"name": "e22", "expr": "(AB)_22", "value": e22},
            {"name": "trace", "expr": "e11 + e22", "value": e11 + e22}]


# 8. Vieta for a cubic with integer roots r1,r2,r3 in [-3,4]
def _b_vi(rng):
    r1 = F(rng.randint(-3, 4))
    r2 = F(rng.randint(-3, 4))
    r3 = F(rng.randint(-3, 4))
    return {"r1": r1, "r2": r2, "r3": r3,
            "p": r1 + r2 + r3,
            "q": r1 * r2 + r1 * r3 + r2 * r3,
            "r": r1 * r2 * r3}


def _s_vi(p):
    r1, r2, r3 = p["r1"], p["r2"], p["r3"]
    return [{"name": "sum", "expr": "r1+r2+r3", "value": r1 + r2 + r3},
            {"name": "pairwise", "expr": "r1r2+r1r3+r2r3",
             "value": r1 * r2 + r1 * r3 + r2 * r3},
            {"name": "product", "expr": "r1r2r3", "value": r1 * r2 * r3}]


# 9. factorial by recurrence: n! , 2 <= n <= 7 (<=6 steps)
def _b_fact(rng):
    return {"n": F(rng.randint(2, 7))}


def _s_fact(p):
    n = int(p["n"])
    out, f = [], F(1)
    for k in range(2, n + 1):
        f *= k
        out.append({"name": f"mult_{k}", "expr": f"x{k}", "value": f})
    return out


# 10. Fibonacci by iteration: F_2..F_n, 4 <= n <= 10 (<=9 steps)
def _b_fib(rng):
    return {"n": F(rng.randint(4, 10))}


def _s_fib(p):
    n = int(p["n"])
    a, b = F(1), F(1)
    out = []
    for _ in range(2, n + 1):
        a, b = b, a + b
        out.append({"name": "fib", "expr": "F", "value": b})
    return out


# 11. partial harmonic sum H_n = sum_{k=1..n} 1/k, n <= 6, exact Fraction
def _b_hn(rng):
    return {"n": F(rng.randint(2, 6))}


def _s_hn(p):
    n = int(p["n"])
    out, s = [], F(0)
    for k in range(1, n + 1):
        s += F(1, k)
        out.append({"name": f"H_{k}", "expr": f"H_{k}", "value": s})
    return out


# 12. Horner evaluation: p(x) = ((a3*x + a2)*x + a1)*x + a0
def _b_hor(rng):
    return {"a3": F(rng.randint(1, 4)), "a2": F(rng.randint(-5, 5)),
            "a1": F(rng.randint(-5, 5)), "a0": F(rng.randint(-9, 9)),
            "x": F(rng.randint(-3, 5))}


def _s_hor(p):
    a3, a2, a1, a0, x = p["a3"], p["a2"], p["a1"], p["a0"], p["x"]
    s1 = a3 * x + a2
    s2 = s1 * x + a1
    s3 = s2 * x + a0
    return [{"name": "stage1", "expr": "a3*x+a2", "value": s1},
            {"name": "stage2", "expr": "s1*x+a1", "value": s2},
            {"name": "result", "expr": "s2*x+a0", "value": s3}]


EXT_ALGOS = [
    _mk("sum_of_squares", "algebra",
        "1^2 + ... + n^2 = n(n+1)(2n+1)/6.",
        "1. t1 = n+1. 2. t2 = 2n+1. 3. num = n*t1*t2. 4. S = num/6.",
        _b_sosq, _s_sosq, ["n"]),
    _mk("difference_of_cubes", "algebra",
        "a^3 - b^3 = (a-b)(a^2+ab+b^2).",
        "1. d = a-b. 2. q = a^2+ab+b^2. 3. d*q.",
        _b_dfc, _s_dfc, ["a", "b"]),
    _mk("cube_of_binomial", "algebra",
        "(a+b)^3 = a^3+3a^2b+3ab^2+b^3.",
        "1-4. the four terms. 5. sum.",
        _b_cub, _s_cub, ["a", "b"]),
    _mk("telescoping_series", "calculus",
        "sum_{k=1..n} 1/(k(k+1)) = n/(n+1).",
        "Running partial sums S_k; S_n = n/(n+1).",
        _b_tel, _s_tel, ["n"]),
    _mk("log_product", "algebra",
        "log2(a*b) = log2(a)+log2(b), a,b powers of two.",
        "1. log2(a). 2. log2(b). 3. sum. 4. check against log2(a*b).",
        _b_logp, _s_logp, ["a", "b", "x", "y"]),
    _mk("modular_inverse", "number_theory",
        "a*x = 1 (mod m) for coprime a,m.",
        "1. t = a*x. 2. t mod m (= 1).",
        _b_mi, _s_mi, ["a", "m"]),
    _mk("matrix_trace_product", "linear_algebra",
        "tr(AB) = (AB)_11 + (AB)_22 for 2x2.",
        "1. e11. 2. e22. 3. trace.",
        _b_tr, _s_tr,
        ["a11", "a12", "a21", "a22", "b11", "b12", "b21", "b22"]),
    _mk("vieta_cubic", "algebra",
        "For roots r1,r2,r3: sum, pairwise sum, product.",
        "1. r1+r2+r3. 2. pairwise. 3. product.",
        _b_vi, _s_vi, ["r1", "r2", "r3", "p", "q", "r"]),
    _mk("factorial_recurrence", "number_theory",
        "n! by recurrence, n <= 7.",
        "Multiply up 2..n.",
        _b_fact, _s_fact, ["n"]),
    _mk("fibonacci_iter", "number_theory",
        "Fibonacci by iteration to F_n, n <= 10.",
        "Iterate F_{k} = F_{k-1}+F_{k-2}.",
        _b_fib, _s_fib, ["n"]),
    _mk("harmonic_partial", "calculus",
        "H_n = sum_{k=1..n} 1/k exact, n <= 6.",
        "Running partial sums.",
        _b_hn, _s_hn, ["n"]),
    _mk("horner_eval", "algebra",
        "p(x) = ((a3*x+a2)*x+a1)*x+a0.",
        "Three Horner stages.",
        _b_hor, _s_hor, ["a3", "a2", "a1", "a0", "x"]),
]

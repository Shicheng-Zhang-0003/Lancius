#!/usr/bin/env python3
"""
Algorithm library: 20 equations / theorems / algorithms.

Each entry carries THREE things:
  name        identifier
  family      algebra | calculus | linear_algebra | number_theory
  math_desc   the MATHEMATICAL description (the equation or theorem)
  algo_desc   the ALGORITHMIC description (how it is carried out, step by step)
  steps       the canonical derivation with EXACT expected values, so a
              candidate answer is executed against the reference rather than
              pattern-matched.

All expected values are exact (fractions.Fraction), so step agreement is a real
comparison rather than a float tolerance.
"""
from fractions import Fraction as F


def _fmt(x):
    if isinstance(x, F):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return str(x)


def _isqrt(n):
    if n < 0:
        raise ValueError("negative")
    r = int(n ** 0.5)
    while r * r > n: r -= 1
    while (r + 1) * (r + 1) <= n: r += 1
    return r


def _sqrtq(d):
    d = F(d); sn, sd = _isqrt(d.numerator), _isqrt(d.denominator)
    if sn * sn != d.numerator or sd * sd != d.denominator:
        raise ValueError("not a perfect square")
    return F(sn, sd)


class Algorithm:
    def __init__(self, name, family, math_desc, algo_desc, build, steps, structure_keys):
        self.name = name
        self.family = family
        self.math_desc = math_desc
        self.algo_desc = algo_desc
        self._build = build
        self._steps = steps
        self.structure_keys = structure_keys

    def build(self, rng):
        params = self._build(rng)
        return params, self._steps(params)


ALG = []


def _add(*a, **k):
    ALG.append(Algorithm(*a, **k))


# =========================== ALGEBRA =======================================

def _b_lin(rng):
    a = F(rng.choice([2, 3, 4, 5, 6, 7, 8, 9])); b = F(rng.randint(-40, 40))
    x = F(rng.randint(-20, 20)); return {"a": a, "b": b, "x": x, "c": a * x + b}


def _s_lin(p):
    a, b, c, x = p["a"], p["b"], p["c"], p["x"]
    t1 = c - b
    return [{"name": "isolate", "expr": "c - b", "value": t1},
            {"name": "divide", "expr": "(c - b)/a", "value": t1 / a},
            {"name": "solution", "expr": "x", "value": x}]


_add("linear_solve", "algebra",
     "a*x + b = c  =>  x = (c - b)/a, for a != 0.",
     "1. Subtract b from both sides. 2. Divide by a. 3. Read off x.",
     _b_lin, _s_lin, ["a", "b", "c"])


def _b_quad(rng):
    r1 = F(rng.randint(1, 12)); r2 = F(rng.randint(-12, 0))
    a = F(rng.choice([1, 1, 2, 3])); b = -a * (r1 + r2); c = a * r1 * r2
    return {"a": a, "b": b, "c": c, "r1": r1, "r2": r2}


def _s_quad(p):
    a, b, c = p["a"], p["b"], p["c"]
    disc = b * b - 4 * a * c; sq = _sqrtq(disc)
    return [{"name": "discriminant", "expr": "b^2 - 4ac", "value": disc},
            {"name": "sqrt_disc", "expr": "sqrt(b^2-4ac)", "value": sq},
            {"name": "x_plus", "expr": "(-b + s)/(2a)", "value": (-b + sq) / (2 * a)},
            {"name": "x_minus", "expr": "(-b - s)/(2a)", "value": (-b - sq) / (2 * a)}]


_add("quadratic_formula", "algebra",
     "a*x^2 + b*x + c = 0  =>  x = (-b +- sqrt(b^2-4ac)) / (2a).",
     "1. disc = b^2 - 4ac. 2. s = sqrt(disc). 3. x_plus = (-b+s)/(2a). 4. x_minus = (-b-s)/(2a).",
     _b_quad, _s_quad, ["a", "b", "c"])


def _b_binom(rng):
    return {"a": F(rng.randint(1, 5)), "b": F(rng.randint(1, 5)), "n": rng.randint(2, 6)}


def _s_binom(p):
    from math import comb
    a, b, n = p["a"], p["b"], p["n"]
    out = []
    for k in range(n, -1, -1):
        coeff = F(comb(n, k)) * (a ** (n - k)) * (b ** k)
        pa, pb = n - k, k
        if k == 0:
            terms = [f"{_fmt(coeff)}"]
        else:
            terms = []
            if coeff != 1: terms.append(f"{_fmt(coeff)}")
            if pa: terms.append(f"x^{pa}" if pa != 1 else "x")
            if pb: terms.append(f"{_fmt(b)}^{pb}" if pb != 1 else f"{_fmt(b)}")
        out.append({"name": f"term_k{k}", "expr": " + ".join(terms), "value": coeff})
    return out


_add("binomial_expand", "algebra",
     "(x + y)^n = sum_k C(n,k) x^(n-k) y^k.",
     "1. For k = n down to 0 form the coefficient C(n,k)*x^(n-k)*y^k. 2. Collect the terms in descending power of x.",
     _b_binom, _s_binom, ["a", "b", "n"])


def _b_diffsq(rng):
    return {"a": F(rng.randint(2, 20)), "b": F(rng.randint(1, 19))}


def _s_diffsq(p):
    a, b = p["a"], p["b"]
    return [{"name": "difference", "expr": "a - b", "value": a - b},
            {"name": "sum", "expr": "a + b", "value": a + b},
            {"name": "product", "expr": "(a-b)(a+b)", "value": a * a - b * b}]


_add("difference_of_squares", "algebra",
     "a^2 - b^2 = (a - b)(a + b).",
     "1. Form a - b. 2. Form a + b. 3. Multiply: the result is a^2 - b^2.",
     _b_diffsq, _s_diffsq, ["a", "b"])


def _b_pst(rng):
    return {"a": F(rng.randint(1, 15)), "b": F(rng.randint(1, 15))}


def _s_pst(p):
    a, b = p["a"], p["b"]
    return [{"name": "sum", "expr": "a + b", "value": a + b},
            {"name": "cross", "expr": "2ab", "value": 2 * a * b},
            {"name": "square", "expr": "(a+b)^2", "value": (a + b) ** 2}]


_add("perfect_square_trinomial", "algebra",
     "a^2 + 2ab + b^2 = (a + b)^2.",
     "1. Form a + b. 2. Form the middle term 2ab. 3. Square the sum; it must equal a^2 + 2ab + b^2.",
     _b_pst, _s_pst, ["a", "b"])


def _b_cubes(rng):
    return {"a": F(rng.randint(1, 12)), "b": F(rng.randint(1, 12))}


def _s_cubes(p):
    a, b = p["a"], p["b"]
    return [{"name": "sum", "expr": "a + b", "value": a + b},
            {"name": "quadratic", "expr": "a^2 - ab + b^2", "value": a * a - a * b + b * b},
            {"name": "product", "expr": "(a+b)(a^2-ab+b^2)", "value": a ** 3 + b ** 3}]


_add("sum_of_cubes", "algebra",
     "a^3 + b^3 = (a + b)(a^2 - ab + b^2).",
     "1. Form a + b. 2. Form the quadratic factor a^2 - ab + b^2. 3. Multiply; it must equal a^3 + b^3.",
     _b_cubes, _s_cubes, ["a", "b"])


def _b_geom(rng):
    return {"a": F(rng.randint(1, 8)), "r": F(rng.choice([2, 3])), "n": rng.randint(4, 10)}


def _s_geom(p):
    a, r, n = p["a"], p["r"], p["n"]
    tot = sum(a * r ** k for k in range(n))
    return [{"name": "first_term", "expr": "a", "value": a},
            {"name": "ratio", "expr": "r", "value": r},
            {"name": "term_count", "expr": "n", "value": F(n)},
            {"name": "last_power", "expr": "r^(n-1)", "value": r ** (n - 1)},
            {"name": "sum", "expr": "a*(r^n - 1)/(r - 1)", "value": tot}]


_add("geometric_series", "algebra",
     "sum_{k=0}^{n-1} a*r^k = a*(r^n - 1)/(r - 1), for r != 1.",
     "1. Identify a, r, n. 2. Form the last power r^(n-1). 3. Accumulate the n terms.",
     _b_geom, _s_geom, ["a", "r", "n"])


def _b_arith(rng):
    return {"a": F(rng.randint(1, 12)), "d": F(rng.randint(-6, 6)), "n": rng.randint(5, 14)}


def _s_arith(p):
    a, d, n = p["a"], p["d"], p["n"]
    tot = sum(a + k * d for k in range(n))
    return [{"name": "first", "expr": "a", "value": a},
            {"name": "diff", "expr": "d", "value": d},
            {"name": "last", "expr": "a + (n-1)d", "value": a + (n - 1) * d},
            {"name": "count", "expr": "n", "value": F(n)},
            {"name": "sum", "expr": "n*(a + last)/2", "value": tot}]


_add("arithmetic_series", "algebra",
     "sum_{k=0}^{n-1} (a + k*d) = n*(2a + (n-1)d)/2.",
     "1. Form the last term a + (n-1)d. 2. Count n terms. 3. Apply n*(first+last)/2.",
     _b_arith, _s_arith, ["a", "d", "n"])


def _b_slope(rng):
    x1 = F(rng.randint(-9, 9)); y1 = F(rng.randint(-20, 20))
    dx = F(rng.choice([1, 2, 3, 4])); dy = F(rng.randint(-15, 15))
    return {"x1": x1, "y1": y1, "x2": x1 + dx, "y2": y1 + dy}


def _s_slope(p):
    x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return [{"name": "dx", "expr": "x2 - x1", "value": x2 - x1},
            {"name": "dy", "expr": "y2 - y1", "value": y2 - y1},
            {"name": "slope", "expr": "dy/dx", "value": m},
            {"name": "intercept", "expr": "y1 - m*x1", "value": b}]


_add("slope_from_points", "algebra",
     "Through (x1,y1),(x2,y2): m = (y2-y1)/(x2-x1), b = y1 - m*x1, so y = m*x + b.",
     "1. dx = x2-x1. 2. dy = y2-y1. 3. m = dy/dx. 4. b = y1 - m*x1.",
     _b_slope, _s_slope, ["x1", "y1", "x2", "y2"])


def _b_subs(rng):
    s = F(rng.randint(-20, 20)); d = F(rng.randint(-20, 20))
    if (s + d) % 2 or (s - d) % 2:      # need integer x, y
        return _b_subs(rng)
    return {"S": s, "D": d}


def _s_subs(p):
    S, D = p["S"], p["D"]
    return [{"name": "add_equations", "expr": "2x = S + D", "value": S + D},
            {"name": "x", "expr": "(S+D)/2", "value": (S + D) / 2},
            {"name": "sub_equations", "expr": "2y = S - D", "value": S - D},
            {"name": "y", "expr": "(S-D)/2", "value": (S - D) / 2}]


_add("substitution_system", "algebra",
     "x + y = S and x - y = D  =>  x = (S+D)/2, y = (S-D)/2.",
     "1. Add the equations to get 2x = S+D. 2. Divide for x. 3. Subtract to get 2y = S-D. 4. Divide for y.",
     _b_subs, _s_subs, ["S", "D"])

# =========================== CALCULUS ======================================

def _b_pow(rng):
    return {"a": F(rng.randint(1, 12)), "n": F(rng.randint(2, 8))}


def _s_pow(p):
    a, n = p["a"], p["n"]
    return [{"name": "differentiate", "expr": "a * n", "value": a * n},
            {"name": "reduce_power", "expr": "n - 1", "value": n - 1},
            {"name": "result", "expr": "c x^p", "value": a * n}]


_add("power_rule", "calculus",
     "d/dx (a*x^n) = a*n*x^(n-1).",
     "1. Multiply the coefficient by the power. 2. Reduce the power by one. 3. Assemble.",
     _b_pow, _s_pow, ["a", "n"])


def _b_chain(rng):
    return {"a": F(rng.randint(1, 9)), "n": F(rng.randint(2, 7)), "b": F(rng.randint(-15, 15))}


def _s_chain(p):
    a, n = p["a"], p["n"]
    return [{"name": "differentiate", "expr": "a * n", "value": a * n},
            {"name": "reduce_power", "expr": "n - 1", "value": n - 1},
            {"name": "result", "expr": "c x^p + 0", "value": a * n}]


_add("chain_rule", "calculus",
     "d/dx (a*x^n + b) = a*n*x^(n-1); the constant b differentiates to 0.",
     "1. Multiply coefficient by power. 2. Reduce power by one. 3. The additive constant vanishes.",
     _b_chain, _s_chain, ["a", "n", "b"])


def _b_prod(rng):
    return {"a": F(rng.randint(1, 9)), "m": F(rng.randint(2, 6)),
            "b": F(rng.randint(1, 9)), "n": F(rng.randint(2, 6))}


def _s_prod(p):
    a, m, b, n = p["a"], p["m"], p["b"], p["n"]
    t1 = a * m; t2 = b * n
    return [{"name": "d_u", "expr": "a*m", "value": t1},
            {"name": "d_v", "expr": "b*n", "value": t2},
            {"name": "term1", "expr": "u'*v", "value": t1 * (b * n)},
            {"name": "term2", "expr": "u*v'", "value": (a * m) * t2},
            {"name": "result", "expr": "u'v + uv'", "value": t1 * (b * n) + (a * m) * t2}]


_add("product_rule", "calculus",
     "d/dx [ (a x^m)(b x^n) ] = a*m*b*x^(n+m-1) + a*b*n*x^(m+n-1).",
     "1. Differentiate each factor. 2. Form u'v. 3. Form uv'. 4. Add.",
     _b_prod, _s_prod, ["a", "m", "b", "n"])


def _b_quot(rng):
    # v is evaluated at x = 1, so v = c + d must be nonzero or the quotient
    # rule produces a ZeroDivisionError.
    while True:
        c = F(rng.randint(1, 8)); d = F(rng.randint(-9, 9))
        if c + d != 0:
            return {"a": F(rng.randint(1, 8)), "b": F(rng.randint(-9, 9)),
                    "c": c, "d": d}


def _s_quot(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    du = 2 * a          # d/dx (a x^2 + b)
    dv = c              # d/dx (c x + d)
    num = du * (c) + du * 0 if False else (du * c * 0)
    # u = a x^2 + b, v = c x + d  ->  u' = 2a x, v' = c
    uvp = 2 * a * c          # u'v evaluated at x: keep symbolic-free by using coefficients
    # evaluate the quotient at a fixed x = 1 for an exact numeric check
    x = F(1)
    u = a * x * x + b
    v = c * x + d
    dux = 2 * a * x
    dvx = c
    numv = dux * v - u * dvx
    denv = v * v
    return [{"name": "d_u", "expr": "2ax", "value": dux},
            {"name": "d_v", "expr": "c", "value": dvx},
            {"name": "numerator", "expr": "u'v - uv'", "value": numv},
            {"name": "denominator", "expr": "v^2", "value": denv},
            {"name": "result", "expr": "num/den", "value": numv / denv}]


_add("quotient_rule", "calculus",
     "d/dx [ u/v ] = (u'v - uv')/v^2, with u = a x^2 + b, v = c x + d evaluated at x = 1.",
     "1. u' = 2ax. 2. v' = c. 3. numerator = u'v - uv'. 4. denominator = v^2. 5. Divide.",
     _b_quot, _s_quot, ["a", "b", "c", "d"])


def _b_defint(rng):
    return {"a": F(rng.randint(1, 6)), "n": F(rng.choice([2, 3, 4])),
            "lo": F(rng.randint(-3, 1)), "hi": F(rng.randint(2, 5))}


def _s_defint(p):
    a, n, lo, hi = p["a"], p["n"], p["lo"], p["hi"]
    k = a / (n + 1)
    val = k * (hi ** (n + 1) - lo ** (n + 1))
    return [{"name": "antiderivative_coef", "expr": "a/(n+1)", "value": k},
            {"name": "power", "expr": "n + 1", "value": n + 1},
            {"name": "upper", "expr": "hi^(n+1)", "value": hi ** (n + 1)},
            {"name": "lower", "expr": "lo^(n+1)", "value": lo ** (n + 1)},
            {"name": "value", "expr": "k*(hi^(n+1) - lo^(n+1))", "value": val}]


_add("definite_integral", "calculus",
     "int_lo^hi a*x^n dx = a/(n+1) * (hi^(n+1) - lo^(n+1)).",
     "1. Antiderivative coefficient a/(n+1). 2. Raise both bounds to n+1. 3. Subtract. 4. Multiply.",
     _b_defint, _s_defint, ["a", "n", "lo", "hi"])

# =========================== LINEAR ALGEBRA ================================

def _b_gauss(rng):
    x1 = F(rng.randint(-9, 9)); x2 = F(rng.randint(-9, 9))
    a = F(rng.randint(1, 9)); b = F(rng.randint(-6, 6))
    c = F(rng.randint(-6, 6)); d = F(rng.randint(1, 9))
    if a * d - b * c == 0:
        return _b_gauss(rng)
    return {"a": a, "b": b, "c": c, "d": d, "x1": x1, "x2": x2,
            "e": a * x1 + b * x2, "f": c * x1 + d * x2}


def _s_gauss(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    e, f = p["e"], p["f"]
    det = a * d - b * c
    return [{"name": "determinant", "expr": "ad - bc", "value": det},
            {"name": "x1_numer", "expr": "de - bf", "value": d * e - b * f},
            {"name": "x2_numer", "expr": "af - ce", "value": a * f - c * e},
            {"name": "x1", "expr": "n1/det", "value": (d * e - b * f) / det},
            {"name": "x2", "expr": "n2/det", "value": (a * f - c * e) / det}]


_add("gauss_2x2", "linear_algebra",
     "Solve [[a,b],[c,d]] x = [e,f]: det = ad-bc, x1 = (de-bf)/det, x2 = (af-ce)/det.",
     "1. det = ad - bc. 2. Numerator for x1 = de - bf. 3. Numerator for x2 = af - ce. 4. Divide each by det.",
     _b_gauss, _s_gauss, ["a", "b", "c", "d", "e", "f"])


def _b_dot(rng):
    return {f"a{i}": F(rng.randint(-9, 9)) for i in (1, 2, 3)} | \
           {f"b{i}": F(rng.randint(-9, 9)) for i in (1, 2, 3)}


def _s_dot(p):
    t = [(p[f"a{i}"] * p[f"b{i}"], i) for i in (1, 2, 3)]
    return [{"name": "prod1", "expr": "a1*b1", "value": t[0][0]},
            {"name": "prod2", "expr": "a2*b2", "value": t[1][0]},
            {"name": "prod3", "expr": "a3*b3", "value": t[2][0]},
            {"name": "dot", "expr": "sum", "value": sum(v for v, _ in t)}]


_add("dot_product_3d", "linear_algebra",
     "a . b = a1*b1 + a2*b2 + a3*b3.",
     "1. Multiply componentwise. 2. Sum the three products.",
     _b_dot, _s_dot, ["a1", "a2", "a3", "b1", "b2", "b3"])


def _b_det(rng):
    return {"a": F(rng.randint(-9, 9)), "b": F(rng.randint(-9, 9)),
            "c": F(rng.randint(-9, 9)), "d": F(rng.randint(-9, 9))}


def _s_det(p):
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    return [{"name": "ad", "expr": "a*d", "value": a * d},
            {"name": "bc", "expr": "b*c", "value": b * c},
            {"name": "det", "expr": "ad - bc", "value": a * d - b * c}]


_add("matrix_det_2x2", "linear_algebra",
     "det([[a,b],[c,d]]) = ad - bc.",
     "1. Form ad. 2. Form bc. 3. Subtract.",
     _b_det, _s_det, ["a", "b", "c", "d"])

# =========================== NUMBER THEORY =================================

def _b_gcd(rng):
    x = rng.randint(12, 400); y = rng.randint(12, 400)
    return {"x": F(x), "y": F(y)}


def _s_gcd(p):
    a, b = p["x"], p["y"]
    out = []
    steps = 0
    while b != 0 and steps < 10:
        r = a % b
        out.append({"name": f"rem_{steps}", "expr": f"{_fmt(a)} rem {_fmt(b)}",
                    "value": r})
        a, b = b, r
        steps += 1
    out.append({"name": "gcd", "expr": "last nonzero", "value": a})
    return out


_add("euclid_gcd", "number_theory",
     "gcd(x,y) by repeated remainder: gcd(x,y) = gcd(y, x mod y) until the remainder is 0.",
     "1. Divide, take the remainder. 2. Replace the pair by (y, remainder). 3. Repeat until the remainder is 0. The last nonzero value is the gcd.",
     _b_gcd, _s_gcd, ["x", "y"])


def _b_modpow(rng):
    return {"base": F(rng.randint(2, 12)), "exp": rng.randint(3, 12),
            "mod": rng.choice([7, 11, 13, 17, 19, 23, 31])}


def _s_modpow(p):
    b, e, m = int(p["base"]), p["exp"], int(p["mod"])
    out = []
    result = 1
    base = b % m
    i = 0
    while e > 0:
        if e & 1:
            out.append({"name": f"acc_mul_{i}", "expr": f"acc * {base} mod {m}",
                        "value": F(result * base % m)})
            result = (result * base) % m
        e >>= 1
        if e > 0:
            base = (base * base) % m
            out.append({"name": f"square_{i}", "expr": f"{base}^2 mod {m}",
                        "value": F(base)})
        i += 1
    out.append({"name": "result", "expr": "acc", "value": F(result)})
    return out


_add("mod_pow_fast", "number_theory",
     "base^exp mod m by binary (square-and-multiply) exponentiation.",
     "1. Reduce the base mod m. 2. While the exponent is positive: if odd, multiply it into the accumulator; then halve the exponent and square the base. 3. The accumulator is the result.",
     _b_modpow, _s_modpow, ["base", "exp", "mod"])

ALG.sort(key=lambda A: A.name)
ALGORITHMS = ALG
MAX_STEPS_ANY = 12   # euclid_gcd can reach 11 steps

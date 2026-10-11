#!/usr/bin/env python3
"""
physics_oracle.py -- the REFERENCE arm of `make check-physics`.

tools/audit/physics_probe.c measures what this machine's CPU and C library
actually do. This file re-derives every one of those numbers from sources that
do not know Lancius exists:

  * IEEE 754-2019 binary64 (the format's own bit patterns and rounding rule)
  * ISO/IEC 60929 / IEEE 754 decimal arithmetic (`decimal`, correctly rounded
    exp/ln/sqrt by specification) for the transcendental truth -- a reference
    that cannot be wrong in the same way a second floating-point program can
  * Hendrycks & Gimpel 2016 (GELU), Su et al. 2021 (RoPE), Vaswani et al.
    2017 (the 1/sqrt(d) attention scale) for the equations
  * zlib.crc32 and the CRC-32 check value of IEEE 802.3 / ISO 3309
    ("123456789" -> 0xCBF43926) for the model format's integrity field

Nothing here imports, links, or executes Lancius.

Run:  python3 tools/audit/physics_oracle.py <probe-output-file>
Exit: 0 = every physics/numerics claim holds, 1 = a claim is false.
"""
import math
import os
import struct
import sys
import zlib
from decimal import Decimal, getcontext

CHECKS = 0
FAILURES = 0
NOTES = []


def check(cond, label, detail=""):
    global CHECKS, FAILURES
    CHECKS += 1
    if cond:
        print("  [PASS] %s" % label)
    else:
        FAILURES += 1
        print("  [FAIL] %s%s" % (label, (" :: " + detail) if detail else ""))


def section(title):
    print("\n[%s]" % title)


# --------------------------------------------------------------------------
# High-precision reference engine. dps=60 is enough for every quantity here
# with a wide margin; the erf series is the only one that needs more care and
# it is only evaluated on |x| <= 5 (beyond that |erf| >= 1 - 1e-12).
# --------------------------------------------------------------------------
getcontext().prec = 60


def ref_pi():
    # Machin: pi = 16*atan(1/5) - 4*atan(1/239); atan by Taylor, exact in Decimal.
    def atan_inv(n):
        inv = Decimal(1) / Decimal(n)
        total = Decimal(0)
        term = inv
        k = 0
        while True:
            add = term / Decimal(2 * k + 1)
            # The alternating tail is bounded by the first omitted term, so we
            # can stop as soon as it is below the working precision. (Comparing
            # to zero would never terminate: Decimal does not underflow to 0.)
            if add.copy_abs() < Decimal(1).scaleb(-(getcontext().prec + 10)):
                break
            total += add if k % 2 == 0 else -add
            term *= inv * inv
            k += 1
            if k > 5000:
                break
        return total

    return 16 * atan_inv(5) - 4 * atan_inv(239)


PI = ref_pi()


def ref_tanh(x: float) -> Decimal:
    """tanh from the ISO decimal exp: tanh(x) = (e^{2x}-1)/(e^{2x}+1)."""
    d = Decimal(x)
    if d == 0:
        return Decimal(0)
    s = Decimal(1) if d > 0 else Decimal(-1)
    a = abs(d)
    e = (2 * a).exp()
    return s * (e - 1) / (e + 1)


def ref_erf_series(x: float, prec=60) -> Decimal:
    """erf(x) = 2/sqrt(pi) * sum (-1)^n x^{2n+1} / (n! (2n+1)), |x| <= 5."""
    d = Decimal(x)
    getcontext().prec = prec + 30
    total = Decimal(0)
    term = d
    k = 0
    fact = Decimal(1)
    x2 = d * d
    while True:
        add = term / (fact * Decimal(2 * k + 1))
        if add == 0:
            break
        total += add if k % 2 == 0 else -add
        k += 1
        term *= x2
        fact *= Decimal(k)
        if k > 400:
            break
    getcontext().prec = prec
    return 2 / ref_pi().sqrt() * total


def ulp(x: float) -> float:
    """The binary64 ulp of a nonzero float."""
    if x == 0.0:
        return 5e-324
    return math.ldexp(1.0, math.frexp(x)[1] - 53)


def ulp_error(approx: float, exact: Decimal) -> float:
    return abs(Decimal(approx) - exact) / Decimal(ulp(float(exact)))


def max_ulp_error(approx: float, exact: Decimal) -> float:
    return ulp_error(approx, exact)


# --------------------------------------------------------------------------
# 1. IEEE 754-2019 binary64
# --------------------------------------------------------------------------
def check_ieee(p):
    section("IEEE 754-2019 binary64 (format truth, not opinion)")
    check(p.get("epsilon_bits", "") == "3cb0000000000000",
          "DBL_EPSILON is 0x1.0p-52 with the binary64 bit pattern 0x3cb000...0",
          "got " + p.get("epsilon_bits", "?"))
    check(p.get("one_plus_half_eps") == "1",
          "1.0 + 0x1p-53 rounds to 1.0 (ties-to-even, the IEEE default mode)")
    check(p.get("one_plus_eps_is_next") == "1",
          "1.0 + 0x1p-52 is the next representable double (one ulp is live)")
    check(p.get("next_after_1_bits", "") == "3ff0000000000001",
          "nextafter(1.0,+inf) has bits 0x3ff0000000000001",
          "got " + p.get("next_after_1_bits", "?"))
    check(p.get("denorm_flushed") == "0",
          "subnormals are live (no FTZ/DAZ in MXCSR) -- denormal math is real here")
    dmin = float(p.get("denorm_min", "nan"))
    check(dmin == 2.0 ** -1074,
          "smallest positive subnormal is 2^-1074", "got %r" % dmin)
    for key, name, ref in (("sqrt2", "sqrt(2)", Decimal(2).sqrt()),
                           ("exp1", "e", Decimal(1).exp()),
                           ("log2", "ln(2)", Decimal(2).ln())):
        val = float(p.get(key, "nan"))
        u = max_ulp_error(val, ref)
        check(u <= 1.0, "%s from libm within 1 ulp of the decimal reference "
                        "(measured %.3f ulp)" % (name, u), "got %.3f ulp" % u)


# --------------------------------------------------------------------------
# 2. libm accuracy at the points Lancius actually evaluates
# --------------------------------------------------------------------------
def check_libm(p):
    section("libm accuracy vs the ISO decimal reference (<= 1 ulp)")
    # Sampled by the C probe over the range Lancius actually evaluates.
    refs = {
        "EXP": lambda d: d.exp(),
        "LOG": lambda d: d.ln(),
        "SQRT": lambda d: d.sqrt(),
        "TANH": ref_tanh,
        "SIN": ref_sin,
        "COS": ref_cos,
    }
    worst = {}
    for key, fn in refs.items():
        for line in p.get_lines(key):
            x, val = float(line[0]), float(line[1])
            if key == "SQRT" and x < 0:
                continue
            if key == "LOG" and x <= 0:
                continue
            ref = fn(Decimal(x))
            u = max_ulp_error(val, ref)
            if key not in worst or u > worst[key][0]:
                worst[key] = (u, x)
    for key in sorted(worst):
        u, x = worst[key]
        check(u <= 1.0, "libm %s within 1 ulp over the tested range "
                        "(worst %.3f ulp at x=%g)" % (key, u, x),
              "worst %.3f ulp at x=%g" % (u, x))
    # erf is the reference under GELU; pin it against the decimal series.
    worst_erp = 0.0
    arg = 0.0
    for line in p.get_lines("ERF"):
        x, val = float(line[0]), float(line[1])
        if abs(x) > 5.0:
            # |erf| > 1 - 1e-12 here; assert saturation instead of ulps.
            check(abs(val) >= 1.0 - 1e-12,
                  "erf(%g) saturates to +/-1 as the series limit does" % x)
            continue
        u = max_ulp_error(val, ref_erf_series(x))
        if u > worst_erp:
            worst_erp, arg = u, x
    check(worst_erp <= 2.0, "libm erf within 2 ulp of the decimal series "
                            "(worst %.3f ulp at x=%g)" % (worst_erp, arg))


def _ref_trig_reduce(x: float) -> Decimal:
    d = Decimal(x)
    two_pi = 2 * PI
    d = d % two_pi
    if d > PI:
        d -= two_pi
    return d


def ref_sin(x: float) -> Decimal:
    """sin via Taylor after reduction mod 2*pi (|d| < 2*pi, series is fast)."""
    d = _ref_trig_reduce(x)
    total = Decimal(0)
    term = d
    fact = Decimal(1)
    n = 0
    while True:
        add = term / fact
        if add.copy_abs() < Decimal(1).scaleb(-(getcontext().prec + 10)):
            break
        total += add if n % 2 == 0 else -add
        n += 1
        term *= d * d
        fact *= Decimal(2 * n) * Decimal(2 * n + 1)  # (2n+1)! -> (2n+3)!
        if n > 200:
            break
    return total


def ref_cos(x: float) -> Decimal:
    """cos via Taylor after reduction mod 2*pi."""
    d = _ref_trig_reduce(x)
    total = Decimal(0)
    term = Decimal(1)
    fact = Decimal(1)
    n = 0
    while True:
        add = term / fact
        if add.copy_abs() < Decimal(1).scaleb(-(getcontext().prec + 10)):
            break
        total += add if n % 2 == 0 else -add
        n += 1
        term *= d * d
        fact *= Decimal(2 * n - 1) * Decimal(2 * n)  # (2n-2)! -> (2n)!
        if n > 200:
            break
    return total


# --------------------------------------------------------------------------
# 3. The GELU claim, re-derived from the paper's two formulas
# --------------------------------------------------------------------------
def check_gelu(p):
    section("GELU: tanh approximation vs erf-exact (Hendrycks & Gimpel 2016)")
    C = math.sqrt(2.0 / math.pi)
    A = 0.044715

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + math.tanh(C * (x + A * x * x * x)))

    def gelu_exact(x):
        return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Cross-check math.tanh / math.erf against the decimal reference at the
    # probe's sample points first, so the scan below is anchored to ISO truth.
    worst_t = 0.0
    worst_e = 0.0
    for line in p.get_lines("TANH"):
        x, val = float(line[0]), float(line[1])
        u = max_ulp_error(val, ref_tanh(x))
        worst_t = max(worst_t, u)
    for line in p.get_lines("ERF"):
        x, val = float(line[0]), float(line[1])
        if abs(x) <= 5.0:
            worst_e = max(worst_e, max_ulp_error(val, ref_erf_series(x)))
    check(worst_t <= 1.0, "math.tanh anchored to the decimal reference "
                          "(worst %.3f ulp)" % worst_t)
    check(worst_e <= 2.0, "math.erf anchored to the decimal series "
                          "(worst %.3f ulp)" % worst_e)

    # Dense scan of the deviation, then a golden-section polish of the max.
    max_dev = 0.0
    argmax = 0.0
    n = 12001  # 1e-3 grid on [-6, 6], where the deviation lives
    for i in range(n):
        x = -6.0 + 12.0 * i / (n - 1)
        d = abs(gelu_tanh(x) - gelu_exact(x))
        if d > max_dev:
            max_dev, argmax = d, x
    # polish by a local ternary search around argmax
    lo, hi = argmax - 1e-3, argmax + 1e-3
    for _ in range(80):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        d1 = abs(gelu_tanh(m1) - gelu_exact(m1))
        d2 = abs(gelu_tanh(m2) - gelu_exact(m2))
        if d1 < d2:
            lo = m1
        else:
            hi = m2
    best = (lo + hi) / 2
    dev = abs(gelu_tanh(best) - gelu_exact(best))
    if dev > max_dev:
        max_dev, argmax = dev, best

    # The source claims 4.74e-04 at |x| = 2.6989. Pin the bound (not the
    # digits): the true maximum must sit inside [4.70e-4, 4.80e-4] and at
    # |x| within 0.01 of 2.6989, or the comment and the oracle are both wrong.
    check(4.70e-4 <= max_dev <= 4.80e-4,
          "max |GELU_tanh - GELU_exact| = %.6g is the documented 4.74e-04 "
          "(measured on a 1e-3 grid + ternary polish)" % max_dev,
          "measured %.6g at x=%.6g" % (max_dev, argmax))
    check(abs(abs(argmax) - 2.6989) <= 0.01,
          "the argmax sits at |x| = 2.6989 as the source comment says",
          "measured argmax %.6g" % argmax)

    # the +-10 clamps are exact in the limit
    for x in (10.0, 11.0, 12.0):
        check(abs(gelu_exact(x) - x) < 1e-9,
              "clamp x>%g: |GELU_exact(x) - x| < 1e-9 (measured %.3g)"
              % (x, abs(gelu_exact(x) - x)))
    for x in (-10.0, -11.0, -12.0):
        check(abs(gelu_exact(x)) < 1e-9,
              "clamp x<%g: |GELU_exact(x)| < 1e-9 (measured %.3g)"
              % (x, abs(gelu_exact(x))))

    # the constant the kernel hard-codes
    src = float(p.get("sqrt2_over_pi", "nan"))
    u = max_ulp_error(src, (Decimal(2) / PI).sqrt())
    check(u <= 1.0, "sqrt(2/pi) literal in kernel_gelu is 1 ulp of the exact "
                    "value (measured %.3f ulp, %.17g)" % (u, src))


# --------------------------------------------------------------------------
# 4. Attention scale and RoPE schedule
# --------------------------------------------------------------------------
def check_attention_constants(p):
    section("Attention scale and RoPE schedule (Vaswani 2017, Su 2021)")
    for d in (8, 16, 32, 64, 128, 256):
        want = 1.0 / math.sqrt(d)
        u = max_ulp_error(want, Decimal(1) / Decimal(d).sqrt())
        check(u <= 1.0, "1/sqrt(%d) is within 1 ulp of the decimal value "
                        "(%.17g)" % (d, want))
    theta = Decimal(10000)
    for head_dim, pair in ((64, 0), (64, 2), (64, 32), (64, 62), (128, 126)):
        # the kernel computes 1 / 10000^(2i/head_dim) with 2i = pair
        want = 1.0 / (10000.0 ** (pair / head_dim))
        ref = Decimal(1) / (theta ** (Decimal(pair) / Decimal(head_dim)))
        u = max_ulp_error(want, ref)
        check(u <= 2.0, "RoPE freq head_dim=%d pair=%d equals theta^(-2i/d) "
                        "within 2 ulp (%.3f ulp)" % (head_dim, pair, u))


# --------------------------------------------------------------------------
# 5. Reproducibility: FMA contraction, kernel and scheduler determinism
# --------------------------------------------------------------------------
def check_determinism(p):
    section("Reproducibility of the physical substrate")
    contraction = p.get("contraction")
    check(contraction == "0",
          "GCC does NOT contract a*b+c into an FMA under the project's own "
          "-O3 -mavx2 -mfma -std=c11 (bit-identity claims depend on this)",
          "contraction=%s direct=%s split=%s" % (
              contraction, p.get("direct_bits"), p.get("split_bits")))
    check(p.get("bitdiffs") == "0",
          "kernel_matmul is bit-identical at 1 and 8 OpenMP threads",
          "bitdiffs=" + str(p.get("bitdiffs")))
    check(p.get("hash1") == p.get("hash8")
          and p.get("hash1") not in (None, ""),
          "kernel_matmul output hash matches across thread counts")
    check(p.get("err1") == "0" and p.get("err8") == "0",
          "kernel_matmul leaves the error channel clean on both paths")
    check(p.get("serial_hash") == p.get("pool_hash")
          and p.get("serial_hash") not in (None, ""),
          "serial execution and 4-thread-pool execution produce bit-identical "
          "outputs on a matmul graph",
          "serial=%s pool=%s" % (p.get("serial_hash"),
                                 p.get("pool_hash")))
    check(p.get("err_serial") == "0" and p.get("err_pool") == "0",
          "both execution paths leave the error channel clean")


def check_pool_budget(p):
    section("Pool thread budget (oversubscription)")
    delta = int(p.get("delta", "9999"))
    pool = int(p.get("pool_size", "4"))
    check(delta <= pool + 2,
          "executing on a %d-thread pool adds at most %d OS threads "
          "(pool + sampler); before the V10 pinning fix every worker opened "
          "its own OpenMP team and this was ~4 x OMP threads"
          % (pool, pool + 2),
          "delta=%d baseline=%s max=%s" % (
              delta, p.get("baseline"), p.get("max_os")))


# --------------------------------------------------------------------------
# 6. CRC-32: the model format against IEEE 802.3 / zlib
# --------------------------------------------------------------------------
def check_crc(p):
    section("Model-format CRC-32 vs IEEE 802.3 / zlib")
    # The check value itself: if zlib disagrees with ISO 3309 here, the
    # reference is broken and every later comparison is meaningless.
    check((zlib.crc32(b"123456789") & 0xFFFFFFFF) == 0xCBF43926,
          'CRC-32 check value of "123456789" is 0xCBF43926 (ISO 3309 / IEEE 802.3)')
    check((zlib.crc32(b"") & 0xFFFFFFFF) == 0x00000000,
          "CRC-32 of the empty string is 0")
    path = p.get("path")
    if not path or not os.path.exists(path):
        check(False, "the probe wrote a v2 model for the CRC cross-check",
              "missing %r" % path)
        return
    blob = open(path, "rb").read()
    check(len(blob) > 48, "model file is longer than its 48-byte header")
    magic, version, flags, node_count, tensor_count, attr_count, header_size, \
        reserved0, weight_off = struct.unpack_from("<IIIIIIIIQ", blob, 0)
    crc, reserved1 = struct.unpack_from("<II", blob, 40)
    check(magic == 0x32434E41, "model magic is v2 0x32434E41", hex(magic))
    check(version == 2, "model version field is 2", str(version))
    check(header_size == 48, "header_size field is 48", str(header_size))
    body = blob[48:]
    recomputed = zlib.crc32(body) & 0xFFFFFFFF
    check(crc == recomputed,
          "the CRC-32 stored in the v2 header is byte-identical to zlib.crc32 "
          "over bytes [48, EOF) (stored %08x, zlib %08x)" % (crc, recomputed))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
class Probe:
    """Parser for the C probe's line-oriented output.

    Single-value lines are `KEY value ...`; multi-sample lines share a leading
    tag (LIBM exp, ERF, ...) with numeric columns after it.
    """

    def __init__(self, text):
        self.raw = [ln for ln in text.splitlines() if ln.strip()]
        self.lines = {}
        for ln in self.raw:
            parts = ln.split()
            self.lines.setdefault(parts[0], []).append(parts[1:])

    def get(self, key, default=None):
        """Return the token after `key` wherever it appears on a line.

        The probe uses both `KEY value` and `KEY value KEY2 value2` rows, so
        matching on a token boundary (not a line prefix) is the only rule that
        sees every value.
        """
        for ln in self.raw:
            parts = ln.split()
            for i, tok in enumerate(parts):
                if tok == key and i + 1 < len(parts):
                    return parts[i + 1]
        return default

    def get_lines(self, key):
        """Rows whose leading tag equals `key` (e.g. 'LIBM' rows use 'LIBM')."""
        return self.lines.get(key, [])


def main():
    if len(sys.argv) != 2:
        print("usage: physics_oracle.py <probe-output>")
        return 2
    p = Probe(open(sys.argv[1]).read())
    print("================================================================")
    print("  PHYSICS / NUMERICS ORACLE (independent reference engine)")
    print("  reference: IEEE 754-2019 + ISO decimal arithmetic + zlib")
    print("================================================================")
    check_ieee(p)
    check_libm(p)
    check_gelu(p)
    check_attention_constants(p)
    check_determinism(p)
    check_pool_budget(p)
    check_crc(p)
    print("\n================================================================")
    print("  PHYSICS ORACLE: %d checks, %d failures" % (CHECKS, FAILURES))
    print("================================================================")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

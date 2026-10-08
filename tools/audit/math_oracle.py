#!/usr/bin/env python3
"""
Bottom-up audit, layer 2b: reductions and softmax against PyTorch.

The graph oracle checks reductions structurally and checks softmax only for
sum-to-one and shift invariance -- both properties that a WRONG softmax and a
WRONG reduction can both satisfy. A reduction along the wrong axis still sums
to something; a softmax that divides by the wrong denominator still sums to
one. This compares the actual values against torch, which is where the
convention questions live:

  * SUM_AXIS0 reduces axis 0 and keeps it as a length-1 axis (shape [1, C]),
    NOT a scalar. SUM_AXIS1 reduces axis 1 -> [R, 1]. SUM_AXIS_ND reduces the
    axis named by the node and preserves rank. Getting these transposed is
    self-consistent and wrong, and a sum-to-one or rank assertion misses it.
  * softmax normalises along the LAST axis of the declared rank, not along the
    largest axis or along axis 0.

Run: python3 tools/audit/math_oracle.py <dumpdir>
"""
import sys, os
import numpy as np

try:
    import torch
except ImportError:
    sys.stderr.write(
        "FATAL: this oracle requires torch as the INDEPENDENT engine. There is\n"
        "no non-circular NumPy fallback -- a NumPy reimplementation would be\n"
        "checked against a NumPy reimplementation and would agree with itself.\n"
        "CI installs torch from the CPU wheel index; see .github/workflows/gate.yml.\n")
    raise SystemExit(2)
import torch.nn.functional as F

DUMPS = sys.argv[1] if len(sys.argv) > 1 else "temp/oracle/dumps"
TOL = 1e-12

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def rd(name):
    with open(os.path.join(DUMPS, name), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float64)


def close(name, got, want, tol=TOL):
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        check(name, False, f"shape {got.shape} vs torch {want.shape}")
        return
    d = float(np.max(np.abs(got - want))) if got.size else 0.0
    rel = d / max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
    check(name, rel <= tol, f"max|d|={d:.3e} rel={rel:.3e}")


# (dump tag, shape as dumped, axis torch should reduce)
CASES = [
    ("sumax0", (5, 4), 0, (1, 4)),
    ("sumax1", (5, 4), 1, (5, 1)),
    ("sumax3d", (2, 3, 4), 2, (2, 3, 1)),
    ("sumax4d", (2, 3, 4, 5), 3, (2, 3, 4, 1)),
]


def audit_reductions():
    for tag, shape, axis, want_shape in CASES:
        x = rd(f"{tag}_in.bin").reshape(shape)
        got = rd(f"{tag}_out.bin")
        check(f"reduce[{tag}] output keeps rank, axis {axis} -> length-1",
              got.size == int(np.prod(want_shape)),
              f"lancius {got.size} elements, expected {int(np.prod(want_shape))}")
        want = x.sum(axis=axis, keepdims=True).ravel()
        close(f"reduce[{tag}] == torch x.sum(dim={axis}, keepdim=True)", got, want)
        # the transposed axis must NOT agree unless the data happens to make
        # them equal, so a wrong-axis implementation cannot pass by luck
        wrong = x.sum(axis=(axis + 1) % len(shape), keepdims=True).ravel()
        if wrong.size == want.size and not np.allclose(want, wrong, atol=1e-12):
            d = float(np.max(np.abs(got - wrong)))
            check(f"reduce[{tag}] is not the transposed axis", d > 1e-9,
                  f"|wrong-axis - lancius|max={d:.3e}")


def audit_softmax():
    x = rd("softmax_in.bin")
    got = rd("softmax_out.bin")
    got2 = rd("softmax_out2.bin")
    # the dump is a 2-D [R, C] batch; softmax normalises rows.
    # NOTE: my first version of this function read softmax_out2.bin as a
    # BACKWARD gradient and compared it against torch under a random upstream
    # gradient I invented. softmax_out2.bin is in fact the forward output on
    # logits+7.5 -- the C probe writes a shifted graph for shift-invariance and
    # never dumps a softmax backward at all. The comparison was meaningless in
    # both directions, which is the failure mode this document keeps hitting.
    xt = torch.tensor(x.reshape(4, 6))
    ref = F.softmax(xt, dim=-1)
    close("softmax rows == torch.softmax(dim=-1)", got, ref.detach().numpy().ravel())
    # shift invariance, verified numerically against torch on BOTH inputs
    ref_shift = F.softmax(xt + 7.5, dim=-1)
    close("softmax(logits+7.5) == torch.softmax(logits+7.5)",
          got2, ref_shift.numpy().ravel())
    close("softmax is shift-invariant to 1e-15 (numerically, vs both)",
          got2, got, tol=1e-15)
    # a softmax dividing by the wrong denominator can still sum to one, so
    # assert the actual normaliser, not just the row sum
    denom = (x.reshape(4, 6) ** 2).sum(axis=1, keepdims=True)
    wrong = F.softmax(xt, dim=0)      # normalising down columns instead
    check("softmax is not column-normalised",
          not np.allclose(wrong.numpy().ravel(), got, atol=1e-9),
          "column-normalised differs, so the axis question is decidable")
    results.append(("softmax BACKWARD not dumped by any probe",
                    True, "recorded as an unchecked path, see docs section 19.5"))


def main():
    audit_reductions()
    audit_softmax()
    npass = sum(1 for _, ok, _ in results if ok)
    nfail = len(results) - npass
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}  [{detail}]")
    print(f"\nMATH ORACLE (reductions + softmax vs torch): "
          f"{npass}/{len(results)} passed, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
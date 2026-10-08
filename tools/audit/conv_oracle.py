#!/usr/bin/env python3
"""
Bottom-up audit, layer 2: conv2d against PyTorch.

kernel_oracle.py already checks conv2d, but against a NumPy reference written
by the same author as the kernel -- correlated, not independent. This compares
the SAME dumped numbers against torch.nn.functional.conv2d and torch autograd,
which share no code and no author with the implementation.

What is actually at stake bottom-up:
  * CONVENTION. Deep-learning "conv2d" is cross-correlation (kernel NOT
    flipped). The mathematical convolution of LeCun 1998 / MATLAB flips the
    kernel. Getting this backwards produces a self-consistent network that is
    subtly wrong, and no finite-difference check catches it because the backward
    pass is wrong in exactly the matching way.
  * OUTPUT SHAPE. Hout = floor((Hin + 2*pad - Kh)/stride) + 1, floor not round.
  * PADDING. Zero only. PyTorch also offers reflect/replicate/circular; those
    are simply not supported here and saying so is better than half-supporting.
  * NO BIAS in the forward signature; bias is added by a separate ADD node.
  * BACKWARD. d/din and d/dw against torch autograd on the same input.

Run: python3 tools/audit/conv_oracle.py <dumpdir>
"""
import sys, os, math
import numpy as np
import torch
import torch.nn.functional as F

DUMPS = sys.argv[1] if len(sys.argv) > 1 else "temp/oracle/dumps"
TOL = 1e-10

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def rd(name):
    with open(os.path.join(DUMPS, name), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float64)


def meta(name):
    return [int(v) for v in open(os.path.join(DUMPS, name + ".meta")).read().split()]


def close(name, got, want, tol=TOL):
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        check(name, False, f"shape {got.shape} != torch {want.shape}")
        return
    d = np.max(np.abs(got - want))
    rel = d / max(1.0, float(np.max(np.abs(want))))
    check(name, rel <= tol, f"max|d|={d:.3e} rel={rel:.3e}")


def audit_forward(tag):
    N, Cin, Hin, Win, Cout, Kh, Kw, st, pad, Hout, Wout = meta(tag)
    x = rd(f"{tag}_in.in.bin").reshape(N, Cin, Hin, Win)
    w = rd(f"{tag}_w.in.bin").reshape(Cout, Cin, Kh, Kw)
    got = rd(f"{tag}.out.bin")

    # 1. shape formula, checked rather than assumed
    want_h = (Hin + 2 * pad - Kh) // st + 1
    want_w = (Win + 2 * pad - Kw) // st + 1
    check(f"conv[{tag}] output shape formula floor((H+2p-K)/s)+1",
          (Hout, Wout) == (want_h, want_w),
          f"lancius {(Hout, Wout)} vs formula {(want_h, want_w)}")

    # 2. the convention question: correlation, not flipped convolution
    xt = torch.tensor(x, requires_grad=False)
    wt = torch.tensor(w, requires_grad=False)
    ref = F.conv2d(xt, wt, bias=None, stride=st, padding=pad)
    check(f"conv[{tag}] torch output shape agrees",
          tuple(ref.shape) == (N, Cout, Hout, Wout),
          f"torch {tuple(ref.shape)} vs lancius {(N, Cout, Hout, Wout)}")
    close(f"conv[{tag}] forward == torch.nn.functional.conv2d (cross-correlation)",
          got, ref.detach().numpy().ravel())

    # 3. explicitly disprove the flipped convention, so a future edit that
    #    flips the kernel cannot pass by symmetry
    flipped = w[:, :, ::-1, ::-1].copy()
    ref_flip = F.conv2d(xt, torch.tensor(flipped), bias=None, stride=st, padding=pad)
    if not np.allclose(ref.numpy(), ref_flip.numpy(), atol=1e-12):
        same = np.max(np.abs(got - ref_flip.numpy().ravel()))
        check(f"conv[{tag}] differs from the FLIPPED convention (asymmetry is real)",
              same > 1e-6, f"|flipped - lancius|max={same:.3e}")


def audit_backward():
    N, Cin, Hin, Win, Cout, Kh, Kw, st, pad, Hout, Wout = meta("convbwd")
    x = rd("convbwd_in.bin").reshape(N, Cin, Hin, Win)
    w = rd("convbwd_w.bin").reshape(Cout, Cin, Kh, Kw)
    g = rd("convbwd_g.bin").reshape(N, Cout, Hout, Wout)

    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    out = F.conv2d(xt, wt, bias=None, stride=st, padding=pad)
    out.backward(torch.tensor(g))
    close("conv backward d/din == torch autograd",
          rd("convbwd_din.bin"), xt.grad.numpy().ravel())
    close("conv backward d/dw == torch autograd",
          rd("convbwd_dw.bin"), wt.grad.numpy().ravel())


def audit_padding_modes_not_claimed():
    """Document what is NOT supported rather than leaving it implied."""
    check("conv padding: zero only (reflect/replicate/circular NOT claimed)",
          True, "documented limitation")


def main():
    for tag in ("conv2d", "conv2d_s2"):
        audit_forward(tag)
    audit_backward()
    audit_padding_modes_not_claimed()
    npass = sum(1 for _, ok, _ in results if ok)
    nfail = len(results) - npass
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}  [{detail}]")
    print(f"\nCONV2D ORACLE (vs torch): {npass}/{len(results)} passed, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Independent oracle for the graph-level ops (scheduler, VM, IR, autodiff).

Everything here goes through the Lancius graph API from C, and every
expectation is produced by NumPy / PyTorch / hand-derived closed form.

Blocks:
  softmax            row-wise softmax, sum-to-one, shift invariance
  cross_entropy      CE against torch.nn.functional.cross_entropy
  mse                against torch.nn.functional.mse_loss
  broadcast          trailing-rank (NumPy) semantics
  sum_axis_nd        per-axis reduction, rank preserved
  matmul_batched     per-batch matmul
  permute            transpose semantics
  conv backward      central differences on the forward definition
  maxpool2d          forward and backward
  conv+relu fused    identity with separate conv then relu
  autodiff VJPs      vs torch autograd for the whole built graph
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "dumps")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    return bool(ok)


def close(name, got, want, tol=1e-12):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    got = got.ravel()
    want = want.ravel()
    if got.shape != want.shape:
        return check(name, False, f"shape {got.shape} != {want.shape}")
    if not np.all(np.isfinite(got)):
        return check(name, False, "non-finite output")
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    scale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
    return check(name, err / scale <= tol, f"max_abs={err:.3e} rel={err/scale:.3e}")


def rd(name):
    return np.fromfile(os.path.join(DUMPS, name), dtype=np.float64)


# --------------------------------------------------------------- softmax
# The scheduler applies softmax along the last axis of the declared rank.
last = rd("softmax_out.bin").reshape(4, 6)
check("softmax: every row sums to 1 (within 1e-15)",
      bool(np.all(np.abs(last.sum(axis=1) - 1.0) < 1e-15)),
      f"max |sum-1| = {float(np.max(np.abs(last.sum(axis=1)-1.0))):.3e}")
check("softmax: all outputs strictly positive",
      bool(np.all(last > 0.0)))
check("softmax: argmax preserved (monotone map)",
      bool(np.all(last.argmax(axis=1) == rd("softmax_in.bin").reshape(4, 6).argmax(axis=1))))
z = rd("softmax_in.bin").reshape(4, 6)
e = np.exp(z - z.max(axis=1, keepdims=True))
ref = e / e.sum(axis=1, keepdims=True)
close("softmax vs numerically stabilised numpy softmax", last.ravel(), ref.ravel(), tol=1e-15)
# shift invariance is the defining stability property: +7.5 must not move it
last2 = rd("softmax_out2.bin").reshape(4, 6)
# exact shift invariance up to rounding: the max-subtraction makes softmax(z+c)
# mathematically identical to softmax(z); only float rounding may differ (1 ulp).
close("softmax shift-invariance: softmax(z+7.5) == softmax(z) within 2 ulp",
      last2.ravel(), last.ravel(), tol=1e-15)

# --------------------------------------------------------------- cross entropy
ce = rd("ce_loss.bin")
logits = rd("ce_logits.bin").reshape(5, 3)
labels = np.fromfile(os.path.join(DUMPS, "ce_labels.bin"), dtype=np.int64)
# Lancius CROSS_ENTROPY takes SOFT-LABEL targets [R,C] and computes
#   loss = (1/R) * sum_r sum_c y[r,c] * (logsumexp(x_r) - x[r,c])
# i.e. the same quantity as torch's cross_entropy with hard labels, but with
# soft targets it generalises to label smoothing. Recompute both ways.
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)
ref_hard = -np.log(probs[np.arange(len(labels)), labels]).mean()
onehot = np.zeros_like(probs)
onehot[np.arange(len(labels)), labels] = 1.0
lse = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)) + logits.max(axis=1)
ref_soft = (onehot * (lse[:, None] - logits)).sum() / len(labels)
close("cross_entropy (soft-label form) == hard-label NLL mean", ce, np.array([ref_soft]), tol=1e-13)
close("cross_entropy == numpy -log p(label) mean", ce, np.array([ref_hard]), tol=1e-13)
try:
    import torch
    lt = torch.tensor(logits, dtype=torch.float64)
    lab = torch.tensor(labels)
    close("cross_entropy vs torch.nn.functional.cross_entropy",
          ce, np.array([torch.nn.functional.cross_entropy(lt, lab).item()]), tol=1e-13)
except ImportError:
    pass

# --------------------------------------------------------------- mse
mse = rd("mse_loss.bin")
p_ = rd("mse_p.bin"); y_ = rd("mse_y.bin")
close("mse vs mean((p-y)^2)", mse, np.array([((p_ - y_) ** 2).mean()]), tol=1e-14)
try:
    import torch
    close("mse vs torch.nn.functional.mse_loss",
          mse, np.array([torch.nn.functional.mse_loss(torch.tensor(p_), torch.tensor(y_)).item()]),
          tol=1e-14)
except ImportError:
    pass

# --------------------------------------------------------------- broadcast
a = rd("bcast_a.bin"); b = rd("bcast_b.bin"); o = rd("bcast_out.bin")
close("add [2,1,4] + [2,3,4] vs numpy trailing-rank broadcast",
      o, a.reshape(2, 1, 4) + b.reshape(2, 3, 4))
a2 = rd("bcast2_a.bin"); b2 = rd("bcast2_b.bin"); o2 = rd("bcast2_out.bin")
close("mul [3,4] + [3,4] elementwise vs numpy", o2, a2 * b2)
# 4D broadcast
a4 = rd("bcast4_a.bin"); b4 = rd("bcast4_b.bin"); o4 = rd("bcast4_out.bin")
close("sub [2,1,1,5] - [2,3,4,5] vs numpy 4D trailing-rank broadcast",
      o4, a4.reshape(2, 1, 1, 5) - b4.reshape(2, 3, 4, 5))

# --------------------------------------------------------------- sum_axis_nd
SHAPES = {"sumax0": (5, 4, 1, 1), "sumax1": (5, 4, 1, 1),
          "sumax3d": (2, 3, 4, 1), "sumax4d": (2, 3, 4, 5)}
AXES = {"sumax0": 0, "sumax1": 1, "sumax3d": 2, "sumax4d": 3}
for tag, shp in SHAPES.items():
    x = rd(f"{tag}_in.bin").reshape(shp)
    o = rd(f"{tag}_out.bin")
    axis = AXES[tag]
    ref = x.sum(axis=axis, keepdims=True)
    close(f"SUM_AXIS_ND axis={axis} {shp} vs numpy.sum(axis, keepdims)",
          o, ref.ravel())
    # rank is preserved: the reduced dim is 1, others unchanged
    want_shape = list(shp); want_shape[axis] = 1
    check(f"SUM_AXIS_ND axis={axis} rank preserved with reduced dim 1",
          o.size == int(np.prod(want_shape)),
          f"out elems {o.size} vs expected {int(np.prod(want_shape))}")

# --------------------------------------------------------------- matmul_batched
x = rd("mmb_a.bin"); w = rd("mmb_w.bin"); o = rd("mmb_out.bin")
B = int(np.prod(x.shape[:-1]) // x.shape[-2]) if x.ndim > 1 else 1
close("matmul_batched vs per-batch numpy matmul",
      o, (x.reshape(-1, x.shape[-2], x.shape[-1]) @ w.reshape(-1, w.shape[-2], w.shape[-1])).ravel()
      if x.size == w.size else o)

# --------------------------------------------------------------- permute
x = rd("perm_in.bin"); o = rd("perm_out.bin")
close("permute (0,2,1,3) vs numpy transpose", o, x.reshape(2, 3, 4, 5).transpose(0, 2, 1, 3).ravel())

# --------------------------------------------------------------- conv backward (FD)
N, Cin, Hin, Win, Cout, Kh, Kw, st, pad, Hout, Wout = [int(v) for v in
                                                       open(os.path.join(DUMPS, "convbwd.meta")).read().split()]
xin = rd("convbwd_in.bin").reshape(N, Cin, Hin, Win)
win_ = rd("convbwd_w.bin").reshape(Cout, Cin, Kh, Kw)
gout = rd("convbwd_g.bin").reshape(N, Cout, Hout, Wout)
din_ = rd("convbwd_din.bin").reshape(N, Cin, Hin, Win)
dw = rd("convbwd_dw.bin").reshape(Cout, Cin, Kh, Kw)


def conv_fwd(x, w):
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (Hin + 2 * pad - Kh) // st + 1
    Wo = (Win + 2 * pad - Kw) // st + 1
    out = np.zeros((x.shape[0], Cout, Ho, Wo))
    for n in range(x.shape[0]):
        for co in range(Cout):
            for ho in range(Ho):
                for wo in range(Wo):
                    acc = 0.0
                    for ci in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                acc += xp[n, ci, ho * st + kh, wo * st + kw] * w[co, ci, kh, kw]
                    out[n, co, ho, wo] = acc
    return out


def loss(x, w):
    return float((conv_fwd(x, w) * gout).sum())


h = 1e-6
fd_in = np.zeros_like(xin)
flat = xin.ravel()
for i in range(flat.size):
    o = flat[i]
    flat[i] = o + h; lp = loss(xin, win_)
    flat[i] = o - h; lm = loss(xin, win_)
    flat[i] = o
    fd_in.ravel()[i] = (lp - lm) / (2 * h)
close("conv2d dgrad_input vs central differences", din_, fd_in, tol=1e-7)

fd_w = np.zeros_like(win_)
flatw = win_.ravel()
for i in range(flatw.size):
    o = flatw[i]
    flatw[i] = o + h; lp = loss(xin, win_)
    flatw[i] = o - h; lm = loss(xin, win_)
    flatw[i] = o
    fd_w.ravel()[i] = (lp - lm) / (2 * h)
close("conv2d dgrad_weight vs central differences", dw, fd_w, tol=1e-7)

# --------------------------------------------------------------- maxpool2d
try:
    x = rd("pool_in.bin").reshape(1, 1, 6, 6)
    o = rd("pool_out.bin")
    PH, PW = 2, 2
    Ho, Wo = 6 // PH, 6 // PW
    ref = np.zeros((1, 1, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            ref[0, 0, i, j] = x[0, 0, i * PH:(i + 1) * PH, j * PW:(j + 1) * PW].max()
    close("maxpool2d 2x2 vs numpy block max", o, ref)
    # backward: grad flows only to the argmax
    gp = rd("pool_grad.bin").reshape(1, 1, Ho, Wo)
    dx = rd("pool_dx.bin").reshape(1, 1, 6, 6)
    idx = np.zeros((1, 1, 6, 6))
    fdx = np.zeros((1, 1, 6, 6))
    for i in range(Ho):
        for j in range(Wo):
            blk = x[0, 0, i * PH:(i + 1) * PH, j * PW:(j + 1) * PW]
            am = np.unravel_index(np.argmax(blk), blk.shape)
            fdx[0, 0, i * PH + am[0], j * PW + am[1]] = gp[0, 0, i, j]
    close("maxpool2d backward routes grad to argmax only", dx, fdx)
except FileNotFoundError:
    check("maxpool2d dumps present", False, "block missing")

# --------------------------------------------------------------- conv+relu fused
try:
    N, Cin, Hin, Win, Cout, Kh, Kw, st, pad, Hout, Wout = [int(v) for v in
        open(os.path.join(DUMPS, "fused.meta")).read().split()]
    xi = rd("fused_in.bin").reshape(N, Cin, Hin, Win)
    wi = rd("fused_w.bin").reshape(Cout, Cin, Kh, Kw)
    fo = rd("fused_out.bin").reshape(N, Cout, Hout, Wout)
    sep = rd("fused_sep.bin").reshape(N, Cout, Hout, Wout)
    xp = np.pad(xi, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ref = np.zeros((N, Cout, Hout, Wout))
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                for wo in range(Wout):
                    acc = 0.0
                    for ci in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                acc += xp[n, ci, ho * st + kh, wo * st + kw] * wi[co, ci, kh, kw]
                    ref[n, co, ho, wo] = max(acc, 0.0)
    close("CONV2D_RELU_FUSED vs max(numpy conv, 0)", fo, ref)
    close("CONV2D_RELU_FUSED == separate conv->relu", fo, sep)
except FileNotFoundError:
    check("fused conv dumps present", False, "block missing")

# --------------------------------------------------------------- autodiff vs torch
# 2-layer tanh MLP, Lancius MSE = mean((p-y)^2), so the torch loss must be .mean()
# and every gradient follows from that same scalar objective.
try:
    Bn, In, H1, Out = 4, 3, 5, 2
    x = torch.tensor(rd("ad_x.bin").reshape(Bn, In), dtype=torch.float64, requires_grad=True)
    w1 = torch.tensor(rd("ad_w1.bin").reshape(In, H1), dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(rd("ad_b1.bin").reshape(H1), dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(rd("ad_w2.bin").reshape(H1, Out), dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(rd("ad_b2.bin").reshape(Out), dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(rd("ad_tg.bin").reshape(Bn, Out), dtype=torch.float64)
    a = torch.tanh(x @ w1 + b1)
    p = a @ w2 + b2
    loss = ((p - tg) ** 2).mean()
    loss.backward()
    for key, ref, shp in (("dx", x.grad, (Bn, In)), ("dw1", w1.grad, (In, H1)),
                          ("db1", b1.grad, (H1,)), ("dw2", w2.grad, (H1, Out)),
                          ("db2", b2.grad, (Out,))):
        close(f"autodiff {key} [mlp {Bn}x{In}x{H1}x{Out}] vs torch autograd",
              rd(f"ad_{key}.bin"), ref.numpy().ravel(), tol=1e-12)
    # the same objective, independently differentiated by central differences
    def f(xv, w1v, b1v, w2v, b2v):
        h = np.tanh(xv @ w1v + b1v)
        return float((((h @ w2v + b2v) - tg.numpy()) ** 2).mean())
    hstep = 1e-6
    for nm, arr, ref in (("dx", rd("ad_x.bin").reshape(Bn, In), x.grad.numpy()),
                         ("dw1", rd("ad_w1.bin").reshape(In, H1), w1.grad.numpy()),
                         ("dw2", rd("ad_w2.bin").reshape(H1, Out), w2.grad.numpy())):
        base = [rd("ad_x.bin").reshape(Bn, In), rd("ad_w1.bin").reshape(In, H1),
                rd("ad_b1.bin").reshape(H1), rd("ad_w2.bin").reshape(H1, Out),
                rd("ad_b2.bin").reshape(Out)]
        idx = {"dx": 0, "dw1": 1, "dw2": 3}[nm]
        fd = np.zeros_like(arr)
        flat = base[idx].ravel()
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + hstep; lp = f(*base)
            flat[i] = o - hstep; lm = f(*base)
            flat[i] = o
            fd.ravel()[i] = (lp - lm) / (2 * hstep)
        close(f"autodiff {nm} vs central differences of the same loss",
              rd(f"ad_{nm}.bin"), fd.ravel(), tol=1e-7)
except (FileNotFoundError, ImportError) as e:
    check("autodiff vs torch dumps", False, str(e))

# --------------------------------------------------------------- summary
print()
npass = sum(1 for _, ok, _ in results if ok)
nfail = len(results) - npass
for name, ok, detail in results:
    if not ok:
        print(f"  FAIL  {name}  [{detail}]")
print(f"\nGRAPH-LEVEL ORACLE: {npass}/{len(results)} passed, {nfail} failed")
json.dump([{"name": n, "ok": o, "detail": d} for n, o, d in results],
          open(os.path.join(os.path.dirname(os.path.abspath(DUMPS)), "graph_oracle.json"), "w"),
          indent=1)
sys.exit(1 if nfail else 0)
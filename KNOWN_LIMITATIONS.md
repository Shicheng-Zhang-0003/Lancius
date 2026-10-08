# Lancius v12R2 Known Limitations

This document defines the explicit boundaries of the v12R2
development milestone.

A development milestone is not defined by having every feature. It is defined
by having a clear and reliable contract.

## Supported

-   CPU inference execution
-   Static graph execution
-   Stable C API
-   Core tensor operations
-   Selected transformer inference operators
-   Serialization and runtime loading

## Experimental or Deferred

The following areas are intentionally not considered stable:

-   GPU acceleration
-   Dynamic shape execution
-   Distributed execution
-   Full training ecosystem
-   GGUF export pipeline

## Training Status

Lancius v12R2 is inference-first, mute-mathematician-first (<100M, scores never speaks).

Training-related components may exist in the codebase but should be
considered experimental development preview, not production-grade.

## Backend Support

Primary supported environment:

-   Linux
-   x86_64 CPU

Additional architectures may require validation.

## Fatal Invariants

The following conditions are guaranteed to abort the process.
They represent internal memory-safety invariants that are NOT
reachable through the stable C API (builders, loaders, and stable
API validate before reaching them):

-   Tensor ndim exceeds 4 (max rank violation)
-   Tensor element count overflows or exceeds 100,000,000
-   Tensor byte count exceeds 800,000,000

All user-reachable errors (unsupported ops, shape mismatches,
malformed models) return error codes via `lancius_set_error()`
and do NOT abort.

## Narrow-but-honest components

-   Bytecode VM: 2D MLP ops only (`MATMUL/ADD/SUB/MUL/RELU/
    BROADCAST/SOFTMAX/SUM`); anything else fails `compile` with NULL.
-   ONNX converter: 9 ops only
    (`Conv/Relu/MaxPool/Flatten/MatMul/Add/Reshape/Gemm/Transpose`);
    anything else raises instead of emitting partial graphs.
    Conv with `dilations!=1`/`group!=1`/`auto_pad!=NOTSET` raises;
    MaxPool with `pads!=0`/`dilations!=1`/`ceil_mode!=0`/`auto_pad` raises.
    MatMul is 2D-only (N-D batch would silently collapse — rejected by scope).
-   Converter: Reshape carries the data input only (shape tensors never
     emitted; 1D helpers would be unloadable); Relu/Add/Flatten/MatMul
     ndim follows data rank; `allowzero=1` explicit-0 and `0+-1` mixes
     raise. Micromodels accept any feat 1..64 (distill 16-dim loads).
     Bare `make` builds all (`.DEFAULT_GOAL := all`; depfiles trailed).
     Quantizer: 4D FP64 conv weights only; all-zero weights stay FP64.
    Per-tensor and per-channel quantization supported; dequantization
    supported. INT8 Add is row-bias (`[1,N]+[R,N]`) only; other INT8
    broadcasts fall through to exact FP64 broadcast, never miscompute.
-   `lancius_read_output` (stable API): FP64 outputs only; FP32/INT8
    outputs report `UNSUPPORTED_OP`.
-   v2 models always carry non-zero CRC32; files with `checksum == 0`
    are rejected by default (set `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` to
    load legacy pre-hardening files). Saver never emits 0.
-   Degenerate denominators fail loud: all-`-inf` softmax/CE rows,
    attention/KV/GQA `NaN` denominators, and degenerate LayerNorm/RMSNorm
    variances return `NUMERICAL` instead of values; fully-masked zero-sum
    attention rows emit zeros for causal safety.
-   `lancius_matmul` builds 2D-only graphs (the executor is 2D); N-D inputs
    fail at build time instead of silently dropping batch dims at execution.
-   Broadcast follows trailing-rank (NumPy) semantics for `ADD`/`SUB`/`MUL`;
    incompatible shapes are rejected, never read out of bounds.
-   Autodiff: `SUM` (any 1..4-D via `broadcast_to_shape`), `SUM_AXIS0/1`,
    `SUM_AXIS_ND` (any axis, any 1..4-D), `RESHAPE` VJPs exact. N-dim
    partial broadcast reduction (e.g. `[2,1,4]` vs `[2,3,4]`) reduces per
    axis via `SUM_AXIS_ND` (was fail-loud). `MATMUL_BATCHED` backward is
    exact via the batched transpose. `GELU`/`LayerNorm`/`RMSNorm` backward
    exact incl. gamma/beta grads (finite-diff ~1e-9..1e-11). Transformer
    attention/GQA/SwiGLU/RoPE backward still **fails loud** (returns NULL)
    instead of training as zero — staged scope, not silent. BROADCAST
    backward reduces over broadcast dimensions. See
    `docs/DESPOT_TRUTH_V2.md`.
-   GELU is tanh-approx (GPT-2/BERT variant, ~2e-3 vs erf-exact), not erf-exact.
-   Norm eps pinned to `LANCIUS_NORM_EPS=1e-5`; not per-node tunable.
-   Optimizer moments are caller-owned arrays (the train lib is stateless:
    no alloc/I/O by contract); weight checkpoints persist via v2 models
    and resume bit-consistently (`audit_train_converge` proves resume
    matches uninterrupted 1e-9). Global-norm clipping covers multi-tensor
    grad lists; per-tensor clip unchanged.
-   Trainers exit 1 on raw `NaN/>1000/<0` and on accuracy ≤ chance (10%);
    `make check` does not run `train_mnist/cifar10` (green says nothing
    about their convergence); LR parity across PyTorch/C batch/scale gaps
    is unproven by design.
-   `conv_bwd_w critical` FP-sum order is last-bit nondeterministic (training only).
-   Bytecode VM materializes `CONST` nodes at execute time; graphs mixing
    `CONST` with computed ops execute identically to the scheduler on the
    2D MLP subset.
-   `export_lancius_onnx.py` refuses `checksum == 0` files unless passed
    `--allow-legacy` (mirrors the C loader default).
-   Operator (`./lancius`): `quickstart`/`status`/`models`/`demo`/`doctor`/`eval` are
    informational or self-contained; `train` shells only fixed binaries,
    user paths always exec without a shell. Long trains (cifar10, hours)
    are confirmed interactively in the TUI.

-   Stable handles carry magic tags; wrong-type/stale/destroyed handles
    return `INVALID_HANDLE`. Tensor handles are magic-tagged wrappers
    borrowed from their graph (graph must outlive tensors; destroy
    invalidates all). Forged pointers fail loud; use-after-graph-free
    remains caller-UB by contract.
-   Bytecode VM sets error codes on every reject (OOM vs corrupt tape
    distinguishable); `pool_wait` timeout reports `LIMIT`/`INTERNAL`.
-   KV-cache, save paths, layernorm/rmsnorm/attention/GQA workers, and
    conv overflow guards all report through the error channel (no silent
    success); OpenMP worker errors propagate via shared flags.
-   `conv2d_bwd`/`conv2d_bwd_w`/`maxpool2d_bwd` builders + executors verify grad
    N/C/H_out/W_out (wrong-shaped grads rejected, never silently wrong).
-   RoPE `2*head_dim` overflow guarded; `vectors_to_lancius.py` raises
    `ValueError` (no `assert`).

## v12R1 hardening batches V5/V6 — boundary impact only (enumeration in `CHANGELOG.md`)

-   V5 (2026-09-30): no scope change; 12 correctness fixes within existing
    boundaries.
-   V6 (2026-10-01): no scope change; 45 correctness fixes. Notable
    boundaries tightened: VM is 2D-only (`ndim!=2` rejected); broadcast
    backward exact at every rank via `SUM_AXIS_ND` (the "still no
    `SUM_AXIS_ND`" note here was stale: R3-1 added the op and V7 §16.4 made
    its VJP reachable and proved it against central differences);
    `CE_BWD` requires 2D + scalar grad; v2 per-channel save refused
    (format unchanged); static plan path requires 32B-aligned base
    (bump path aligns internally); `CONST` 1..4-D and `ROPE` now persist.

## Philosophy

Limitations are documented intentionally to prevent unsupported
assumptions.


## Gate scope (V7)

These are the boundaries of the *evidence*, not of the code. A property not
listed as proven here is not proven; the honest default is "unchecked".

Proven by `make check-oracle` against NumPy / torch autograd / hand-derived
closed form / central differences:

- Every pure kernel: matmul, FP32 matmul, conv2d (two stride/pad settings),
  INT8 conv, LayerNorm and RMSNorm forward and all backwards, GELU forward and
  backward, SwiGLU, RoPE, causal attention, GQA, KV-cache attention
- Every graph-level op: softmax, cross-entropy, MSE, `ADD`/`SUB`/`MUL`
  broadcast at 2-D/3-D/4-D, `SUM_AXIS_ND` on every axis of every rank,
  `MATMUL_BATCHED`, permute, `CONV2D_BWD`, `CONV2D_BWD_W`, `MAXPOOL2D` and its
  backward, fused conv+relu
- The trainers: SGD, SGDM, AdamW (against `torch.optim.AdamW`), global-norm
  clip, both LR schedules
- The full autodiff VJP set of a two-layer tanh MLP, against torch autograd and
  against central differences on the same objective

Proven by `make check-sanitizers` (ASan + UBSan + LeakSanitizer over the
**instrumented library**, every audit):

- No heap or stack error, no undefined behaviour, no leak in any gate binary

Proven by `make check-ubstrict` (UBSan alone, `-fno-sanitize-recover=all`):

- No signed-integer overflow, shift error, bad float-to-int conversion,
  misaligned or out-of-bounds access in any gate binary

Proven by `make check-mutation`:

- The gate catches 18 of 18 injected defects; 1 further mutation was verified
  behaviourally equivalent and is reported as neutral, not as a pass

**Not proven, and therefore not claimed:**

- The 3463-LDFD submodule's own gates are its responsibility; this repo runs
  `make ldfd-test` as a gate member and reports an honest SKIP when the library
  is absent
- ONNX conversion beyond LeNet-class graphs, dynamic shapes, GPU paths,
  multi-process or multi-threaded *user* code, and the FP32 path beyond matmul
- Performance. Nothing here measures throughput; the oracles measure
  correctness only

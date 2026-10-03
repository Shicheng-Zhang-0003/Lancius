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
-   Quantizer: 4D FP64 conv weights only; all-zero weights stay FP64.
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
    `RESHAPE` VJPs exact. N-dim partial broadcast reduction (e.g.
    `[2,1,4]` vs `[2,3,4]`) has no `SUM_AXIS_ND` op yet and **fails loud**
    (returns NULL) instead of training as zero. Transformer and
    `MATMUL_BATCHED` backward fail loud. BROADCAST backward now correctly
    reduces over broadcast dimensions. See `docs/DESPOT_TRUTH_V2.md`.
-   GELU is tanh-approx (GPT-2/BERT variant, ~2e-3 vs erf-exact), not erf-exact.
-   Norm eps pinned to `LANCIUS_NORM_EPS=1e-5`; not per-node tunable.
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
    return `INVALID_HANDLE`. Tensor handles borrow from their graph.
-   Bytecode VM sets error codes on every reject (OOM vs corrupt tape
    distinguishable); `pool_wait` timeout reports `LIMIT`/`INTERNAL`.
-   KV-cache, save paths, layernorm/rmsnorm/attention/GQA workers, and
    conv overflow guards all report through the error channel (no silent
    success); OpenMP worker errors propagate via shared flags.
-   `conv2d_bwd_w`/`maxpool2d_bwd` builders + executors verify grad
    N/C/H_out/W_out (wrong-shaped grads rejected, never silently wrong).
-   RoPE `2*head_dim` overflow guarded; `vectors_to_lancius.py` raises
    `ValueError` (no `assert`).

## v12R1 hardening batches V5/V6 — boundary impact only (enumeration in `CHANGELOG.md`)

-   V5 (2026-09-30): no scope change; 12 correctness fixes within existing
    boundaries.
-   V6 (2026-10-01): no scope change; 45 correctness fixes. Notable
    boundaries tightened: VM is 2D-only (`ndim!=2` rejected); broadcast
    backward exact for 2D/4D else `UNSUPPORTED_OP` (still no `SUM_AXIS_ND`);
    `CE_BWD` requires 2D + scalar grad; v2 per-channel save refused
    (format unchanged); static plan path requires 32B-aligned base
    (bump path aligns internally); `CONST` 1..4-D and `ROPE` now persist.

## Philosophy

Limitations are documented intentionally to prevent unsupported
assumptions.

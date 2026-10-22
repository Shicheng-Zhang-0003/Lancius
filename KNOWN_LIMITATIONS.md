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

### Trainer step size divided by `pe` twice (fixed; effect recorded)

All three `examples/` trainers (`train_prm_verifier.c`, `train_verifier5.c`,
`train_gen1.c`) built a graph of `mean((logits - onehot)^2)`, whose VJP is
already `(2/pe)*(z-T)` with `pe = rows*NCLASS`, and then stepped with
`lr_eff = LR * 2/pe` — applying `2/pe` a second time. The effective step
therefore scaled as **1/rows^2**, so larger batches learned *worse*.

Measured on the 3-class verifier, same data and LR, 25 epochs:

| batch | shipped (`LR*2/pe`) | fixed (`LR*C/2`) |
|---|---|---|
| 64 | 0.4913 | 0.5055 |
| 256 | 0.4821 | 0.5030 |
| 1024 | **0.3306** (chance) | **0.5155** |

At batch 1024 the shipped step was `2.1e-8` per parameter per step, leaving the
model at chance and *below* the 0.4418 majority baseline. Corrected to
`lr_eff = LR * NCLASS/2`, which is the unique constant turning the step into
`LR*(1/rows)*sum_r (z_r - T_r)*x_r` — the form a mean-CE trainer takes. The
correction is worth ~2 points at batch 64 (the 16-byte-feature ceiling is the
binding constraint there) but is the difference between a trained model and
noise at batch 1024. `tools/prm/ref_prm.c` was updated in lockstep.

Note what the fix does **not** do: `C/2` matches the *step form*, not the
*gradient*. The MSE-on-logits VJP and a true CE gradient differ by the softmax
Jacobian, which is input-dependent, so no constant converts one into the other.
Both trainers minimising mean-MSE-on-logits also cap the achievable CE at
`ln(1 + (C-1)/e)` (~0.555 for C=3) because the MSE optimum is a hard one-hot
logit vector, not a calibrated distribution.

**Resolved (superseded).** `examples/train_verifier.c` is now the canonical
verifier trainer and sidesteps this entire class of defect: it trains the
framework's own fused softmax cross-entropy (`lancius_cross_entropy`, exact VJP
`(softmax-y)/R`), reads the loss every step, clips global grad norm, and uses
the framework's `lancius_lr_warmup_cosine`. Gradient-checked against central
differences (worst scale-relative error < 1e-7). `train_verifier5.c`,
`train_gen1.c` and `train_prm_verifier.c` are legacy and superseded by it; a
corpus consumed by the new trainer must carry a `corpus.hdr` (FEAT, NCLASS,
NTRAIN, NTEST) and graphs are built once per run (persistent training graph,
buffers bound by pointer).

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
-   GELU is tanh-approx (GPT-2/BERT variant; measured max deviation 4.74e-04, rms 1.42e-04 vs erf-exact), not erf-exact.
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
-   `tools/onnx/export_lancius_onnx.py` refuses `checksum == 0` files unless passed
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
-   RoPE `2*head_dim` overflow guarded; `tools/data/vectors_to_lancius.py` raises
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


## Gate scope (V7, V9 deltas)

These are the boundaries of the *evidence*, not of the code. A property not
listed as proven here is not proven; the honest default is "unchecked".

Proven by `make check-oracle` against NumPy / torch autograd / hand-derived
closed form / central differences (284 total, 0 skipped with torch present;
`oracle_gate.sh` fails if torch present but skips>0):

- Every pure kernel: matmul, FP32 matmul, conv2d (two stride/pad settings),
  INT8 conv, LayerNorm and RMSNorm forward and all backwards, GELU forward and
  backward, SwiGLU, RoPE, causal attention, GQA, KV-cache attention
- Every graph-level op: softmax, softmax backward (`dx = y*(dy - sum(dy*y))`
  against torch autograd on ordinary and saturated rows, plus the two
  structural properties a wrong axis or a missing `dot` subtraction breaks),
  cross-entropy, MSE, `ADD`/`SUB`/`MUL`
  broadcast at 2-D/3-D/4-D, `SUM_AXIS_ND` on every axis of every rank,
  `MATMUL_BATCHED`, permute, `CONV2D_BWD`, `CONV2D_BWD_W`, `MAXPOOL2D` and its
  backward, fused conv+relu
- The trainers: SGD, SGDM, AdamW (against `torch.optim.AdamW`), global-norm
  clip, both LR schedules
- The full autodiff VJP set of a two-layer tanh MLP, against torch autograd and
  against central differences on the same objective
- V9: CE corrupt-label NUMERICAL + INT8-matmul scale NUMERICAL pinned by
  `temp/despot-v9/` reproducers (NaN now loud, valid targets torch-exact)

Proven by `make check-sanitizers` (ASan + UBSan + LeakSanitizer over the
**instrumented library**, every audit):

- No heap or stack error, no undefined behaviour, no leak in any gate binary
- V9: 28/28 (`test_diamond_memory`, `train_verifier_head` added; `distill`
  standalone documented, not omitted silently)

Proven by `make check-ubstrict` (UBSan alone, `-fno-sanitize-recover=all`):

- No signed-integer overflow, shift error, bad float-to-int conversion,
  misaligned or out-of-bounds access in any gate binary
- V9: 28/28 (`fuzz_lancius`, `test_path_bg`, diamond added; the V9 note
  said 29 -- the sweep's own AUDITS list has 28 members and is the truth)

Proven by `make check-mutation`:

- The gate catches 18 of 18 injected defects (19 tries); 1 further mutation was verified
  behaviourally equivalent and is reported as neutral, not as a pass

**V9 contract tightenings (no format break, additive API only):**

- Saves are v2-only: v2 refusal returns -1 with cause preserved (was silent
  v1 downgrade with lost per-channel scales). Legacy v1 writer kept as
  explicit `lancius_graph_save_v1_legacy` opt-in, never implicit.
- Handles carry a per-process secret cookie + owner graph: forged/cross-graph
  handles return `INVALID_HANDLE` (was magic-only forgery).
- VM has a bounded form `lancius_vm_execute_checked(prog,ins,out,out_len,scratch)`
  requiring `out_len>=out_elements` else `LIMIT` (was unbounded `memcpy`);
  trailing `HALT` no longer falsely fails.
- Planner returns NULL on any error (was partial plan + sticky error).
- CLI `--topk 1..100`, display scans at most 100k cols (was 1..1M `O(topk*C)`).
- Python: open+`fstat`+capped reads; inflate capped at 8GB (was TOCTOU +
  unbounded disk-fill).

**Not proven, and therefore not claimed:**

- The 3463-LDFD submodule's own gates are its responsibility; this repo runs
  `make ldfd-test` as a gate member and reports an honest SKIP when the library
  is absent
- ONNX conversion beyond LeNet-class graphs, dynamic shapes, GPU paths,
  multi-process or multi-threaded *user* code, and the FP32 path beyond matmul
- Performance. Throughput is now *measured* by `make bench_kernels`
  (`examples/bench_kernels.c`, median of 7 after a warm-up, FLOP formula and
  thread environment printed beside every rate) and the numbers are recorded
  in `docs/DESPOT_TRUTH_V2.md` §21. It is not a gate and it makes no claim
  about any other framework: a benchmark that beats a strawman is a lie in
  the making. Still unchecked: end-to-end training throughput at model scale,
  and any comparison against torch/TF on the same hardware.

## Softmax backward had no external oracle (closed, V10)

`lancius_softmax_bwd` was reachable through autodiff and exercised indirectly
wherever an MLP trained, but no external oracle compared it against torch
autograd: no probe dumped a softmax backward, and `softmax_out2.bin` -- which
looks like a gradient by name -- is the *shifted forward output* written for
the shift-invariance probe. That hole is closed by the `softmax backward` block
in `tools/audit/graph_oracle_dump.c` (ordinary and saturated rows) and the
matching block in `graph_oracle.py`, which pins `dx = y*(dy - sum(dy*y))`
three ways: the two structural properties a wrong axis or a missing `dot`
subtraction breaks (each row of dx sums to 0; adding a constant to dy leaves dx
unchanged), the closed form, and torch autograd. Recorded rather than deleted,
because the fact that a limitation existed and was found by inspection -- not
by a failing test -- is the part worth keeping.

`conv_oracle.py`, `math_oracle.py` and `trainlib_oracle.py` additionally require
PyTorch and have no NumPy fallback, because torch is the independent engine for
those layers; a NumPy reimplementation of SGD checked against a NumPy
reimplementation of SGD would agree with itself.

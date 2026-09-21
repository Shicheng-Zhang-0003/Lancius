# Lancius Changelog

## v12R2 (unreleased) — learn to verify

First work toward the R2 mandate (see `docs/v12R2_SCOPE.md`): the framework
gains only generic trainable primitives so models can learn the
extended-boolean verifier scheme. Truth semantics stay models-side.

- Core: `TANH`/`TANH_BWD` (bounded activation saturating the fluid scale)
  and `MSE`/`MSE_BWD` (regression against step labels in [-1,1]) across
  IR builders, scheduler executors, autodiff VJPs, and v2 loader
  forward cases (ids appended at the end; v11S+ files load identically).
  `_BWD` nodes stay non-serializable, like all training artifacts.
- Gates: 6 new known-answer checks (tanh(0/±1), MSE == 1/3); new
  `train_verifier_head` example trains a tanh-headed MLP by MSE with
  weakest-link credit through the argmin step — loss falls, analytic
  gradients match finite differences to ~1e-11, scores stay in [-1,1],
  all in `make check`.
- Pipeline: `distill_prm800k.py` converts PRM800k step rows to fixed-dim
  numeric vectors + {-1,0,+1} targets with manifests and sha256;
  honestly reports the vendored subset is single-class (+1 only).
- Python-to-C: stdlib-only `distill_prm800k.py` retired in favor of
  `examples/distill_prm800k.c` (self-contained JSON + sha256, no
  dependencies) — byte-identical `.bin` output on all 14,564 vendored
  steps, ASan/UBSan clean including hostile inputs, `--selftest` proofs
  in `make check`. Third-party-bound scripts (torch/onnx/ort/network)
  stay in Python: porting those buys nothing but difficulty.

## v12R1 / V1.2RC1 — 2026-09-21 — hardening plus numerical correctness

First development milestone of the v12 cycle (R1 phase), built on the v11S
stable baseline. Three batches, no format / stable-ABI break beyond additive
error codes:

Hardening batch:

- Scheduler: broadcast column/identity fix, RMSNorm 3D hidden fix,
  parallel FP32 dispatch + FP32 reset hygiene, cycle returns NULL +
  GRAPH_CYCLE, matmul overflow guards, XEnt-bwd numerical guard,
  ROPE even-dim check, permute/batched validation, pool offset bounds
  (`plan->max_id`), parallel/CONST OOM errors.
- Bytecode VM fails loud on unsupported ops; VM input/dim checks.
- `onnx_to_lancius.py`: strict (unknown op / unmapped input / bad perm /
  unresolvable Reshape raise), real CRC32 in header.
- Arena/IR: create/alloc/track/realloc guards; builders validate shapes
  (attention, layernorm/rmsnorm gamma, swiglu, broadcast); fixed
  `bind_external_int8`, `set_owner`, `release_owned` FP32 leak.
- Kernels: null/zero/overflow guards everywhere; attention/GQA
  `-INFINITY` causal sentinel; GELU clamp removed; RoPE odd-dim refuse;
  KV-cache zero-len guards; MaxPool `-INFINITY` + NaN propagate;
  INT8 zero-scale is now a loud error; quantizer clamp + skip-if-INT8.
- Planner/threadpool: all allocs checked, id bounds, overflow-safe
  offsets, threadpool create/teardown hardening.
- Persistence: v2 save CRC fail-closed + `w+b` read-back fix (was silent
  crc=0), v1 loader ndim/duplicate-id/unknown-op hardening, v2 header
  reserved-field + duplicate-NOP rejection, stable API error-map
  completion + FP64-only `read_output` honesty + checked counts.
- Audits updated to `-INFINITY` / unclamped GELU references.
- Removed 15 stale `*.bak*` / `*backup*` files.

Numerical-correctness batch (hostile audit of every numeric path; each
confirmed defect fixed and re-proven by independent execution):

- Scheduler/IR/VM: N-dim broadcast `ADD`/`SUB`/`MUL` (IR emits
  `max`-per-dim output shape; scheduler + VM execute strided broadcast;
  was flat-loop wrong + OOB); softmax zero-sum guard in scheduler and VM.
- Persistence: v2 CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`
  opts into legacy unverified loads); O(1) duplicate-ID detection;
  sparse-ID bounds; streaming weight skip; sticky-error clearing on success.
- Runtime: attention cache/heads/dim validation, cache-less long-context
  decode rejected, GQA Q/K/V shape checks, RMSNorm gamma/divisibility
  checks, `_checked` counters on execution/planning/copy paths,
  `FLATTEN`/`RESHAPE` element-equality verification.
- Kernels: INT8 `int64` accumulators (conv + mixed-precision matmul), real
  overflow guards (dead `&& 0` removed), OOM error reporting for
  thread-local and Flash/GQA scratch buffers.
- Core: arena failures carry error codes; `dtype_size(invalid) = 0` with
  `bytes_checked` rejection; `checked_product_shape(NULL)` fails.
- Stable API: additive error codes `GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/
  `INVALID_HANDLE` replace lossy collapsing.
- ONNX: Reshape `0`-copy vs `-1`-infer, Gemm `transA`/`alpha`/`beta`
  rejection + transpose clone-on-write, symmetric-only Conv pads/strides
  and square-only MaxPool, symbolic-dim rejection, static-batch export.
- Training: CIFAR-10 `[-1,1]` normalization matching PyTorch reference,
  He init on MNIST, identity-based parameter binding, per-batch grad
  zeroing, identity-tracked eval logits, raw-vs-clamped loss logging.
- Audits: `audit_modern_llm`, `audit_flash_attention`,
  `audit_threadpool_parity`, `audit_ffi`, `audit_pytorch_parity.py` now
  propagate failures via exit codes.

Despot truth batch (every remaining lie found and implemented):

- Planner records freed offsets: diamond-graph reuse no longer overlaps
  live tensors (was silent corruption on second reuse).
- Broadcast upgraded to trailing-rank (NumPy) semantics in validator, IR
  builders, and scheduler, with per-dim compat guards and bounds checks;
  incompatible shapes rejected, never read out of bounds.
- Conv index math in `int64` (was `int` truncation); KV-cache attention OOM
  reports instead of emitting zeros; cross-entropy forward degenerate
  denominator is `NUMERICAL`, matching backward (was `1e30` sentinel).
- Autodiff: scalar-reduction OOB fixed, 4D `CONST` cloning preserved,
  incompatible accumulation fails loud instead of dropping gradients.
- Scheduler fail-closed on wave-cap overflow; static executor uses
  `_checked` sizing with FP32 binding and NOP skipping; sanitizer gate
  restores a clean build afterwards.
- Audits: `audit_nan_injection`, `audit_memory_pool`, `test_grad_check`,
  `test_path_bg`, `test_torture` (real cycle test), `fuzz_lancius`
  (safe-rejects separated), `audit_internals` (softmax normalization),
  `audit_modern_llm` (GQA values) all propagate failures; new
  `audit_despot_probe` pins broadcast/planner/CE truth in `make check`.
- ONNX: Gemm always clones on transpose (never mutates the shared
  initializer — was order-dependent double-transpose); Reshape preserves
  the batch dim (was forced to 1); `audit_trained_reality.py` exits
  nonzero below 95/100.
- Stable API oversized-input hole proven closed by `audit_fault_injection`
  (11/11).

## v11S / V1.1 — Stable Release

### Changed
-   Model format v2 frozen with CRC32 body integrity verification
-   EXTERNAL_WEIGHTS flag reserved and rejected by loader
-   Stable C API expanded: model load/save, tensor introspection
-   `lancius_read_output` rejects undersized buffers (no silent truncation)
-   Scratch arena auto-sized from liveness analysis
-   Version identity consistent across all documents
-   Full validation suite green (make check, check-long, sanitizers)

### Security
-   CRC32 integrity check on model body (bytes 48..EOF)
-   Reserved header flags rejected at load time

## v11A3 / V1.1-AlphaRC3 — Freeze & Hardening

### Changed
-   Feature freeze: no new operators, runtime subsystems, or training features
-   Loader and model-format validation hardening
-   Sanitizer and fuzz validation expansion
-   Regression-test expansion
-   Documentation cleanup and version-identity consistency
-   Release-candidate preparation for v11S

## v11A2 / V1.1-AlphaRC2 — Transformer Runtime Usability

### Added
-   Dedicated KV-cache runtime object
-   Cache-aware attention execution
-   Explicit prefill and generation execution flows
-   First-class 3D transformer tensor construction (`lancius_input_3d`)
-   Transformer known-answer audit (LayerNorm, RMSNorm, GELU, SwiGLU, RoPE,
    attention, KV-cache parity, prefill/generation parity, GQA)
-   FP32 execution foundation (matmul kernel, scheduler dispatch,
    serialization, path audit)
-   Stronger v2 loader validation

### Improved
-   Reduced demo-style graph mutation in transformer examples
-   Scheduler buffer lifecycle and repeated-execution hygiene

## v11A1 / V1.1-AlphaRC1 — Foundation & Runtime Honesty

### Added

-   Development milestone contract
-   Hardened runtime validation
-   Expanded testing infrastructure
-   Stable API boundary
-   Improved documentation

### Improved

-   Memory safety checks
-   Graph validation
-   Numerical stability
-   Runtime reliability

### Removed

-   Experimental GGUF export path from stable distribution

### Deferred To Future Releases

-   GPU backends
-   Dynamic execution
-   Expanded training support
-   Additional deployment targets

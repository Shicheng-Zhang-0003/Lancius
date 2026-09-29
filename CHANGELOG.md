# Lancius Changelog

## hardening batch V4 (2026-09-28) — race conditions, memory safety, quantization, portability

Comprehensive bug-fix campaign across autodiff, kernels, serialization, CLI,
build system, and Python tooling. 26 defects fixed and re-proven by
`make check`, `check-sanitizers`, and pytorch parity.

- Autodiff: BROADCAST backward now correctly reduces over broadcast dimensions
  (was passing grad_out unchanged); NULL checks on `fwd_n->inputs` throughout;
  OOB reads on shape/axes arrays fixed (now pads to 4D); off-by-one in node
  capacity check fixed.
- Kernels: race condition in `kernel_conv2d_bwd_in` fixed (thread-local
  accumulators); race condition in MaxPool2D backward fixed (thread-local
  accumulators).
- Memory: memory leaks in IR node allocation fixed; `abort()` removed from
  library code (replaced with `lancius_set_error` returns); all `fprintf`/`printf`
  in library code replaced with `lancius_set_error`.
- Stable API: dangling `wrapper->sched` fixed; `set_owner` now updates
  `int8_owner`.
- Serialization: portability fixed (`uint64_t` sizing, byte swapping);
  CRC32 table init race fixed (`call_once`); NOP IDs no longer mapped to NULL;
  double-read for CRC eliminated (now computed during parsing).
- Threadpool: `lancius_pool_wait` now accepts a timeout parameter.
- Bytecode VM: overflow checks added.
- Quantization: per-channel quantization support added; dequantization support
  added.
- CLI: command injection fixed (now uses `fork+execvp` instead of `system`).
- Python scripts: security fixes, stale version references removed, error
  handling improved.
- Build system: version consistency enforced, `-Werror` added, Threads
  dependency fixed; `train_cifar10` now links `-lpthread`; dependency tracking
  improved; `.gitignore` updated with missing entries; `lancius.pc.in` version
  fixed.
- Code quality: magic numbers replaced with named constants throughout.

## despot audit V3 (2026-09-28) — every live bug found and implemented

Four forensic sweeps, ~70 code-backed defects, all fixed and re-proven
(`make check`, `check-sanitizers`, despot probe, pytorch parity 3.42e-07):

- Kernels: matmul/conv index-overflow guards, bwd_in stride guards,
  layernorm/rmsnorm degenerate is NUMERICAL (was silent beta/zeros), RoPE
  int-wrap guard, KV-cache hidden/malloc guards, NULL paths set errors.
- Scheduler: CE fwd R/C + shape guards, FP32/INT8 matmul K-match (was OOB),
  FP64 M*K/K*N overflow, elementwise input-size checks, ROPE qk-size +
  offset checks, softmax zero-guard + Inf, permute shape correspondence +
  checked indices, static-plan skips failed assignments, peak/required set
  errors (bounded executor cannot under-alloc), abort-free hot paths kept.
- Vision: inputs/input_count guards on every branch (was NULL+0 deref),
  conv H_out verification vs n->shape, pool index guards, INT8 scale NaN/Inf
  rejected, silent returns set errors.
- IR/autodiff: alloc_node rt-OOM returns NULL, matmul sets errors and stays
  2D (executor is 2D), f32 ownership split (was: leak + free of external),
  broadcast_4d true compat check, bwd_w validation, fused-clone rt attach +
  no aliasing + id-cap guards, loss-node membership/range checks, seed NULL
  aborts, inputs[2] guarded, INPUT clone keeps int8/f32/dtype/scale, NULL
  grad aborts (was neutral-skip), unhandled forward op aborts, permute axes
  validated, builders set SHAPE_MISMATCH (was silent NULL).
- Planner/threadpool/VM/optimizer/quantize: zero-size plan aborts (was
  offset-0 overlap), pool queue grows (was racy inline run) + shutdown
  reject + OOM error, VM materializes CONST (was garbage regs) + inputs
  checks, fusion validates stolen inputs, quantizer syncs rt scale + frees
  stale int8 + skips non-finite max.
- Persistence: v1 checked writes + tmp-less partials unlinked + NULL guards
  + no abort() + ndim 1/3 loads (was 2D coercion) + view-source validation +
  double-free removed (both weight paths + fail path); v2 tmp+rename saves,
  streamed CRC (no 800MB malloc, no long truncation, no seek-bypass),
  duplicate-NOP seen-list, invalid dtype fails, u64 narrowing kept.
- Interop/python: converter true input ranks (was: padded count always 4),
  Reshape rank from resolved dims, packed-field range checks, Gemm byte
  context, write errors + crc0->1; ONNX exporter 2GB cap + header checks +
  checksum==0 refuses (--allow-legacy opts in); pytorch exporter None/cap/
  arity/perm/stride checks + multi-input trace dummies; datasets no-shell
  git clone + cwd-jail cleanup + tar/zip-slip guards + download try/except.
- Operator/examples: CLI fork+exec for user paths (was: 4 shell-injection
  sites) + truncation fails + fread short-read fails + strtol topk/show +
  TUI drain + TUI show passed; trainers validate files/magic/counts/sizes,
  check OOM/graph/schedule, guard grad/loss NULLs, fix cifar eval counting
  (+evaluated denominator) + save check, fix verifier arena use-after-reset
  (heap copies) + worst-step max metric + per-iter g2 destroy + full cleanup
  on fatal paths; run_edge caps + cleanup + rc=1; generate_text cleanup +
  rc propagation; run_llm 3D builders + overflow + output checks.
- Proven: `make check` green (73/73, 265/265, despot probe, verifier with
  corrected worst-step 1.34→0.0013), `check-sanitizers` clean, pytorch
  parity exact.

## operator TUI perfection (2026-09-28) — ordinary users guided, offline honest

- New verbs: `status` (operational snapshot), `models [--check]` (list plus
  validate local models), `demo` (one-command install proof), `help [verb]`
  (plus per-verb `--help` everywhere; `run --help` no longer unknown-arg).
- `doctor` tells the truth: core MISSING fails, training data absent is an
  optional-data warning (was blanket READY), plus network (curl, 5s cap) and
  disk-free sections.
- `train` pre-checks data before work (suggests the exact pull command),
  `--dry` shows binary plus data plus time hint; verifier seconds, mnist
  minutes, cifar10 hours with an explicit are-you-sure in the TUI.
- `run` validates mode/fill/topk/show with suggestions, checks model and
  `--input` files first, explains `--input` size mismatches in bytes.
- `info` flags training-only (`_BWD`) and reserved (`EMBEDDING`, `KV_*`)
  ops, prints human dtype names, hints at `lancius models` on missing files.
- `convert`/`export` pre-check inputs, scripts, and python deps (no raw
  tracebacks); `datasets pull` validates names and refuses cleanly offline.
- TUI rebuilt around direct calls (no `./lancius` re-exec for internal
  verbs, error codes preserved), with header status, plain-language menu
  plus time guide, `help` entry, per-prompt defaults and back (empty),
  validation loops, cifar10 confirm, pause-after-command on ttys, last-status
  line, and `NO_COLOR`/`TERM=dumb`/non-tty safe output.

## despot truth V2 (2026-09-28) — every remaining lie closed, re-proven

- Autodiff truth: `broadcast_to_shape` (any 1..4-D scalar lift); `SUM` grad
  exact for any ndim (was `[1,1]`-for-3D); `SUM_AXIS0/1` + `RESHAPE` VJPs
  added (were silent drops); transformer forward cloned so `fwd_to_full`
  stays complete; `_BWD` in forward returns NULL; N-dim partial broadcast
  reduction fails loud (no `SUM_AXIS_ND` yet) with sticky-error abort of the
  whole training graph; `accum_grad` returns `1/0` with all `NULL` paths
  checked; `add/sub/mul` set `SHAPE_MISMATCH` (was silent NULL).
- Executors: permute stride products checked; batched-matmul batch offsets
  checked (`M·K`, `K·N`, `M·N` + `batch·elems`); liveness/static/parallel
  paths use `_checked` bytes/elements (no abort); `SUM`/`SUM_AXIS`/`TRANSPOSE`
  shape-validated; INT8 Add row-bias-only with FP64 fallback; INT8 matmul
  `scale_a=1.0` for all-zero (consistent with quantizer skip).
- Kernels: attention/GQA `NaN→NUMERICAL` (was silent zeros); KV-cache
  `max/sum NaN→NUMERICAL` (`-inf`/zero stay zeros); GELU documented as
  tanh-approx; `LANCIUS_NORM_EPS=1e-5` pinned.
- Vision: INT8 zero-scale reports `NUMERICAL` (was `INVALID_SHAPE`);
  `MAXPOOL_BWD` output bytes checked.
- Persistence: `checksum==0` doc fixed to rejected-by-default; `u64→size_t`
  narrowing checked; `BROADCAST` loads any 1..4-D.
- ONNX: Conv `dilations/group/auto_pad` rejected; MaxPool
  `pads/dilations/ceil_mode/auto_pad` rejected; Reshape dead code removed.
- Training honesty: CIFAR raw-loss abort (`NaN/>1000/<0` returns 1);
  MNIST raw-loss abort + accuracy gate; both trainers exit 1 at ≤ chance;
  `test_ffi_error` exits 1 on unexpected success/non-NULL handle.
- Docs: new `docs/DESPOT_TRUTH_V2.md` (full math/programming/operational
  audit with formulas and proofs); `docs/ARCHITECTURE.md`,
  `KNOWN_LIMITATIONS.md`, `MANIFEST.md`, `STATUS.md` updated; temp execution
  under `/tmp/opencode/lancius-despot-logs`.
- Proven: `make check` green; `test_grad_check` (`8.6e-10`, `5.8e-8`);
  `probe_v2` (SUM-3D `BROADCAST ndim3`, RESHAPE, SUM_AXIS, `broadcast_to_shape`,
  N-D partial fail-loud, attention NaN→NUMERICAL) all truth holds.

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

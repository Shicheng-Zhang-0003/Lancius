# Lancius Current Status

Current internal milestone: **v12R1**
Previous internal milestone: **v11S**
Public equivalent: **V1.2RC1**
Release line: **12 R1 development**

## Phase

v12R1 is the first development milestone of the v12 cycle
(progression `S → R1 → R2 → R3 → S`; v12R1 is the R1 phase),
built on the v11S stable baseline.

Historical baseline (v11A3 gate, complete):
- `make check` green
- `make check-long` green
- sanitizer validation green
- malformed model loading safely rejected
- version identity consistent
- model format v2 frozen with CRC32 integrity
- stable C API covers core inference workflow

Live v12R1 gate is Validation batch below (`make check/long/sanitizers`
+ despot probes + `probe_v6`); historical checklist above is not the
current gate.

## v12R1 theme: hardening plus numerical correctness

Bottom-up correctness pass over all layers, then a hostile,
formula-by-formula audit of every numeric path. Each confirmed
defect fixed and re-proven by independent execution.
Full per-batch record lives in `CHANGELOG.md`; essence only here
(see also `docs/DESPOT_TRUTH_V2.md`):
- N-dimensional broadcast `ADD`/`SUB`/`MUL` correct in scheduler, IR shape
  inference, and bytecode VM (was flat-loop wrong + OOB)
- Softmax zero-sum guard in scheduler and VM
- v2 CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into
  legacy); O(1) duplicate detection; sparse-ID bounds; streaming skip
- Attention cache/heads/dim validation; cache-less long-context decode
  rejected; GQA shape validation
- INT8 `int64` accumulators; real overflow guards; OOM error reporting
- Aborting counters replaced with `_checked` + error returns on execution,
  planning, and copy paths
- ONNX strictness (Reshape `0`/`-1`, Gemm clone + `alpha`/`beta`/`transA`,
  symmetric Conv/Pool, static batch)
- Training alignment (`[-1,1]` CIFAR norm, He init, identity binding,
  identity-tracked eval logits)
- Widened stable FFI error codes; false-green audits propagate failures
- Despot truth batch: planner offsets recorded (diamond reuse sound),
  trailing-rank cross-rank broadcast with compat guards, int64 conv indices,
  KV-cache OOM reporting, CE degenerate is NUMERICAL, Gemm always-clone,
  Reshape preserves batch, torture cycle test is real, GQA values verified
- Despot truth V2 (2026-09-28): `broadcast_to_shape` any 1..4-D; `SUM`
  (any ndim), `SUM_AXIS0/1`, `RESHAPE` VJPs exact; N-dim partial broadcast
  fails loud with whole-graph abort; permute/batched-matmul offsets checked;
  no abort on hot paths; INT8 Add row-bias-only; INT8 `scale=1.0` all-zero;
  attention/KV/GQA NaN→NUMERICAL; GELU tanh-approx documented;
  `LANCIUS_NORM_EPS` pinned; vision `NUMERICAL` scale + checked pool-bwd;
  `u64→size_t` checked; ONNX dilation/group/pads/ceil rejected; trainers
  raw-abort + chance-gate; `test_ffi_error` honest; see
  `docs/DESPOT_TRUTH_V2.md` and `probe_v2` (`/tmp/opencode/lancius-despot-logs`)
- Hardening batch V4 (2026-09-28): BROADCAST backward race-condition fixes
  (thread-local accumulators in conv2d bwd_in + MaxPool2D bwd); autodiff
  NULL checks + OOB fixes; IR memory leaks fixed; `abort()` removed from
  library code; stable API dangling pointer fixed; serialization portability
  (uint64_t, byte swapping, CRC32 call_once); threadpool timeout; VM overflow
  checks; per-channel quant + dequant; CLI fork+execvp; Python security fixes;
  build system hardening (-Werror, Threads, version consistency)
- Hardening batch V5 (2026-09-30): threadpool queue growth use-before-init fixed;
  command injection in distill_prm800k fixed; IR silent NULL returns fixed (6 locations);
  optimizer error clearing on success fixed; quantizer zero-scale check added;
  scheduler cross-entropy consistency fixed; memory planner alignment fixed;
  stable API duplicate include removed; serializer fprintf/printf removed;
  vision ops fprintf removed; example files hardened (parity_runner, run_trained_batch);
  ONNX converter shape filtering fixed; autodiff NOP comment clarified
- Despot audit V6 (2026-10-01, 45 defects): exact 4D broadcast backward
  (permute+reshape), CE_BWD ctor/exec guards, conv_bwd_w/LN/RMSN guards;
  VM ndim!=2/tape/out_reg/shape; scheduler NULL-deref/errors; pool
  malloc+free/errors; arena checked grow; static 32B align + posix_memalign;
  v2 mkstemp/ftello/CONST/ROPE/scale; v1 tmp+FP32; quantizer/optimizer/vision;
  capped fetches; distill checks; tar-slip; .d purge. See CHANGELOG §V6
  and `docs/DESPOT_TRUTH_V2.md` §12 + `probe_v6`.

## Feature freeze

v11S/v11A3 historical freeze (not v12R1 scope):
- no new operators
- no new runtime subsystems
- no new training features
- no new model-format changes

v12R1 adds correctness within scope plus additive primitives only
(`broadcast_to_shape`, TANH/MSE, per-channel quant/dequant, pool timeout);
see `CHANGELOG.md` for deltas.

## Validation batch — v12R1

Build clean under `-Wall -Wextra -Werror`; `audit_regression_13c` 49/49,
`audit_known_answer` 73/73, `audit_transformer_known_answer` 265/265,
`audit_fp32_path` 19/19, `audit_fault_injection` 11/11 green;
`make check`, `make check-long`, and `make check-sanitizers` green with
failure-propagating exit codes; broadcast/planner/softmax/CE fixes verified
by the despot probe (`audit_despot_probe`, in the `make check` gate);
V2 fixes verified by `probe_v2` (SUM-3D, RESHAPE, SUM_AXIS, N-D fail-loud,
attention NaN→NUMERICAL) + `test_grad_check` (`8.6e-10`, `5.8e-8`);
V3 fixes verified by the same gate (no new gates needed: every finding was
a hardening of an already-gated path) plus pytorch parity
(`3.42e-07`, FP64 limits);
V4 fixes verified by the same gate plus new audits for race conditions,
quantization (per-channel + dequant), serialization portability, and CLI
security (fork+execvp).
V6 fixes verified by the same gates (all green from clean tree) plus
`probe_v6` (4D dim0/1/2/3 + multi-dim, CE_BWD, VM rank) and
`test_grad_check` still `8.6e-10`, `5.8e-8`.

## Operator status

`./lancius` (`examples/lancius_cli.c`) is the ordinary-user operator:
`doctor`/`status`/`models [--check]`/`demo`/`help`, per-verb `--help`,
`datasets`/`train`/`run`/`info`/`convert`/`export`/`generate`, and a `tui`
that calls verbs directly (no re-exec), validates inputs, confirms long
trains, and pauses on ttys. User paths exec without a shell. `demo`
proves an install in one command; `make check` covers `info`/`run`.

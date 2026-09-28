# Lancius Current Status

Current internal milestone: **v12R1**
Previous internal milestone: **v11S**
Public equivalent: **V1.2RC1**
Release line: **12 R1 development**

## Phase

v12R1 is the first development milestone of the v12 cycle
(progression `S → R1 → R2 → R3 → S`; v12R1 is the R1 phase),
built on the v11S stable baseline.

The v11A3 hardening gate is complete:
- `make check` green
- `make check-long` green
- sanitizer validation green
- malformed model loading safely rejected
- version identity consistent
- model format v2 frozen with CRC32 integrity
- stable C API covers core inference workflow

## v12R1 theme: hardening plus numerical correctness

Bottom-up correctness pass over all layers, then a hostile,
formula-by-formula audit of every numeric path. Each confirmed
defect fixed and re-proven by independent execution (see CHANGELOG):
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

## Feature freeze

v11S inherits the v11A3 feature freeze:
- no new operators
- no new runtime subsystems
- no new training features
- no new model-format changes

Only critical bug fixes are accepted post-release. v12R1 adds no scope,
only correctness within existing scope.

## Validation batch — v12R1

Build clean under `-Wall -Wextra -Werror`; `audit_regression_13c` 49/49,
`audit_known_answer` 73/73, `audit_transformer_known_answer` 265/265,
`audit_fp32_path` 19/19, `audit_fault_injection` 11/11 green;
`make check`, `make check-long`, and `make check-sanitizers` green with
failure-propagating exit codes; broadcast/planner/softmax/CE fixes verified
by the despot probe (`audit_despot_probe`, in the `make check` gate);
V2 fixes verified by `probe_v2` (SUM-3D, RESHAPE, SUM_AXIS, N-D fail-loud,
attention NaN→NUMERICAL) + `test_grad_check` (`8.6e-10`, `5.8e-8`).

# Lancius Current Status

Current internal milestone: **v12A2**
Previous internal milestone: **v12R1**
Public equivalent: **TBD**
Release line: **12 A2 development**

## Phase

v12A2 is the second development milestone of the v12 cycle
(progression `S → R1 → R2 → R3 → S`; v12A2 is the R2 phase),
built on the v11S stable baseline and the v12R1 snapshot.

The v11A3 hardening gate is complete:
- `make check` green
- `make check-long` green
- sanitizer validation green
- malformed model loading safely rejected
- version identity consistent
- model format v2 frozen with CRC32 integrity
- stable C API covers core inference workflow

## v12A2 theme: numerical correctness

Hostile, formula-by-formula audit of every numeric path. Each confirmed
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

## Feature freeze

v11S inherits the v11A3 feature freeze:
- no new operators
- no new runtime subsystems
- no new training features
- no new model-format changes

Only critical bug fixes are accepted post-release. v12A2 adds no scope,
only correctness within existing scope.

## Validation batch — v12A2

Build clean under `-Wall -Wextra -Werror`; `audit_regression_13c` 49/49,
`audit_known_answer` 67/67, `audit_transformer_known_answer` 265/265,
`audit_fp32_path` 19/19 green; `make check` green with failure-propagating
exit codes; broadcast/CRC/softmax fixes verified by dedicated probes.

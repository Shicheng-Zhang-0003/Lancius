# Lancius Compatibility Manifest

## V1.2-RC3.4 compatibility and assets (2026-11-25)

RC3.4 keeps the v2 model-file format and stable C ABI. It corrects training
and numerical behavior, so retrained weights and loss trajectories may
change. The new I6 family experts use a separate seven-predicate semantic
feature contract and their own train-only normalizer; do not use the RC3.3 or
I1–I3 feature meanings/scalers with these weights. The GitHub release
attaches all four exports, checkpoints, logs, the exact corpus and normalizer,
and the expanded scenario/step audit bundle. Per-release provenance is in
`docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.4.md`.

## Dataset acquisition

`tools/data/manage_datasets.py` downloads datasets with Python's URL, gzip,
tar, and zip libraries. Downloaded payloads have size caps, and archive
members are checked before extraction. Dataset tooling is independent of the
C runtime build; it does not change the model format or C ABI.

## RC3.x model-release line (V1.2-RC3.1 → V1.2-RC3.4, 2026-10-24 → 2026-11-25)

RC3.x points are development releases. RC3.4 corrects training/runtime
numerics in `src/` but does not change the v2 serialization format or stable
C ABI. Its I6 models have a new feature/scaler contract. The current point is
`V1.2-RC3.4`.

Per-release provenance:

- `V1.2-RC3.1` — I3 verifier and I4 fire-risk assets attached to the GitHub
  release page; provenance in
  `docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.1.md`.
- `V1.2-RC3.2` — I6 trainer/corpus/eval tooling and the I5 rules engine enter
  the tree; generated corpora and checkpoints stay gitignored. Release notes
  in `docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.2.md`.
- `V1.2-RC3.3` — I6 corpus label-noise floor removed, feature 57 becomes
  `n_wrong`, the verdict becomes continuous, and all four I6 family experts
  are trained and exported. The family checkpoints are reproducible from
  `generation_1/I6/corpus_clean/`; release notes in
  `docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.3.md`.
- `V1.2-RC3.4` — compact sparse-schema family experts, algorithm-held-out
  evaluation, independently computed continuous MAE/MSE, expanded
  150-algorithm stress audit, trainer numerical fixes, and removal of the
  obsolete external dataset bridge. Complete I6 reproducibility assets are
  attached to the release; release notes in
  `docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.4.md`.

Assets, when attached, live on the GitHub release page; provenance and
per-file hashes live in the release notes for that point.

## v12R2 Milestone (development, current) + v12R1 baseline (2026-09-21) + despot truth V2/V3 (2026-09-28) + hardening V4 (2026-09-28) + V5 (2026-09-30) + despot audit V6 (2026-10-01) + despot audit V7 (2026-10-03) + despot audit V9 (2026-10-09) + despot audit V10 (2026-10-11)

v12R2 is the second development milestone of the v12 cycle, not a stable
release. Base public github tag: `V1.2-RC3`; the current point release is
`V1.2-RC3.4` (CMake/pkg-config `1.2.0` remains the same development line).
RC3 adds the unified trainer (`examples/train_verifier.c`, fused softmax
cross-entropy behind a persistent training graph, checkpoint/resume, corpus.hdr
contract, row-agnostic export) and ships its trained models as GitHub release
assets — the shapes are unchanged from RC2, so every RC2 model still loads.
It inherits the v11S contract and the v12R1/V1.2RC1 additions below, plus R2-1..R2-6 (train-lib, char-v1, micromodel, eval_verifier, sandbox, CLI eval) and despot audit V7 error-channel + grad-shape + per-channel honesty fixes + V9 math/programming/operational truth (13 defects) + V10 substrate truth (physics gate, FP32 matmul loop order, pool thread budget, softmax-bwd oracle). Enumeration in `CHANGELOG.md`; proofs in `docs/HOW_IT_WORKS.md` (concrete mechanics: pipeline, ops, training, verification) and `docs/DESPOT_TRUTH_V2.md` §12-13, §20:

- v2 model format unchanged: files written by v11S+ load with mandatory
  CRC32 body integrity (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into
  legacy unverified loads). Saver never emits `checksum==0`; loader checks
  `u64→size_t` narrowing; `BROADCAST` persists any 1..4-D shape.
- Stable C API unchanged in shape; error causes widened
  (`GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/`INVALID_HANDLE` preserved
  instead of collapsed). `lancius_add/sub/mul` set `SHAPE_MISMATCH` on
  broadcast incompat (was silent NULL).
- Broadcast `ADD`/`SUB`/`MUL` follow trailing-rank (NumPy) semantics.
  INT8 Add is row-bias only, else exact FP64 broadcast.
- Degenerate softmax/CE/attention/KV/GQA/LayerNorm/RMSNorm denominators
  report `NUMERICAL` on NaN/degenerate variance (zero-sum causal attention
  rows stay zeros).
- `lancius_matmul` is 2D-only: N-D inputs fail at build (they previously
  built a 2D node that silently dropped batch dims at execution, so no
  correctly-executing model changes behavior).
- v2 saves are tmp-file + rename (no observable change on success; crashed
  saves no longer leave truncated files); duplicate NOP ids are rejected.
- Autodiff: `broadcast_to_shape` for any 1..4-D scalar lift; `SUM`
  (any ndim), `SUM_AXIS0/1`, `RESHAPE` VJPs exact; N-dim partial broadcast
  reduction is trainable via `SUM_AXIS_ND` (R3-1) with an exact VJP (V7
  §16.4 — it was unreachable until `broadcast_to_shape` was corrected to the
  same NumPy rule its 4-D sibling already used); every unhandled forward op
  fails loud; sticky shape errors abort the whole training graph.
- Kernels: `LANCIUS_NORM_EPS=1e-5` pinned; GELU documented as tanh-approx;
  INT8 `scale=1.0` for all-zero (consistent with quantizer skip).
- Executors: checked strides/offsets/bytes on permute, batched-matmul,
  SUM/transpose/broadcast, static executors, liveness. No abort on hot paths.
- ONNX: dilations/group/auto_pad/pads/ceil_mode rejected loud.
- Trainers: raw-loss abort + accuracy≤chance exits 1; `test_ffi_error`
  exits 1 on unexpected success.
- No new operators, subsystems, training features, or format changes
  beyond additive `broadcast_to_shape` constructor (same `BROADCAST` opcode).

Hardening batch V4 (2026-09-28) — 26 defects fixed: contract deltas only,
enumeration in `CHANGELOG.md` (autodiff reduce, races, leaks, abort→errors,
dangling sched, portability, CRC call_once, pool timeout, VM overflow,
per-channel+dequant, CLI fork+execvp, Python/build).

Hardening batch V5 (2026-09-30) — 12 defects fixed: contract deltas only,
enumeration in `CHANGELOG.md` (queue use-before-init, IR NULLs→errors,
optimizer mask, dequant scale, CE R/C, 32B split, silent lib, examples,
converter 1s).

R3 training-wrap (landed; verified in V7) — additive correctness: SUM_AXIS_ND (42),
TRANSPOSE_BATCHED (43), norm/activation _BWD (44..49); old files load
identically (ids appended), new-op files require a new loader; v1 loader
rejects new ids like all post-v11S ops.

Despot audit V6 (2026-10-01) — 45 defects fixed: contract deltas (no format
break): 4D broadcast backward exact; CE_BWD 2D+scalar; conv/LN/RMSN guards;
VM ndim!=2/tape/out_reg/shape; scheduler NULL-deref/errors; pool malloc+free
+errors; arena checked grow; static 32B align (plan requires aligned base,
bump aligns start); v2 mkstemp/ftello/empty/CONST/ROPE/scale; v1 tmp+FP32;
quantizer/optimizer/vision; capped fetches; tar-slip; .d purge.
Enumeration in `CHANGELOG.md`, proofs in `docs/DESPOT_TRUTH_V2.md` §12.

Despot audit V9 (2026-10-09) — 13 defects, no format break, additive API only:
CE corrupt-label NUMERICAL, INT8-matmul scale NUMERICAL, v2-only saves
(`lancius_graph_save_v1_legacy` explicit opt-in), v1 trunc-leak + LIMIT,
handle cookies + owner (`INVALID_HANDLE` without 64-bit guess, opaque ABI
unchanged), `pool_submit` NULL_PTR, `lancius_vm_execute_checked` additive
(legacy `vm_execute` delegates with `SIZE_MAX`), planner NULL-on-error,
Python open+fstat+capped reads + 8GB inflate caps, CLI topk 1..100.
Enumeration in `CHANGELOG.md`, proofs in `docs/DESPOT_TRUTH_V2.md` §20.

Despot audit V10 (2026-10-11) — no format break, no API change, no numerical
behaviour change: `kernel_matmul_f32` is restructured (IKJ over a
register-blocked column tile) and proven bit-identical over 12 shapes
including the non-multiples-of-8 tile boundaries, so no model's outputs move;
the FP64-accumulate convention the external oracle pins is unchanged. The
thread pool pins each worker's OpenMP team to 1, which changes only how many
OS threads exist and makes pool execution bit-deterministic — results are the
same, the thread count and the reproducibility are not. Two new gates
(`make check-physics`, and the softmax-backward block inside the existing
graph oracle) and one new measurement (`make bench_kernels`) are additive and
run nowhere near the ABI. Enumeration in `CHANGELOG.md`, proofs in
`docs/DESPOT_TRUTH_V2.md` §21.

Internal headers and experimental paths (transformer builders, training
loops, ONNX converter, quantizer) may change in v12R2 without notice.

---

# Lancius v11S Compatibility Manifest (stable, retained)

Version: v11S
Release Line: 1.1 Stable

## Purpose

This document defines the compatibility contract of Lancius v11S.

## Stable Components

### Runtime

-   Graph execution
-   Tensor management
-   Memory planning
-   Scheduler integration

### API
The stable public interface is provided through the Lancius stable API
headers (`lancius_stable_api.h`).

Stable API functions (v11S):
-   `lancius_create_context` / `lancius_destroy_context`
-   `lancius_graph_create_stable` / `lancius_graph_destroy_stable`
-   `lancius_graph_load_stable` / `lancius_graph_save_stable`
-   `lancius_add_input` / `lancius_add_matmul` / `lancius_add_relu`
-   `lancius_bind_data` / `lancius_compile_and_run` / `lancius_read_output`
-   `lancius_tensor_element_count` / `lancius_tensor_get_dtype`
-   `lancius_get_last_error` / `lancius_get_error_string`

Internal headers may change.

### Model Execution

Supported:

-   Static computation graphs
-   Supported operator set
-   CPU execution path

## Compatibility Guarantees

v11S guarantees:

-   reproducible builds
-   stable public API behavior
-   documented limitations
-   regression-tested core execution

## Non-Guarantees

The following are not guaranteed:

-   binary compatibility with experimental components
-   unsupported operators
-   unfinished backends

## Versioning

v11S represents the first stable release of the 1.1 cycle.

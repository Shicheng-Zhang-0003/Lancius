# Lancius Compatibility Manifest

## v12R1 Milestone (development, 2026-09-21) + despot truth V2/V3 (2026-09-28) + hardening batch V4 (2026-09-28) + hardening batch V5 (2026-09-30)

v12R1 is the first development milestone of the v12 cycle, not a stable
release. Public github tag: `V1.2RC1`. It inherits the v11S contract below, with the following additions:

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
  reduction fails loud (no `SUM_AXIS_ND` yet); every unhandled forward op
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

Hardening batch V4 (2026-09-28) — 26 defects fixed:

- Autodiff: BROADCAST backward correctly reduces over broadcast dimensions;
  NULL checks on `fwd_n->inputs`; OOB reads on shape/axes arrays fixed (pads
  to 4D); off-by-one in node capacity check fixed.
- Kernels: race conditions in `kernel_conv2d_bwd_in` and MaxPool2D backward
  fixed (thread-local accumulators).
- Memory: IR node allocation leaks fixed; `abort()` removed from library code;
  all `fprintf`/`printf` replaced with `lancius_set_error`.
- Stable API: dangling `wrapper->sched` fixed; `set_owner` updates `int8_owner`.
- Serialization: portability fixed (`uint64_t`, byte swapping); CRC32 table
  init race fixed (`call_once`); NOP IDs no longer mapped to NULL; double-read
  for CRC eliminated.
- Threadpool: `lancius_pool_wait` timeout support added.
- Bytecode VM: overflow checks added.
- Quantization: per-channel quantization and dequantization support added.
- CLI: command injection fixed (`fork+execvp`).
- Python: security fixes, stale versions removed, error handling improved.
- Build: version consistency, `-Werror`, Threads dependency; `train_cifar10`
  links `-lpthread`; `.gitignore` updated; `lancius.pc.in` version fixed.
- Code quality: magic numbers replaced with named constants.

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

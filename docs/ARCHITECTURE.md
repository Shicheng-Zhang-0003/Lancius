# Lancius v12R2 Architecture Overview

## High Level Pipeline

    Model Input
        |
        v
    Frontend / Loader
        |
        v
    Intermediate Representation
        |
        v
    Compiler Passes
        |
        v
    Memory Planning
        |
        v
    Scheduler
        |
        v
    Runtime Executor
        |
        v
    Kernel Backend

## Core Principles

### Separation

The architecture separates:

-   model representation (IR DAG, `src/ir/lancius_ir.c`)
-   optimization (fusion `lancius_optimizer.c`, quantization `lancius_quantize.c`)
-   execution (scheduler `lancius_scheduler.c`, VM `lancius_bytecode.c`)
-   hardware interaction (kernels `lancius_kernels.c`, threadpool, arena)
-   persistence (v2 serializer `lancius_serialize_v2.c`, validators, checked math)

### Runtime First

Lancius v12R2 prioritizes predictable execution over maximum feature
count. Every hot path returns errors (`OOM/OVERFLOW/NUMERICAL/SHAPE_MISMATCH`)
instead of aborting or emitting silent values. The despot truth batch V2
closed the last silent-drop paths (see `docs/DESPOT_TRUTH_V2.md`).
Hardening batches V4 and V5 extended this to threadpool, IR, examples,
and Python tooling. R3 training-wrap adds per-axis N-dim reduction, batched transpose,
and norm/activation backwards (all pre-router executors, all fail-loud
outside scope). Despot audit V7 (2026-10-03) closed the remaining
silent-success paths: OpenMP worker errors, save/KV/VM/pool error
channels, grad-shape guards, per-channel refuse, stable-handle magic.

### Memory Awareness

Memory planning is a core subsystem responsible for reducing unnecessary
allocations. The linear-scan planner assigns every intermediate tensor a
recorded flat-buffer offset with wave-liveness reuse (`birth`=producing wave,
`death`=max consuming wave, sinks pinned, `death<birth` expiry, first-fit +
32B align); pooled execution is verified value-identical to direct execution
on diamond graphs. Arena uses 32B footprints with `SIZE_MAX`-guarded
`uintptr_t` align math. All profiling uses `_checked` sizing, never aborting.

### Numerical Honesty

v12R2 executes N-dimensional trailing-rank broadcast
(`out[I]=A[bcast(I)] OP B[bcast(I)]`, `out=max(a,b)`), max-subtracted
softmax/CE with `NUMERICAL` zero-sum guards, Flash/GQA/KV-cache attention
with `NaN→NUMERICAL` (zero stays zeros for causal safety), tanh-approx GELU
(measured 4.74e-04 vs erf-exact, pinned by the oracle), `LANCIUS_NORM_EPS=1e-5` norms, int64 INT8
accumulation (`scale=1.0` for all-zero, not `1e-8`), and fail-loud autodiff
(`broadcast_to_shape` for any 1..4-D scalar lift; `SUM/SUM_AXIS0/1/RESHAPE`
VJPs exact; N-dim partial reduction fails loud instead of training as zero;
BROADCAST backward correctly reduces over broadcast dimensions).
Loaders enforce v2 CRC32 integrity (`checksum==0` rejected by default);
corrupt shapes return errors, never silent values. ONNX strictly rejects
dilations/groups/auto_pad/pads/ceil_mode instead of silent dense compute.
Trainers abort on raw `NaN/>1000/<0` and exit 1 at ≤ chance accuracy.
Per-channel quantization and dequantization are supported alongside
per-tensor quantization.

## Subsystem Contracts (despot V2/V3 + hardening batches V4/V5 + despot V6)

- **Kernels:** pure pointers+dims, FP64-first, FP32 matmul with FP64 accum,
  INT8 symmetric per-tensor and per-channel, dequantization supported,
  no hidden quantization. Thread-local accumulators in `kernel_conv2d_bwd_in`
  and MaxPool2D backward eliminate race conditions.
- **Scheduler:** strided N-dim broadcast, checked strides/offsets/bytes on
  every path (permute, batched-matmul, SUM, transpose, static executors).
  INT8 Add is row-bias only, else exact FP64 broadcast.
- **Autodiff:** every forward op has an explicit VJP or fails loud;
  `accum_grad` returns `1/0`, sticky `SHAPE_MISMATCH` aborts the whole
  training graph. BROADCAST backward correctly reduces over broadcast
  dimensions. NULL checks on `fwd_n->inputs`; OOB reads on shape/axes arrays
  fixed (pads to 4D). See `docs/DESPOT_TRUTH_V2.md §3`.
- **Persistence:** v2 frozen, 48B header + 104B nodes, LE-only, CRC over
  `48..EOF`, sparse-ID bounds, O(1) duplicate detection, `u64→size_t`
  narrowing checked, `BROADCAST` any 1..4-D. Serialization portability
  fixed (`uint64_t`, byte swapping); CRC32 table init race fixed
  (`call_once`); NOP IDs no longer mapped to NULL.
- **Threadpool:** `lancius_pool_wait` timeout support; race conditions
  eliminated via thread-local accumulators in backward kernels.
- **Stable API:** opaque handles, widened error codes; dangling
  `wrapper->sched` fixed; `set_owner` updates `int8_owner`.
- **CLI:** user paths executed via `fork+execvp` (no shell injection).
- **Quantization:** per-tensor and per-channel quantization; dequantization
  support; V6 never clears sticky error, per-tensor drops stale per-channel,
  dequant checks scales + frees stale FP64.
- **Build:** `-Werror` enforced; version consistency; Threads dependency;
  V6 `.d` purge, tar-slip validation, capped fetches.
- **Validation:** every audit in `make check` propagates failures; `probe_v2`
  pins SUM-3D, RESHAPE, SUM_AXIS, `broadcast_to_shape`, N-D partial fail-loud,
  attention NaN→NUMERICAL; V6 `probe_v6` pins 4D dim0/1/2/3, CE_BWD, VM rank.
- **V6 deltas:** VM `ndim!=2`/tape/`out_reg`/shape; scheduler NULL-deref/errors;
  pool `malloc+free` + errors; arena checked grow; static 32B align;
  v2 `mkstemp`/`ftello`/CONST/ROPE/scale; v1 tmp+FP32; fusion shape equality;
  builders/vision validated. See `docs/DESPOT_TRUTH_V2.md` §12.

## Future Direction

v12 development may expand:

-   ~~`SUM_AXIS_ND` for per-axis N-dim broadcast grad reduction~~ — shipped in
    R3-1 and made reachable in V7 (`docs/DESPOT_TRUTH_V2.md` §16.4); the op
    trains and its VJP is exact to ~1e-9 by central difference
-   FP32 operator expansion + FP32 KV-cache
-   broader ONNX coverage + dynamic shape exploration
-   backend support, optimization passes, ecosystem integration


## V7: invariants the runtime now guarantees

These are contracts, not implementation notes. Each is enforced by a check in
`examples/audit_v7_hardening.c` and each was found by breaking it first.

### The sticky-error contract

`lancius_set_error` writes `_Thread_local` state. The runtime therefore
guarantees:

1. **A successful op leaves `lancius_get_error()` at `OK`.** Every branch in
   `execute_node_math` that computes a result terminates with `return`.
   `SUM_AXIS_ND` (opcode 42) once did not: it fell through to the
   `op >= LANCIUS_OP_CONV2D` router, which rejected it with `UNSUPPORTED_OP`
   after the values were already correct.
2. **An error raised inside an OpenMP worker is invisible to the caller.**
   Every parallel kernel that can fail propagates through an explicit shared
   flag that the master re-sets after the region (`omp_err_*` in
   `lancius_kernels.c`, `omp_alloc_failed` in `lancius_vision_ops.c`).
3. **A sticky error aborts the whole training graph.**
   `lancius_ir_autodiff` clears at entry and returns `NULL` on any error, so a
   NULL grad can never train as zero.

Together these mean an error check after a successful call is a *meaningful*
assertion, which is what makes them testable.

### Builder and executor must agree

A builder is a proof that a node is executable. When they disagree, the op is
either unbuildable or unloadable, and the executor's capability is a lie.

`lancius_broadcast`, `lancius_broadcast_4d` and the BROADCAST executor all
implement one rule: trailing-rank alignment with per-dim `da == 1 || da ==
out[i]`. `lancius_broadcast_to_shape` implemented a stricter one (exact shape
equality), which silently disabled `SUM_AXIS_ND`'s documented VJP and made
1-D/3-D expanding broadcasts unbuildable while their 2-D/4-D twins worked.

Any new builder is expected to accept exactly what its executor accepts, and
`H8` of the V7 audit pins that at every rank with a v2 save/load round-trip.

### Arena contract

- Every allocation is 32-byte aligned (AVX2), whether the caller asks for it or
  takes the default. Verified across 64 independent arenas, because glibc only
  guarantees 16 and a single-arena check passes by luck half the time.
- Requested alignment must be a power of two and at most 1 MiB. An uncapped
  alignment wraps `ALIGN_UP` and forces a gigantic grow — an OOM-DoS.
- `size == 0` allocates one byte, so two zero-size allocations never alias.

### Autodiff teardown

`lancius_ir_autodiff` has one allocation of `grad_nodes` and roughly fifty ways
to fail. All fifty route through `autodiff_abort()`, which frees `grad_nodes`,
`fwd_to_full`, `grad_map`, the cloned graph and the struct. A new failure
branch that rolls its own free list is a review failure, because that is exactly
how 45 branches leaked the same buffer.

### Gate integrity

The gates are themselves audited. `make check-mutation` injects real defects
and requires the gate to go red; `make check-oracle` recomputes every kernel and
graph op from a source that does not link Lancius. See
`docs/DESPOT_TRUTH_V2.md` §16 for the per-mutation record.

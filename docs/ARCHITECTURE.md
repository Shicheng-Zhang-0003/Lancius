# Lancius v12R1 Architecture Overview

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

Lancius v12R1 prioritizes predictable execution over maximum feature
count. Every hot path returns errors (`OOM/OVERFLOW/NUMERICAL/SHAPE_MISMATCH`)
instead of aborting or emitting silent values. The despot truth batch V2
closed the last silent-drop paths (see `docs/DESPOT_TRUTH_V2.md`).

### Memory Awareness

Memory planning is a core subsystem responsible for reducing unnecessary
allocations. The linear-scan planner assigns every intermediate tensor a
recorded flat-buffer offset with wave-liveness reuse (`birth`=producing wave,
`death`=max consuming wave, sinks pinned, `death<birth` expiry, first-fit +
32B align); pooled execution is verified value-identical to direct execution
on diamond graphs. Arena uses 32B footprints with `SIZE_MAX`-guarded
`uintptr_t` align math. All profiling uses `_checked` sizing, never aborting.

### Numerical Honesty

v12R1 executes N-dimensional trailing-rank broadcast
(`out[I]=A[bcast(I)] OP B[bcast(I)]`, `out=max(a,b)`), max-subtracted
softmax/CE with `NUMERICAL` zero-sum guards, Flash/GQA/KV-cache attention
with `NaN→NUMERICAL` (zero stays zeros for causal safety), tanh-approx GELU
(documented, ~2e-3 vs erf), `LANCIUS_NORM_EPS=1e-5` norms, int64 INT8
accumulation (`scale=1.0` for all-zero, not `1e-8`), and fail-loud autodiff
(`broadcast_to_shape` for any 1..4-D scalar lift; `SUM/SUM_AXIS0/1/RESHAPE`
VJPs exact; N-dim partial reduction fails loud instead of training as zero).
Loaders enforce v2 CRC32 integrity (`checksum==0` rejected by default);
corrupt shapes return errors, never silent values. ONNX strictly rejects
dilations/groups/auto_pad/pads/ceil_mode instead of silent dense compute.
Trainers abort on raw `NaN/>1000/<0` and exit 1 at ≤ chance accuracy.

## Subsystem Contracts (despot V2)

- **Kernels:** pure pointers+dims, FP64-first, FP32 matmul with FP64 accum,
  INT8 symmetric per-tensor, no hidden quantization.
- **Scheduler:** strided N-dim broadcast, checked strides/offsets/bytes on
  every path (permute, batched-matmul, SUM, transpose, static executors).
  INT8 Add is row-bias only, else exact FP64 broadcast.
- **Autodiff:** every forward op has an explicit VJP or fails loud;
  `accum_grad` returns `1/0`, sticky `SHAPE_MISMATCH` aborts the whole
  training graph. See `docs/DESPOT_TRUTH_V2.md §3`.
- **Persistence:** v2 frozen, 48B header + 104B nodes, LE-only, CRC over
  `48..EOF`, sparse-ID bounds, O(1) duplicate detection, `u64→size_t`
  narrowing checked, `BROADCAST` any 1..4-D.
- **Validation:** every audit in `make check` propagates failures; `probe_v2`
  pins SUM-3D, RESHAPE, SUM_AXIS, `broadcast_to_shape`, N-D partial fail-loud,
  attention NaN→NUMERICAL.

## Future Direction

v12 development may expand:

-   `SUM_AXIS_ND` for per-axis N-dim broadcast grad reduction (currently
    fails loud, honestly)
-   FP32 operator expansion + FP32 KV-cache
-   broader ONNX coverage + dynamic shape exploration
-   backend support, optimization passes, ecosystem integration

# Lancius Changelog

## v12R1 (unreleased) — hardening batch — 2026-09-12

Bottom-up correctness pass over all layers (no format / stable-ABI break):

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
  KV-cache OOM + zero-len guards; MaxPool `-INFINITY` + NaN propagate;
  INT8 zero-scale is now a loud error; quantizer clamp + skip-if-INT8.
- Planner/threadpool: all allocs checked, id bounds, overflow-safe
  offsets, threadpool create/teardown hardening.
- Persistence: v2 save CRC fail-closed + `w+b` read-back fix (was silent
  crc=0), v1 loader ndim/duplicate-id/unknown-op hardening, v2 header
  reserved-field + duplicate-NOP rejection, stable API error-map
  completion + FP64-only `read_output` honesty + checked counts.
- Audits updated to `-INFINITY` / unclamped GELU references.
- Removed 15 stale `*.bak*` / `*backup*` files.

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

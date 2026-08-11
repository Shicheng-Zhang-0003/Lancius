# Lancius Changelog

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

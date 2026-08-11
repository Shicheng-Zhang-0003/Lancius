# Lancius v11S Release Notes

## Lancius 1.1 Stable

Lancius v11S marks the first stable release of the 1.1 development cycle.

This release represents the completion of the v11A3 hardening gate:
feature freeze, loader hardening, model-format freeze with CRC32 integrity,
sanitizer and fuzz validation, and full regression defense.

## Highlights

-   Model format v2 frozen with CRC32 body integrity verification
-   Stable C API covers core inference workflow (load, save, build, run, read)
-   Zero-allocation static graph execution via linear-scan memory planner
-   Transformer inference: KV-cache, Flash Attention, RoPE, GQA, SwiGLU
-   FP32 matmul execution path with FP64 accumulation
-   Full validation suite: 400+ checks, zero failures
-   O(1) arena allocator with 32-byte SIMD alignment
-   Threadpool wave-parallel execution with parity verification

## Who Should Use This Release

Recommended for:
-   edge deployment and bare-metal inference
-   developers building on a stable C API
-   researchers needing deterministic, reproducible execution

## Migration

Users moving from v11A3 should verify:
-   model files (v2 format unchanged, CRC32 now written on save)
-   stable API usage (new functions added, no breaking changes)
-   ONNX converter output (writes v2 with checksum=0, accepted as legacy)

## Known Limitations

See `KNOWN_LIMITATIONS.md` for the full list.

## Future

The v12 development cycle will focus on:
-   FP32 operator expansion beyond matmul
-   FP32 KV-cache storage
-   Broader ONNX operator coverage
-   Dynamic shape exploration

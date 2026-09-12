# Lancius v12R1 Known Limitations

This document defines the explicit boundaries of the v12R1
development milestone.

A development milestone is not defined by having every feature. It is defined
by having a clear and reliable contract.

## Supported

-   CPU inference execution
-   Static graph execution
-   Stable C API
-   Core tensor operations
-   Selected transformer inference operators
-   Serialization and runtime loading

## Experimental or Deferred

The following areas are intentionally not considered stable:

-   GPU acceleration
-   Dynamic shape execution
-   Distributed execution
-   Full training ecosystem
-   Advanced quantization workflows
-   GGUF export pipeline

## Training Status

Lancius v12R1 is inference-first.

Training-related components may exist in the codebase but should not be
considered development preview.

## Backend Support

Primary supported environment:

-   Linux
-   x86_64 CPU

Additional architectures may require validation.

## Fatal Invariants

The following conditions are guaranteed to abort the process.
They represent internal memory-safety invariants that are NOT
reachable through the stable C API (builders, loaders, and stable
API validate before reaching them):

-   Tensor ndim exceeds 4 (max rank violation)
-   Tensor element count overflows or exceeds 100,000,000
-   Tensor byte count exceeds 800,000,000

All user-reachable errors (unsupported ops, shape mismatches,
malformed models) return error codes via `lancius_set_error()`
and do NOT abort.

## Narrow-but-honest components

-   Bytecode VM: 2D MLP ops only (`MATMUL/ADD/SUB/MUL/RELU/
    BROADCAST/SOFTMAX/SUM`); anything else fails `compile` with NULL.
-   ONNX converter: 9 ops only
    (`Conv/Relu/MaxPool/Flatten/MatMul/Add/Reshape/Gemm/Transpose`);
    anything else raises instead of emitting partial graphs.
-   Quantizer: 4D FP64 conv weights only; all-zero weights stay FP64.
-   `lancius_read_output` (stable API): FP64 outputs only; FP32/INT8
    outputs report `UNSUPPORTED_OP`.
-   v2 models always carry non-zero CRC32; `checksum == 0` means legacy
    pre-hardening file, accepted without verification.

## Philosophy

Limitations are documented intentionally to prevent unsupported
assumptions.

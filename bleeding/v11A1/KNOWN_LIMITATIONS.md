# Lancius v11A1 Known Limitations

This document defines the explicit boundaries of the stable v11A1
release.

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

Lancius v11A1 is inference-first.

Training-related components may exist in the codebase but should not be
considered development preview.

## Backend Support

Primary supported environment:

-   Linux
-   x86_64 CPU

Additional architectures may require validation.

## Philosophy

Limitations are documented intentionally to prevent unsupported
assumptions.

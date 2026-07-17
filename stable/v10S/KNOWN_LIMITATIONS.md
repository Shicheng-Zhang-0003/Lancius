# Lancius v10S Known Limitations

This document defines the explicit boundaries of the stable v10S
release.

A stable release is not defined by having every feature. It is defined
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

Lancius v10S is inference-first.

Training-related components may exist in the codebase but should not be
considered production-ready.

## Backend Support

Primary supported environment:

-   Linux
-   x86_64 CPU

Additional architectures may require validation.

## Philosophy

Limitations are documented intentionally to prevent unsupported
assumptions.

# Lancius Compatibility Manifest

## v12R1 Milestone (development, 2026-09-21)

v12R1 is the first development milestone of the v12 cycle, not a stable
release. It inherits the v11S contract below, with the following additions:

- v2 model format unchanged: files written by v11S+ load with mandatory
  CRC32 body integrity (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into
  legacy unverified loads).
- Stable C API unchanged in shape; error causes widened
  (`GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/`INVALID_HANDLE` preserved
  instead of collapsed).
- Broadcast `ADD`/`SUB`/`MUL` follow trailing-rank (NumPy) semantics.
- Degenerate softmax/cross-entropy denominators report `NUMERICAL`.
- No new operators, subsystems, training features, or format changes.

Internal headers and experimental paths (transformer builders, training
loops, ONNX converter, quantizer) may change in v12R2 without notice.

---

# Lancius v11S Compatibility Manifest (stable, retained)

Version: v11S\
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

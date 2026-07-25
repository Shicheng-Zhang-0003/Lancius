# Lancius v11A1

> **Development milestone — not a stable release.**

Lancius `v11A1` is the first development milestone in the v11 cycle.
It focuses on landing the v11 foundation, improving runtime honesty, and strengthening validation.

This release is intended for development, testing, and contribution workflows.
It is **not** intended as a production-stable release.

---

## Highlights

- Unified `v11A1` version identity across docs and build tooling
- Improved Makefile consistency and library composition
- Finished the buffer ownership contract:
  - external
  - arena
  - pool
  - owned heap
- `lancius_graph_destroy()` now releases graph-owned heap buffers
- Unsupported operations now fail loudly instead of silently doing nothing
- Implemented `Permute` and `MatMulBatched`
- Introduced the v2 model-format direction:
  - explicit fixed-width format
  - C-side v2 support
  - legacy v1 load fallback retained
- Added basic shape and serialization hardening
- Improved static execution and memory-plan correctness
- Removed hidden execution-time INT8 quantization side effects
- Improved transformer-path honesty:
  - experimental transformer demos clearly marked
  - unsupported transformer backward ops fail loudly
- Cleaned up the stable C API boundary
- Added stronger test discipline:
  - `make check`
  - known-answer audits
  - regression audits
  - sanitizer-oriented validation

---

## Important fixes

- Fixed a use-after-free caused by stale runtime-state pointers after runtime table reallocation
- Fixed scheduler fallthrough where transformer ops could incorrectly enter the vision router
- Fixed small C correctness issues:
  - signed-shift UB in dataset readers
  - misleading cross-entropy loss display hack
  - unimplemented kernel declarations removed from public headers

---

## Compatibility

`v11A1` is a development milestone.

- The v2 model format is introduced, but long-term binary compatibility is **not guaranteed yet**
- Legacy v1 models can still be loaded via fallback
- Stable API improvements are ongoing and may still change before `v11S`

---

## Known limitations

The following remain intentionally out of scope for `v11A1`:

- GPU acceleration
- dynamic shapes
- full production training support
- complete ONNX operator coverage
- full LLM serving runtime
- final binary compatibility guarantees

---

## Validation

Typical validation:

```bash
make clean
make
make check

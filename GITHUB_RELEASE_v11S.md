# Lancius v11S — Stable Release

**Tag:** `v11S`
**Public version:** V1.1
**License:** GPL-3.0-or-later

---

## Overview

Lancius v11S is the **first stable release** of the Lancius 1.1 cycle. It represents the completion of the v11A3 hardening gate: feature freeze, loader hardening, model-format freeze with CRC32 integrity, sanitizer and fuzz validation, and full regression defense.

Lancius is a lightweight C machine-learning compiler and runtime focused on bare-metal inference, static graph execution, memory planning, and low-level runtime control.

---

## Highlights

### Model Format v2 — Frozen with Integrity
- **CRC32 body integrity check** on all v2 model files (bytes 48..EOF)
- Reserved header flags rejected at load time
- Binary compatibility **guaranteed** for v2 models written by v11S+
- Legacy v1 loading available as deprecated fallback

### Stable C API
- Opaque handles with thread-local error states (safe for Python/Rust/Go FFI)
- Model load/save via `lancius_graph_load_stable` / `lancius_graph_save_stable`
- Graph construction, data binding, execution, and output reading
- Output reading with truncation protection (no silent data loss)
- Tensor introspection (element count, dtype query)

### Zero-Allocation Static Execution
- Linear-scan liveness analysis with graph coloring
- Static flat-buffer execution (zero heap allocation during inference)
- 90% RAM reduction on deep sequential chains vs naive allocation

### Transformer Inference (Experimental)
- KV-cache runtime object with prefill/generation flows
- Flash Attention (online softmax, O(D) memory per head)
- RoPE positional encoding, GQA, SwiGLU, RMSNorm, LayerNorm
- 265-check known-answer audit with zero failures

### Validation
- **400+ checks, zero failures** across the full validation suite
- AddressSanitizer + UndefinedBehaviorSanitizer clean
- 10,000 malformed binary injection test — zero crashes
- Finite-difference gradient verification (< 1e-4 threshold)
- Threadpool parity audit — zero race conditions

---

## Breaking Changes from v11A3

- `lancius_graph_save()` now returns `int` (0 = success, -1 = failure)
- `lancius_read_output()` returns `LANCIUS_ERR_BUFFER_TOO_SMALL` instead of silently truncating
- `lancius_vm_execute()` returns `int` (0 = success, -1 = OOM) instead of `void`
- Reserved `EXTERNAL_WEIGHTS` header flag is now rejected by the loader

---

## Platform Support

- **Primary:** Linux x86_64 (GCC/Clang, OpenMP)
- **Build:** `make clean && make -j$(nproc)`
- **Validation:** `make check && make check-long && make check-sanitizers`

---

## Known Limitations

- CPU-only (no GPU acceleration)
- Static graph execution only (no dynamic shapes)
- Transformer inference is experimental (FP64-only KV-cache)
- FP32 execution scoped to matmul only
- ONNX conversion experimental (LeNet-class graphs only)
- Training is not part of the stable contract

See `KNOWN_LIMITATIONS.md` for the full list.

---

## Building from Source

```bash
git clone <repo-url> && cd lancius
make clean && make -j$(nproc)
make check
```

### CMake (alternative)
```bash
cmake -B build && cmake --build build
```

---

## Full Changelog

See `CHANGELOG.md` for the complete v11 cycle changelog.

---

*Built with ❤️ and a deep distrust of undefined behavior.*

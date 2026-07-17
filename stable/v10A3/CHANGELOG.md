# Changelog

## [1.0.0-rc1] - 2024-07-14 (v10A3 Release Candidate)
### Added
- **Stable FFI API:** Production-ready opaque handles and thread-local error states for Python/Rust/Go bindings.
- **Ecosystem Bridge:** ONNX/PyTorch interoperability tools and GGUF export compatibility (`lancius_bridge.py`).
- **INT8 Symmetric Quantization:** Post-training quantization and dynamic mixed-precision edge deployment.
- **Wave Scheduler:** Dependency-aware topological scheduling with OpenMP parallel execution.
- **Modern LLM Kernels:** Flash Attention, RMSNorm, SwiGLU, GQA, and RoPE.
- **Linear Scan Memory Planner:** Liveness analysis and graph coloring for static flat-buffer pooling (90% RAM reduction).

### Hardened (v10A3 Zero-Tolerance Code Freeze)
- **SIMD Boundary Trap:** Enforced strict 32-byte alignment in the Arena Allocator to prevent AVX2 General Protection Faults.
- **Cyclic Graph Abort:** Implemented strict cycle detection in the Wave Scheduler to prevent infinite recursion and stack overflows on malformed payloads.
- **Numerical Clamping:** Added `[-50.0, 50.0]` safety envelopes to SwiGLU and GELU kernels to prevent `exp()` infinity explosions during high-variance inference.
- **Adversarial Soak Gauntlet:** Verified zero memory leaks under ASan when subjected to 10,000 corrupted binaries, 2,000 KV-Cache autoregressive steps, and extreme FP64 value injection.
- **Deserialization Hardening:** Implemented strict bounds checking and safe teardowns for truncated or malicious `.lancius` files.

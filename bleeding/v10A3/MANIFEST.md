# Lancius Compatibility Manifest (v10A3 Release Candidate)

This document defines the strict compatibility contract for the Lancius v1.0 Release Candidate.

## 1. Binary Format Contract
* **Magic Number:** `0x21434E41` ("LANC!" in little-endian)
* **Format Version:** `1`
* **Endianness:** Little-Endian
* **Tensor Alignment:** 32-byte boundary (AVX2/SIMD safe)

## 2. Supported Platforms
* **Primary:** Linux x86_64 (GCC/Clang, AVX2/FMA3 required for optimal performance)
* **Secondary:** Linux ARM64 (NEON fallback)
* **OS:** POSIX-compliant (Requires `pthreads`, standard `malloc`)

## 3. Ecosystem Interoperability
* **ONNX:** Opset 17 (via `onnx_to_lancius.py`)
* **GGUF:** Export supported via `lancius_bridge.py`

## 4. Supported Operators (v10A3)
* **Vision:** Conv2D, MaxPool2D, Flatten, Reshape, Conv2D+ReLU (Fused)
* **Linear:** MatMul, Add, Sub, Mul, Broadcast, Transpose
* **Activations:** ReLU, Softmax, GELU, SwiGLU
* **LLM/Transformer:** LayerNorm, RMSNorm, Multi-Head Attention (Causal), Grouped-Query Attention (GQA), RoPE
* **Training:** Cross-Entropy, Symbolic Autodiff (VJP)

# Lancius v11A1 Operator Support

This matrix describes operator support in the current v11A1 tree.

## Core Ops

| Op | Forward | Backward | Serialization | Notes |
|---|---:|---:|---:|---|
| Add | yes | yes | yes | |
| Sub | yes | yes | yes | |
| Mul | yes | yes | yes | |
| MatMul | yes | yes | yes | |
| Relu | yes | yes | yes | |
| Softmax | yes | yes | yes | 2D only |
| Sum | yes | yes | yes | |
| Broadcast | yes | yes | yes | |
| Transpose | yes | yes | yes | 2D only |

## Vision Ops

| Op | Forward | Backward | Serialization | Notes |
|---|---:|---:|---:|---|
| Conv2D | yes | yes | yes | |
| MaxPool2D | yes | yes | yes | |
| Flatten | yes | yes | yes | |
| Conv2D+ReLU fused | yes | yes | yes | |
| Reshape | yes | yes | yes | |

## Training Ops

| Op | Forward | Backward | Serialization | Notes |
|---|---:|---:|---:|---|
| CrossEntropy | yes | yes | yes | |
| Conv2D backward | yes | yes | yes | |
| Conv2D weight backward | yes | yes | yes | |
| MaxPool backward | yes | yes | yes | |
| Relu backward | yes | yes | yes | |
| Softmax backward | yes | yes | yes | |

## Transformer Ops

| Op | Forward | Backward | Serialization | Notes |
|---|---:|---:|---:|---|
| LayerNorm | yes | no | yes | experimental |
| GELU | yes | no | yes | experimental |
| RoPE | partial | no | yes | experimental |
| Attention | partial | no | yes | KV-cache path incomplete |
| RMSNorm | yes | no | yes | experimental |
| SwiGLU | yes | no | yes | experimental |
| GQA | yes | no | yes | experimental |

## Unsupported in v11A1

These ops must fail loudly at runtime:

| Op | Status |
|---|---|
| Permute | implemented in Task 5 |
| MatMulBatched | implemented in Task 5 |
| Embedding | unsupported |
| KVCacheRead | unsupported |
| KVCacheWrite | unsupported |

## Transformer honesty (Task 10)

- Attention execution uses the KV-cache kernel for single-query generation.
- Mismatched Q/KV sequence lengths fail loudly.
- Transformer backward ops are unsupported and fail loudly in autodiff.
- `generate_text` and `run_llm` are experimental demos, not stable LLM serving.

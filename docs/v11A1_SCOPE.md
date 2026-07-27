# Lancius v11A1 Scope

## Theme

Land the v11 foundation.

## Included in v11A1

- Version identity is v11A1 everywhere
- Stable API compiled into main library
- Memory planner compiled into main library
- Graph-owned memory released by graph destruction
- Unsupported runtime ops fail loudly
- Permute implemented
- MatMulBatched implemented
- Model format status documented
- Transformer kernels marked experimental
- Tests verify values, not just allocation

## Not included in v11A1

- Full LLM runtime
- Dynamic shapes
- GPU
- Production training
- Full ONNX coverage
- Binary compatibility guarantee

## Deferred to v11A2

- KV-cache runtime object
- Full transformer execution pipeline
- FP32 performance kernels
- Expanded ONNX operator support
- Quantization calibration workflow

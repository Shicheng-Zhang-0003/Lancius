<!-- SECTION:HEADER -->
# Lancius v11A2

> **Internal milestone:** `v11A2`
> **Public release:** `V1.1-AlphaRC2`
> **Status:** Development milestone — **not a stable release**

Lancius is a lightweight C machine-learning compiler and runtime focused on
bare-metal inference, static graph execution, memory planning, and low-level
runtime control.

`v11A2` is the second development milestone in the Lancius v11 cycle.

Its theme is:

> **Transformer runtime usability.**
<!-- /SECTION:HEADER -->

<!-- SECTION:RELEASE_IDENTITY -->
## Release Identity

| Internal Version | Public Version       | Release Type      |
|------------------|----------------------|-------------------|
| `v11A2`          | `V1.1-AlphaRC2`      | Development Alpha |

Lancius uses the following internal milestone progression:

```text
S → A1 → A2 → A3 → S
```

Where:

- `S` is a stable release
- `A1` is the first development milestone after the previous stable release
- `A2` is the second development milestone, where new subsystems begin becoming independent
- `A3` is the freeze, hardening, and bug-hunting milestone
- the next `S` is the stable release candidate

> This is a development milestone.
> Binary compatibility is not guaranteed yet.
<!-- /SECTION:RELEASE_IDENTITY -->

<!-- SECTION:HIGHLIGHTS -->
## v11A2 Highlights

`v11A2` focuses on making Lancius transformer execution more structurally
honest, more testable, and more runtime-oriented.

This milestone is still a development milestone. It is not a stable release.

### Transformer Runtime Usability

- Added a dedicated **KV-cache runtime object**
- Added cache-aware attention execution
- Added explicit **prefill** and **generation** execution flows
- Improved RoPE position handling through cache state
- Added first-class 3D transformer tensor construction
- Reduced demo-style graph mutation in transformer examples

### Transformer Validation

- Added a dedicated transformer known-answer audit
- Validates core transformer math and runtime behavior, including:
  - LayerNorm
  - RMSNorm
  - GELU
  - SwiGLU
  - RoPE
  - full causal attention
  - KV-cache step parity
  - prefill + generation parity
  - grouped-query attention, GQA

The KV-cache parity test is especially important because it verifies that
stepwise cache-backed generation matches full causal attention over the same
sequence.

### FP32 Execution Foundation

- Added FP32 runtime buffer support
- Added an FP32 matmul kernel
- Added scheduler dispatch for FP32 matmul
- Added FP32 model serialization support
- Added an FP32 path audit

The current FP32 matmul path uses FP32 inputs and outputs with FP64
accumulation for numerical stability.

FP32 execution is currently scoped primarily to matmul. Wider FP32 operator
support is deferred.

### Runtime and Loader Hardening

- Improved repeated-execution hygiene
- Improved malformed-model rejection
- Continued hardening of the v2 model format
- Continued separation between runtime state and demo-level graph mutation

The v2 model format remains the active development format.

Binary compatibility is still **not guaranteed** during the v11 development
cycle.
<!-- /SECTION:HIGHLIGHTS -->

<!-- SECTION:WHATS_CHANGED -->
## What Changed Since v11A1

`v11A2` continues the v11 development cycle.

The focus of this milestone is not broad feature expansion. The focus is
making the transformer execution path more real, more testable, and less
dependent on demo-level graph mutation.

### Added

- Dedicated transformer runtime support:
  - `include/lancius/lancius_transformer.h`
  - `src/runtime/lancius_transformer.c`

- A real **KV-cache runtime object**
  - maximum sequence length
  - active sequence length
  - head count
  - head dimension
  - K/V buffer management
  - cache reset / append / query operations

- Cache-aware attention execution
  - attention execution can consult cache state
  - single-token generation can run against an active KV-cache
  - reduces reliance on mutating IR shapes during generation

- Explicit prefill and generation flow
  - prompt prefill via full causal attention
  - prompt K/V append into the cache
  - single-token generation against the active cache

- RoPE position contract
  - cache active sequence length is used as the position source
  - RoPE helper validates cache state and head dimension parity

- First-class 3D transformer tensor construction
  - added `lancius_input_3d()`
  - reduces manual `ndim` and shape patching in examples

- Transformer known-answer audit
  - LayerNorm
  - RMSNorm
  - GELU
  - SwiGLU
  - RoPE
  - full causal attention
  - KV-cache step parity
  - prefill + generation parity
  - grouped-query attention, GQA

- FP32 execution foundation
  - FP32 runtime buffer support
  - FP32 tensor ownership helpers
  - FP32 matmul kernel
  - scheduler dispatch for FP32 matmul
  - FP32 path audit

- FP32 serialization support
  - v2 save path can write FP32 weights
  - v2 load path can accept FP32 tensors
  - FP32 roundtrip validation

- Stronger v2 loader validation
  - reject invalid dtype
  - reject malformed weight length
  - reject impossible tensor shapes
  - reject malformed model structure more cleanly

### Improved

- Transformer execution is less demo-like
  - static graph shapes are preferred
  - sequence state lives in runtime objects
  - examples are closer to real runtime usage

- Scheduler buffer lifecycle is more explicit
  - FP32 buffers are handled separately
  - repeated execution hygiene is improved
  - pool and arena ownership behavior is clearer

- Validation coverage is stronger
  - transformer math is checked against known answers
  - KV-cache step parity is validated
  - FP32 execution is validated
  - malformed model loading is tested

### Deferred

The following remain intentionally deferred:

- full FP32 operator coverage
- FP32 KV-cache storage
- general ONNX converter usability
- dynamic shape execution
- GPU acceleration
- production LLM serving
- final binary compatibility guarantees

`v11A2` is still a development milestone.

It is not a stable release.
<!-- /SECTION:WHATS_CHANGED -->

<!-- SECTION:BUILDING -->
## Building

Lancius is primarily built with GNU Make.

### Requirements

- C11-compatible compiler, such as GCC or Clang
- GNU Make
- OpenMP support
- Linux x86_64 is the primary supported environment

On Debian/Ubuntu-like systems:

```bash
sudo apt install build-essential
```

### Build with Make

From inside the `v11A2/` directory:

```bash
make clean
make
```

This builds:

- `liblancius.a`
- internal examples
- audit binaries
- test binaries

For a parallel build:

```bash
make -j$(nproc)
```

### Build with CMake

A CMake build is also available:

```bash
cmake -B build
cmake --build build
```

The Makefile remains the canonical build path for running the full validation
suite.

### Optional Python Tooling

Python is not required to build or run the core C runtime.

It is only needed for optional ONNX interoperability workflows:

```bash
python3 -m pip install onnx onnxruntime numpy
```
<!-- /SECTION:BUILDING -->

<!-- SECTION:VALIDATION -->
## Validation

Lancius `v11A2` uses a layered validation suite.

The minimum development gate is:

```bash
make check
```

The stronger pre-hardening gates are:

```bash
make check-long
make check-sanitizers
```

### Core Validation

```bash
make check
```

This runs the primary regression and correctness suite, including:

- arena stress testing
- malformed model rejection
- serialization roundtrip tests
- finite-difference gradient checking
- stable C API / FFI audit
- threadpool parity audit
- NaN / Inf injection audit
- memory planner audit
- diamond-graph memory audit
- flash attention audit
- modern transformer kernel audit
- known-answer correctness audit
- regression hardening audit
- transformer known-answer audit
- FP32 path audit

### Long Validation

```bash
make check-long
```

This extends the core suite with longer-running adversarial tests, including:

- soak fuzzing
- deterministic fuzz execution

### Sanitizer Validation

```bash
make check-sanitizers
```

This rebuilds selected stress binaries with sanitizer instrumentation and runs:

- AddressSanitizer
- UndefinedBehaviorSanitizer

### Targeted Audits

Transformer validation:

```bash
./audit_transformer_known_answer
```

FP32 validation:

```bash
./audit_fp32_path
```

### Example Runtime Demos

Transformer prefill/generation demo:

```bash
./generate_text
```

Adversarial soak demo:

```bash
./soak_fuzz
```

### Optional ONNX Parity Workflow

Python dependencies are required for the optional ONNX interoperability path:

```bash
python3 -m pip install onnx onnxruntime numpy
```

Example workflow:

```bash
python3 export_pytorch_onnx.py
python3 audit_pytorch_parity.py
```

> Passing `make check` is the minimum development gate.
> Passing `make check-long` and `make check-sanitizers` is expected before
> release-candidate hardening.
<!-- /SECTION:VALIDATION -->

<!-- SECTION:FEATURE_STATUS -->
## Feature Status

Lancius `v11A2` is a development milestone.

The following table describes the current status of major subsystems.

| Area | Status | Notes |
|---|---|---|
| Core tensor ops | Development | Add, Sub, Mul, MatMul, ReLU, Softmax, Sum, Broadcast, Transpose |
| Vision ops | Development | Conv2D, MaxPool2D, Flatten, fused Conv2D+ReLU |
| Training ops | Experimental | CrossEntropy backward, Conv backward, MaxPool backward |
| Transformer kernels | Experimental | LayerNorm, RMSNorm, GELU, RoPE, Attention, KV-cache attention, SwiGLU, GQA |
| KV-cache runtime | Experimental | Stateful cache object introduced in v11A2, FP64-only for now |
| Prefill / generation flow | Experimental | Explicit prefill and generation paths introduced in v11A2 |
| FP32 execution | Experimental | FP32 matmul kernel, scheduler dispatch, and serialization support |
| Stable C API | Partial | Opaque handles and limited graph builders; not full runtime coverage yet |
| Model format v2 | Development | Active format; binary compatibility is not guaranteed yet |
| ONNX conversion | Experimental | Validated primarily against LeNet-class graphs |
| Memory planner | Development | Linear-scan liveness planning and static flat-buffer execution |
| Threadpool execution | Development | Wave-parallel execution with parity validation |
| GPU acceleration | Not supported | CPU-only runtime |
| Dynamic shapes | Not supported | Static graph execution only |
| Production LLM serving | Not supported | Research and development milestone only |

> v11A2 is intended for development, validation, and experimentation.
> It is not intended as a production-stable release.
<!-- /SECTION:FEATURE_STATUS -->

<!-- SECTION:KNOWN_LIMITATIONS -->
## Known Limitations

Lancius `v11A2` is a development milestone.

Its limitations are intentional boundaries. They define what this release is
not claiming to be.

> `v11A2` is not a production-stable release.

### Production Status

- Not intended for production deployment
- Not intended as a finalized public SDK
- Not intended for long-term binary compatibility
- Internal APIs and runtime structures may still change

### Runtime Limitations

- CPU-only execution
- No GPU acceleration
- No distributed execution
- Static graph execution only
- No dynamic shape execution
- No general runtime shape mutation contract

### Transformer Limitations

Transformer support is experimental.

- Transformer inference is experimental
- Transformer backward passes are not supported
- KV-cache runtime is FP64-only for now
- FP32 execution is currently scoped primarily to matmul
- FP32 KV-cache storage is not supported
- This is not a full LLM serving runtime

### Model Format Limitations

The active model format is v2.

However:

- The v2 format is not frozen
- Binary compatibility is not guaranteed during the v11 development cycle
- Models produced by `v11A1` or `v11A2` should be treated as development artifacts
- Legacy v1 loading remains available as a fallback, but v1 is not the active format

### ONNX Interoperability Limitations

ONNX conversion is experimental.

- Operator coverage is limited
- The converter is validated primarily against LeNet-class graphs
- Broader ONNX usability is not guaranteed in `v11A2`
- Unsupported ONNX behavior should be treated as experimental, not stable

### API Limitations

The stable C API exists, but it is partial.

- It does not yet cover the full runtime surface
- Examples may still use internal headers
- The stable API should not yet be treated as a complete public SDK

### Training Limitations

Training-related code exists in the repository, but `v11A2` is inference-first.

- Training components are experimental
- Training workflows are not production-grade
- Training is not part of the stable release contract

### Platform Support

Primary supported environment:

- Linux
- x86_64 CPU

Other architectures may work, but they require additional validation.

### Hardening Status

`v11A2` includes validation and testing, but it is not fully hardened.

The next milestone, `v11A3`, is intended to focus on:

- feature freeze
- loader hardening
- model-format hardening
- sanitizer and fuzz validation
- release-candidate preparation
<!-- /SECTION:KNOWN_LIMITATIONS -->

<!-- SECTION:MODEL_FORMAT -->
## Model Format

The active model format is **v2**.

v2 improves on v1 by using:

- explicit magic
- explicit version
- fixed-width fields
- little-endian encoding
- explicit header flags
- stronger loader validation

Legacy v1 loading remains available as a fallback.

However:

> Binary compatibility is **not guaranteed** during the v11 development cycle.
<!-- /SECTION:MODEL_FORMAT -->

<!-- SECTION:ONNX_INTEROPERABILITY -->
## ONNX Interoperability

Lancius includes an experimental ONNX conversion workflow.

The current path is:

```text
PyTorch / ONNX model
        ↓
ONNX export
        ↓
onnx_to_lancius.py
        ↓
Lancius binary model
        ↓
Lancius C runtime execution
```

This workflow is intended for validation and interoperability testing.

It is not a general-purpose ONNX runtime.

### Current Status

ONNX support is:

- experimental
- validated primarily against LeNet-class convolutional graphs
- limited in operator coverage
- not guaranteed to handle arbitrary ONNX models

The converter currently writes Lancius v2 binary models.

### Supported Converter Operators

The ONNX converter currently handles a small operator set:

- `Conv`
- `Relu`
- `MaxPool`
- `Flatten`
- `MatMul`
- `Add`
- `Reshape`
- `Gemm`
- `Transpose`

Other ONNX operators are not part of the validated conversion path.

### Optional Python Dependencies

Core Lancius does not require Python.

Python is only needed for ONNX conversion and parity validation:

```bash
python3 -m pip install onnx onnxruntime numpy
```

PyTorch-based workflows additionally require PyTorch and torchvision.

### Example Workflow

Generate a pure ONNX model:

```bash
python3 build_pure_onnx.py
```

or export a PyTorch LeNet-style model:

```bash
python3 export_pytorch_onnx.py
```

Convert the ONNX model to a Lancius binary:

```bash
python3 onnx_to_lancius.py pytorch_lenet.onnx pytorch_lenet.lancius
```

Run parity validation against ONNX Runtime:

```bash
python3 audit_pytorch_parity.py
```

### Important Limitations

- Dynamic shapes are not supported.
- Operator coverage is limited.
- Attribute handling is simplified.
- The converter is currently FP64-oriented.
- FP32 ONNX export is deferred.
- Unsupported ONNX graphs may fail conversion or produce incomplete models.

> ONNX support should be treated as an experimental interoperability path,
> not a stable model import guarantee.
<!-- /SECTION:ONNX_INTEROPERABILITY -->

<!-- SECTION:DOCUMENTATION -->
## Documentation

Relevant documents in this tree:

- `docs/v11A2_SCOPE.md` — current milestone scope
- `docs/v11A1_SCOPE.md` — previous milestone scope
- `docs/v11A1_MODEL_FORMAT.md` — model format direction
- `docs/v11A1_OPS.md` — operator support matrix
- `docs/ARCHITECTURE.md` — high-level architecture overview
- `docs/releases/v10S/RELEASE_NOTES_v10S.md` — historical v10S release notes
- `v11A1_Release_notes.md` — historical v11A1 release notes
- `KNOWN_LIMITATIONS.md` — explicit limitations and non-goals
- `SECURITY.md` — security reporting policy
- `CHANGELOG.md` — changelog

> Some documents may still reference `v11A1`.
>
> Where that happens, treat them as historical unless they explicitly describe
> `v11A2` behavior.
<!-- /SECTION:DOCUMENTATION -->

<!-- SECTION:ROADMAP -->
## Roadmap

Lancius is currently in the v11 development cycle.

The internal milestone progression is:

```text
S → A1 → A2 → A3 → S
```

For public GitHub releases, internal milestones are mapped as follows:

| Internal Milestone | Public Release       | Purpose                              |
|--------------------|----------------------|--------------------------------------|
| `v11A1`            | `V1.1-AlphaRC1`      | Foundation and runtime honesty       |
| `v11A2`            | `V1.1-AlphaRC2`      | Transformer runtime usability        |
| `v11A3`            | `V1.1-AlphaRC3`      | Freeze, hardening, and bug hunting   |
| `v11S`             | `V1.1`               | Stable release candidate             |

### Current Milestone

This release is:

```text
v11A2 / V1.1-AlphaRC2
```

Its theme is:

> Transformer runtime usability.

It introduces the KV-cache runtime object, explicit prefill/generation flow,
transformer known-answer validation, and the beginning of the FP32 execution
path.

### Next Milestone

The next milestone is:

```text
v11A3 / V1.1-AlphaRC3
```

`v11A3` is intended to be a freeze and hardening milestone.

Its focus will be:

- feature freeze
- loader hardening
- model-format hardening
- sanitizer and fuzz validation
- regression defense
- documentation cleanup
- release-candidate preparation

`v11A3` is not intended to introduce major new features.

### Stable Release

After `v11A3`, the next stable milestone is:

```text
v11S / V1.1
```

The stable release will be cut only after the v11A3 hardening gate is
complete.
<!-- /SECTION:ROADMAP -->

<!-- SECTION:SECURITY -->
## Security

Lancius `v11A2` is a development milestone.

Security issues should be reported privately before public disclosure.

Security-relevant concerns include:

- memory corruption
- unsafe deserialization
- malformed model loading
- arbitrary execution risks
- dependency vulnerabilities

Because `v11A2` is not a stable release, models and inputs should be treated
as development artifacts. Do not load untrusted models in production
environments.

For the current reporting policy, see:

- `SECURITY.md`
<!-- /SECTION:SECURITY -->

<!-- SECTION:LICENSE -->
## License

Lancius is free software.

This project is licensed under the **GNU General Public License v3.0**.

SPDX-License-Identifier: `GPL-3.0-or-later`

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or, at your option,
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see:

<https://www.gnu.org/licenses/gpl-3.0.html>

### Source Availability

Under GPLv3, if you distribute binaries built from this project, you must
also provide recipients with access to the complete corresponding source
code under the same license terms.

For this repository, the corresponding source code is the complete contents
of the source tree used to build the distributed binaries.
<!-- /SECTION:LICENSE -->

<!-- README_END -->

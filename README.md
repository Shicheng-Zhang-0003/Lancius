<!-- SECTION:HEADER -->
# Lancius v12R1

> **Internal milestone:** `v12R1`
> **Public release:** `V1.2RC1`
> **Status:** Development milestone (R1 — first v12 milestone)

Lancius is a lightweight C machine-learning compiler and runtime focused on
bare-metal inference, static graph execution, memory planning, and low-level
runtime control.

`v11S` is the first stable release of the Lancius 1.1 cycle.
It represents the completion of the v11A3 hardening gate:
feature freeze, loader hardening, model-format freeze with CRC32 integrity,
sanitizer and fuzz validation, and full regression defense.

`v12R1` builds on the `v11S` stable baseline.
Its themes are **hardening** (loader integrity, execution contracts, stability
guards) and **mathematical correctness**: a hostile,
formula-by-formula audit of every numeric path, with each confirmed defect
fixed and re-proven by independent execution.
<!-- /SECTION:HEADER -->

<!-- SECTION:RELEASE_IDENTITY -->
## Release Identity

| Internal Version | Public Version       | Release Type      |
|------------------|----------------------|-------------------|
| `v12R1`          | `V1.2RC1`      | Development Milestone    |

Lancius uses the following internal milestone progression:

```text
S → R1 → R2 → R3 → S
```

Where:

- `S` is a stable release
- `R1` is the first development milestone after the previous stable release
- `R2` is the second development milestone, where new subsystems begin becoming independent
- `R3` is the freeze, hardening, and bug-hunting milestone
- the next `S` is the stable release candidate

`v12R1` is the first development milestone of the v12 cycle (the R1 phase
in R-series numbering), built on the `v11S` stable baseline.

> This is a development milestone.
> Binary compatibility is guaranteed for v2 models written by v11S+.
<!-- /SECTION:RELEASE_IDENTITY -->

<!-- SECTION:HIGHLIGHTS -->
## v12R1 Highlights

`v12R1` makes Lancius numerically honest: every kernel, executor, gradient,
shape formula, serializer field, and converter mapping was independently
re-derived and re-executed. Plausible outputs were not accepted as proof.

### Correctness Fixes (audited, fixed, re-proven)

- **Broadcast elementwise math is now correct.** Direct `ADD`/`SUB`/`MUL`
  on broadcastable shapes (e.g. `[2,2]` with `[1,2]`) previously ran a flat
  `a[k] OP b[k]` loop — wrong values plus an out-of-bounds read.
  IR builders now emit the true broadcast output shape
  (`out[i] = max(a[i], b[i])`), and both the scheduler and the bytecode VM
  execute N-dimensional strided broadcast. Verified both argument orders:
  `[1,2,3,4] OP [10,20] → [10,40,30,80]` / `[11,22,13,24]`.
- **Softmax zero-sum guard.** All-`-inf` rows previously produced silent
  `NaN`. Both the scheduler and the VM now return a numerical error instead.
- **Model integrity is now mandatory.** A zero CRC field previously disabled
  verification entirely (4 zeroed bytes bypassed integrity). The v2 loader
  now rejects `checksum == 0` by default; legacy unverified loads require
  explicit opt-in via `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`. Duplicate-ID
  detection is O(1), sparse IDs are bounded, and weight skipping streams
  instead of truncating through `long`.
- **Attention execution is validated.** Cache heads/dim must match the query,
  K/V heads/dim must match, and single-token decode without a bound cache is
  rejected instead of attending over garbage. GQA Q/K/V shapes are checked
  against declared heads/dim.
- **INT8 accumulation is 64-bit.** The `int32` accumulator overflowed past
  132104 terms; both the INT8 conv kernel and the mixed-precision matmul now
  use `int64`. The dead `&& 0` overflow guard is a real check.
- **No more silent OOM.** Thread-local weight buffers and Flash/GQA scratch
  allocations now report `OOM` instead of emitting zeros or uninitialized
  tiles. Arena allocation failures report error codes.
- **No more `abort()` on the execution hot paths.** Scheduler, planner, and
  vision copy paths use the `_checked` element/byte counters and return
  errors on corrupt shapes.

### ONNX Strictness

The converter now fails loud instead of emitting silently wrong graphs:

- `Reshape` follows ONNX semantics: `0` copies the input dim, `-1` infers.
  The two were previously conflated.
- `Gemm` rejects `transA` and non-unit `alpha`/`beta`, and transposed weights
  are cloned per use instead of mutating a shared initializer in place.
- Asymmetric Conv pads/strides and non-square MaxPool kernels are rejected
  instead of silently keeping only the first element.
- Symbolic (dynamic-batch) dimensions are rejected instead of being silently
  frozen to 1. Export static batches for conversion.

### Training Alignment

- CIFAR-10 inputs are now `((x/255) - 0.5) / 0.5` in `[-1,1]`, matching the
  PyTorch `Normalize((0.5,), (0.5,))` reference. The old `[-0.5,0.5]` range
  halved the effective scale versus tuned learning rates.
- MNIST uses He initialization (correct for ReLU) instead of
  Xavier-uniform, binds training parameters by buffer identity instead of
  shape-sniffing, zeroes gradient buffers per batch, and documents the true
  cross-entropy mean scale (`1/R`, i.e. `1/64` — not `1/640`).
- CIFAR-10 evaluation tracks logits by graph identity instead of picking the
  first `ADD` with 10 columns (the autodiff graph contains many), and logs
  raw versus clamped loss so divergence can no longer hide at `2.3025`.

### API and Harness Honesty

- The stable FFI error space now preserves causes: `GRAPH_CYCLE`,
  `OVERFLOW`, `NUMERICAL`, and `INVALID_HANDLE` instead of collapsing them
  into shape-mismatch/OOM/null.
- Audits that printed PASS/FAIL but always exited 0 now propagate failures
  (`audit_modern_llm`, `audit_flash_attention`, `audit_threadpool_parity`,
  `audit_ffi`, `audit_pytorch_parity.py`).
- `lancius_dtype_size()` returns 0 on invalid codes and
  `checked_product_shape(NULL)` fails instead of masking caller bugs.

### Inherited Baseline (v11S)

- Dedicated **KV-cache runtime object** with explicit **prefill** and
  **generation** flows, `lancius_input_3d()`, and a 265-check transformer
  known-answer audit (LayerNorm, RMSNorm, GELU, SwiGLU, RoPE, full causal
  attention, KV-cache step parity, prefill+generation parity, GQA).
- FP32 foundation: FP32 buffers, FP64-accumulation matmul kernel, scheduler
  dispatch, serialization roundtrip, and path audit. FP32 remains
  matmul-scoped; there is no FP32 LLM path yet.
- v2 model format with CRC32 body integrity, reserved-flag rejection, and
  fail-closed malformed-model handling.

The v2 model format remains the active development format.

Binary compatibility is **guaranteed** for v2 models written by v11S and later.
<!-- /SECTION:HIGHLIGHTS -->

<!-- SECTION:WHATS_CHANGED -->
## What Changed Since v11S

### Fixed

- N-dimensional broadcast `ADD`/`SUB`/`MUL` in scheduler, IR shape inference,
  and bytecode VM (was flat-loop wrong + OOB).
- Softmax zero-sum guard in scheduler and VM.
- v2 loader: CRC required by default, O(1) duplicate detection, sparse-ID
  bounds, streaming weight skip, sticky-error clearing.
- Attention cache/heads/dim validation; single-token-without-cache rejected;
  GQA shape validation; RMSNorm gamma/divisibility checks.
- INT8 `int64` accumulators; real overflow guards; OOM error reporting in
  kernels, arena, and thread-local paths.
- Aborting element/byte counters replaced with `_checked` + error returns on
  all execution, planning, and copy paths; `FLATTEN`/`RESHAPE` verify
  element-count equality.
- ONNX: Reshape `0`/`-1`, Gemm clone-on-write + `alpha`/`beta`/`transA`
  rejection, symmetric-only Conv/Pool, symbolic-dim rejection.
- Training: `[-1,1]` CIFAR normalization, He init on MNIST, identity-based
  parameter binding, per-batch grad zeroing, identity-tracked eval logits,
  raw-vs-clamped loss logging, static ONNX export.
- Stable API: widened error codes; `dtype_size`/`product_shape` fail loud.
- False-green audits now exit non-zero on divergence.

### Improved

- Arena failures carry error codes for downstream diagnosis.
- Bytecode VM validates registers, guards `rows*cols` overflow, and rejects
  unbroadcastable binary shapes instead of miscomputing.
- Parity scripts fail CI on divergence.

### Deferred

The following remain intentionally deferred:

- full FP32 operator coverage (LLM ops are FP64-only)
- FP32 KV-cache storage
- general ONNX converter usability beyond LeNet-class graphs
- dynamic shape execution
- GPU acceleration
- production LLM serving
- final binary compatibility guarantees
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

From inside the `v12R1/` directory:

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

Lancius `v12R1` uses a layered validation suite.

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
- fault-injection audit
- despot truth probe (broadcast values, cross-rank broadcast, diamond
  pooled parity, cross-entropy NUMERICAL contract)
- internal x-ray audit (mini-CNN execution, softmax normalization)

Every audit in the gate propagates failures through its exit code: a green
`make check` means every check passed, not just that binaries ran.

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

The gate restores a clean non-instrumented build afterwards
(`make -B all`), so a stale sanitized binary can never leak into `make check`.

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

Lancius `v12R1` is a development milestone.

The following table describes the current status of major subsystems.

| Area | Status | Notes |
|---|---|---|
| Core tensor ops | Development | Add/Sub/Mul (N-dim broadcast-correct), MatMul, ReLU, Softmax (zero-sum guarded), Sum, Broadcast, Transpose |
| Vision ops | Development | Conv2D, MaxPool2D, Flatten, fused Conv2D+ReLU; `FLATTEN`/`RESHAPE` verify element equality |
| Training ops | Experimental | CrossEntropy backward, Conv backward, MaxPool backward; He init, `[-1,1]` CIFAR norm |
| Transformer kernels | Experimental | LayerNorm, RMSNorm, GELU, RoPE, Attention, KV-cache attention, SwiGLU, GQA (validated shapes) |
| KV-cache runtime | Experimental | Stateful cache object, FP64-only for now |
| Prefill / generation flow | Experimental | Explicit prefill; single-token decode requires a bound cache |
| FP32 execution | Experimental | FP32 matmul kernel (FP64 accumulation), scheduler dispatch, serialization; no FP32 LLM path |
| Stable C API | Partial | Opaque handles, widened error codes (`GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/`INVALID_HANDLE`); builders still cover core inference only; `read_output` FP64-only |
| Model format v2 | Development | CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into legacy); sparse-ID bounds |
| ONNX conversion | Experimental | Strict LeNet-class path: correct Reshape/Gemm semantics, symmetric Conv/Pool only, static batch |
| Memory planner | Development | Linear-scan liveness planning and static flat-buffer execution |
| Threadpool execution | Development | Wave-parallel execution with parity validation |
| GPU acceleration | Not supported | CPU-only runtime |
| Dynamic shapes | Not supported | Static graph execution only |
| Production LLM serving | Not supported | Research and development milestone only |

> v12R1 targets honest numerics and strict boundaries, not expanded scope.
<!-- /SECTION:FEATURE_STATUS -->

<!-- SECTION:KNOWN_LIMITATIONS -->
## Known Limitations

Lancius `v12R1` is a development milestone.

Its limitations are intentional boundaries. They define what this release is
not claiming to be.

> `v12R1` is a development milestone. The limitations below define its supported scope.

### Production Status

- Intended for edge deployment and bare-metal inference within documented scope
- Stable C API covers the core inference workflow
- Binary compatibility guaranteed for v2 models written by v11S+
- Internal APIs beyond the stable API may change in future cycles

### Runtime Limitations

- CPU-only execution
- No GPU acceleration
- No distributed execution
- Static graph execution only
- No dynamic shape execution
- No general runtime shape mutation contract
- Single-token attention decode requires a bound KV-cache; cache-less
  decode of a longer context is rejected rather than executed over garbage

### Transformer Limitations

Transformer support is experimental.

- Transformer inference is experimental
- Transformer backward passes are not supported (autodiff fails loud)
- Batched-matmul backward is not supported (autodiff fails loud)
- KV-cache runtime is FP64-only for now
- FP32 execution is currently scoped to matmul; there is no FP32 LLM path
- The `ROPE` graph opcode has no public builder; use `kernel_rope()` or
  `lancius_transformer_apply_rope_token()` for position handling
- `EMBEDDING`, `KV_CACHE_READ`, and `KV_CACHE_WRITE` opcodes are reserved
  and unimplemented; graphs using them fail loud
- This is not a full LLM serving runtime

### Model Format Limitations

The active model format is v2.

However:

- The v2 format is **frozen** as of v11S for compatible writers
- Binary compatibility is guaranteed for v2 models written by v11S+
- Models produced by `v11A1` or `v11A2` should be treated as development artifacts
- `checksum == 0` files are rejected by default; set
  `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` to load legacy unverified files
- Legacy v1 loading remains available as a fallback, but v1 is not the active format

### ONNX Interoperability Limitations

ONNX conversion is experimental and deliberately strict.

- Operator coverage is limited to `Conv`, `Relu`, `MaxPool`, `Flatten`,
  `MatMul`, `Add`, `Reshape`, `Gemm`, `Transpose`
- Validated primarily against LeNet-class graphs
- `Reshape` supports rank 2 and rank 4 targets with `0`-copy and single-`-1`
  inference only
- Conv requires symmetric pads/strides expressible as a single pad/stride;
  MaxPool requires square kernels
- `Gemm` requires `transA == 0` and `alpha == beta == 1.0`
- Symbolic (dynamic) dimensions are rejected; export static batches
- The converter is currently FP64-oriented
- Unsupported ONNX graphs fail conversion instead of producing partial models

### API Limitations

The stable C API covers the core inference workflow.
- Model loading and saving via `lancius_graph_load_stable` / `lancius_graph_save_stable`
- Graph construction (input, matmul, relu)
- Data binding and execution
- Output reading with truncation protection (FP64 outputs only)
- Tensor introspection (element count, dtype)
- Opaque handles and thread-local error states (widened codes)
- Transformer ops, conv2d builders, and advanced ops remain internal-only

### Training Limitations

Training-related code exists in the repository, but `v12R1` is inference-first.

- Training components are experimental
- Training workflows are not production-grade
- Evaluated on MNIST/CIFAR-10 style graphs only
- Training is not part of any stable release contract

### Platform Support

Primary supported environment:

- Linux
- x86_64 CPU

Other architectures may work, but they require additional validation.
The Makefile targets AVX2/FMA; non-x86 builds need flag adjustments.

### Hardening Status

`v12R1` includes the hardening, numerical-correctness, and despot truth
batches described above. It is a development milestone, not a frozen
release.

The next milestone (`v12R2`) is intended to focus on:

- feature decisions for the remainder of the v12 cycle
- continued loader and format hardening
- sanitizer and fuzz validation
- release-candidate preparation
<!-- /SECTION:KNOWN_LIMITATIONS -->

<!-- SECTION:MODEL_FORMAT -->
## Model Format

The active model format is **v2**.

v2 improves on v1 by using:

- explicit magic (`0x32434E41`)
- explicit version (`2`)
- fixed-width fields
- little-endian encoding
- explicit header flags
- stronger loader validation
- mandatory CRC32 body integrity check (v12R1; opt-out only via
  `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`)

Loader defenses include duplicate-ID rejection, forward-reference rejection,
weight-length-vs-shape agreement, sparse-ID bounds, reserved-flag rejection,
and `EXTERNAL_WEIGHTS` rejection.

Legacy v1 loading remains available as a deprecated fallback.

However:

> Binary compatibility is **guaranteed** for v2 models written by v11S and later.
> The v2 format is frozen. CRC32 integrity verification is required by default.
<!-- /SECTION:MODEL_FORMAT -->

<!-- SECTION:ONNX_INTEROPERABILITY -->
## ONNX Interoperability

Lancius includes an experimental ONNX conversion workflow.

The current path is:

```text
PyTorch / ONNX model
        ↓
ONNX export (static batch)
        ↓
onnx_to_lancius.py (strict)
        ↓
Lancius binary model
        ↓
Lancius C runtime execution
```

This workflow is intended for validation and interoperability testing.

It is not a general-purpose ONNX runtime.

### Current Status

ONNX support is:

- experimental and strict: violations raise instead of degrading silently
- validated primarily against LeNet-class convolutional graphs
- limited in operator coverage
- not guaranteed to handle arbitrary ONNX models

The converter currently writes Lancius v2 binary models.

### Supported Converter Operators

The ONNX converter currently handles a small operator set:

- `Conv` (symmetric pads/strides only)
- `Relu`
- `MaxPool` (square kernels only)
- `Flatten`
- `MatMul`
- `Add`
- `Reshape` (rank 2/4, `0`-copy, single-`-1`)
- `Gemm` (`transA == 0`, `alpha == beta == 1.0`, per-use transpose clones)
- `Transpose` (`[1,0]` / `[1,0,2,3]` only)

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

or export a PyTorch LeNet-style model (static batch):

```bash
python3 export_pytorch_onnx.py
```

Convert the ONNX model to a Lancius binary:

```bash
python3 onnx_to_lancius.py pytorch_lenet.onnx pytorch_lenet.lancius
```

Run parity validation against ONNX Runtime (fails non-zero on divergence):

```bash
python3 audit_pytorch_parity.py
```

### Important Limitations

- Dynamic shapes are not supported; export static batches.
- Operator coverage is limited.
- Conv/Pool attribute handling is symmetric-only by design.
- The converter is currently FP64-oriented.
- FP32 ONNX export is deferred.
- Unsupported ONNX graphs fail conversion instead of producing partial models.

> ONNX support should be treated as an experimental interoperability path,
> not a stable model import guarantee.
<!-- /SECTION:ONNX_INTEROPERABILITY -->

<!-- SECTION:DOCUMENTATION -->
## Documentation

Relevant documents in this tree:

- `docs/v11A3_SCOPE.md` — last frozen milestone scope
- `docs/v11A2_SCOPE.md` — previous milestone scope
- `docs/v11A1_SCOPE.md` — historical milestone scope
- `docs/v11A1_MODEL_FORMAT.md` — model format direction
- `docs/v11A1_OPS.md` — operator support matrix
- `docs/ARCHITECTURE.md` — high-level architecture overview
- `docs/releases/v10S/RELEASE_NOTES_v10S.md` — historical v10S release notes
- `docs/releases/v11S/GITHUB_RELEASE_v11S.md` — v11S release notes
- `docs/releases/v12R1/GITHUB_RELEASE_v12R1.md` — v12R1 release notes
- `KNOWN_LIMITATIONS.md` — explicit limitations and non-goals
- `SECURITY.md` — security reporting policy
- `CHANGELOG.md` — changelog (see the `v12R1` entry for this milestone)
- `STATUS.md` — current milestone status

> Some documents may still reference `v11A1` or `v11A2`.
>
> Where that happens, treat them as historical unless they explicitly describe
> current (`v12R1`) behavior.
<!-- /SECTION:DOCUMENTATION -->

<!-- SECTION:ROADMAP -->
## Roadmap

Lancius is currently in the v12 development cycle.

The internal milestone progression is:

```text
S → R1 → R2 → R3 → S
```

For public GitHub releases, internal milestones are mapped as follows:

| Internal Milestone | Public Release       | Purpose                              |
|--------------------|----------------------|--------------------------------------|
| `v11A1`            | `V1.1-AlphaRC1`      | Foundation and runtime honesty       |
| `v11A2`            | `V1.1-AlphaRC2`      | Transformer runtime usability        |
| `v11A3`            | `V1.1-AlphaRC3`      | Freeze, hardening, and bug hunting   |
| `v11S`             | `V1.1`               | Stable release                       |
| `v12R1`            | `V1.2RC1`            | Current development milestone: hardening plus numerical correctness |
| `v12R2`            | `V1.2RC2`            | Next development milestone (R2 phase) |

### Current Milestone

This release is:

```text
v12R1
```

Its theme is:

> Hardening and numerical correctness: loader integrity, execution
> contracts, and a hostile audit of every math path, with each
> confirmed defect fixed and re-proven.

Previous milestone: v11S / V1.1 (stable).

### Next Milestone

The next milestone is `v12R2` (the R2 phase, public `V1.2RC2`), followed by the v12 freeze,
hardening, and bug-hunting phase (R3) and the `v12S` stable release
candidate (public `V1.2`).

Candidate v12 work (not committed): FP32 operator expansion, FP32 KV-cache
storage, broader ONNX coverage, and dynamic shape exploration.

### Stable Release

`v11S` **is** the stable release.

The next stable release will be `v12S` (public `V1.2`), cut after the v12 development
cycle completes its hardening gate.
<!-- /SECTION:ROADMAP -->

<!-- SECTION:SECURITY -->
## Security

Lancius `v12R1` is a development milestone, not a hardened release.

Security issues should be reported privately before public disclosure.

Security-relevant concerns include:

- memory corruption
- unsafe deserialization
- malformed model loading
- arbitrary execution risks
- dependency vulnerabilities

Model handling notes:

- v2 CRC32 integrity verification is required by default; files with a zero
  checksum are rejected unless `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` is set.
- Models loaded from untrusted sources should still be validated before
  use. Loading arbitrary untrusted binaries is not recommended.

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

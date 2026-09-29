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

Each fix below was re-derived formula-by-formula and re-proven by
independent execution. The full per-batch record lives in `CHANGELOG.md`;
only the essence is stated here so this list cannot drift from it.

- **Broadcast elementwise math is now correct** (was flat-loop wrong + OOB):
  true N-dim trailing-rank broadcast in IR, scheduler, and VM.
- **Softmax zero-sum guard** (was silent `NaN`): scheduler and VM return a
  numerical error instead.
- **Model integrity is now mandatory** (was 4 zeroed bytes bypassed it):
  `checksum == 0` rejected by default, O(1) duplicate IDs, sparse-ID bounds,
  streaming weight skip.
- **Attention execution is validated** (was attending over garbage):
  cache/K/V heads/dim checks, cache-less long-context decode rejected.
- **INT8 accumulation is 64-bit** (was `int32` overflow past 132104 terms).
- **No more silent OOM** (was zeros/uninitialized tiles): error codes
  everywhere including arena and thread-local paths.
- **No more `abort()` on the execution hot paths** (was aborting counters):
  `_checked` arithmetic with error returns.

### ONNX Strictness

The converter fails loud instead of emitting silently wrong graphs:
`Reshape` `0`-copy vs `-1`-infer, `Gemm` clone-on-write with
`transA`/`alpha`/`beta` rejection, symmetric-only Conv/Pool,
symbolic-dim rejection. Details in `CHANGELOG.md`.

### Training Alignment

CIFAR-10 `[-1,1]` normalization matching PyTorch, He init on MNIST,
identity-based parameter binding, per-batch grad zeroing, `1/R`
cross-entropy scale, identity-tracked eval logits. Details in
`CHANGELOG.md`.

### API and Harness Honesty

Widened FFI error codes (`GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/
`INVALID_HANDLE`), failure-propagating audit exit codes, loud
`dtype_size`/`product_shape`. Details in `CHANGELOG.md`.

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

This section used to repeat the Highlights above bullet-for-bullet. The
single record is `CHANGELOG.md` (v12R1 batches, despot truth V2/V3,
hardening batch V4); the machine-checked proofs are
`docs/DESPOT_TRUTH_V2.md` and `make check`.
In short: N-dim broadcast, softmax guards, mandatory CRC integrity,
validated attention, `int64` INT8, OOM errors, abort-free hot paths,
strict ONNX, training alignment, widened API codes, honest audit exits,
despot truth V2/V3 gradient and loader truth, hardening batch V4
(race conditions, memory safety, quantization, portability, CLI injection).

### Improved

- Arena failures carry error codes for downstream diagnosis.
- Bytecode VM validates registers, guards `rows*cols` overflow, and rejects
  unbroadcastable binary shapes instead of miscomputing.
- Parity scripts fail CI on divergence.
- `lancius_add/sub/mul` set `SHAPE_MISMATCH` (was silent NULL); autodiff
  clears sticky errors at entry and aborts partial graphs.

### Deferred

The following remain intentionally deferred:

- per-axis N-dim broadcast grad reduction (`SUM_AXIS_ND`; currently fails loud)
- full FP32 operator coverage (LLM ops are FP64-only)
- FP32 KV-cache storage
- general ONNX converter usability beyond LeNet-class graphs
- dynamic shape execution
- GPU acceleration
- production LLM serving
- final binary compatibility guarantees

### New in Hardening Batch V4

- **Per-channel quantization** support (in addition to existing per-tensor)
- **Dequantization** support
- **Threadpool timeout** on `lancius_pool_wait` (no longer blocks indefinitely)
- **Race-condition fixes**: `kernel_conv2d_bwd_in` and MaxPool2D backward now
  use thread-local accumulators
- **CLI command-injection fix**: user paths now `fork+execvp` (no shell)
- **Serialization portability**: `uint64_t` sizing, byte swapping, CRC32
  `call_once` init
- **Build hardening**: `-Werror`, version consistency, Threads dependency
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
- race-condition audits (conv2d bwd_in, MaxPool2D bwd thread-local accumulators)
- quantization audits (per-channel quant, dequant roundtrip)
- serialization portability audit (uint64_t, byte swapping, CRC32 call_once)
- CLI security audit (fork+execvp, no shell injection)

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
| Threadpool execution | Development | Wave-parallel execution with parity validation; `lancius_pool_wait` timeout support |
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

The binding contract lives in one place: **`KNOWN_LIMITATIONS.md`**. It is
restated here only as essence, so the two can never drift apart:

- Edge/bare-metal inference within documented scope; stable C API covers
  the core inference workflow; v2 binary compatibility guaranteed for
  models written by v11S+; internal APIs may change.
- CPU-only, static graphs, no dynamic shapes; Linux x86_64 primary
  (Makefile targets AVX2/FMA).
- Transformers, FP32-beyond-matmul, training, and broad ONNX are
  experimental or deferred — each fails loud outside its scope.
- v2 integrity required by default (`checksum == 0` rejected unless
  `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`).

The next milestone (`v12R2`) is scoped in `docs/v12R2_SCOPE.md`.
<!-- /SECTION:KNOWN_LIMITATIONS -->

<!-- SECTION:MODEL_FORMAT -->
## Model Format

The active model format is **v2** (frozen since v11S; binary compatibility
guaranteed for v2 models written by v11S+; CRC32 integrity required by
default). The contract — magic, versions, flags, loader defenses,
legacy opt-in — is stated once in **`MANIFEST.md`**; the historical
direction note is `docs/v11A1_MODEL_FORMAT.md`.
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

Support is experimental and strict (violations raise; LeNet-class only;
converter writes v2 models). Operator scope and boundaries are stated once
in **`KNOWN_LIMITATIONS.md`**; the strictness record is `CHANGELOG.md`.

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
<!-- /SECTION:ONNX_INTEROPERABILITY -->

<!-- SECTION:DOCUMENTATION -->
## Documentation

Each document owns one thing; start here, then follow pointers:

- `STATUS.md` — current milestone status (the only live status)
- `CHANGELOG.md` — per-batch fix history (the only fix list)
- `KNOWN_LIMITATIONS.md` — binding boundaries (the only contract)
- `MANIFEST.md` — compatibility contract (the only compat statement)
- `docs/DESPOT_TRUTH_V2.md` — audit proofs incl. V3 addendum
- `docs/ARCHITECTURE.md` — subsystem contracts and pipeline
- `docs/v12R2_SCOPE.md` — next-milestone scope (binding for v12R2)
- `docs/releases/v12R1/GITHUB_RELEASE_v12R1.md` — v12R1 release notes
- `docs/releases/v11S/GITHUB_RELEASE_v11S.md` — v11S release notes
- `docs/releases/v10S/RELEASE_NOTES_v10S.md` — historical v10S notes
- `docs/v11A3_SCOPE.md`, `docs/v11A2_SCOPE.md`, `docs/v11A1_SCOPE.md` — frozen history
- `docs/v11A1_MODEL_FORMAT.md`, `docs/v11A1_OPS.md` — historical direction
  (bannered as superseded; do not quote for v12R1 behavior)
- `SECURITY.md` — security reporting policy

> Historical files (`v11A*`, `v10S`) describe their own milestones, not
> v12R1. Quoting them for current behavior is a documentation bug —
> report it.
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
candidate (public `V1.2`). Scope — not just direction — is fixed in
`docs/v12R2_SCOPE.md`; candidate work listed anywhere else is stale
unless that file says so.

### Stable Release

`v11S` **is** the stable release.

The next stable release will be `v12S` (public `V1.2`), cut after the v12 development
cycle completes its hardening gate.
<!-- /SECTION:ROADMAP -->

<!-- SECTION:SECURITY -->
## Security

Lancius `v12R1` is a development milestone, not a hardened release.
Report issues privately before public disclosure; treat untrusted model
files as untrusted input (CRC is integrity, not trust). The policy lives
in **`SECURITY.md`** — stated there, not repeated here.
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

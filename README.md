<!-- SECTION:HEADER -->
# Lancius v12R2

> **Internal milestone:** `v12R2`
> **Public release:** `V1.2RC2`
> **Status:** Development milestone (R2 — second v12 milestone)

Lancius is a lightweight C machine-learning compiler and runtime focused on
bare-metal inference, static graph execution, memory planning, and low-level
runtime control.

`v11S` is the first stable release of the Lancius 1.1 cycle.
It represents the completion of the v11A3 hardening gate:
feature freeze, loader hardening, model-format freeze with CRC32 integrity,
sanitizer and fuzz validation, and full regression defense.

`v12R2` builds on the `v12R1` hardening baseline (itself on `v11S`).
Its themes are **learning** (R2-1..R2-6) on top of inherited **hardening** (loader integrity, execution contracts, stability
guards) and **mathematical correctness**: a hostile,
formula-by-formula audit of every numeric path, with each confirmed defect
fixed and re-proven by independent execution. Hardening batches V4 and V5
extended this to threadpool, IR, examples, and Python tooling (38 defects
total).
<!-- /SECTION:HEADER -->

<!-- SECTION:RELEASE_IDENTITY -->
## Release Identity

| Internal Version | Public Version       | Release Type      |
|------------------|----------------------|-------------------|
| `v12R2`          | `V1.2RC2`      | Development Milestone    |

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

`v12R2` is the second development milestone of the v12 cycle (the R2 phase
in R-series numbering), built on the `v11S` stable baseline via `v12R1`.

> This is a development milestone.
> Binary compatibility is guaranteed for v2 models written by v11S+.
<!-- /SECTION:RELEASE_IDENTITY -->

<!-- SECTION:HIGHLIGHTS -->
## v12R2 Highlights

`v12R2` keeps v12R1 numerical honesty and proves Lancius can learn: every kernel, executor, gradient,
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

- Dedicated **KV-cache runtime object** with `lancius_input_3d()`, and a
  265-check transformer known-answer audit (LayerNorm, RMSNorm, GELU, SwiGLU,
  RoPE, full causal attention, KV-cache step parity, GQA). Generation demos
  scrapped in v12R2 (mute mathematician does not speak); kernels stay as
  exact math primitives.
- FP32 foundation: FP32 buffers, FP64-accumulation matmul kernel, scheduler
  dispatch, serialization roundtrip, and path audit. FP32 remains
  matmul-scoped.
- v2 model format with CRC32 body integrity, reserved-flag rejection, and
  fail-closed malformed-model handling.

The v2 model format remains the active development format.

Binary compatibility is **guaranteed** for v2 models written by v11S and later.

R2-1..R2-6 (learn to verify): train-lib (SGD/SGDM/AdamW), char-v1
problem encoder, micromodel bridge (loss 0.291094->0.000010),
eval_verifier harness, sandbox caps, CLI eval verb. See `CHANGELOG.md`.
<!-- /SECTION:HIGHLIGHTS -->

<!-- SECTION:WHATS_CHANGED -->
## What Changed Since v11S

### Fixed

The single record is `CHANGELOG.md` (v12R1 batches, despot truth V2/V3,
hardening batches V4/V5, despot audit V6); the machine-checked proofs are
`docs/DESPOT_TRUTH_V2.md` and `make check`.
In short: N-dim broadcast, softmax guards, mandatory CRC integrity,
validated attention, `int64` INT8, OOM errors, abort-free hot paths,
strict ONNX, training alignment, widened API codes, honest audit exits,
despot truth V2/V3 gradient and loader truth, hardening batches V4/V5
(race conditions, memory safety, quantization, portability, CLI injection,
queue use-before-init, IR NULLs, optimizer, planner align),
despot audit V6 (exact 4D broadcast backward, CE_BWD guards, VM/scheduler
NULL-deref, pool UAF, arena wrap, 32B align, mkstemp/fsync, per-channel
refuse, quantizer/optimizer/vision checks, capped fetches, tar-slip).

### Improved

- Arena failures carry error codes for downstream diagnosis.
- Bytecode VM validates registers, guards `rows*cols` overflow, and rejects
  unbroadcastable binary shapes instead of miscomputing.
- Parity scripts fail CI on divergence.
- `lancius_add/sub/mul` set `SHAPE_MISMATCH` (was silent NULL); autodiff
  clears sticky errors at entry and aborts partial graphs.

### Deferred

The following remain intentionally deferred:

- ~~per-axis N-dim broadcast grad reduction~~ — **closed in R3-1 and verified in
  V7**: `SUM_AXIS_ND` is trainable, its VJP is exact against central differences
  at ~1e-9, and it runs through every rank (see `docs/DESPOT_TRUTH_V2.md` §16)
- full FP32 operator coverage (transformer math ops are FP64-only; no generation path by design)
- FP32 KV-cache storage
- general ONNX converter usability beyond LeNet-class graphs
- dynamic shape execution
- GPU acceleration
- language generation / production serving (scrapped in v12R2: mute mathematician scores, never speaks)
- final binary compatibility guarantees

### New in Hardening Batches V4/V5 + Despot V6

V4 essence (full list in `CHANGELOG.md`):

- **Per-channel quantization** support (in addition to existing per-tensor)
- **Dequantization** support
- **Threadpool timeout** on `lancius_pool_wait` (no longer blocks indefinitely)
- **Race-condition fixes**: `kernel_conv2d_bwd_in` and MaxPool2D backward now
  use thread-local accumulators
- **CLI command-injection fix**: user paths now `fork+execvp` (no shell)
- **Serialization portability**: `uint64_t` sizing, byte swapping, CRC32
  `call_once` init
- **Build hardening**: `-Werror`, version consistency, Threads dependency

V5 essence: queue use-before-init, IR NULLs→errors, optimizer mask,
dequant scale, CE R/C, 32B split, silent lib, examples, converter 1s.

V6 essence: exact 4D broadcast backward (permute+reshape), CE_BWD guards,
conv_bwd_w/LN/RMSN guards, VM `ndim!=2`/tape/`out_reg`/shape checks,
scheduler NULL-deref/errors, pool malloc+free + errors, arena checked grow,
static 32B align + `posix_memalign`, v2 `mkstemp`/`ftello`/CONST/ROPE/scale,
v1 tmp+FP32, quantizer/optimizer/vision checks, capped fetches, tar-slip,
`.d` purge. Proven by `make check/long/sanitizers` + `probe_v6`.
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

From inside the `v12R2/` directory:

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

Lancius `v12R2` uses a layered validation suite.

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
- N-dim reduction audit (per-axis sums 1..4-D, broadcast training exactness)
- norm/activation/batched backward audit (finite-diff ~1e-9..1e-11)
- training convergence audit (XOR solve, determinism, checkpoint resume)
- V7 hardening audit (1226 checks): sticky-error contract per op, softmax /
  cross-entropy / attention stability at |logit| up to 1e5, arena 32B alignment
  across 64 fresh arenas, the alignment cap, INT8 quantizer scale exactness and
  round-trip error bound, conv+relu fusion shape guard, a representative op
  table, the `broadcast_to_shape` contract at every rank plus its v2 round-trip,
  and INT8 64-bit accumulation proven at 204800 taps

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

**V7 change:** this gate used to rebuild three binaries (`stress_test`,
`test_torture`, `fuzz_lancius`) against a `liblancius.a` that was **not
instrumented**, so nothing inside the library was ever checked, and 25 of the 28
audits never ran under a sanitizer at all. It now instruments the library and
every audit in the gate and runs all of them under ASan + UBSan +
LeakSanitizer. It still restores a clean non-instrumented build afterwards
(`make -B all`), so a stale sanitized binary can never leak into `make check`.

A standalone UBSan pass runs the checks that only exist outside the combined
mode (signed-integer-overflow, shift, bool, enum, float-cast-overflow,
integer-divide-by-zero, object-size, bounds) with `-fno-sanitize-recover=all`,
so a finding aborts instead of warning:

```bash
make check-ubstrict
```

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

Mute-mathematician micromodel demo (<100M, scores, never speaks):

```bash
./train_micromodel
./eval_verifier
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

### Gate Integrity Gates

These exist because the gates above were written by the same person who wrote
the code, and a gate nobody checked for its own ability to fail is decoration.

```bash
make check-oracle     # recompute every kernel and graph op in NumPy/PyTorch/closed form
make check-sanitizers # ASan + UBSan + LeakSanitizer over the INSTRUMENTED library and every audit
make check-ubstrict   # UBSan alone with -fno-sanitize-recover=all (signed overflow, shift, casts)
make check-mutation   # inject real defects and require this gate to go red
```

`make check-mutation` is the one that matters most: it took 18 injected defects
and required the gate to catch every one. The first run caught 8 and **missed 7**;
each of those 7 became a permanent check in `audit_v7_hardening`. The current
tally is 18/18 caught, 1 verified behaviourally neutral, 0 holes. Details and the
per-mutation record are in `docs/DESPOT_TRUTH_V2.md` §16.

### Primary-source verification

A finite difference proves the derivative of whatever function you implemented.
It does not prove you implemented the intended function — a wrong constant is
perfectly self-consistent and passes every internal test. So every constant and
equation is also checked against its published source: Hendrycks & Gimpel for
GELU, Ba/Kiros/Hinton for LayerNorm, Zhang & Sennrich for RMSNorm,
Vaswani for the `1/sqrt(d)` scale, Su et al. for RoPE, Loshchilov & Hutter for
AdamW and the cosine schedule, Pascanu et al. for gradient clipping, Ainslie
et al. for GQA grouping, He et al. for init, the TFLite spec for INT8
quantization, IEEE 802.3 for CRC-32, and the datasets' own specifications for
MNIST and CIFAR-10.

That pass found one real error: the GELU comment claimed a max deviation from
erf-exact GELU of `~2e-3`, when the true value is **4.74e-04**. The bound had
never been checked against anything — and the oracle had inherited the same
unverified number. Both are now measured and pinned. The full citation table,
including the one deliberate divergence from the TFLite spec and the claims that
remain unchecked, is `docs/DESPOT_TRUTH_V2.md` §17.

All five gates run on every push (`.github/workflows/gate.yml`).
<!-- /SECTION:VALIDATION -->

<!-- SECTION:FEATURE_STATUS -->
## Feature Status

Lancius `v12R2` is a development milestone.

The following table describes the current status of major subsystems.

| Area | Status | Notes |
|---|---|---|
| Stress testing | Available | `./run_stress.sh` sweeps `fuzz_lancius` seeds, repeats the deterministic suites to catch cross-invocation state leaks, enforces a per-case timeout, and treats "prints FAIL but exits 0" as a failure. Results under `stress-logs/` |
| Dataset acquisition | Development | `manage_datasets.py` fetches through the 3463-LDFD submodule when `libsnapshot.so` is present (streaming gzip + tar, atomic land, HTTP>=400 as error), else the hardened urllib path. `manage_datasets.py status` reports which is live |
| Core tensor ops | Development | Add/Sub/Mul (N-dim broadcast-correct), MatMul, ReLU, Softmax (zero-sum guarded), Sum, Broadcast, Transpose |
| Vision ops | Development | Conv2D, MaxPool2D, Flatten, fused Conv2D+ReLU; `FLATTEN`/`RESHAPE` verify element equality |
| Training ops | Development | N-dim broadcast reduction (`SUM_AXIS_ND`) with an exact VJP proven against central differences, batched-matmul/GELU/LayerNorm/RMSNorm backward incl. gamma/beta (finite-diff ~1e-9..1e-11), global-norm clip, XOR convergence + checkpoint-resume gates; attention/GQA/SwiGLU/RoPE backward staged (fail loud) |
| Transformer kernels | Experimental | LayerNorm, RMSNorm, GELU, RoPE, Attention, KV-cache attention, SwiGLU, GQA (validated shapes; math primitives only, no generation flows) |
| KV-cache runtime | Experimental | Stateful cache object, FP64-only for now; step parity audited, generation demos scrapped |
| Language generation | Scrapped | No prefill/generation demos, no streaming generation, no `generate` verb (`lancius generate` fails loud → use `lancius eval`) |
| FP32 execution | Experimental | FP32 matmul kernel (FP64 accumulation), scheduler dispatch, serialization; matmul-scoped |
| Stable C API | Partial | Opaque handles, widened error codes (`GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/`INVALID_HANDLE`); builders still cover core inference only; `read_output` FP64-only |
| Model format v2 | Development | CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into legacy); sparse-ID bounds |
| ONNX conversion | Experimental | Strict LeNet-class path: correct Reshape/Gemm semantics, symmetric Conv/Pool only, static batch |
| Memory planner | Development | Linear-scan liveness planning and static flat-buffer execution |
| Threadpool execution | Development | Wave-parallel execution with parity validation; `lancius_pool_wait` timeout support |
| GPU acceleration | Not supported | CPU-only runtime |
| Dynamic shapes | Not supported | Static graph execution only |
| Micromodels | Development | Mute-mathematician target <100M params: `train_micromodel` + `eval_verifier` + train-lib + sandbox gates |

> v12R2 targets a mute mathematician (<100M, scores never speaks), not fluency.
<!-- /SECTION:FEATURE_STATUS -->

<!-- SECTION:KNOWN_LIMITATIONS -->
## Known Limitations

Lancius `v12R2` is a development milestone.

Its limitations are intentional boundaries. They define what this release is
not claiming to be.

> `v12R2` is a development milestone. The limitations below define its supported scope.

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

The current milestone (`v12R2`) is scoped in `docs/v12R2_SCOPE.md`.
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
- `docs/DESPOT_TRUTH_V2.md` — audit proofs incl. V3/V4/V5/V6 addenda
- `docs/ARCHITECTURE.md` — subsystem contracts and pipeline
- `3463-LDFD/readme.md` — Live Data Feeding Framework (submodule): its own
  status banner is the only truth about what is built there
- `3463-LDFD/tests/` — 4 dependency-free suites plus a loopback fetch suite
- `docs/v12R2_SCOPE.md` — current-milestone scope (binding for v12R2)
- `docs/releases/v12R1/GITHUB_RELEASE_v12R1.md` — v12R1 release notes
- `docs/releases/v11S/GITHUB_RELEASE_v11S.md` — v11S release notes
- `docs/releases/v10S/RELEASE_NOTES_v10S.md` — historical v10S notes
- `docs/v11A3_SCOPE.md`, `docs/v11A2_SCOPE.md`, `docs/v11A1_SCOPE.md` — frozen history
- `docs/v11A1_MODEL_FORMAT.md`, `docs/v11A1_OPS.md` — historical direction
  (bannered as superseded; do not quote for v12R1 behavior)
- `SECURITY.md` — security reporting policy

> Historical files (`v11A*`, `v10S`) describe their own milestones, not
> v12R2. Quoting them for current behavior is a documentation bug —
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
| `v12R1`            | `V1.2RC1`            | Previous development milestone: hardening plus numerical correctness |
| `v12R2`            | `V1.2RC2`            | Current development milestone (R2 phase: mute mathematician learns) |

### Current Milestone

This release is:

```text
v12R2
```

Its theme is:

> Mute mathematician learns: v12R1 honesty kept, plus train-lib,
> char-v1 encoder, micromodel bridge, verifier eval, sandbox caps,
> CLI eval — scores, never speaks.

Previous milestone: v12R1 / V1.2RC1 (hardening plus numerical correctness).
Stable baseline: v11S / V1.1.

### Next Milestone

The next milestone is `v12R3` (the R3 freeze/hardening phase), followed by the v12 freeze,
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

Lancius `v12R2` is a development milestone, not a hardened release.
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

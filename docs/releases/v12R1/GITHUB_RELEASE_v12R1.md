# Lancius v12R1 — Development Milestone

**Tag:** `V1.2RC1`
**Public version:** V1.2RC1
**License:** GPL-3.0-or-later

---

## Overview

Lancius v12R1 is the **first development milestone** of the Lancius 1.2 cycle, built on the v11S stable baseline. Its themes are **hardening** (loader integrity, execution contracts, stability guards) and **mathematical correctness** (a hostile, formula-by-formula audit of every numeric path, each confirmed defect fixed and re-proven by independent execution, plus a despot truth batch that closed every remaining lie).

Lancius is a lightweight C machine-learning compiler and runtime focused on bare-metal inference, static graph execution, memory planning, and low-level runtime control.

This is a development milestone, not a stable release. `v11S` remains the stable release.

---

## Highlights

### Hardening

- v2 model integrity mandatory: CRC32 required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into legacy), O(1) duplicate-ID detection, sparse-ID bounds, streaming weight skip
- Execution contracts: attention/GQA/RMSNorm shape validation, cache-less long-context decode rejected, `_checked` counters on all execution/planning/copy paths
- Stability guards: softmax zero-sum errors, int64 INT8 accumulators, real overflow checks, OOM reporting (including KV-cache attention)

### Numerical Correctness

- N-dim trailing-rank broadcast `ADD`/`SUB`/`MUL` in IR, scheduler, and VM (was flat-loop wrong + OOB)
- Degenerate cross-entropy denominators report `NUMERICAL`, matching backward (no `1e30` masking)
- int64 conv index math; KV-cache OOM reporting; honest autodiff (OOB fixed, 4D `CONST` preserved, loud accumulation)
- Memory planner records freed offsets: diamond reuse verified value-identical to direct execution
- Strict ONNX converter: Reshape `0`/`-1`, Gemm always-clone (never mutates shared weights) + `alpha`/`beta`/`transA` rejection, symmetric Conv/Pool, static batch, batch-dim preserving Reshape

### Honest Validation

- **Every audit in the gate propagates failures through its exit code** — no false-green
- New `audit_despot_probe` pins broadcast values, cross-rank broadcast, diamond pooled parity, and the CE contract inside `make check`
- `audit_internals` verifies softmax normalization; `test_torture` runs a real cycle test; `fuzz_lancius` separates safe-rejects from true failures; `audit_modern_llm` verifies GQA values
- Sanitizer gate restores a clean build afterwards, so instrumented binaries can never leak into `make check`
- V3 forensic sweep (see `CHANGELOG.md`): ~70 defects fixed across kernels,
  scheduler, autodiff, persistence, interop, trainers, and operator;
  sanitizers clean, PyTorch parity exact

### Hardening Batch V5 (2026-09-30)

- Threadpool queue growth use-before-initialization fixed
- Command injection in `distill_prm800k` fixed (`system()` → `mkdir()`)
- IR silent NULL returns fixed (6 locations now set error codes)
- Optimizer no longer clears errors on success
- Quantizer zero-scale check added
- Scheduler cross-entropy consistency fixed
- Memory planner 32-byte alignment on free-block split
- Serializer/Vision `fprintf`/`printf` removed (library is silent)
- Examples hardened (`parity_runner`, `run_trained_batch`)
- Python `onnx_to_lancius.py` shape filtering fixed

### Ordinary-User Operator

- `./lancius demo` proves an install in one command; `./lancius tui`
  guides every workflow with validation, defaults, and offline honesty
- Every verb has `--help`; user file paths never pass through a shell;
  long trains confirm before burning hours; `doctor` distinguishes
  missing-core (fail) from missing-training-data (warning)

---

## Validation

- `audit_known_answer` 73/73, `audit_transformer_known_answer` 265/265, `audit_regression_13c` 49/49, `audit_fp32_path` 19/19, `audit_fault_injection` 11/11
- `make check`, `make check-long` (soak 3/3, fuzz 500/0), `make check-sanitizers` green from a clean tree
- Despot truth V2: `probe_v2` pins SUM-3D, RESHAPE, SUM_AXIS, `broadcast_to_shape`, N-D partial fail-loud, attention NaN→NUMERICAL; `test_grad_check` (`8.6e-10`, `5.8e-8`); see `docs/DESPOT_TRUTH_V2.md`

---

## Breaking Changes from v11S

- Degenerate softmax/cross-entropy rows return `LANCIUS_ERROR_NUMERICAL` instead of values
- `lancius_add`/`sub`/`mul` accept trailing-rank broadcast shapes (superset of v11S same-rank behavior)
- Files with v2 `checksum == 0` are rejected by default (opt-in via `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`)
- Additive stable-API error codes: `GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/`INVALID_HANDLE`

---

## Platform Support

- **Primary:** Linux x86_64 (GCC/Clang, OpenMP, AVX2/FMA)
- **Build:** `make clean && make -j$(nproc)`
- **Validation:** `make check && make check-long && make check-sanitizers`

---

## Known Limitations

- CPU-only (no GPU acceleration)
- Static graph execution only (no dynamic shapes)
- Transformer inference is experimental (FP64-only KV-cache, no backward passes)
- FP32 execution scoped to matmul only
- ONNX conversion experimental (LeNet-class graphs only; dilations/groups/pads/ceil rejected loud)
- Training is not part of any release contract; N-dim partial broadcast grads fail loud (no `SUM_AXIS_ND` yet); trainers exit 1 at ≤ chance
- GELU is tanh-approx; norm eps pinned to `1e-5`

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

(CMake builds the libraries only; the Makefile is the canonical path for the full validation suite.)

---

## Bug-Fix Campaign (pre-R2)

A comprehensive bug-fix campaign closed **38 defects (26 in V4 + 12 in V5)** across the
codebase before R2 development began, plus **45 defects in despot audit V6 (2026-10-01)**.
Full enumeration lives in `CHANGELOG.md`; essence only here:

- **Autodiff**: BROADCAST backward, race conditions in conv2d_bwd_in/maxpool2d_bwd, NULL checks, OOB reads on shape/axes, off-by-one in node capacity, dangling pointer in stable API; V6 exact 4D permute+reshape reduction, CE_BWD guards
- **Safety**: command injection in CLI, `abort()` removed from library code, memory leaks in IR, overflow checks in bytecode VM; V6 VM tape/out_reg, scheduler NULL-deref, pool UAF, arena wrap, 32B align
- **Correctness**: `fprintf`/`printf` → `lancius_set_error`, serialization portability, CRC32 race condition, NOP handling in serializer, threadpool timeout; V6 mkstemp/fsync, per-channel refuse, ndim==0/CONST/ROPE/scale, quantizer/optimizer/vision checks
- **Quantization**: per-channel quantization, dequantization
- **Build**: Python scripts, build system, Makefile; V6 capped fetches, distill checks, tar-slip, `.d` purge

All fixes validated by `make check`, `check-sanitizers`, and despot truth probes.

---

## Full Changelog

See `CHANGELOG.md` for the complete v12R1 changelog.

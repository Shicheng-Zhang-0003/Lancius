# Lancius Current Status

Current internal milestone: **v12R2** (point release **v12R2.4**)
Previous internal milestone: **v12R1**
Public equivalent: **V1.2-RC3.4**
Release line: **12 R2 development — RC3.x model line**

V1.2-RC3.4 publishes the compact four-family I6 verifier set, its corpus,
checkpoints, and expanded stress audit. It also corrects training/runtime
numerics and removes the obsolete dataset-bridge integration. The current
release metrics and scope limits are recorded in `CHANGELOG.md` and
`generation_1/I6/README.md`.

## Phase

v12R2 is the second development milestone of the v12 cycle
(progression `S → R1 → R2 → R3 → S`; v12R2 is the R2 phase),
proving Lancius can learn. Previous v12R1/V1.2RC1 (hardening plus
numerical correctness) is the baseline, all gates green.

Historical baseline (v11A3 gate, complete):
- `make check` green
- `make check-long` green
- sanitizer validation green
- malformed model loading safely rejected
- version identity consistent
- model format v2 frozen with CRC32 integrity
- stable C API covers core inference workflow

Live v12R2 gate is Validation batch below (`make check/long/sanitizers`
+ despot probes + `probe_v6`); historical checklist above is not the
current gate.

## RC3.x point releases (V1.2-RC3.1 → V1.2-RC3.4)

The RC3.x line carries development point releases. RC3.4 changes training and
runtime numerical behavior but keeps the v2 model format and stable C ABI;
it makes no `v12S` scope claim. The current point is `V1.2-RC3.4`.

### V1.2RC3.1 (2026-10-24): shell fixes + model-release lineage

The first point release fixes four composition-layer defects only (I2's
removed-sentinel repo resolution + CLI locator backport, the CLI's three stale
root-relative binary probes reporting 0/5 on a built tree, the
`train_i3`/`train_i4` missing-BINS makefile omission, and the
`generation_1` README/card count drift including the I1 58M claim and the
"12 clones" pool wording), and puts I4 in the tree ahead of its release
assets. Verified per fix in `CHANGELOG.md`; `make check` RC=0 with the batch
in tree. The I3 verifier and the I4 export are the RC3.1 assets; provenance
blocks in `docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.1.md`.

### V1.2RC3.2 (2026-11-06): I6 flagship verifier lands; I5 rules engine

RC3.2 brings the I6 lineage into the tree — `examples/train_i6.c`
(65→4096→2560→5, fused cross-entropy, warm start from the I3 ALU pool, pinned
normalizer, hard-negative mining, checkpoint/resume, `--verify-lancius`),
`generation_1/I6/gen_data_i6.py`, `run_big_i6.py`, `paths.py`, and the model
card — alongside the I5 generic decision-surface trainer and its evaluation
and acquisition tooling. Corpus/model/result directories remain generated and
gitignored. Full enumeration in `CHANGELOG.md`; release notes in
`docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.2.md`.

### V1.2-RC3.3 (2026-11-23): I6 corpus clean + all four experts (historical)

The corpus label-noise floor is removed
(`verifier5.level_realized()` is the single source of truth; the generator
refuses unrealized labels; 0 conflicting labels across 11,349 train / 3,655
test overall feature keys), feature 57 is `n_wrong` instead of `mean_rel`,
and the verdict is continuous — expected level under the 5-class softmax,
clipped to `(-1+1e-6, 1-1e-6)`, with `-2` reserved for non-logic faults.
All four family experts are trained and exported; see the table below. Full
enumeration in `CHANGELOG.md`; release notes in
`docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.3.md`.

### V1.2-RC3.4 (2026-11-25): compact I6 flagship + complete audit bundle

RC3.4 uses four fresh 65→128→64→5 experts (17,029 parameters each), seven
checker-derived features, algorithm-held-out splits, and independent
continuous MAE/MSE. Each expert scored 100% exact agreement on 741,544
overall and 2,785,441 local-step generated labels; 68,056 condition slices
were all exact. These are generated feature-rule verdict results, not a
human-judged math benchmark. The release attaches exports, training
checkpoints/logs, the 233,381-row corpus and normalizer, and the full
scenario-/step-level stress outputs. Numerical training corrections and
obsolete external-data integration removal are also included. Details are in
`docs/releases/v12R2/GITHUB_RELEASE_V1.2-RC3.4.md`.

## I6 verifier status (historical V1.2RC3.3 release)

I6 keeps the proven `65 → 4096 → 2560 → 5` shape (10,771,461 parameters),
warm-starts from `generation_1/I3/alu_pool/alu_01.lancius`, and routes rows
to four family experts by `FAM_ID` (feature index 58): algebra, calculus,
linear algebra, number theory. The corpus is
`generation_1/I6/corpus_clean/` — 186,824 train / 46,706 test rows over 150
algorithms and 5 levels.

The previously published sub-millilevel results are withdrawn: the MAEs were
computed from argmax labels, and the MSEs were estimated as `MAE²`. A direct
continuous-output audit of the saved artifacts found MAE ranges of
0.252607–0.258127 (algebra), 0.005315–0.010808 (calculus),
0.003623–0.012003 (linear algebra), and 0.417701–0.420918 (number theory)
across two evaluation paths. All miss the `1e-3` target. Corrected MSE for
those artifacts has not been established. Training settings and export
history are in `I6_TRAINING_STATUS.md` and `generation_1/I6/README.md`.

## I6 current release status (V1.2-RC3.4)

The current release uses four 65 → 128 → 64 → 5 experts, fresh
initialization, seven checker-derived semantic coordinates, and
algorithm-held-out train/validation/test splits. The 233,381-row corpus has
zero label-rule mismatches and no algorithm overlap between splits. The
held-out test metrics for all four experts pass the strict `1e-3` gates for
both continuous expected-level MAE and independently computed per-row MSE,
for overall rows and local-step rows. The 150-algorithm stress sweep matched
all generated labels exactly across five levels and three tiers; see the I6
README for condition-level continuous metrics and the I3 baseline comparison.

## v12R2 theme: mute mathematician learns (R2-1..R2-6)

v12R1 honesty kept; R2 proves learning: R2-1 train-lib (SGD/SGDM/AdamW-decoupled/clip/cosine),
R2-2 char-v1 encoder + manifests, R2-3 micromodel bridge (0.291094->0.000010),
R2-4 eval_verifier harness, R2-5 sandbox caps + replay, R2-6 CLI eval + ABI/binding + FP32 parity.
Generation scrapped (mute scores, never speaks). See `CHANGELOG.md` R2-1..R2-6 and `docs/v12R2_SCOPE.md`.

## v12R1 theme: hardening plus numerical correctness (inherited baseline)

Bottom-up correctness pass over all layers, then a hostile,
formula-by-formula audit of every numeric path. Each confirmed
defect fixed and re-proven by independent execution.
Full per-batch record lives in `CHANGELOG.md`; essence only here
(see also `docs/DESPOT_TRUTH_V2.md`):
- N-dimensional broadcast `ADD`/`SUB`/`MUL` correct in scheduler, IR shape
  inference, and bytecode VM (was flat-loop wrong + OOB)
- Softmax zero-sum guard in scheduler and VM
- v2 CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into
  legacy); O(1) duplicate detection; sparse-ID bounds; streaming skip
- Attention cache/heads/dim validation; cache-less long-context decode
  rejected; GQA shape validation
- INT8 `int64` accumulators; real overflow guards; OOM error reporting
- Aborting counters replaced with `_checked` + error returns on execution,
  planning, and copy paths
- ONNX strictness (Reshape `0`/`-1`, Gemm clone + `alpha`/`beta`/`transA`,
  symmetric Conv/Pool, static batch)
- Training alignment (`[-1,1]` CIFAR norm, He init, identity binding,
  identity-tracked eval logits)
- Widened stable FFI error codes; false-green audits propagate failures
- Despot truth batch: planner offsets recorded (diamond reuse sound),
  trailing-rank cross-rank broadcast with compat guards, int64 conv indices,
  KV-cache OOM reporting, CE degenerate is NUMERICAL, Gemm always-clone,
  Reshape preserves batch, torture cycle test is real, GQA values verified
- Despot truth V2 (2026-09-28): `broadcast_to_shape` any 1..4-D; `SUM`
  (any ndim), `SUM_AXIS0/1`, `RESHAPE` VJPs exact; N-dim partial broadcast
  fails loud with whole-graph abort; permute/batched-matmul offsets checked;
  no abort on hot paths; INT8 Add row-bias-only; INT8 `scale=1.0` all-zero;
  attention/KV/GQA NaN→NUMERICAL; GELU tanh-approx documented;
  `LANCIUS_NORM_EPS` pinned; vision `NUMERICAL` scale + checked pool-bwd;
  `u64→size_t` checked; ONNX dilation/group/pads/ceil rejected; trainers
  raw-abort + chance-gate; `test_ffi_error` honest; see
  `docs/DESPOT_TRUTH_V2.md` and `probe_v2` (`/tmp/opencode/lancius-despot-logs`)
- Hardening batch V4 (2026-09-28): BROADCAST backward race-condition fixes
  (thread-local accumulators in conv2d bwd_in + MaxPool2D bwd); autodiff
  NULL checks + OOB fixes; IR memory leaks fixed; `abort()` removed from
  library code; stable API dangling pointer fixed; serialization portability
  (uint64_t, byte swapping, CRC32 call_once); threadpool timeout; VM overflow
  checks; per-channel quant + dequant; CLI fork+execvp; Python security fixes;
  build system hardening (-Werror, Threads, version consistency)
- Hardening batch V5 (2026-09-30): threadpool queue growth use-before-init fixed;
  command injection in distill_prm800k fixed; IR silent NULL returns fixed (6 locations);
  optimizer error clearing on success fixed; quantizer zero-scale check added;
  scheduler cross-entropy consistency fixed; memory planner alignment fixed;
  stable API duplicate include removed; serializer fprintf/printf removed;
  vision ops fprintf removed; example files hardened (parity_runner, run_trained_batch);
  ONNX converter shape filtering fixed; autodiff NOP comment clarified
- Despot audit V6 (2026-10-01, 45 defects): exact 4D broadcast backward
  (permute+reshape), CE_BWD ctor/exec guards, conv_bwd_w/LN/RMSN guards;
  VM ndim!=2/tape/out_reg/shape; scheduler NULL-deref/errors; pool
  malloc+free/errors; arena checked grow; static 32B align + posix_memalign;
  v2 mkstemp/ftello/CONST/ROPE/scale; v1 tmp+FP32; quantizer/optimizer/vision;
  capped fetches; distill checks; tar-slip; .d purge. See CHANGELOG §V6
  and `docs/DESPOT_TRUTH_V2.md` §12 + `probe_v6`.

## Feature freeze

v11S/v11A3 historical freeze (not v12R1 scope):
- no new operators
- no new runtime subsystems
- no new training features
- no new model-format changes

v12R1 adds correctness within scope plus additive primitives only
(`broadcast_to_shape`, TANH/MSE, per-channel quant/dequant, pool timeout);
see `CHANGELOG.md` for deltas.

## V7 truth batch (this pass): the gate audited against itself

Four defects found and fixed, each by a method that does not trust the author:

- **`SUM_AXIS_ND` fell through to the vision router.** The branch computed
  correct values, had no terminating `return`, reached
  `if (n->op >= LANCIUS_OP_CONV2D)`, and `lancius_execute_vision_op` rejected
  opcode 42 with `UNSUPPORTED_OP`. Correct answer, then a false failure in the
  thread-local error that autodiff reads as "abort the whole graph".
- **45 autodiff abort paths leaked `tg->grad_nodes`** (72 bytes per failing
  call; ASan proved it; `audit_fault_injection` hit it on every run). Every
  abort now funnels through one `autodiff_abort()` helper.
- **`lancius_broadcast_to_shape` was stricter than the op it builds.** It
  demanded exact shape equality while its 4-D sibling and the executor accept
  NumPy trailing-rank with per-dim 1-or-equal. Consequence: the documented
  `SUM_AXIS_ND` VJP could never build, so R3-1's headline feature was not
  trainable end to end. Now exact against central differences at ~1e-9.
- **`test_path_bg` (960 B) and `audit_modern_llm` (640+ B) leaked their own
  buffers** by assigning `runtime_data` instead of binding ownership.

Four gates added, because the first three could be green while the library was
wrong:

| Gate | What it proves | Tally |
|---|---|---|
| `make check` | the whole gate: every audit, every example, every known-answer | see `docs/DESPOT_TRUTH_V2.md` §16 |
| `make check-oracle` | every kernel, graph op, and train-lib entry point recomputed in NumPy / PyTorch autograd / closed form / central differences | **284 comparisons, 0 skipped (torch present)**: 46/46 kernels + 45/45 graph ops + 169/169 train-lib + 11/11 conv2d-vs-torch + 13/13 reductions+softmax-vs-torch |
| `make check-physics` | the physics/numerics substrate -- IEEE 754 constants, libm accuracy, the GELU/attention/RoPE constants, FMA contraction, kernel and scheduler determinism, the pool thread budget and the v2 model CRC -- against an independent reference that never executes Lancius | **56 checks, 0 failures** (ISO decimal arithmetic + zlib + the source papers; python3 stdlib only, no torch, no network) |
| `make check-sanitizers` | ASan + UBSan + LSan over the **instrumented library** and every audit (the old gate ran 3 binaries against an uninstrumented archive) | 28/28 clean (V9: +diamond +verifier_head; distill standalone) |
| `make check-ubstrict` | UBSan alone, `-fno-sanitize-recover=all`, so signed overflow / shift / float-cast abort instead of warn | 28/28 clean (V9: +fuzz/path_bg/diamond; the 29 in the V9 tally was 25+3 rounded up, and the sweep list is the source of truth) |
| `make check-mutation` | real defects injected; the gate must go red | **18/18 caught, 1 verified-neutral, 0 holes, 19 tries** (first run: 8 caught, 7 holes) |

Six gates run on every push (`check`, `check-long`, `check-sanitizers`,
`check-oracle`, `check-physics`, `check-ubstrict`); only `check-mutation` is
sampled to `main` + dispatch, because it rebuilds the library once per
mutation. Full record: `docs/DESPOT_TRUTH_V2.md` §16 (V9 §20, V10 §21).

Throughput is measured, not claimed, by `make bench_kernels`
(`examples/bench_kernels.c`): median of 7 timed calls after a warm-up, on
this machine, with the thread environment and the FLOP formula printed
beside every rate. It is deliberately not a gate member, because timing is
machine-dependent and must never be able to turn the suite red.

## V8 truth batch: every constant checked against its primary source

A finite difference proves the derivative of whatever function was implemented,
not that the intended function was implemented. If GELU's coefficient were
0.0447 rather than 0.044715, every finite difference and every known-answer test
would still pass. So §17 of `docs/DESPOT_TRUTH_V2.md` checks each constant,
equation and dataset fact against its published source.

**Verified exact:** GELU tanh form (Hendrycks & Gimpel 2016); LayerNorm biased
`1/n` variance and eps 1e-5 (Ba, Kiros & Hinton 2016); RMSNorm with no mean
subtraction (Zhang & Sennrich 2019); the `1/sqrt(d)` attention scale (Vaswani
2017); RoPE's `10000^(-2i/d)` schedule and its relative-position property (Su
et al. 2021); SiLU and SwiGLU (Hendrycks & Gimpel, Shazeer 2020); AdamW
decoupled decay and `sqrt(v_hat)+eps` placement (Loshchilov & Hutter 2019,
matching `torch.optim.AdamW` to 5.6e-16 over 20 steps); the SGDR cosine
schedule; gradient-norm clipping (Pascanu et al. 2013); GQA head grouping
(Ainslie et al. 2023); He init `sqrt(2/fan_in)` with conv `fan_in =
C_in*K_h*K_w` (He et al. 2015); the INT32 accumulator threshold, verified
arithmetically as `2147483647/(128*127) = 132104` exactly as the comment claims;
CRC-32 byte-identical to `zlib.crc32`; the online-softmax rescaling exact to
2.2e-16 against direct softmax; and the MNIST and CIFAR-10 container formats by
content invariants (IDX magic numbers, record counts, and the published class
balances — CIFAR-10's test split is exactly 1000 per class).

**One real error found and fixed:** the GELU comment claimed max deviation from
erf-exact GELU of `~2e-3`. Measured over a dense scan the true maximum is
**4.74e-04** at x = 2.699 (rms 1.42e-04) — the claim overstated the error 4x,
and the external oracle had inherited the same unverified number as its
acceptance bound. The comment is corrected and the oracle now **measures and
pins** both figures, so the next drift is caught rather than passed along.

**One divergence recorded, not changed:** the INT8 quantizer clamps to
`[-128,127]` where the TFLite spec and TensorRT both specify `[-127,127]`. It is
numerically inert, because `scale = max_abs/127` makes -128 unreachable, and
§17.2 explains why it is recorded rather than silently altered.

**Still unchecked, and said so:** the GELU `±10` clamps are a local guard rather
than part of the paper (verified exact in the limit); CIFAR-10 is normalised to
`[-1,1]` rather than PyTorch's per-channel statistics, a deliberate choice that
makes local numbers non-comparable to torchvision baselines without adjustment;
and `.tar.gz` digests cannot be compared across mirrors because gzip is not
byte-reproducible, so datasets are verified by content rather than digest.

## Validation batch — v12R2

Build clean under `-Wall -Wextra -Werror`; `audit_v7_hardening` 346/346
(H1-H9: sticky-error contract, softmax/attention stability at |z|=1e5, arena
32B alignment sweeping 4KB..1MB across the mmap/heap boundary, alignment cap,
quantizer exactness and error bound, fusion shape guard, full op table,
`broadcast_to_shape` contract at every rank, INT8 64-bit accumulation proven
at 204800 taps); `audit_regression_13c` 49/49,
`audit_known_answer` 165/165, `audit_transformer_known_answer` 265/265,
`audit_fp32_path` 19/19, `audit_fault_injection` 12/12,
`audit_sum_axis_nd` 21/21, `audit_train_bwd` 32/32,
`audit_train_converge` 12/12 green;
`make check`, `make check-long`, and `make check-sanitizers` green with
failure-propagating exit codes; broadcast/planner/softmax/CE fixes verified
by the despot probe (`audit_despot_probe`, in the `make check` gate);
V2 fixes verified by `probe_v2` (SUM-3D, RESHAPE, SUM_AXIS, N-D fail-loud,
attention NaN→NUMERICAL) + `test_grad_check` (`8.6e-10`, `5.8e-8`);
V3 fixes verified by the same gate (no new gates needed: every finding was
a hardening of an already-gated path) plus pytorch parity
(`3.42e-07`, FP64 limits);
V4 fixes verified by the same gate plus new audits for race conditions,
quantization (per-channel + dequant), serialization portability, and CLI
security (fork+execvp).
R3 training-wrap verified by the same gates plus the new training gates (N-dim reduction exactness, norm/batched finite-diffs ~1e-9..1e-11, XOR convergence + determinism + checkpoint resume, global-norm clip).
V6 fixes verified by the same gates (all green from clean tree) plus
`probe_v6` (4D dim0/1/2/3 + multi-dim, CE_BWD, VM rank) and
`test_grad_check` still `8.6e-10`, `5.8e-8`.
V9 fixes verified by the same gates (all green from clean tree) plus
`temp/despot-v9/` reproducers (CE-NaN NUMERICAL, pool NULL_PTR, VM LIMIT +
trailing-HALT exactness) and oracle 276/276 0 skipped (torch present).

## Despot V9 (2026-10-09): the gate audited against physics truth

Three parallel hostile sweeps (math / programming / operational) + NumPy 2.5.1
+ torch 2.13 CPU + closed-form re-execution. 13 defects fixed: CE corrupt-label
NUMERICAL, INT8-matmul scale NUMERICAL, v2-only saves, v1 trunc-leak + LIMIT,
handle cookies + owner, pool_submit NULL_PTR, VM checked + HALT, planner
fail-closed, Python TOCTOU + inflate caps, CLI topk 1..100. Full record
`docs/DESPOT_TRUTH_V2.md` §20 + `CHANGELOG.md` V9.

## Operator status

`./lancius` (`examples/lancius_cli.c`) is the ordinary-user operator:
`quickstart`/`doctor`/`status`/`models [--check]`/`demo`/`help`, per-verb `--help`,
`datasets`/`train`/`run`/`info`/`convert`/`export`/`eval`/`generate (fails loud -> eval)`, and a `tui`
that calls verbs directly (no re-exec), validates inputs, confirms long
trains, and pauses on ttys. User paths exec without a shell. `demo`
proves an install in one command; `make check` covers `info`/`run`/`eval`. `train_mnist/cifar10` stay manual-only
(see `KNOWN_LIMITATIONS.md`); `run_edge`/`parity_runner`/`run_trained_batch` are manual
parity runners, not gate members.

## Where to start reading

`docs/HOW_IT_WORKS.md` -- the canonical guide to how a model is defined,
processed, executed, trained, and proved correct. `docs/ARCHITECTURE.md` is the
shape of the system; that document is the mechanics.

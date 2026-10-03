# Lancius v12R2 Scope — Development Milestone (R2 phase)

Internal: `v12R2` · Public github tag: `V1.2RC2` (internal code stays `vXYRZ`;
public tags read `VX.YRCZ`; github tags/releases only).

Previous milestone: v12R1 / V1.2RC1 (hardening plus numerical correctness,
all gates green, no false-green).

## R2 mandate

v12R1 proved Lancius computes honestly. v12R2 proves Lancius can **learn**:
every role below must stand up as an independent subsystem with its own
gate. R3 freezes and hardens; v12S (public `V1.2`) ships.

Target class: mute-mathematician micromodels below 100M parameters that
decide mathematical input with precision rivaling 400B-class models — by
exact computation, checkable reasoning steps, and deterministic execution,
not by fluency. No language generation: the model never speaks, it scores.
A mute mathematician that solves olympiad/millennium problems outranks a
fluent speaker that cannot solve.

Language generation is scrapped in v12R2: `generate_text`/`run_llm`
deleted, CLI `generate` fails loud pointing to `eval`, no streaming
generation, no prefill/generation demos. Transformer kernels stay as exact
math primitives with known-answer audits; generation flows do not.

## Architectural constraint (binding)

The **extended boolean architecture is for the models only**. Lancius must
be able to *train it into models*, but it is not a component of the
Lancius framework itself. Concretely:

- The C core gains only **generic trainable primitives**: bounded
  activations (tanh), regression losses (MSE), and the builders /
  executors / gradients / loader cases they require. No truth-algebra
  opcodes, no validity gate, no residual-map opcodes in the IR.
- Truth semantics (polarity/confidence split, weakest-link aggregation,
  residual map, $\bot$ handling, thresholds) live in **model weights,
  example training loops, and the Python data pipeline** — all outside
  `src/` and `include/`, all replaceable without touching the framework.
- Any future reviewer must be able to delete every extended-boolean
  artifact and still have a complete, coherent framework. If a change
  fails that test, it does not land.

## Workstreams (each independent, each gated)

### R2-1 · Training library (`src/train/` prospects + optimizer primitives)

AdamW + SGD + gradient clipping + cosine/warmup schedules as a real
library, not example-inline code, each routine proven by finite-difference
tests that exit nonzero on divergence.
Gate: MNIST/CIFAR parity vs PyTorch reference within tolerance; loss
monotonicity tests; zero new warnings under `-Werror`.

### R2-2 · Problem encoder (reading, not speaking + batching + manifests)

Mute mathematician reads but never speaks. Byte-level problem encoder
`char-v1` is the decision (BPE rejected for v12R2: speaking needs BPE,
scoring does not). Dataset manifests with SHA256 and reproducible splits
for every set `manage_datasets.py` pulls.
Gate: byte-reproducible datasets from manifest alone; encoder roundtrip
tests.

### R2-3 · Step-distillation bridge (the missing link)

`manage_datasets.py` already pulls GSM8K/MATH/MiniF2F/PRM800k/SVAMP/
ProofWriter/RuleTaker, and its docstring already names the gap: problem
sets must become fixed-size numeric vectors for the mute scorer.
v12R2 builds that bridge (byte-level problem encoding, explicitly marked;
no language-model encoding).
Gate: vendored PRM800k rows → vectors → C training loop end to end,
deterministic.

### R2-4 · Verifier head + eval harness (extended boolean, models-only)

A scalar $\tanh$ verifier head trained with MSE against step labels in
$[-1,1]$, weakest-link credit assignment through the argmin step, eval on
GSM8K/SVAMP with per-step accuracy. The full truth-value specification
(negation, Gödel min/max connectives, weakest-link theorem, residual map
$v(r) = (\delta-r)/(\delta+r)$ with anchors at $r \in \{0, \delta/3,
\delta, 3\delta, \infty\}$, $\bot$(-2) abstention channel, accept/flag/
reject thresholds) is models-side doctrine, recorded below, not framework
law.
Gate: published eval numbers reproducible by one command; anchor-point
calibration as a test assertion; no cherry-picked splits.

### R2-5 · Sandbox (budgets + determinism + abstention)

Peak-memory cap enforced pre-execution, op-count/step cap, deterministic
replay (bit-identical across runs and thread counts), closure of the
remaining `abort()` reachability on internal paths. The $\bot$ channel
gives untrusted-model handling a name: the checker abstains instead of
scoring.
Gate: malicious-model suite passes under ASan/UBSan inside bounded
resources; replay test bit-identical.

### R2-6 · Runtime growth + kernel packaging

FP32 matmul parity (today FP64-first math, FP32 matmul only), no streaming
generation (scrapped), `lancius` CLI (run/train/eval verbs, no generate);
versioned ABI with break-test, Python binding smoke test, clean-container
install verification.
Gate: FP32-vs-FP64 parity audit; CLI runs a vendored micromodel end to
end; downstream binding builds against installed headers only.

### R2-7 · N-dim training completeness (the broadcast hole)

`SUM_AXIS_ND` (id 42): per-axis reduction for any 1..4-D tensor, closing
the N-dim partial-broadcast fail-loud (e.g. `[2,1,4]` vs `[2,3,4]`).
Gate: values 1..4-D, 2D equivalence, bad-axis reject, v2 roundtrip, 3D
exact grads, finite-diff (`audit_sum_axis_nd` in `make check`).

### R2-8 · Norm/activation/batched backwards (affine params train)

Batched transpose (id 43, forward + persistable) unlocks `MATMUL_BATCHED`
backward; dedicated `LAYERNORM_BWD[_GAMMA,_BETA]` (44..46),
`RMSNORM_BWD[_GAMMA]` (47..48), `GELU_BWD` (49) with pre-router
executors. Attention/GQA/SwiGLU/RoPE backward stays fail-loud (staged).
Gate: finite-diff ~1e-9..1e-11 per VJP incl. gamma/beta
(`audit_train_bwd` in `make check`).

### R2-9 · Convergence + determinism + checkpoint honesty

`lancius_clip_global_norm` multi-tensor clip; `audit_train_converge` in
`make check`: 2-4-1 tanh XOR through graph+autodiff+SGD solves 4/4,
same-seed rerun agrees 1e-12, v2 checkpoint at half-time resumes to the
same loss 1e-9. Moments stay caller-owned (stateless-lib contract).

## Extended boolean model — specification (models-side)

Value space $V = [-1,1] \cup \{\bot\}$, $\bot$ wire-encoded as $-2$,
disjoint from the truth channel by construction. Score decomposes into
polarity $\mathrm{sign}(v)$ and confidence $|v|$; $0$ is maximal
uncertainty and the unique fixpoint of negation $\neg v = -v$.

- Anchors: $1$ verified true · $0.5$ probably true · $0$ fifty-fifty ·
  $-0.5$ probably false · $-1$ false · $\bot$ "that ain't math" (reject
  with cause, never score).
- Connectives: Gödel $\land = \min$, $\lor = \max$ (idempotent,
  De Morgan-consistent with $\neg$; Łukasiewicz rejected — its
  $0 \land 0 = 0$ yields the same uncertainty as Gödel, but its
  non-idempotent $\land$ complicates the weakest-link theorem).
  Implication is the Gödel residuum ($v \to w = 1$ if $v \le w$ else $w$).
- Weakest-link theorem: $V(D) = \min_i v_i$ over derivation steps, with
  $\bot$ absorbing. One false step caps the derivation; one $\bot$ step
  voids it. Monotone, idempotent, parameter-free.
- Residual map (canonical): $v(r) = (\delta-r)/(\delta+r)$ for residual
  $r \ge 0$ and derived tolerance $\delta$. Anchors fall out exactly at
  $r \in \{0, \delta/3, \delta, 3\delta, \infty\}$. $\delta$ is computed
  per step (flop bound × epsilon × condition estimate), never a magic
  constant.
- Pipeline per item: validity gate (math vs $\bot$; unknown types route
  to $\bot$, never a guess) → type classification (closed arithmetic,
  algebraic, calculus, rule steps, word problems reduced to the former)
  → exact FP64 computation → comparison → score → min-aggregation.
- Learning: PRM800k labels $\{-1,0,+1\}$ embed directly as
  $\{-1,0,1\}$; verifier head is a scalar $\tanh$ trained by MSE with
  subgradient credit through the weakest step; validity gate is a
  separate binary classifier. Native reading of the scale is comparative
  (ordering + anchors); probabilistic calibration is an open question,
  not an assertion.
- Policy defaults: accept $\ge 0.5$ · repair/flag $(-0.5, 0.5)$ ·
  reject $\le -0.5$ · $\bot$ rejects with cause.
- Open problems: per-type $\delta$ for symbolic/logical steps;
  regression vs ordinal vs two-headed scorer; multi-step credit beyond
  argmin-subgradients; whether $\bot$ covers only ill-formed input or
  also out-of-competence abstention (recommended: both).

## Explicitly out of R2

GPU backend, dynamic-shape generality, serving infrastructure, v2 format
breaks, language generation of any kind (scrapped: no generate/prefill
demos, no streaming, no BPE speaking-side), full transformer pre-training
in C (verifier-staged scope is the commitment; anything larger stays
Python-orchestrated). Target stays below 100M: anything larger is out.

## v12R1 bug-fix campaign (applied before R2 work)

A comprehensive bug-fix campaign closed **38 defects** before R2 work
(26 in V4 + 12 in V5: autodiff reduce/races/NULLs/OOB, CLI injection,
abort→errors, leaks, VM overflow, fprintf→errors, portability, CRC
call_once, pool timeout, per-channel+dequant, Python/build). No scope
change. Full enumeration lives once in `CHANGELOG.md` (single-owner);
essence only here.

## Validation contract (inherited from v12R1, extended)

`make check` / `check-long` / `check-sanitizers` all green from a clean
tree, every gate exits nonzero on divergence, new primitives carry
known-answer + finite-difference proofs, eval numbers are published with
seeds and misses. No false-green, no cherry-picks.

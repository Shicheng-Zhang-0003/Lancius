# V1.2-RC3 — unified trainer, ALU conglomeration, trained model artifacts

Source archives (auto-generated from the `V1.2-RC3` tag) plus **twenty-two** trained
model artifacts. This is the release the title line advertises in full: the
trained models are the point, the source is included so every number below is
reproducible rather than asserted, and RC2's models still load unchanged —
no format break, no operator change, no ABI change.

**Pre-release, deliberately.** Every accuracy quoted below measures agreement
with a symbolic checker on synthetic derivations with labels known by
construction. None of these models has ever seen human-written mathematics.

---

## The finding that started this release

`examples/train_verifier5.c` and `examples/train_gen1.c` — the two copy-forked
drivers that produced every verifier shipped in RC2 — predated the library's
own training tools (`src/train/lancius_train.c`) in nine specific ways, and
none of the nine fails loud:

1. **MSE-on-logits with an `lr_eff = LR*NCLASS/2` step-size fudge.** The step
   *form* was right; the gradient was wrong — a true CE gradient carries the
   input-dependent softmax Jacobian, and no constant converts one into the
   other. The loss was never read: both trainers printed it as `-`.
2. **A non-Fisher–Yates shuffle** — `swap(idx[i-1], idx[nfit-1-j])` draws each
   partner from a shrinking tail instead of the full prefix, a biased
   permutation.
3. **A graph rebuilt from scratch per batch** — build + autodiff + arena reset
   + backward (which re-runs the forward internally) + destroy, every step:
   ~2× forward waste and allocation churn for one forward's worth of signal.
4. **No instrumentation** — no grad clip, no NaN guards, constant LR while
   `lancius_lr_warmup_cosine` already existed in the library.
5. **Selection on overall-exact only, step rows skipped;** no checkpoint, no
   resume; `--lancius-rows` frozen at 256 into every export; dims hardcoded
   against one corpus.

`examples/train_verifier.c` supersedes the lineage as one parameterized
trainer (any FEAT/NCLASS/H1/H2 from `corpus.hdr`):

- the framework's own **fused softmax cross-entropy** (`lancius_cross_entropy`,
  exact VJP `(softmax−y)/R`), with the loss **read every step**;
- a **persistent training graph** — built once, INPUT buffers bound by pointer
  (the autodiff mirror shares `runtime_data`, verified in
  `src/math/lancius_autodiff.c`), so a step is one memcpy plus one fused
  fwd+bwd schedule execute: no per-batch graph build, no double forward;
- **Fisher–Yates** epoch shuffle; per-step **`lancius_lr_warmup_cosine`**, with
  resume stretching the schedule over the remaining plan instead of clamping;
- **`lancius_clip_global_norm` every step**, fail-loud guards on batch data,
  grads and weights — NaN dies with the tensor index, not two epochs later;
- model selection on the **deployed metric** — mean of val overall- and
  per-step-exact, MAE tiebreak — twice per epoch (`--eval-frac`);
- **checkpoint/resume**: `best.bin`/`last.bin` carry weights, momentum, mu/sd,
  rng state and step counter;
- export defaults to **`--lancius-rows 1`** — the row count is no longer
  frozen into the artifact (it stays available at 256 for the legacy bus);
- the loader validates bin sizes against `corpus.hdr` and rejects
  non-finite features **with row indices at load time**.

## Verification numbers

Gradient check of the CE objective against central differences: **7.3e-08**
at 4096×2560, **2.2e-08** at 512×256.

Full-scale run, `data_vec/v5` (20 algorithms, 41,689 rows), 65→4096→2560→5,
5 epochs, 555 steps, ~40 min:

```
   step   lr       CE(ema)   val-exact  step-exact   MAE
     55   0.0494    0.6576     0.9947     0.9980    0.0026  <- best
    220   0.0346    0.0249     1.0000     1.0000    0.0000  <- best
    555   0.0005    0.0001     1.0000     1.0000    0.0000
```

Held-out test split, best snapshot restored (never touched during training):

```
overall    n=1574   exact=0.9968  within1=0.9987  macroF1=0.9968  MAE=0.0029  kappa=0.9960
per-step   n=6764   exact=0.9999  within1=1.0000  macroF1=0.9996  MAE=0.0001  kappa=0.9998
```

The same LR that required the `lr_eff` rescue under the MSE step anneals
monotonically `0.0494 → 0.0005` with **zero divergence events** — that is the
difference between matching a step-size form and owning the gradient.

Resume was exercised, not asserted: a 1-epoch run checkpoints at step 111
(best 0.9974); `--resume` continues step numbering, anneals `0.0088 →
0.000524` across the resumed epoch, and improves the best each eval.

The 3×3×2 wire sweep (H1 ∈ {64,96,128} × seeds {20261007,11,22}, 10 epochs):
**9/9 trained, exported, and round-trip exact**, test-exact
0.9970 ± 0.0012. Width 64 is no longer fragile: the RC2-era toolchain scored
48% vs 99% on two H defaults of the *same* architecture because defaults and
models drifted apart; under the repaired objective the width gap is
0.9981 vs 0.9962 and every configuration is usable.

`.lancius` round-trips, reloaded in a fresh process from disk: **7.1e-15**
(flagship, 86 MB), **3.6e-15** (512×256), **5.3e-15** (legacy 256-row export).

## The conglomeration — 65→4096→2560→5 verifier as ALU stack

`generation_1/I1/` is the heritage suite (big sweeps, 0.9356 model, model
card); `generation_1/I2/` is the build that consumes the verifier as a
*literal mixture of experts*: `isa/` (LAIS protocol: FEED/VERDICT/NONE,
`isa.py` + the C header `include/lancius/lancius_alu.h`), `bus/` (Lenstra-style
weighted verdict aggregation), `orchestrator/` (fabric_net/fabric_live/
fabric_coherent/fabric_chat), `paths.py` (single resolution of the repo root
for every consumer), `chat.py` (one entry point). The control unit is an
external language model (`qwen2.5-coder:3b` via ollama); the *computation* is
speech-free — counting is routed to the verifier by construction, not by
prompt. Its 24-case suite (prompt-injection, FEED exact-65, memory
contradiction, retraction honesty, counter arithmetic with random arrays)
passes **24/24**.

The ALU pool (`alu_00`..`alu_07`) are resurrection clones of the gen-1 0.9356
verifier; `alu_expert.lancius` is the merged expert graph the fabric bundles.

## Assets

**Flagship verifier — 65→4096→2560→5, v2 format, CRC32 integrity**

| asset | size | provenance |
|---|---|---|
| `verifier5-unified-4096x2560.lancius` | 86 MB | `examples/train_verifier.c`, CE objective, held-out 0.9968/0.9999; row-agnostic (batch 1 in-file) |
| `gen1-lr002.lancius` | 83 MB | RC2-era gen-1 driver, corrected LR=0.002 run, big-suite 0.9356; batch 256 in-file |

**The ALU runtime — what the I2 fabric loads**

| asset | size | provenance |
|---|---|---|
| `alu_expert.lancius` | 194 MB | merged expert graph for the conglomeration fabric |
| `alu_00.lancius` … `alu_07.lancius` | 86 MB each | 8-slot ALU resurrection pool (gen1-lr002 lineage) |

**Efficacy sweep — unified trainer, 9 wire-verified configurations**

`verifier5-sweep-h{64,96,128}-s{20261007,11,22}.lancius` — nine graphs,
71–138 KB, test-exact 0.9970 ± 0.0012, round-trip verified 9/9.

**Legacy reference — RC2 lineage, still loading**

| asset | size | provenance |
|---|---|---|
| `verifier5-legacy-h96.lancius` | 55 KB | RC2 single-hidden verifier (mean-MSE era), for contract parity |
| `verifier5-legacy-h96.weights.txt` | 163 KB | same model as portable ASCII (weights + scaler) |

The ASCII dump of the flagship is 255–266 MB and is **not** an asset: it is a
build product (`--save-dir` in `examples/train_verifier.c` regenerates it), and
uploading a dump that can be regenerated from the source + checkpoints is
history-weight without information.

## Reproduce

```sh
make train_verifier                       # the unified trainer
printf 'FEAT 65\nNCLASS 5\nNTRAIN 33351\nNTEST 8338\n' > data_vec/v5/corpus.hdr
./train_verifier data_vec/v5 4096 2560 5 \
    --export-lancius verifier5.lancius --verify-lancius
./train_verifier data_vec/v5 512 256 1    # then --resume its last.bin
python3 tools/prm/sweep_configs.py --epochs 10   # 3x3 wire sweep
python3 generation_1/I2/chat.py --alus 4 --demo    # the conglomeration (needs ollama)
python3 generation_1/I2/orchestrator/fabric_test_chat.py --part all
```

## Known limitations

- **No real data.** Synthetic injected defects only; accuracies are agreement
  with the symbolic checker, not verifier accuracy on human writing.
- **CE vs MSE contract note:** RC2's `verifier5-legacy-h96.lancius` was trained
  *mean-MSE on logits* with the step-size defect family; it loads and runs fine
  (same graph), but its numbers are the RC2-documented ones, not the CE numbers.
- **The conglomeration needs an ollama server** (`qwen2.5-coder:3b`) — the CU
  is external, by design. Without it, `fabric_test_chat.py` fails loudly; the
  verifier math itself does not need it (FEED/VERDICT routes are deterministic).
- Row-agnostic export is new: models saved with default `--lancius-rows 1`
  refuse a 256-row bind by shape mismatch; `--lancius-rows 256` preserves the
  legacy contract.
- 20 algorithms at TRAIN time in v5; the I1 big suite (100 algorithms) remains
  the gen-1 model's evaluation regime (`generation_1/I1/BIG_SUITE.md`).

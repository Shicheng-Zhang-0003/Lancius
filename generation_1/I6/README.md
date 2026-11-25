# Generation 6 (I6) — verifier research line

I6 is the verifier lineage **I1 → I2 → I3 → I6**. V1.2-RC3.4 publishes the
compact sparse-schema experts described here. The earlier RC3.3 results and
reproduction commands below remain as historical release records; their
large-model feature contract does not describe the RC3.4 models.
Download the [RC3.4 I6 reproducibility bundle](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-reproducibility-bundle.tar.gz)
for the four exported experts, checkpoints, logs, corpus, normalizer, and
complete stress outputs.

## V1.2-RC3.4 current model (2026-11-25)

- Four family experts train from fresh initialization on algorithm-held-out
  train/validation/test splits. I3 warm starts are rejected because their
  full feature schema and pinned normalizer are incompatible.
- Each expert is a 65 → 128 → 64 → 5 tanh network with 17,029 parameters;
  the four-family export contains 68,116 parameters total. The four `.lancius`
  files are 137,896 bytes each and use 128-row static batches.
- I6 uses seven semantic inputs at coordinates 50, 53, 55, 57, 59, 60, and
  61; every other coordinate is zero. They encode structural validity,
  final-step validity, undecidability, any wrong step, overall-row,
  final-step-row, and local-step validity.
- The generator checks every emitted target against the feature rule. The
  current corpus has 233,381 rows with zero rule mismatches; algorithms do
  not cross split boundaries. Its normalizer is fit on training algorithms
  only.
- MAE and MSE are computed independently from continuous expected-level
  errors. Validation selects a positive logit scale, folded into the exported
  output layer. Final test metrics are computed once after selection; export
  parity separately probes the first 128 test rows.
- The initial full-feature neural run missed the target. The sparse-schema
  experts pass held-out overall and local-step MAE and independently computed
  MSE gates. Test results are below.
- `run_big_i6.py` loads all 12 I3 baselines using their own architectures and
  legacy normalizer, and reports overall and per-step MAE/MSE by family,
  family × tier, level, tier, and algorithm.
- These ALUs are verifier models. They consume checker-derived semantic
  features and emit a five-level verdict; they do not parse natural-language
  questions or compute arbitrary math from text.

### Reproduce the current experiment

The corpus, exports, checkpoints, logs, and complete stress manifests are
attached as `I6-RC3.4-reproducibility-bundle.tar.gz` to the GitHub release.
They are generated outputs and remain gitignored in the source tree. To
recreate training, use the same corpus seed and a fresh `models_verified/`
directory; the trainer otherwise resumes from an existing `last.bin`.

```bash
make build/bin/train_i6
python3 generation_1/I6/gen_data_i6.py 8000 20261125 \
  generation_1/I6/corpus_verified
python3 generation_1/I6/split_corpus_by_family.py \
  generation_1/I6/corpus_verified
python3 generation_1/I6/orchestrate_i6.py --skip-eval --epochs 80 --cores 2
```

Then run the randomized condition sweep below. The orchestrator passes
`--strict-targets` to all four trainers; evaluation also gates aggregate and
condition-level MAE/MSE.

### Held-out test results

The test labels are generated from injected derivation defects and the
seven-feature contract. The algorithm-held-out split tests transfer to
unseen algorithm templates within this generator/checker setup; it is not a
human-annotated benchmark of arbitrary proof reasoning.

Each expert used validation-only calibration (`logit_scale=8`); the scale was
folded into the exported output layer before the test pass. The test split was
not used for training, checkpoint choice, or calibration. MAE is the mean of
absolute row errors; MSE is independently the mean of each row error squared.

| Family | Test rows | Overall MAE | Overall MSE | Local-step MAE | Local-step MSE | Runtime parity max diff* |
|---|---:|---:|---:|---:|---:|---:|
| Algebra | 5,840 | 3.88316151e-7 | 3.88316151e-13 | 6.91616766e-7 | 6.91616767e-13 | 1.421e-14 |
| Calculus | 5,422 | 3.97312860e-7 | 3.97312860e-13 | 7.10045662e-7 | 7.10045662e-13 | 1.421e-14 |
| Linear algebra | 4,181 | 4.19385797e-7 | 4.19385797e-13 | 6.84294361e-7 | 6.84294361e-13 | 2.132e-14 |
| Number theory | 22,192 | 4.07578254e-7 | 4.07578254e-13 | 7.47768440e-7 | 7.47768440e-13 | 2.842e-14 |

All four experts pass all four strict test gates: overall MAE, overall MSE,
local-step MAE, and local-step MSE are each below `1e-3`. Exact class
agreement is 1.0 for both row types in these test splits. *The runtime parity
check compares the exported Lancius model with the trainer on the first 128
test rows per expert; it is not a full test-split runtime replay.

### Randomized 150-algorithm stress sweep

The saved run used this command, with all four I6 experts, the 12-model I3
ALU pool, and each generation's own normalizer:

```bash
python3 generation_1/I6/run_big_i6.py 200 \
  generation_1/I6/models_verified/alu_algebra.lancius 20261125 \
  temp/i6-review-final-stress \
  --alu-pool generation_1/I3/alu_pool \
  --musd generation_1/I6/corpus_verified/mu_sd.txt \
  --i3-musd generation_1/I3/gen1_mu_sd.txt \
  --experts generation_1/I6/models_verified --strict-targets
```

That run passed aggregate strict gates. A separate audit of its saved matrix
confirmed all 348 condition slices and full coverage. The current
`--strict-targets` implementation now enforces both the aggregate gates and
every nonempty condition slice, and fails if expected condition coverage is
missing.

The sweep generated 26,430 overall rows and 96,487 local-step rows. The I6
experts matched all generated labels exactly for both row types, all five
target levels, and all three stress tiers. It skipped 3,219 unbuildable or
invalid cells; 351 unrealizable target requests were discarded. All 348
nonempty continuous condition slices passed both metric gates: 174 overall
conditions and 174 local-step conditions, each grouped by family,
family × tier, target level, tier, or algorithm. The largest overall slice
was level `-1.0` at MAE `1e-6` / MSE `1e-12`; the largest local-step slice was
algorithm `divisor_sq_two_primes` at MAE `7.90087464e-7` / MSE
`7.90087464e-13`. Reaggregated stress metrics are overall MAE
`4.03859251e-7` / MSE `4.03859251e-13` and local-step MAE
`6.90580078e-7` / MSE `6.90580078e-13`. The full per-condition matrix and
all 12 I3 baseline comparisons are preserved in
[`docs/audits/I6_STRESS_2026-11-25.txt`](../../docs/audits/I6_STRESS_2026-11-25.txt).

### Individual-expert cross-family stress sweep

Using master seed `20261126`, each of the four exported I6 experts was
evaluated separately on the same generated cases across all 150 algorithms.
Each expert processed
26,294 overall cases and 95,746 local-step records. This is 488,160 expert
predictions across the four models, counting overall and local-step records
separately. Every model had exact class agreement on both row types.

| Expert | Overall MAE | Overall MSE | Local-step MAE | Local-step MSE |
|---|---:|---:|---:|---:|
| Algebra | 4.03247889e-7 | 4.03247889e-13 | 6.91047146e-7 | 6.91047146e-13 |
| Calculus | 4.03247889e-7 | 4.03247889e-13 | 6.91047146e-7 | 6.91047146e-13 |
| Linear algebra | 4.03247889e-7 | 4.03247889e-13 | 6.91047146e-7 | 6.91047146e-13 |
| Number theory | 4.03247889e-7 | 4.03247889e-13 | 6.91047146e-7 | 6.91047146e-13 |

For every expert, all 348 condition slices passed both `1e-3` gates: 174
overall slices and 174 local-step slices, grouped by family, family × tier,
level, tier, and algorithm. Exact agreement was 1.0 in every slice. The
largest slice errors were MAE `1e-6` and MSE `1e-12` (at level `-1.0`). MSE
was computed independently as the mean of per-row squared errors; it was not
derived from MAE².

The generator skipped 3,359 invalid or unbuildable cells and dropped 347
unrealizable target requests. The run used `--skip-alu-baselines`, so it
compares the four I6 experts with one another on the shared generated set and
does not include I3 baseline results. These are checker-generated cases for
I6's seven semantic predicates; passing this sweep measures the learned
verdict rule under these conditions, not general proof understanding on
arbitrary math text.

Reproduce this run with the following command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 generation_1/I6/run_big_i6.py 200 \
  generation_1/I6/models_verified/alu_algebra.lancius 20261126 \
  temp/i6-review-each-expert-20261126 \
  --musd generation_1/I6/corpus_verified/mu_sd.txt \
  --i3-musd generation_1/I3/gen1_mu_sd.txt \
  --experts generation_1/I6/models_verified \
  --each-expert --skip-alu-baselines --strict-targets
```

The full summary and all expert-by-condition metrics are in
[`docs/audits/I6_EACH_EXPERT_STRESS_SEED_20261126.txt`](../../docs/audits/I6_EACH_EXPERT_STRESS_SEED_20261126.txt).

### Crossed algorithm/input stress suite

The expanded suite sampled **1,000 valid parameter sets from each of the 150
algorithm builders** (150,000 draws total). Those draws contained 57,300
distinct algorithm/parameter vectors; the builders have finite or discrete
domains, so repeated vectors are expected and are counted explicitly in the
per-input and parameter-vector reports. Each sampled parameter set was tested
at every realizable one of the five verdict levels, so the same coefficients
and derivation were checked against clean, minor-error, missing-step,
wrong-answer, and fatal candidate cases. It also adds six explicit edge
inputs, including a zero sequence, small rational coefficients (`1/1000`, `-1/3`), and large
coefficients (`99991`, `99989`). The test does not mutate derived parameters
independently; each randomized input comes from its algorithm's valid
parameter builder.

| Expert | Overall correct | Overall MAE | Overall MSE | Local-step correct | Local-step MAE | Local-step MSE |
|---|---:|---:|---:|---:|---:|---:|
| Algebra | 741,544 / 741,544 (100%) | 4.04574779e-7 | 4.04574779e-13 | 2,785,441 / 2,785,441 (100%) | 6.94728770e-7 | 6.94728770e-13 |
| Calculus | 741,544 / 741,544 (100%) | 4.04574779e-7 | 4.04574779e-13 | 2,785,441 / 2,785,441 (100%) | 6.94728770e-7 | 6.94728770e-13 |
| Linear algebra | 741,544 / 741,544 (100%) | 4.04574779e-7 | 4.04574779e-13 | 2,785,441 / 2,785,441 (100%) | 6.94728770e-7 | 6.94728770e-13 |
| Number theory | 741,544 / 741,544 (100%) | 4.04574779e-7 | 4.04574779e-13 | 2,785,441 / 2,785,441 (100%) | 6.94728770e-7 | 6.94728770e-13 |

Across **68,056 condition slices**, every expert had 100% exact-class
agreement. The largest continuous slice errors were MAE
`1.0000000001e-6` and independently computed MSE `1.0000000001e-12`; every
slice passed both `1e-3` gates. The suite also reports class precision,
recall, and F1; correctness by algorithm, verdict, candidate type, coefficient
and output profile, derivation depth, step position and status, and sampled
coefficient extrema; plus per-parameter-set accuracy. The per-input report
shows 100% accuracy for all four experts. It records 150,006 input sets:
150,000 randomized draws and six explicit edge inputs.

Eight algorithm/verdict combinations have no `+0.5` case: those algorithms
produce one-step derivations, so there is no non-final working step to corrupt
while preserving the final answer. The suite records these as unrealizable
instead of treating them as model errors. It dropped 8,482 such requests,
had zero builder or candidate-generation failures, and retained full
per-algorithm coverage for all other realizable labels.

The detailed audit tables are preserved here:

- [Run summary](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/summary.txt)
- [All condition accuracy, MAE, and MSE slices](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/condition_accuracy.csv.gz)
- [Per-input-set accuracy across all four experts](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/input_accuracy.csv.gz)
- [Per-class precision, recall, and F1](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/class_accuracy.csv)
- [Sampled parameter ranges and sign/zero/fraction coverage](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/parameter_coverage.csv)
- [Sample count versus distinct parameter vectors by algorithm and input source](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/parameter_vector_coverage.csv)
- [Algorithm × source × verdict coverage, including empty strata](../../docs/audits/I6_EXPANDED_STRESS_SEED_20261128/strata_coverage.csv)

The compressed scenario-by-scenario and step-by-step prediction manifests are
in the ignored run directory `temp/i6-expanded-final-20261128/`.

Reproduce the full run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python3 generation_1/I6/stress_i6_expanded.py \
  --inputs-per-algorithm 1000 --seed 20261128 \
  --outdir temp/i6-expanded-final-20261128 \
  --strict-targets --require-perfect-accuracy
```

**Scope limit:** I6 receives seven checker-derived semantic predicates, not
raw problem text or coefficients. This suite varies valid inputs and verifies
the learned verdict rule across the checker outputs; it does not establish
general math reasoning or prove the checker correct.

## Historical RC3.3 release record

RC3.3 I6 results did not establish the `1e-3` target. Published MAE used
argmax classes, and published MSE was set to `MAE²`; both claims were
withdrawn. The continuous-output audit of those saved artifacts found MAEs
from 0.0036 to 0.4209. The notes below document that tagged release only.

## Release Identity

| Internal Version | Public Version       | Release Type      |
|------------------|----------------------|-------------------|
| `v12R2.3` (base `v12R2`) | `V1.2RC3.3` | Development Milestone (RC3.x point release) |

Lancius uses the following internal milestone progression:

```text
S → R1 → R2 → R3 → S
```

Where:

- `S` is a stable release
- `R1` is the first development milestone after the previous stable release
- `R2` is the second development milestone, where new subsystems begin becoming independent
- `R3` is the freeze, hardening, and bug-hunting milestone

`v12R2` is the second development milestone of the v12 cycle (the R2 phase
in R-series numbering), built on the `v11S` stable baseline via `v12R1`.

> This is a development milestone.
> Binary compatibility is guaranteed for v2 models written by v11S+.

**RC3.x model line (starts at `V1.2-RC3.1`; current point `V1.2-RC3.4`).**
RC3.4 keeps the v2 model format and stable C ABI, while correcting training
and runtime numerical behavior. The rules and per-release asset provenance live in `MANIFEST.md` and
the release notes under `docs/releases/v12R2/`
(`GITHUB_RELEASE_V1.2-RC3.4.md` is current).

## What's New in V1.2-RC3.3 (historical release)

### Corpus label-noise floor removed
- `verifier5.level_realized()` is now the single source of truth for whether
  a candidate realizes its requested level; the generator refuses unrealized
  labels. The 577 train / 145 test 1-step label mismatches are gone.
- Audit on `corpus_clean`: **0 conflicting labels** across 11,349 train /
  3,655 test feature keys. The overall verdict is therefore an exact function
  of the measured features — the Bayes error is zero.

### Feature 57: `mean_rel` → `n_wrong`
- `mean_rel` shrinks as `1/n_steps`, making a single wrong step in a 12-step
  chain almost invisible after normalization; the `+0.5/+1` boundary landed
  inconsistently between n=11 and n=12.
- `n_wrong` (count of non-exact present steps) preserves the "one step is
  off" signal at every chain length, fixing the n=12 +0.5/+1 boundary.
- The input remains 65-wide. Existing I6 artifacts retain the I3 affine
  transform in `gen1_mu_sd.txt`; coordinate 57 is therefore not z-scored from
  I6's `n_wrong` distribution. Do not swap its scaler without retraining or
  transforming the first layer. On an I3 warm start, the trainer resets the
  inherited W1 row for coordinate 57 because `mean_rel` and `n_wrong` have no
  affine conversion. I1-I3 keep the original `mean_rel` schema.
- In the released training split, `n_wrong` at coordinate 57 has mean 1.518
  and standard deviation 2.546; the retained I3 stats there are `MU=0.184736`
  and `SD=1.358372`. Thus the retained transform is an affine scale, not an
  I6 z-score. `n_wrong=12` maps to 8.698 before training's clamp.
- I6 training clips normalized inputs to `[-8, 8]`. I6 evaluation now applies
  the same clip; I3 baseline evaluation uses its legacy `mean_rel` feature and
  its original unclipped preprocessing.

### Continuous verdict
- Expected level under the 5-class softmax, clipped to `(-1+1e-6, 1-1e-6)`.
- `-2` reserved for non-logic faults, returned before the logic readout.
- Added to `run_big_i6.py`; `continuous_verdict(prob, fault_code=None)`.

### Corrected continuous-output audit

Across the two direct saved-artifact evaluation paths, continuous MAE was:

| Family | Continuous MAE range | `1e-3` target |
|---|---:|---|
| Algebra | 0.252607–0.258127 | **FAIL** |
| Calculus | 0.005315–0.010808 | **FAIL** |
| Linear Algebra | 0.003623–0.012003 | **FAIL** |
| Number Theory | 0.417701–0.420918 | **FAIL** |

The old table used argmax labels for MAE and `MAE²` for MSE. Those values
are withdrawn. The current trainer computes the expected softmax level for
MAE and computes MSE from each squared continuous error; fresh evaluation or
retraining is required before reporting new MSE values.

All warm-started from `generation_1/I3/alu_pool/alu_01.lancius` with a
20-epoch budget, lr 0.0008, `OMP_NUM_THREADS=2`, `OVERALL_REPEAT=4`,
`POS_REPEAT=1`, hard-negative frac 0.50 from epoch 2.

### Trainer sampling fixed
- `OVERALL_REPEAT=4` (was 20× class-1 repeat that starved `+0.5`)
- `POS_REPEAT=1` (was 20×)
- Shuffle before hard-negative replacement so repeats aren't wiped out.

### Orchestration
- `orchestrate_i6.py` defaults to sequential training (laptop thermal-sensitive),
  running one family expert at a time.

## RC3.3 architecture (historical)

| field | value |
|-------|-------|
| architecture | 65 → 4096 → 2560 → 5 (tanh/tanh) |
| parameters | 10,771,461 (~10.8M; the proven I3 verifier shape, still a micromodel) |
| loss | fused softmax cross-entropy, exact VJP (softmax-y)/R |
| optimizer | SGD + momentum + decoupled weight decay |
| LR schedule | warmup + cosine |
| clip | global-norm 5.0 |
| epochs | 20-epoch budget; exported checkpoint is the best validation checkpoint |
| batch | 256 |
| hard negative mining | frac=0.5, starts epoch 2 |
| family routing | hard split by `FAM_ID` (zero-based feature index 58), one expert per family |

The trunk is deliberately the **same shape as I3**, not wider: 4096/2560.
I6's gains come from the cleaned corpus, `n_wrong`, the continuous verdict,
longer training, and hard-negative mining — not from a larger model. It
stays well under the Lancius 100M-element / 800MB tensor cap and stays fast
enough for the ALU bus (static scheduler, arena/flat buffers, real
`lancius run` subprocesses).

## RC3.3 corpus (historical)

Same format as I3: `corpus.hdr` + `{train,val,test}.{X,L,S,A}.bin`,
FEAT=65, NCLASS=5, 0.8/0.2 split.

The released corpus is `generation_1/I6/corpus_clean/`:

| Split | Rows |
|---|---:|
| Train | 186,824 |
| Test | 46,706 |
| Total | 233,530 |

150 algorithms across 5 levels (`-1, -0.5, 0, +0.5, +1`), with 0
unrealizable labels. The family split is by `FAM_ID` (feature index 58) and
is written by `split_corpus_by_family.py`:

| Family | Train | Test |
|---|---:|---:|
| Algebra | 62,647 | 15,860 |
| Calculus | 36,001 | 9,026 |
| Linear Algebra | 22,904 | 5,735 |
| Number Theory | 65,272 | 16,085 |

```
generation_1/I6/gen_data_i6.py [n_per_level] [seed] [out_dir]
  defaults: 8000, 20261026, generation_1/I6/corpus
```

Deltas vs I3's corpus:
- **More rows**: 8000 base cells per level (233,530 total rows before split)
- **Hard-negative mining** (`--hard-neg`): score candidate cells through the
  I3 ALU pool and oversample cells where ALUs disagree or are wrong. The
  released `corpus_clean/meta.json` records `hard_negative: false`; the
  trainer-side hard-negative replacement below is what was used for the
  released experts.
- **Deeper LONG derivations**: long chains (10–12 steps) appear in training,
  not just test
- **Rebalanced regimes**: neg tier drawn at the same rate as base
- **Label-noise floor removed**: 0 unrealizable labels; `n_wrong` feature;
  continuous verdict

Generate the corpus (the released corpus used the base path; add `--hard-neg`
to oversample ALU-disagreement cells):

```
python3 generation_1/I6/gen_data_i6.py 8000 20261026 generation_1/I6/corpus_clean
```

Split it into the four family corpora:

```
python3 generation_1/I6/split_corpus_by_family.py generation_1/I6/corpus_clean
```

## RC3.3 training recipe (historical)

The released configuration is the four-family orchestrated run:

```
make build/bin/train_i6
python3 generation_1/I6/orchestrate_i6.py
```

or one family at a time:

```
build/bin/train_i6 generation_1/I6/corpus_clean/algebra 4096 2560 20 256 \
  --seed 20261026 --lr 0.0008 \
  --init-from generation_1/I3/alu_pool/alu_01.lancius \
  --mu-sd generation_1/I3/gen1_mu_sd.txt \
  --overall-repeat 4 --pos-repeat 1 \
  --hard-negative-frac 0.50 --hard-negative-start 2 \
  --export-lancius generation_1/I6/models_clean/alu_algebra.lancius \
  --verify-lancius --save-dir generation_1/I6/models_clean/algebra
```

- `--init-from`: warm-start the six param tensors from an I3 ALU model
  (alu_00..alu_11). Momentum starts at 0.
- `--mu-sd`: pinned normalizer (shared with I3; do NOT refit)
- `--hard-negative-frac` / `--hard-negative-start`: control hard-negative
  mining (default 0.5 / epoch 2)
- `--overall-repeat` / `--pos-repeat`: sampling weights (4 / 1 in the
  released run)
- `--verify-lancius`: round-trip reload check (max|diff| must be ~1e-15)
- Checkpoints `best.bin`/`last.bin` in `--save-dir`; `--resume` continues

The training budget is 20 epochs. The exported checkpoint for each family is
its best validation checkpoint: Algebra epoch 1, Calculus epoch 1, Linear
Algebra epoch 16, Number Theory epoch 1.

## RC3.3 evaluation recipe (historical)

Multi-expert head-to-head against the I3 ALU pool on freshly generated
cells:

```
python3 generation_1/I6/run_big_i6.py 1000 \
  generation_1/I6/models_clean/alu_algebra.lancius \
  --alu-pool generation_1/I3/alu_pool \
  --musd generation_1/I6/gen1_mu_sd.txt \
  --experts generation_1/I6/models_clean \
  --outdir generation_1/I6/results_big/clean_multi
```

The positional model is only a loader placeholder; routing uses
`--experts`. Use `200` runs/algo for a quick check and `1000` for the final
sweep. The report prints discrete and continuous MAE/MSE, paired agreement,
and per-algorithm deltas. The continuous numbers are the target.

The earlier held-out table and PASS statuses are withdrawn. The saved
artifact audit above reports continuous expected-level MAE. Do not infer MSE
from MAE; it must be computed as the mean of per-row squared errors.

## RC3.3 generated files (historical)

```
generation_1/I6/
  gen_data_i6.py   cleaned corpus builder -> corpus_clean/
  split_corpus_by_family.py  FAM_ID splitter -> four family corpora
  run_big_i6.py    multi-expert eval + continuous verdict
  orchestrate_i6.py  sequential family training/eval orchestrator
  evaluate_i6_multi.py / monitor_i6.py  training/eval automation
  paths.py         repo resolution, pinned MU/SD loader, lancius locator
  README.md        this file
  corpus_clean/    {train,test}.{X,L,S,A}.bin + corpus.hdr + meta.json
                   + algebra/ calculus/ linear_algebra/ number_theory/
  models_clean/    alu_{family}.lancius + {family}/best.bin,last.bin
  results_big/     big_summary.txt + per-epoch prelims
```

## Reproduce RC3.3 (historical)

```bash
# 1. Build the trainer
make build/bin/train_i6

# 2. Generate the cleaned corpus (add --hard-neg for ALU oversampling)
python3 generation_1/I6/gen_data_i6.py 8000 20261026 generation_1/I6/corpus_clean

# 3. Split into the four family corpora (FAM_ID = feature 58)
python3 generation_1/I6/split_corpus_by_family.py generation_1/I6/corpus_clean

# 4. Train all four experts sequentially (warm start from the I3 ALU pool)
python3 generation_1/I6/orchestrate_i6.py

# 5. Evaluate multi-expert against the I3 ALU pool
python3 generation_1/I6/run_big_i6.py 1000 \
  generation_1/I6/models_clean/alu_algebra.lancius \
  --alu-pool generation_1/I3/alu_pool \
  --musd generation_1/I6/gen1_mu_sd.txt \
  --experts generation_1/I6/models_clean
```

## RC3.3 known limits (historical)

- The verifier is trained on the generator's injected-defect corpus, not
  human-written derivations. The scale and machinery are validated; this is
  not a claim of real-world grading accuracy.
- `+0.5` ("mostly true, something wrong") remains a judgement call with a
  chosen defect floor, and `0.0` can mean an omitted step on a 1-step
  algorithm.
- The continuous verdict is the expected level under the 5-class softmax,
  clipped to `(-1+1e-6, 1-1e-6)`; it is an ordinal score, not a calibrated
  probability. `-2` is reserved for non-logic faults.
- Family routing is a hard split by `FAM_ID`; rows outside the four known
  families fall back to a single trunk rather than a learned router.
- The pinned MU/SD must travel with the model; serving with a refit
  normalizer is silent train/serve skew.
- `wilson_check` and similar boundary-band algorithms were the I3 residual;
  hard-negative mining targets them, but they can remain above 0.99.

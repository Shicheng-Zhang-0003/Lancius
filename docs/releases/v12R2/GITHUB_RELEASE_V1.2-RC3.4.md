# Lancius V1.2-RC3.4

| Field | Value |
|---|---|
| Public tag | `V1.2-RC3.4` |
| Internal line | `v12R2.4` on `v12R2` |
| Release date | 2026-11-25 |
| Type | Development point release: I6 model, dataset, audit, and training/runtime corrections |
| Compatibility | No v2 model-format or stable C ABI break; training numerics and outputs are corrected |

RC3.4 publishes a new compact I6 verifier family and the data needed to
inspect and reproduce its reported experiment. It also corrects training
loss/gradient and numerical edge cases discovered during the I6 audit. The
historical RC3.3 I6 headline remains withdrawn; RC3.4 uses fresh models,
different features, and a separate normalizer.

## I6 flagship: what the models do

The release contains four family experts: algebra, calculus, linear algebra,
and number theory. Each export is a tanh `65 → 128 → 64 → 5` network with
**17,029 parameters** and a 128-row static batch. The four exports contain
68,116 parameters in total. Training starts from fresh initialization; I3
warm starts are rejected because their feature schema and normalizer differ.

Each I6 input is a 65-wide vector with only seven nonzero-capable semantic
coordinates: structural validity (50), final-step validity (53),
undecidability (55), any wrong step (57), overall-row flag (59), final-step
flag (60), and local-step validity (61). The remaining coordinates are zero.
The output has five ordered verdict classes (`-1`, `-0.5`, `0`, `+0.5`, `+1`)
and an expected-level continuous score. The train-only scaler is part of the
model contract. Routing is by the dataset's algorithm-family mapping.

**Scope:** these ALUs classify checker-derived features for generated
derivations. They do not read natural-language questions, perform the CU's
language analysis, or solve arbitrary math from text. The stress labels are
generated from injected defects and checked against the feature contract;
100% agreement is not a claim of human-judged proof correctness or end-to-end
math-solving accuracy.

## Results

The 233,381-row corpus holds out whole algorithms, stratified by family, for
train/validation/test. Its feature-rule audit found zero target mismatches.
The train-only normalizer and validation-only calibration are applied before
the held-out test pass. MAE is mean absolute error of the expected class
level. MSE is separately calculated as mean of each row's squared error; it
is not inferred from MAE.

| Family | Held-out rows | Overall MAE | Overall MSE | Local-step MAE | Local-step MSE |
|---|---:|---:|---:|---:|---:|
| Algebra | 5,840 | 3.88316151e-7 | 3.88316151e-13 | 6.91616766e-7 | 6.91616767e-13 |
| Calculus | 5,422 | 3.97312860e-7 | 3.97312860e-13 | 7.10045662e-7 | 7.10045662e-13 |
| Linear algebra | 4,181 | 4.19385797e-7 | 4.19385797e-13 | 6.84294361e-7 | 6.84294361e-13 |
| Number theory | 22,192 | 4.07578254e-7 | 4.07578254e-13 | 7.47768440e-7 | 7.47768440e-13 |

The crossed stress run used 1,000 builder draws per algorithm across all 150
algorithm builders (150,000 draws; **57,300 distinct parameter vectors**),
crossed each with every realizable verdict label, and added six explicit
edge inputs. Each of the four experts scored the same cases:

- Overall exact-class agreement: **741,544 / 741,544 (100%)**.
- Local-step exact-class agreement: **2,785,441 / 2,785,441 (100%)**.
- Condition slices: **68,056 / 68,056 at 100% exact-class agreement**.
- Overall continuous MAE/MSE: `4.04574779e-7` / `4.04574779e-13`.
- Local-step continuous MAE/MSE: `6.94728770e-7` / `6.94728770e-13`.
- Largest condition-slice MAE/MSE: about `1.0000000001e-6` /
  `1.0000000001e-12`; every slice passes both `1e-3` gates.
- Zero builder or candidate failures. 8,482 unrealizable target requests
  were recorded and excluded; these are primarily `+0.5` requests for
  one-step derivations that have no non-final step to corrupt.

The stress harness records per-algorithm, family, verdict, candidate type,
input source, coefficient/sign/zero/fraction/magnitude, output profile,
derivation depth, step position/status/class, fault size, and selected
intersections. It also records per-class precision/recall/F1, per-input
accuracy, sampled parameter coverage, empty/unrealizable strata, and
scenario-/step-level predictions.

## Reproducibility assets

The attached `I6-RC3.4-reproducibility-bundle.tar.gz` contains:

- All four exported `.lancius` models.
- Each family's `best.bin`, `last.bin`, and `train.log`.
- The complete `corpus_verified/` train/validation/test corpus, metadata,
  family splits, and matching `mu_sd.txt` normalizer.
- The complete expanded stress output: summary, all condition rows,
  per-class metrics, per-input accuracy, parameter/strata coverage, and
  compressed per-scenario and per-step prediction manifests.
- A bundle README and a SHA-256 manifest for every included file.

The source tree also contains the reusable harness at
`generation_1/I6/stress_i6_expanded.py` and compressed audit tables under
`docs/audits/I6_EXPANDED_STRESS_SEED_20261128/`.

The published downloads and their SHA-256 checksums are:

| Asset | Size (bytes) | SHA-256 | Download |
|---|---:|---|---|
| Algebra expert | 137,896 | `7653ff9b10ed27a5959fc8b6c4eff38c938aea02c35b7bde996cb07c1aa7fe92` | [`.lancius`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-algebra.lancius) |
| Calculus expert | 137,896 | `7b286e6a63f2e9041bd1d753a1a251170ac4f09b1f273d8d4e26f4179719ecc7` | [`.lancius`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-calculus.lancius) |
| Linear algebra expert | 137,896 | `ed82cf6afa8ad9dd3045946beb01509692cbf809558c19c49a73d090a66b34a1` | [`.lancius`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-linear-algebra.lancius) |
| Number theory expert | 137,896 | `9b320caf506d520285967d8d88cdfb837f226557380d7b24cb15e343462cffc5` | [`.lancius`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-number-theory.lancius) |
| Train-only normalizer | 899 | `53e877b0ce887e95f230173b2da5a2e3bf607474c33a03c866232b0a3189a9d1` | [`mu_sd.txt`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-mu_sd.txt) |
| Full reproducibility bundle | 36,734,008 | `fff50df5981ae445f2a5005f0157e5e6dcbf628e3b886f9e24e1adba2723a276` | [`.tar.gz`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-reproducibility-bundle.tar.gz) |
| Checksum list | 568 | `d5fd4e7b60501f2e45f873300cefe37b06c89c2b89624e33c2dc28ab77de4f92` | [`SHA256SUMS.txt`](https://github.com/Shicheng-Zhang-0003/Lancius/releases/download/V1.2-RC3.4/I6-RC3.4-SHA256SUMS.txt) |

The checksum list covers the four standalone expert files, normalizer, and
reproducibility bundle. The bundle's own manifest covers all 101 files
inside the archive.

Reproduce the expanded model sweep from the repository root with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python3 generation_1/I6/stress_i6_expanded.py \
  --inputs-per-algorithm 1000 --seed 20261128 \
  --outdir temp/i6-expanded-final-20261128 \
  --strict-targets --require-perfect-accuracy
```

The run's summary and every aggregate report are in the bundle; the exact
results and interpretation are also recorded in
[`generation_1/I6/README.md`](../../../generation_1/I6/README.md).

## Verification

- The full `make check` gate passed at the RC3.4 source revision.
- All modified I6 Python modules and the expanded stress harness passed
  `python3 -m py_compile`.
- The expanded I6 stress run completed with `--strict-targets` and
  `--require-perfect-accuracy`; its command and all row-level outputs are in
  the attached bundle.

## Other RC3.4 corrections

- Cross-entropy uses shifted logits for stable forward evaluation and the
  exact VJP for finite nonnegative targets, including rows whose target sum
  is not one.
- MSE computes a true mean of squared errors with scaled accumulation;
  global gradient clipping uses a scaled sum-of-squares method and clips
  large finite gradients without first overflowing their norm.
- Invalid non-finite intermediate values fail with numerical errors, and a
  warmup schedule that consumes the full duration falls back to the defined
  cosine schedule.
- Fixed-size training batches weight replayed tail rows so their gradient is
  the exact mean over real rows. I6 epoch loss reporting uses the same
  weighting.
- I4 average precision groups tied scores, Cohen's kappa uses true/predicted
  marginals, MNIST/CIFAR evaluation includes final partial batches, tanh
  examples use Glorot scaling, and CIFAR uses the shared global-norm clipper.
- The obsolete external data bridge integration, code, build targets,
  dependency requirements, and audit files are removed.

## Upgrade and limits

- The v2 `.lancius` format and stable C ABI are unchanged. The training
  formulas are corrected, so retraining can produce different weights.
- Use the RC3.4 `mu_sd.txt` and seven-predicate feature contract with these
  models. RC3.3/I3 normalizers and legacy 65-feature rows are not compatible
  with the RC3.4 semantic schema, even though the binary model loader is
  compatible.
- The expected-level score is an ordinal verdict, not a calibrated
  probability. Family routing is based on the algorithm map, not a learned
  natural-language router.
- The test cases come from the verifier generator. This release does not
  establish arbitrary proof understanding, CU parsing accuracy, or
  end-to-end math-solving accuracy.

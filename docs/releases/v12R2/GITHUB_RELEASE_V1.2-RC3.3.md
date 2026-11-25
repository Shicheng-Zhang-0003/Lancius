# Lancius V1.2-RC3.3

> **What this release is:** the third point release of the RC3 model line —
> **model, dataset and fabric iteration** cut from an unchanged `v12R2`
> codebase. Not `V1.2RC4` and not `V1.2`: no format break, no stable-API
> change, no numerical behaviour change, so no compatibility claim moved
> (see `MANIFEST.md`, which records this release line explicitly).

| Field | Value |
|---|---|
| Public tag | `V1.2-RC3.3` |
| Base | `V1.2-RC3.2` (internal `v12R2`, unchanged) |
| Date | 2026-11-23 |
| Type | Model/dataset point release (I6 corpus + four family experts) |
| Compatibility | v2 `.lancius` files written by v11S+ load unchanged; RC2/RC3/RC3.1/RC3.2 assets remain valid |

## Headline: the I6 corpus and four family experts

The I6 lineage (`I1 → I2 → I3 → I6`) keeps the proven
`65 → 4096 → 2560 → 5` verifier shape (10,771,461 parameters), warm-starts
from the I3 ALU pool, and has four family experts — algebra, calculus,
linear algebra, number theory — on the cleaned 150-algorithm corpus.

**Metric correction:** the originally published near-zero MAEs were computed
from argmax classes rather than the continuous expected level. The MSE values
were estimated as `MAE²`, not computed from per-row squared errors. Those
results are withdrawn. A direct saved-artifact audit found continuous MAE
ranges of 0.252607–0.258127 (algebra), 0.005315–0.010808 (calculus),
0.003623–0.012003 (linear algebra), and 0.417701–0.420918 (number theory)
across two evaluation paths; every value misses `1e-3`. Corrected continuous
MSE for the saved artifacts has not been established. The trainer now computes
continuous expected-level MAE and true mean squared error.

The experts were trained with a 20-epoch budget, lr 0.0008,
`OMP_NUM_THREADS=2`, `OVERALL_REPEAT=4`, `POS_REPEAT=1`, and hard-negative
frac 0.50 from epoch 2. The exported checkpoint for each family is its best
validation checkpoint.

## Corpus label-noise floor removed

`make_candidate` could not realize `+0.5` for a 1-step derivation (it bumped
the only step, which is the final one), and its `0.0` omission range was
empty for a lone step. The old corpus therefore had 577 train / 145 test
overall rows whose labels contradicted their measurements, all at
`nsteps == 1`.

`verifier5.level_realized()` is now the single source of truth used by both
the generator and the evaluator, and `emit()` refuses to write an unrealized
level. Audit on `generation_1/I6/corpus_clean/`:

- **0 unrealizable overall labels**
- **0 conflicting labels** across 11,349 train / 3,655 test overall feature
  keys (and 0 across 59,780 step keys)
- 186,824 train / 46,706 test rows over 150 algorithms and 5 levels

The overall verdict is therefore an exact function of the measured features;
the Bayes floor is zero, and the `1e-3` target is an optimization problem,
not a noise floor.

## Feature 57: `mean_rel` → `n_wrong`

`mean_rel` shrinks as `1/n_steps`, so one wrong step in a 12-step chain
looked almost like a clean chain after normalization and the `+0.5/+1`
boundary landed inconsistently between `n=11` and `n=12`. `n_wrong` (count of
non-exact present steps) preserves the "one step is off" signal at every
chain length, fixing the `n=12` `+0.5/+1` boundary. The input remains
65-wide; the pinned normalizer from `gen1_mu_sd.txt` is unchanged.

## Continuous verdict

`run_big_i6.py` reports the expected level under the 5-class softmax, clipped
to `(-1+1e-6, 1-1e-6)`. `-2` is reserved for non-logic faults and is returned
by `continuous_verdict(..., fault_code=...)` before the logic readout. The
5-output model shape is preserved; the failed 1-output rewrite is abandoned.

## Trainer sampling fixed

- `OVERALL_REPEAT=4` (was a 20× class-1 repeat that starved `+0.5`)
- `POS_REPEAT=1` (was 20×)
- Shuffle before hard-negative replacement so the repeated verdict rows are
  not wiped out

## Sequential orchestration

`orchestrate_i6.py` defaults to one trainer at a time (laptop thermal-safe),
running one family expert at a time with `OMP_NUM_THREADS=2`, then evaluates
all four through `run_big_i6.py --experts`.

## Architecture and routing

- **Input:** 65 features (feature 57 = `n_wrong`; feature 58 = `FAM_ID`)
- **Trunk:** 65 → 4096 → 2560 → 5 (tanh/tanh), 10,771,461 parameters
- **Loss:** fused softmax cross-entropy, exact VJP `(softmax-y)/R`
- **Routing:** hard split by `FAM_ID` into four experts; rows outside the
  four known families fall back to a single trunk
- **Warm start:** `generation_1/I3/alu_pool/alu_01.lancius`
- **Normalizer:** pinned `generation_1/I3/gen1_mu_sd.txt` (do not refit)

## Verified rather than asserted

- Corpus audit: 0 unrealizable labels, 0 conflicting labels in both splits.
- All four training logs report the epoch metrics in the table above; the
  exports carry the `--verify-lancius` round-trip check (the linear-algebra
  export measured `max|diff| = 7.105e-15`).
- The I6 corpus and model directories are generated and gitignored; they are
  reproducible with `generation_1/I6/gen_data_i6.py`,
  `split_corpus_by_family.py`, and `orchestrate_i6.py`.
- `src/` is untouched by the RC3.x line; the six-gate CI set from RC3 is
  unchanged in content.

## Assets

No binary assets are attached to this release page. The four family experts
are reproducible from the released corpus recipe above; the model files are
~86 MB each as `.lancius` exports and are intentionally not committed.

## What this release does not establish

- The verifier numbers come from the generator's injected-defect labels, not
  human-written derivations. The scale and machinery are validated, not
  real-world grading accuracy.
- `+0.5` ("mostly true, something wrong") remains a judgement call with a
  chosen defect floor; `0.0` can mean an omitted step on a 1-step algorithm.
- The continuous verdict is an ordinal score, not a calibrated probability.
- Family routing is a hard split, not a learned router.
- Generation remains scrapped by design: the model scores and reports; it
  never speaks.

## Upgrade notes

- Pull the tag; no rebuild is required for model use. `make check` remains
  the minimum gate for the framework.
- RC3.1/RC3.2 assets need no re-download: shapes are unchanged.
- MU/SD pinning, the 65-feature layout, and the 5-output shape are the
  serving contract; a refit normalizer is silent train/serve skew.
- The v2 format and the CRC-mandatory default carry over unchanged.

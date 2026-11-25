# Lancius V1.2-RC3.2

> **What this release is:** the second point release of the RC3 model line —
> **model, dataset and fabric iteration** cut from an unchanged `v12R2`
> codebase. Not `V1.2RC4` and not `V1.2`: no format break, no stable-API
> change, no numerical behaviour change, so no compatibility claim moved
> (see `MANIFEST.md`, which records this release line explicitly).

| Field | Value |
|---|---|
| Public tag | `V1.2-RC3.2` |
| Base | `V1.2-RC3.1` (internal `v12R2`, unchanged) |
| Date | 2026-11-06 |
| Type | Model-lineage point release (I6 verifier + I5 rules engine) |
| Compatibility | v2 `.lancius` files written by v11S+ load unchanged; RC2/RC3/RC3.1 assets remain valid |

## What is new versus RC3.1

### I6 flagship verifier enters the tree

I6 returns to the verifier flagship after the I4 fire-risk and I5
decision-surface deviations:

- `examples/train_i6.c` — `65 → 4096 → 2560 → 5` fused cross-entropy
  trainer, warm start from the I3 ALU pool via `--init-from`, pinned
  `--mu-sd` normalizer, hard-negative mining
  (`--hard-negative-frac`/`--hard-negative-start`), persistent training
  graph, checkpoint/resume, `--verify-lancius` round-trip.
- `generation_1/I6/gen_data_i6.py` — expanded 150-algorithm corpus builder
  (8000 base cells per level, hard-negative mining against the I3 ALU pool,
  deeper LONG derivations, rebalanced regimes).
- `generation_1/I6/run_big_i6.py` — head-to-head evaluation against all 12
  I3 ALUs on identical cells.
- `generation_1/I6/paths.py` — repo resolution, pinned normalizer, lancius
  locator.
- `generation_1/I6/README.md` — model card and reproduce instructions.
- `corpus/`, `models/`, `results_big/` are gitignored generated/trained
  artifacts.

### I5 rules engine enters the tree

- `examples/train_i5.c` — generic decision-surface trainer, fused
  cross-entropy, persistent graph, row-agnostic export.
- `generation_1/I5/run_big_i5.py` — binary evaluation: accuracy, MAE, MSE,
  ROC-AUC, threshold sweep.
- `generation_1/I5/paths.py`, `generation_1/I5/README.md` — "Lancius as
  rules engine" framing.
- Acquisition and corpus tools: `tools/acquire_and_train_i5.py`,
  `tools/fire_data_acquire.py`, `tools/make_synthetic_corpus.py`, and the
  checkpoint converter `bin_to_lancius.c`.

### Infrastructure and documentation

- `.gitignore` gains corpus/model/result ignore rules for both generations
  plus `MATH_COMPENDIUM.md` and `I6_TRAINING_STATUS.md` as dev-only
  artifacts.
- The makefile gains `train_i5`/`train_i6` link rules and registers both in
  `BINS` (43 binaries).
- `README.md` and `generation_1/README.md` span I1–I6 with I5 as the rules
  engine and I6 as the flagship verifier.
- `243-BDC/README.md` documents the cloned I4/I5 tracks alongside Idea1.

## Verified rather than asserted

- `make check` RC=0 with the batch in tree; `-Wall -Wextra -Werror` build
  clean.
- `src/` is untouched by the RC3.x line; the six-gate CI set from RC3
  (`check`, `check-long`, `check-sanitizers`, `check-oracle`,
  `check-physics`, `check-ubstrict`; `check-mutation` sampled) is unchanged
  in content.
- The I6 trainer's export path carries the `--verify-lancius` round-trip
  check (max|diff| ~1e-15).

## What this release does not establish

- I6 at RC3.2 was a trainer/corpus/eval landing, not a completed expert
  set; the label-noise floor and the four-family results land in RC3.3.
- I5 is a generic decision-surface trainer, not an attribution or causal
  engine; its scores must be calibrated before being read as probabilities.
- Generation remains scrapped by design: the model scores and reports; it
  never speaks.

## Upgrade notes

- Pull the tag and `make`; `lancius doctor` should show `OK ×5`.
- RC3.1 assets need no re-download: shapes are unchanged.
- The v2 format and the CRC-mandatory default carry over unchanged.

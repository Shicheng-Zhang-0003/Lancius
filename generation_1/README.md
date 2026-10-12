# generation_1 — Gen1 10M prototype, everything related

Mute verifier, second scale-up: 65 feats -> 4096 (tanh) -> 2560 (tanh) -> 5,
10,771,461 weights, trained 2 epochs with the Lancius framework only
(`examples/train_gen1.c`: graph + autodiff + `schedule_execute`, SGDM
model-side; no torch, no onnx anywhere in training).

## Contents

| file | what |
|---|---|
| `gen1_model.txt` | ASCII weights, `prm_model.txt`-style (`H1/H2/FEAT/NCLASS`, `MU/SD`, `W1/B1/W2/B2/W3/B3`), 10,771,596 lines, 255M. Bit-verified against the frozen `.lancius` (`generation_1/results/ascii_verify.log`). |
| `algos_gen1_ext.py` | 12 new theorems beside `tools/prm/algos.py` (20): sum_of_squares, difference_of_cubes, cube_of_binomial, telescoping_series, log_product, modular_inverse, matrix_trace_product, vieta_cubic, factorial_recurrence, fibonacci_iter, harmonic_partial, horner_eval. Same contract: exact Fractions, <=12 steps, 4 families. |
| `run_tests_gen1.py` | Expanded harness: DETAIL (first set of all 32 algos) + SWEEP (32 algos x 3 tiers x 25 params, overall + per-step rows, confusion, mismatches). |
| `results/full.log` | Full sweep output: 2400 overall rows, 13153 step rows. |
| `results/sweep_summary.txt` | One-line tally. |
| `results/ascii_verify.log` | ASCII-vs-`.lancius` proof. |
| `results/dump.log` | Model-dump log. |
| `MODEL_CARD_GEN1.md` | Numbers + limits. |

The frozen binary itself lives at `temp/gen1-runs/gen1-10M.lancius`
(83M, 15 nodes, `crc=d2dee26c`, gitignored artifact); `gen1_model.txt`
here is its portable copy. The trainer is `examples/train_gen1.c`; the
65-dim gauntlet is `tools/prm/bench_gen1.py`.

## Headline numbers (5-epoch prototype — still not converged)

- Held-out test split (12287 rows; 2366 overall + 9921 step): overall exact
  **`0.8960`** (2120/2366), per-step exact **`0.9694`** (9617/9921), chance `0.20`.
- macro-F1 `0.8941` overall / `0.9275` per-step, MAE `0.0949`, kappa `0.8694`
  / `0.9542` per-step. The macro-F1 figures are **recomputed independently**:
  `train_gen1.c` printed 0.6172 / 0.6329 for these very confusion matrices
  because `fn` was accumulated without the `if(r!=c)` guard that `fp` had, so
  recall read `tp/(tp+rowsum)` = 0.5 on a near-perfect model. Accuracy, MAE and
  kappa were never affected by it.
- Per-class exact (overall): `-1 0.958`, `-0.5 0.868`, `0 0.998`, `+0.5 0.733`,
  `+1 0.917` — the `+0.5` class is where verification actually fails.
- Dominant error **`+0.5 -> +1` (84/476)**: "mostly true, something wrong"
  certified as "plain true". Secondary: `+0.5 -> -0.5` 36, `+1 -> 0` 28.
- Worst algos are the long-iteration chains: `fibonacci_iter 0.71`,
  `euclid_gcd 0.78`, `mod_pow_fast 0.78`, `telescoping_series 0.80`.
- Validation curve by epoch: `0.7201 / 0.7993 / 0.8748 / 0.8878 / 0.8957` —
  still climbing, so more epochs before a bigger H.

### The earlier 0.5871 was an undertraining artifact

The `0.5871` / `0.7366` figures previously recorded here came from a 2-epoch,
batch-512 run — about 192 optimiser steps for 10.77M parameters. That is close
to the class prior plus a bias term, and it understates the architecture by
~31 accuracy points. Both the trainer and this card now reflect 5 epochs.

### Compounding defect: the step scaled as 1/rows^2

This trainer also carried the `lr_eff = LR * 2/pe` bug fixed across all three
trainers (`docs/DESPOT_TRUTH_V2.md` §18.2.1): the graph's MSE VJP already
contains `2/pe`, so folding it into the step again divided by `pe` twice —
**1600x** too small at rows=256, NCLASS=5. The 0.8960 above was measured with
that defect present, so it is a lower bound. `--lr` was added to the trainer
for re-tuning under the corrected convention.

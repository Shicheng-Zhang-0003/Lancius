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

The frozen binary lives at `temp/gen1-runs/gen1-lr002.lancius` (83M, 15 nodes,
gitignored artefact), with `gen1_model.txt` here as its portable copy. The
trainer is `examples/train_gen1.c`; the 65-dim gauntlet is
`tools/prm/bench_gen1.py`.

**The older `gen1-10M.lancius` is still in that directory and is the model the
docs used to describe.** It is the 2-epoch run and scores overall 0.4474 on the
big suite against the corrected model's 0.9356 — same architecture, same sweep,
same master seed, different weights. Both print `65 -> 4096 -> 2560 -> 5`, so the
architecture does not separate them and nothing in either banner used to say
which was which. `run_big.py` now refuses to use a default that is not the
newest `.lancius` in that directory, prints each model's build date, and writes
the model identity into `results_big/big_summary.txt`.

## Headline numbers (5-epoch prototype — still not converged)

- Held-out test split (12287 rows; 2366 overall + 9921 step), corrected model:
  overall exact **`0.9903`** (2344/2366), per-step **`0.9980`** (9903/9921),
  chance `0.20`. The `0.8960` / `0.9694` in the same columns is the same 5-epoch
  run with the step-size defect still present.
- `run_big.py`, fully randomized (1000 runs/algo, master 20261007): overall
  exact **`0.9356`** (86,087/92,015), per-step **`0.8979`** (258,084/287,446).
- macro-F1 **`0.9903`** overall / `0.9948` per-step, MAE `0.0082` / `0.0017`,
  kappa `0.9878` / `0.9970` — recomputed independently, because
  `train_gen1.c` printed 0.6623 / 0.6643 for these very confusion matrices
  (`fn` was accumulated without the `if(r!=c)` guard that `fp` had, so recall
  read `tp/(tp+rowsum)` = 0.5 on a near-perfect model). Accuracy, MAE and
  kappa were never affected by that defect.
- Per-class exact (overall, n=2366): `-1 1.000`, `-0.5 0.969`, `0 1.000`,
  `+0.5 0.974`, `+1 0.987`. Under the full randomized suite (92,015 rows) the
  weakest level is `-0.5` at 0.8660 and the weakest class moves again, so
  single-model per-class figures and suite-level figures are not
  interchangeable — see `MODEL_CARD_GEN1.md` for both.
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
**1600x** too small at rows=256, NCLASS=5. The `0.9903` / `0.9980` headline
above is the same 5-epoch run with the defect fixed; the `0.8960` / `0.9694` it
replaced is the same run with the defect present. `--lr` was added for re-tuning
under the corrected convention, where the shipped `LR=0.05` becomes 1600x
hotter and diverges (val exact 0.1014 at LR=0.01).

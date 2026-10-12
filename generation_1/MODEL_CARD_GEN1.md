# Gen1 10M prototype — model card

65 -> 4096 -> 2560 -> 5 (tanh/tanh), 10,771,461 weights. 5 epochs,
batch 256, SGDM (LR 0.002, MOM 0.9, WD 1e-4), seed 20261007,
validation-selected. Lancius-only training. Prototype checkpoint, NOT a
release, but no longer "NOT converged".

## Sweep (held-out test split, 12287 rows; 2366 overall + 9921 step rows)

Chance 0.20. **Epoch 4 selected on validation (val exact 0.9935).**

| slice | shipped LR=0.05 | corrected LR=0.002 |
|---|---|---|
| overall exact | 0.8960 | **0.9903** (2344/2366) |
| per-step exact | 0.9694 | **0.9980** (9903/9921) |

macro-F1 **0.9903** overall / **0.9948** per-step (recomputed independently —
see the metric note below; the trainer binary that produced this run predates
that fix and printed 0.6623/0.6643). MAE 0.0082 / 0.0017, κ 0.9878 / 0.9970.
Within-1 accuracy 0.9949 / 0.9988.

Per-algorithm held-out accuracy is at or near 1.0 for every algorithm except
`factorial_recurrence 0.9224` and `fibonacci_iter 0.9790` — the iteration and
long-chain cases are the only remaining structure. The two defects below were
costing 9.4 accuracy points overall and 28.6 on macro-F1.

### Under full randomization (`run_big.py`, 1000 runs/algo, master 20261007)

The held-out split above is a fixed test set; the big suite re-scores the model
across 100,000 freshly generated derivations with every cell's level, tier and
scale multiplier drawn independently, so nothing is aligned by run index.
92,015 valid overall rows and 287,446 step rows survive 4,513 validity skips and
3,472 dropped unrealizable levels.

| slice | value |
|---|---|
| overall exact | **0.9356** (86,087/92,015) |
| per-step exact | **0.8979** (258,084/287,446) |
| `-1.0` / `-0.5` / `0.0` / `+0.5` / `+1.0` | 0.9138 / 0.8660 / 0.9969 / 0.9231 / 0.9850 |
| base / big / neg tier | 0.9502 / 0.9218 / 0.9342 |

The same suite on the 2-epoch v1 model gave 0.4575 overall. `-0.5` ("probably
false, something right in there") is the weakest level at 0.8660 and its
dominant error is `-0.5` true read as `+0.5` predicted, 1,353 cases — the
opposite direction from the old card's `-0.5 -> +1`. The `big` tier now trails
`base` by 2.8 points, which the old sweep could not see because its scale
multiplier was pinned at `x7` instead of drawn over 2..20.

Per-algorithm overall accuracy: 30 of 32 algorithms at or above 0.78, with
`chain_rule`, `dot_product_3d`, `power_rule`, `perfect_square_trinomial`,
`sum_of_cubes`, `sum_of_squares`, `difference_of_cubes` and `horner_eval` all
at 1.0000. Worst are the long-iteration chains — `fibonacci_iter 0.706`,
`euclid_gcd 0.776`, `mod_pow_fast 0.779`, `telescoping_series 0.796`.

### The previous 0.5871 was an undertrained artifact

The earlier card recorded **0.5871 / 0.7366** from a 2-epoch, batch-512 run.
That is ~192 optimiser steps for 10.77M parameters, so the model had barely
left its initialisation: 0.5871 is close to the class prior plus a bias term,
not a measurement of the architecture. The same architecture trained for 5
epochs (961 steps) reaches **0.8960**. The "2 epochs: undertrained" caveat
below was correct and is now quantified.

### Compounding defect: the step scaled as 1/rows^2

This trainer also carried the `lr_eff = LR * 2/pe` defect documented in
`docs/DESPOT_TRUTH_V2.md` §18.2.1: the graph's MSE VJP already contains
`2/pe`, so folding it into the step again divided by `pe` twice. At
rows=256, NCLASS=5 the factor was **1600x** too small. Corrected to
`lr_eff = LR * NCLASS/2` in `examples/train_gen1.c` (`--lr` override added for
re-tuning).

Under the corrected convention the shipped `LR=0.05` becomes 1600x hotter and
**overshoots into chaos**, so the LR itself had to be re-tuned. One-epoch
sweep at batch 1024 (41 steps), validation exact / MAE:

| LR | `lr_eff` | val-exact | val-MAE | |
|---|---|---|---|---|
| 0.0005 | 0.00125 | 0.8676 | 0.1320 | healthy |
| **0.002** | 0.005 | **0.9489** | **0.0457** | best |
| 0.01 | 0.025 | 0.1014 | 0.5766 | collapsed |
| 0.05 | 0.125 | — | — | (not measured; 0.01 already diverged) |

The corrected run at **LR=0.002** reaches **val-exact 0.9489 in a single
epoch** — 41 steps — against the shipped trainer's **0.7201 in its first epoch**
and **0.8957 after 5 epochs / 961 steps**. Test metrics at that epoch:
overall exact **0.9497**, within1 0.9793, macro-F1 0.6433, MAE 0.0408,
kappa 0.9370.

### The `+0.5` class is where verification actually fails

macro-F1 0.6172 against exact 0.8960 is not noise: the errors concentrate in
the two "mostly wrong / mostly right" classes. Per-class exact accuracy
(overall, n=2366):

| level | exact | n |
|---|---|---|
| -1.0 | 0.958 | 449 |
| -0.5 | 0.868 | 463 |
| 0.0 | 0.998 | 523 |
| +0.5 | **0.733** | 476 |
| +1.0 | 0.917 | 455 |

The single largest error bucket is **`+0.5 -> +1` at 84** — "mostly true,
something wrong" certified as "plain true", the dangerous direction. Second is
`+0.5 -> -0.5` at 36 and `+1.0 -> 0.0` at 28. Per-step (n=9921) the same
pattern holds at lower magnitude: `+0.5` is weakest at 0.835, and the
cross-sign leaks (`-1 -> +0.5` 18, `-0.5 -> +1` 18, `+1 -> -0.5` 33) are the
ones that matter for a verifier, even though they are a small fraction of
total error.

### macro-F1 was miscomputed in every number above

`train_gen1.c` accumulated `fn` without the `if(r!=c)` guard that `fp` had, so
`fn` swallowed the diagonal and recall became `tp/(tp + rowsum)`. The tables
above report the **recomputed** values: overall 0.6172 -> **0.8941** and
per-step 0.6329 -> **0.9275** for the same confusion matrices. The defect was
present in `train_verifier5.c`, `train_gen1.c` and `tools/prm/ref_prm.c` and
absent from `train_prm_verifier.c`; it read ~0.667 for a near-perfect model
where the matrix implied ~0.997. Accuracy, MAE and kappa were unaffected.

## Limits (carry over from the 5-level stack, still true at 10M)

- Synthetic checker-agreement only; no human derivation has passed through it.
- Middle-class blindness (was `-0.5`, now primarily `+0.5` at 0.733 overall /
  0.835 per-step) and long-derivation dilution, both measured above.
- No algebraic equivalence: right answer by another route scores badly.
- Not converged: the validation curve is still climbing at epoch 5
  (0.7201 / 0.7993 / 0.8748 / 0.8878 / 0.8957), so more epochs is the next
  step, not a bigger H.
- `fwd_logits` is cache-hostile (`W2` 84MB streamed with stride `H2` per row),
  so the held-out pass costs ~1TB of memory traffic per call and the trainer
  spends ~15 minutes evaluating after the last epoch.
- `gen1_model.txt` (255M) is portable; the `.lancius` batch dim is fixed at 256.

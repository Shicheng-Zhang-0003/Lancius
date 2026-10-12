# Big suite: 100 algorithms x 1000 runs

`run_big.py`: 100 algos (20 `tools/prm/algos` + 12 `algos_gen1_ext` + 68
`algos_gen1_big`) x 1000 runs = 100,000 derivations (94,354 valid after
5,646 counted validity skips), overall + per-step rows (280,967 step rows),
master seed 20261007 with per-cell sub-seeds (reproducible), tiers and the
big-tier multiplier drawn per run, and **levels drawn per cell rather than
cycling by run index**. Scorer there: Gen1-10M v1 (2 epochs). Results:
`results_big/full.log`, `big_summary.txt`.

## Every value is now a draw, and what that changed

Three things used to be scheduled rather than sampled: the level was
`LEVELS[k % 5]`, which gave all 100 algorithms the *same* level at the same run
index; the walkthrough level was pinned to `+1.0` so it only ever exercised the
defect-free case; and the `big` tier scaled every parameter by exactly `7`.

All three are draws now, all still reproducible from the master seed, and the
actual drawn level and multiplier distributions are printed and persisted so a
reader can see what was measured instead of assuming a balance that is no
longer constructed.

Verifying that this did not smuggle in a bias: across 10 master seeds on the same
model, randomized and cycling agree to within sampling error (deltas +0.0084,
-0.0115, +0.0094, both signs present), and the set of algorithms that cannot be
built at all is *identical* under both. Randomizing the scale multiplier does not
make any algorithm newly unbuildable and actually produces fewer skips, because
a drawn multiplier below 7 stays in range more often than a fixed 7.

## Finding: single-step derivations make some levels unrepresentable

The defect injector cannot express every level for every derivation. `+0.5`
needs a working step that is not the final one, and `0.0` needs a suffix to
omit — a single canonical step offers neither. Asking for `+0.5` there corrupts
the only step, which is the final answer, making it indistinguishable from
`-0.5`; asking for `0.0` omits nothing, making it indistinguishable from `+1.0`.

`cross_2d`, `hadamard_trace`, `diag_trace`, `euler_phi`, `legendre_euler`,
`divisor_count`, `fermat_little` and `wilson_check` are single-step for every
parameter draw, and `bezout_witness`, `factorial_recurrence`, `sum_odd` are
single-step for some. Measured over 1500 cells across all 100 algorithms,
**3.5% of the sweep carried a label its own measurements could not support**,
which capped the accuracy *any* verifier could reach at about 0.9647 — a
property of the harness rather than of the model.

`level_realized()` now re-derives the level from the produced values by exact
Fraction comparison and drops the cells that do not match, counting them as
`dropped unrealizable levels` rather than letting them into the tally. This is
the same failure mode as the 2-step hole below, one step worse in degree: a
2-step derivation is OOD but honestly labelled, whereas these were OOD *and*
mislabelled.

## Two models share a directory, and the older one scores 0.4474

`temp/gen1-runs/` holds both `gen1-lr002.lancius` (the corrected model, built
2026-10-11) and `gen1-10M.lancius` (the 2-epoch run, built 2026-10-09). They are
the same architecture — both print `65 -> 4096 -> 2560 -> 5` — so the banner that
printed only the architecture could not tell them apart, and `run_big.py`'s
default named the older one. Running the documented command with no model
argument therefore scored the 2-epoch weights and reported **overall 0.4474 /
per-step 0.6537**, versus **0.9356 / 0.8979** for the corrected model on the
identical sweep, master seed and 92,015 rows. The 0.4474 in the summary below
`results_big/big_summary.txt` was that result and is the reason the accuracies
looked "a little bad" while the same lines in this file read 0.9356.

Nothing about the harness was wrong. `run_big.py` now refuses a default that is
not the newest `.lancius` in its directory, prints each model's build date beside
its architecture, and writes the model identity into the summary file so a number
can never be detached from the weights that produced it. This is the same defect
as `docs/DESPOT_TRUTH_V2.md` §18.6.2, in a different directory: a path that
outlived the model written into it.

## Headlines (chance 0.20)

Measured on the **corrected** model — `temp/gen1-runs/gen1-lr002.lancius`, built
2026-10-11, the `LR=0.002` run of §18.6.1, 5 epochs, epoch 4 selected on
validation at 0.9935 — under the fully randomized sweep: 1000 runs per algorithm, master seed 20261007,
92,015 valid overall rows and 287,446 step rows after 4,513 validity skips and
3,472 dropped unrealizable levels.

- Overall exact **0.9356** (86,087/92,015); per-step **0.8979**
  (258,084/287,446). The old figures here were **0.4575 / 0.6750** on the
  2-epoch Gen1-10M v1.
- Levels: `-1 0.9138 / -0.5 0.8660 / 0 0.9969 / +0.5 0.9231 / +1 0.9850`.
  `-0.5` — "probably false, something right in there" — is now the weakest
  level by 4.8 points, and it was also the weakest before, so the ordering is
  the model's rather than the schedule's.
- Tiers: base **0.9502** / big **0.9218** / neg **0.9342**. The `big` tier is
  now 2.8 points below `base`, where the old sweep showed it flat (0.439 vs
  0.467) — but that flatness was measured with the scale multiplier pinned at
  `x7`, a single magnitude. Drawn over 2..20, each magnitude appearing
  1,488–2,211 times, scale generalizes *less well than a fixed magnification
  suggested*, which is the point of drawing it.
- Worst: `sum_odd 0.6657`, `partial_fractions 0.6680`, `inverse_2x2 0.6970`,
  `inflection_cubic 0.7000`, `nilpotent_sq 0.7010`. Best: 32 algorithms at
  exactly `1.0000`, including all of `monotonic_quad`, `shadow_rates`,
  `collatz_5`, `triangular_mod` and `square_part`.

The dominant error is `-0.5` true read as `+0.5` predicted, **1,353** cases —
"probably false" read as "mostly true". This is the opposite of the old card's
`-0.5 -> +1` and the more benign direction, but the second is still present:
`+0.5` true read as `-1.0`, **898** cases, and `+0.5` true read as `-0.5`,
263. Both certify a working derivation as false.

## Retracted: "2-step derivations are OOD for the whole stack"

This section used to record that the three worst algorithms (`shadow_rates`,
`lhopital_poly`, `monotonic_quad`) all have exactly 2 canonical steps, that
clean-`+1` recall on them was **0.000 (0/60)** under both the 2-epoch Gen1-10M
and the converged 60-epoch H128, and that no model trained on the base corpus
had ever seen a 2-step derivation — a distribution hole whose fix belonged in
the corpus rather than in H.

The premise was right and the conclusion was wrong. `shadow_rates`,
`lhopital_poly` and `monotonic_quad` do all have 2 canonical steps, and the base
training corpus minimum is 3, so they are outside the training distribution.
But under the corrected model (fixed step size, epoch 4 of 5) all three now
score **1.0000** on 1000 runs each, and so does `taylor_linear`, also 2 steps.
A regime the training data never shows is a regime the model can still
generalize to; 0.000 recall was what *undertraining* looked like, not what a
distribution hole looks like. The "fix belongs in the corpus" recommendation
was a misdiagnosis, and the corrected step size — not a corpus change — is what
moved those three from 0.020–0.054 to 1.0000.

What is actually weak now, and what it is not: `sum_odd 0.6657` (2–5 steps),
`partial_fractions 0.6680` (4 steps), `inverse_2x2 0.6970` (3 steps),
`inflection_cubic 0.7000` (2), `nilpotent_sq 0.7010` (2). `inverse_2x2` and
`partial_fractions` have 3–4 steps, well inside the training range, and two of
the six have the same step count as three algorithms now scoring 1.0000 — so
step count does not separate the weak from the strong. The binding variable is
something else and is not yet identified; recorded as an open question rather
than re-diagnosed from the same evidence that produced the first wrong answer.

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

## Headlines (chance 0.20)

- Overall exact **0.4575** (43,170/94,354); per-step **0.6750** (189,664/280,967).
- Levels: `-1 0.57 / -0.5 0.24 / 0 0.75 / +0.5 0.28 / +1 0.45`.
- Tiers: base 0.467 / big 0.439 / neg 0.467 (flat: scale and sign generalize).
- Best: `cube_of_binomial 0.855`, `sum_fourth_powers 0.852`, `geometric_series 0.816`.
- Worst: `shadow_rates 0.020`, `lhopital_poly 0.040`, `monotonic_quad 0.054`.

## Finding: 2-step derivations are OOD for the whole stack

The three worst algos (plus `taylor_linear`, `nilpotent_sq`) all have exactly
2 canonical steps. Clean-+1 recall on them is **0.000 (0/60)** under BOTH the
2-epoch Gen1-10M and the converged 60-epoch H128 (`temp/gen1-runs/shadow-check.log`).
The base training corpus minimum is 3 steps (`binomial_expand`, `chain_rule`,
...). So no model trained on it has ever seen a 2-step derivation, and all of
them map sparse-clean measurement vectors to non-+1 (mostly 0.0: the big
confusion shows +1-truth -> 0.0-pred 8,126 times).

This is a distribution hole, not a bug: measurements are correct
(exact=1, struct_ok=1), the scorer just never learned the sparse regime.
Fix belongs in the corpus (2-step and 1-step derivations as first-class rows),
not in H. Recorded here so Gen2 training includes them.

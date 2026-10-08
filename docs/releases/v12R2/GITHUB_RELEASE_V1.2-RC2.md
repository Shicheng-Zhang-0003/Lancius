# V1.2-RC2 — PRM800K Veracity Verifier: source + trained model artifacts

Source archives (auto-generated from `7b5b1b2`) plus eleven trained model
artifacts. The models are the point of this release; the source is included so
every number below is reproducible rather than asserted.

**Pre-release, deliberately.** None of these models has ever seen
human-written mathematics. Every training label is an injected typed defect
whose ground truth is known by construction. The accuracies measure agreement
with a symbolic checker on synthetic derivations — they are not estimates of
real-world verifier accuracy.

---

## The finding that started this

`examples/distill_prm800k.c` trains on `chosen_completion` alone, and `chosen`
carries rating `+1` in **12961/12961** train rows and **1603/1603** test rows.
The shipped task is a constant label, so the best achievable "accuracy" is
reproducing a prior. Parsing every completion instead yields 48,672 train and
5,082 test steps with genuine spread (`-1/0/+1` = 19656/10234/18782 and
1970/867/2245). The test majority-class baseline is **2245/5082 = 0.4418** —
that, not 1.0000, is the bar a model has to clear to have demonstrated anything.

Featurisation is shared with the distiller, so a silent divergence would make
every downstream number a fiction that still prints. `c_featurize.c` reimplements
the featuriser independently in C and is proved **bit-identical across all
53,754 rows**.

## 3-class verifier — 0.4957 against 0.4418

`logits = X@W1 + b1 → tanh → @W2 + b2`, minimising `mean((logits − onehot)²)`
with graph autodiff supplying gradients. Four defects had to be fixed, each of
which would have printed a plausible number rather than failing:

1. Scaling logits by `rows/2` "to make MSE `0.5‖·‖²`". MSE is a *mean*, so the
   multiplier enters twice and scales the gradient by **s² = 1024**, not s.
   Diverged to NaN inside one epoch.
2. `softmax_ce` returned mean·log p — the **negative** of cross-entropy — so a
   working run reported −1.04.
3. The graph's MSE VJP is `(2/pe)(z−T)`; CE gives `(1/rows)(z−T)`. The
   relationship is the constant `2/NCLASS`, not `2/pe`.
4. The baseline evaluated `(model_t){0}` with NULL `W1/b1/W2/b2` — segfault;
   and one `metrics_t` was reused across splits, so `train-acc` printed test
   accuracy.

## Gradients verified independently

`ref_prm.c` is a pure-C trainer with hand-derived CE backprop and no framework
anywhere in it, mirroring every hyperparameter, the RNG stream, the He init and
the effective step. It reaches **0.4959** against the framework's **0.4957**,
trajectories tracking. The self-contained gradient check pins the graph's own
mean-MSE objective at **9.5e-09**.

Getting there cost three false alarms, **two of which were my harness rather
than the framework** — the exact failure the external-oracle discipline exists to
catch, reproduced from the inside. `MSE_BWD` was exact to `0e+00` in isolation
and a hand-derived CE gradient matched central differences to all printed
digits; the residual came from converting mean-MSE to CE with the wrong
prefactor (twice) and from an error metric dividing by gradients at the `1e-12`
level where a central difference is pure float noise.

## What this model cannot do — a proof, not a measurement

Across ten complexity levels, integer addition through matrix products:
**balanced accuracy 0.5000 at every level**, with `P(+1|correct)` equal to
`P(+1|wrong)` to four decimals.

The decisive result: **1150 minimal pairs whose 16-dim feature vectors are
bit-identical** — `2 + 2 = 4` against `2 + 2 = 7`, same length, same digit count,
same everything the featuriser sees — received **bit-identical predictions in
1150/1150 cases**. The capability is absent, not weak.

A control training the same architecture directly on this data reaches **0.5215
on the training half**: it cannot fit what it is shown, so the failure is in the
representation, not the optimisation.

This does not contradict 0.4957 on real PRM data — it explains it. PRM800K
ratings correlate with the *surface shape* of the writing, which is the entire
content of this model.

## The 5-level scale, and why disjointness is the whole ballgame

`{−1, −0.5, 0, +0.5, +1}`, separated on `(structure, final answer)`: structure ok
and final ok with every step exact / structure ok and final ok with a working
step off / a step omitted so nothing is judgeable / structure ok with the final
answer wrong / structure wrong and final wrong.

An earlier version made `−0.5` by corrupting an *intermediate* step while
leaving structure and final intact — **indistinguishable from `+0.5` by
construction**. The model responded correctly and sent **74% of true `−0.5` to
`+1`**, i.e. "probably false" read as "plain true", the most damaging error the
scale exists to prevent. Making the classes disjoint moved overall exact
accuracy **0.7097 → 0.8428** and MAE **0.347 → 0.145**, with no change to
architecture or data volume.

Label provenance: ground truth is the **injected defect type**, decided at
generation time and **never re-derived by the checker**. If it were, labels and
features would both be functions of the checker and the network would only
re-encode the checker's output — success that validates nothing.

Two further defects, both silent rather than crashing: a perturbation that was a
**no-op whenever `|x·scale| < 1`**, so answers below ~3 were never corrupted and
whole `−0.5`/`−1` samples were relabelled `+1`; and per-step feature rows with
**no step index**, so the net could not tell which step it was judging and
collapsed to `+1`, sending 99% of true `−0.5` steps there.

## Twenty algorithms, six families, exact arithmetic

`linear_solve`, `quadratic_formula`, `binomial_expand`, `difference_of_squares`,
`perfect_square_trinomial`, `sum_of_cubes`, `geometric_series`,
`arithmetic_series`, `slope_from_points`, `substitution_system`, `power_rule`,
`chain_rule`, `product_rule`, `quotient_rule`, `definite_integral`, `gauss_2x2`,
`dot_product_3d`, `matrix_det_2x2`, `euclid_gcd`, `mod_pow_fast`.

Each carries a mathematical description, an algorithmic description, and a
canonical derivation in exact `fractions.Fraction`, so step agreement is a real
comparison rather than a tolerance argument. `euclid_gcd` (repeated remainder,
up to 11 steps) and `mod_pow_fast` (square-and-multiply) are genuinely iterative,
so multi-step execution is exercised rather than closed forms.

## Trained from scratch, and 2,000 runtime tests

65 → 96 → 5, tanh, softmax CE, SGDM with L2 into the gradient, He init via
Box-Muller. Model selection on a validation split carved from **train**; test
touched once. Gradient check **8.9e-08**.

```
overall    n=1574  exact=0.9377  within1=0.9752  macroF1=0.6385  MAE=0.0562  kappa=0.9218
per-step   n=6764  exact=0.9808  within1=0.9871  macroF1=0.6499  MAE=0.0203  kappa=0.9708
```

`run_tests.py`: 20 algorithms × 100 parameter sets, cycling all five levels —
**1869/2000 overall (93.45%)**, **8230/8380 per-step (98.21%)**.

Per level: `−1` 0.9725, `−0.5` 0.9050, `0` 0.9950, **`+0.5` 0.8075**, `+1`
0.9925. The dominant error is `+0.5 → +1` in 71 cases, then `−0.5 → +1` in 31 —
both certify something wrong as true.

## `.lancius` export

The v2 serialiser writes node **values only for bound `INPUT` nodes**; `CONST`
carries a single scalar, so weights cannot ride along as constants. Weights are
bound INPUTs, features unbound. The stable C API cannot express this graph
(`lancius_graph_handle` is a wrapper, and its builders stop at input/matmul/relu
with no tanh and no bias add), but `lancius_graph_save()` is what
`save_stable` delegates to internally, so calling it directly is the same path.

**The batch dimension is fixed in the file** — node shapes are restored on load,
so the row count cannot be changed afterwards. Use the ASCII weights for
arbitrary batches.

All nine graphs were **reloaded in a fresh process from disk** and reproduce
their reference logits to **≤ 6.7e-16**.

## Efficacy and stability — and a correction to our own headline

3 widths × 3 seeds:

| H | test exact (mean ± sd) | range | kappa |
|---|---|---|---|
| 64 | 0.9070 ± 0.0388 | [0.8793, 0.9619] | 0.8815 ± 0.0501 |
| **96** | **0.9290 ± 0.0170** | [0.9053, 0.9441] | **0.9107 ± 0.0216** |
| 128 | 0.9183 ± 0.0280 | [0.8793, 0.9435] | 0.8973 ± 0.0354 |

All nine: **0.9181 ± 0.0306**. Chance is 0.200. **Ship H=96** — best mean *and*
lowest variance; H=64 has the worst mean and the widest spread.

Two corrections follow, both recorded rather than quietly dropped:

- **The first sweep's width effect was a seed artifact.** Varying width at one
  seed and seed at one width separately made H=64 look best (0.9619) and would
  have shipped a 37 KB model on it. The 3×3 shows 0.9619 is the lucky end of the
  least stable configuration's range. Separating factors one at a time cannot
  detect an interaction.
- **0.9377 quoted earlier was a favourable draw.** Against H=96's
  0.9290 ± 0.0170 it sits ~+0.65 sd above its own mean. The defensible statement
  is the mean with its spread.

## Assets

**Models — 5-level, `.lancius` (recommended: `h96-s20261007`)**

`verifier5-h{64,96,128}-s{20261007,11,22}.lancius` — nine graphs, 37–74 KB.

**Models — portable ASCII**

- `verifier5-h96-s20261007.weights.txt` — same model, plus the feature scaler
- `prm3-h32-s20261007.weights.txt` — the 3-class PRM800K model, plus scaler

The scaler travels with the weights deliberately: a model without the
normalisation its inputs were fitted under is not the model.

## Reproduce

```sh
python3 tools/prm/verifier5.py data_vec/v5 1600 20261007
gcc -O2 -fopenmp -std=c11 -I./include -o train_v5 examples/train_verifier5.c \
    liblancius.a -fopenmp -lm -lpthread
./train_v5 data_vec/v5 96 60 128 --seed 20261007 \
    --export-lancius model.lancius --verify-lancius --lancius-rows 256
python3 tools/prm/sweep_configs.py     # the full 3x3
python3 tools/prm/run_tests.py         # 2,000 runtime tests
```

## Known limitations

- **No real data.** Synthetic defects only.
- **`+0.5` is the weakest level** and its errors go in the dangerous direction.
- **No algebraic equivalence** — a correct answer by another route scores badly.
- 20 algorithms, integer-friendly coefficients; no radicals, no matrices above
  2×2, no systems beyond 2 unknowns.
- **Edge cases are out of scope** and untrained, by request.

Design rationale, every bug found on the way (including three separate wrong
ways to select a graph's output node, all of which produced *false* verification
failures), and the full efficacy/stability analysis are in
`docs/DESPOT_TRUTH_V2.md` §18.

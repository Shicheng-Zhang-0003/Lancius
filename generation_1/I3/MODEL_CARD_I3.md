# MODEL CARD — `temp/i3-runs/i3_alu_00.lancius` ("I3-alu00-warm")

**Family:** Lancius verifier, generation 3, lineage gen1 → I2-`alu_00` → (warm start) → I3.
**Purpose:** accept a 65-dim step-feature row and emit one of five verdict levels.

## Snapshot

| field | value |
|---|---|
| architecture | 65 -> 4096 -> 2560 -> 5 (10,771,461 params) |
| shapes | (65,4096) (1,4096) (4096,2560) (1,2560) (2560,5) (1,5) |
| parameters | 10,771,461 |
| format | .lancius 86,173,352 bytes, fixed eval batch 256 rows |
| warm start | generation_1/I2/alu_pool/alu_00.lancius (norms logged in training log) |
| normalizer | generation_1/I3/gen1_mu_sd.txt (the gen1 fit — do NOT refit MU/SD for this lineage) |
| algorithm scope | 150 algorithms = 20 base ({@link algos.py ALGORITHMS}) + 12 gen1-ext + 68 gen1-big + 50 I3-ext |
| excited weights | not quantized; mobility (momentum) zeroed after warm load |

## Data

Corpus `generation_1/I3/corpus/` with seed 20261019, all 150 algorithms drawn:

* 101,549 rows; train 81,239 (69,054 fit / 12,185 validation), test 20,310.
* Edge rows (whale-number, empty/ident/sign/ambiguity derivations) on every algorithm.
* Regimes: base / wide-fractional / fractional / negative; big tier drawn in lm-corpus clones
  (I3 gen_data_i3.py has its own multi-scale check of `x * scale` injection).

## Training

* epochs 4, batch 256, base LR 0.0008 -> effective 0.002 (one-third of gen1's LR since warm), momentum
  0.9 / adamF, weight decay 1e-4, gradient clipping as in trainer, seed 20261019.
* validation-selected epoch 2 (val exact = 0.9740, tied with epoch 4).
* gradcheck: 6.154e-08 scale-relative worst err on W2 (20 probes vs central differences).

## Held-out test (never trained on; rows 20,310)

| metric | overall | per-step |
|---|---|---|
| exact | 0.9741 | 0.9940 |
| within1 | 0.9797 | 0.9949 |
| macroF1 | 0.9736 | 0.9857 |
| MAE(level) | 0.0249 | 0.0059 |
| kappa | 0.9675 | 0.9909 |

Levels: -1.0 plain false, -0.5 probably false, +0.0 undecidable, +0.5 mostly true, +1.0 plain true.
The same-cell caveat: '+0.0 undecidable' rows are likewise emitted both when a derivation is genuinely
unjudgeable and when the script fails to express a level at all; run_big_i3.py drops those at harness level
("u-nexpressible") — the unexpressible set is 3,440 cells of 150k in the sweep.

## Randomized verification (run_big_i3.py, 1000 cells/algo, both models, identical cells)

* I3: overall 0.9891 (130,197 cells), per-step 0.9946 (479,233 row-steps).
* basis (`alu_00`): overall 0.9556, per-step 0.9741.
* paired: both 124,325 / only-I3 4,453 / only-basis 89 / neither 1,330.
* no algorithm regressed by more than 0.0030 (second_diff_central 0.9920->0.9890 is the largest dip).

## Designated use

* Verdict machine for the 150-algorithm verifier API family: drop-in replacement for `alu_00.lancius`
  wherever verifier5-compatible rows are produced, with equal 65/MU/SD front-end semantics.
* Basis for warm-starting further generations (it exported `generation_1/I3/corpus/model_i3.txt` at the
  end of training for that purpose) — the preferred lineage head going into gen 4.
* **The I3 fabric**: deployed as 12 clones in `generation_1/I3/alu_pool/`, driven by the I2-style
  combined-model protocol (LAIS-CU manual + `phi4-mini:3.8b` control unit via Ollama). The CU may quote
  only packet-delivered numbers (anti-hallucination contract holds in the I3 fork unchanged).

## Deployment notes (I3 fabric)

* Normalization must come from `paths.read_musd()` (`generation_1/I3/gen1_mu_sd.txt`): these weights were
  fitted under those exact stats, and `compute_musd(data_vec/v5)` no longer reproduces them. Serving with
  a refit normalizer is a silent train/serve skew.
* The Lancius CLI solved by `paths.find_lancius()` is `build/bin/lancius` — never assume `<repo>/lancius`.
* CU characteristics observed in smoke runs: phi4-mini can pad fabricated FEED arrays beyond its token
  budget; the fork's deterministic clamps (repeat-reply break, repeated-failing-op retraction) bound the
  damage. Plain arithmetic is directed away from the bus entirely.

## Out-of-scope warnings

* The tooling can only return the 5 levels above; value-val exact issuing / ROS API is not included.
* All accuracy tables come from the same generator family as the corpus (verifier5 feature rows). They say
  nothing about cells derived from different injectors/feature extractors — keep the front-end inheritance in mind.
* Misuse hazard: do not benchmark THIS model against others using a *refit* MU/SD; the results above are
  same-scaling-only, and any rescaling re-implementation must be relabeled a different snapshot.

# 1I5 — Lancius as Rules Engine (generic decision-surface trainer)

**Generation 1, Iteration 5.** I5 generalizes the 1I4 fire-risk pilot into a
reusable "Lancius as rules engine" trainer: Lancius learns a deterministic
decision surface P(out | features) that an external exploration loop can query
repeatedly — perturb inputs, dump scores, compare equal and different
situations, regress deviations, plot association graphs. Causation lives
outside Lancius; Lancius is the mute mathematician that scores, never speaks.

```
1IX lineage: 1I1 -> 1I2 -> 1I3 -> 1I4 -> 1I5 -> 1I6
                        ^ fire-risk pilot       ^ generic decision surface  ^ flagship verifier
```

## What I5 is

A persistent-graph MLP trainer (`examples/train_i5.c`) that reads a binary
corpus from `corpus.hdr` (not hardcoded dims) and trains:

- **Architecture:** FEAT -> H1 -> H2 -> NCLASS (tanh/tanh), NCLASS 2–8
- **Loss:** fused softmax cross-entropy (exact VJP `(softmax-y)/R`) — no MSE-on-logits step-size fudge
- **Optimizer:** SGD + momentum + decoupled weight decay
- **Clip:** global-norm grad clipping
- **LR schedule:** warmup + cosine
- **Warm start:** `--init-from <.lancius>` loads six param tensors; momentum starts at 0
- **Export:** `--verify-lancius` round-trip check (max|diff| ~1e-15)
- **Corpus:** `corpus.hdr` + `{train,val,test}.{X,L,S,A}.bin` (X=float64, L/S/A=int32)

## What I5 does

Produces a `.lancius` model whose output is a **score to be probed**, not an attribution to be trusted. An external loop queries it repeatedly:

```
perturb inputs -> dump scores -> compare equal/different -> regress deviations -> plot association graphs
```

This is the "rules engine" framing: Lancius never speaks causation; it only scores.

## Corpora

| Corpus | FEAT | NCLASS | Rows | Source |
|--------|------|--------|------|--------|
| `data_synthetic/` | 10 | 2 | 300/80/80 | `tools/make_synthetic_corpus.py` (smoke test) |
| `data_i5_real/` | 77 | 2 | 301/58/91 | 1I4 fire-risk data in train_i5 format |

The real corpus (`data_i5_real/`) is the 1I4 rangeland fire-risk data relabeled into train_i5 binary layout — 77 features, 2 classes (burn next year / not), extreme class imbalance (~0.7% positives).

## Files

```
generation_1/I5/
  paths.py          repo resolution, corpus/trainer locators
  README.md         this file
  corpus/           (links to data_synthetic/ and data_i5_real/)
  models/           exported .lancius models + checkpoints
  results_big/      head-to-head evaluation outputs
  tools/            (links to acquisition / corpus tools)

../examples/train_i5.c          the I5 trainer (CE, persistent graph)
../tools/acquire_and_train_i5.py fully automated NASA/Copernicus/GBIS pipeline
../tools/fire_data_acquire.py   I5 fire-data acquisition framework (stubs + GBIF live)
../tools/make_synthetic_corpus.py tiny synthetic corpus generator
```

## Reproduce

```bash
# 1. Build the trainer
make build/bin/train_i5

# 2. Synthetic smoke test (fast)
python3 tools/make_synthetic_corpus.py
build/bin/train_i5 data_synthetic 64 32 20 64 \
  --export-lancius generation_1/I5/models/i5_synthetic.lancius \
  --verify-lancius --save-dir generation_1/I5/models

# 3. Real fire-risk corpus (needs data_i5_real/ already staged)
build/bin/train_i5 data_i5_real 64 32 20 64 \
  --export-lancius generation_1/I5/models/i5_fire.lancius \
  --verify-lancius --save-dir generation_1/I5/models

# 4. Full automated acquisition (needs NASA Earthdata + Copernicus creds)
python3 tools/acquire_and_train_i5.py
```

## Known limits

- Binary metrics (AP/logloss/Brier) only when NCLASS == 2
- Extreme class imbalance in fire data (~0.7% positives) — 20x positive oversampling is the default
- Scores must be calibrated (Platt scaling) before being interpreted as probabilities
- Research prototype: not causal, not operational

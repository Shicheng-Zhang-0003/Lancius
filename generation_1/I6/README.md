# Generation 6 (I6) — the flagship verifier, trained with ALU models

I6 is the flagship Lancius verifier lineage: **I1 → I2 → I3 → I6**.
It is trained with the I3 ALU pool as warm-start substrate and targets
**MAE(level) < 1e-3 and MSE(level) < 1e-3** on the 150-algorithm verifier
corpus.

I4 and I5 were deviations (fire-risk and generic decision-surface
trainers); I6 returns to the verifier flagship with a slightly wider trunk,
more epochs, the fused cross-entropy loss, and hard-negative mining.

```
i6  =  I3 ALU pool (alu_00..alu_11, 150-algo trained)
       --warm start + hard negatives + 20 epochs-->  I6 model
```

## Architecture

| field | value |
|-------|-------|
| architecture | 65 → 5120 → 3072 → 5 (tanh/tanh) |
| parameters | ~16.2M (~1.5x the I3 trunk; still a micromodel) |
| loss | fused softmax cross-entropy, exact VJP (softmax-y)/R |
| optimizer | SGD + momentum + decoupled weight decay |
| LR schedule | warmup + cosine |
| clip | global-norm 5.0 |
| epochs | 20 (I3 did 4) |
| batch | 256 |
| hard negative mining | frac=0.5, starts epoch 2 |

The trunk is deliberately **slightly wider, not much larger**: 5120/3072 vs
I3's 4096/2560. It stays well under the Lancius 100M-element / 800MB tensor
cap and stays fast enough for the ALU bus (static scheduler, arena/flat
buffers, real `lancius run` subprocesses).

## Corpus

Same format as I3: `corpus.hdr` + `{train,val,test}.{X,L,S,A}.bin`,
FEAT=65, NCLASS=5, 0.8/0.2 split.

```
generation_1/I6/gen_data_i6.py [n_per_level] [seed] [out_dir]
  defaults: 8000, 20261026, generation_1/I6/corpus
```

Deltas vs I3's corpus:
- **More rows**: 8000 per level vs 3500 (~400k+ rows before split)
- **Hard negative mining** (`--hard-neg`): score candidate cells through the
  I3 ALU pool and oversample cells where ALUs disagree or are wrong — this
  concentrates training mass on the decision boundary where MAE lives
- **Deeper LONG derivations**: long chains (10–12 steps) appear in training,
  not just test
- **Rebalanced regimes**: neg tier drawn at the same rate as base

With hard negatives:
```
python3 generation_1/I6/gen_data_i6.py 8000 20261026 generation_1/I6/corpus \
    --hard-neg --alu-pool generation_1/I3/alu_pool --neg-per-alu 200
```

## Training

```
make build/bin/train_i6
build/bin/train_i6 generation_1/I6/corpus 5120 3072 20 256 \
  --seed 20261026 --lr 0.0008 \
  --init-from generation_1/I3/alu_pool/alu_01.lancius \
  --mu-sd generation_1/I3/gen1_mu_sd.txt \
  --export-lancius generation_1/I6/models/i6_alu_00.lancius \
  --verify-lancius --save-dir generation_1/I6/models
```

- `--init-from`: warm-start the six param tensors from an I3 ALU model
  (alu_00..alu_11). Momentum starts at 0.
- `--mu-sd`: pinned normalizer (shared with I3; do NOT refit)
- `--hard-negative-frac` / `--hard-negative-start`: control hard-negative
  mining (default 0.5 / epoch 2)
- `--verify-lancius`: round-trip reload check (max|diff| must be ~1e-15)
- Checkpoints `best.bin`/`last.bin` in `--save-dir`; `--resume` continues

## Evaluation

Head-to-head vs all 12 I3 ALUs on identical cells:

```
python3 generation_1/I6/run_big_i6.py 1000 \
  generation_1/I6/models/i6_alu_00.lancius
```

Reports per-model overall/per-step exact, MAE, MSE, paired agreement, and
per-algorithm deltas. Target:

| metric | target |
|--------|--------|
| MAE(level) | < 1e-3 |
| MSE(level) | < 1e-3 |

## Files

```
generation_1/I6/
  gen_data_i6.py   corpus builder -> corpus/
  run_big_i6.py    head-to-head vs all 12 I3 ALUs
  paths.py         repo resolution, pinned MU/SD loader, lancius locator
  README.md        this file
  corpus/          {train,test}.{X,L,S,A}.bin + corpus.hdr + meta.json
  models/          i6_alu_00.lancius (+ best.bin / last.bin checkpoints)
  results_big/     big_summary.txt
```

temp/i6-runs/   training outputs (exported .lancius, round-trip verified)

## Reproduce end to end

```bash
# 1. Build the trainer
make build/bin/train_i6

# 2. Generate the corpus (with hard negatives against the I3 ALU pool)
python3 generation_1/I6/gen_data_i6.py 8000 20261026 generation_1/I6/corpus \
    --hard-neg --alu-pool generation_1/I3/alu_pool --neg-per-alu 200

# 3. Train (warm start from the best I3 ALU, ~2-4h depending on cores)
build/bin/train_i6 generation_1/I6/corpus 5120 3072 20 256 \
  --seed 20261026 --lr 0.0008 \
  --init-from generation_1/I3/alu_pool/alu_01.lancius \
  --mu-sd generation_1/I3/gen1_mu_sd.txt \
  --export-lancius generation_1/I6/models/i6_alu_00.lancius \
  --verify-lancius --save-dir generation_1/I6/models

# 4. Evaluate head-to-head vs the I3 ALU pool
python3 generation_1/I6/run_big_i6.py 1000 \
  generation_1/I6/models/i6_alu_00.lancius
```

## Known limits

- `wilson_check` and similar boundary-band algorithms were the I3 residual;
  hard-negative mining targets them but they may remain above 0.99.
- MAE < 1e-3 is a strict target: on a 5-level scale it means the model is
  essentially never wrong. If it stalls, increase `n_per_level`, add more
  epochs, or widen H1/H2 further (still under the 100M-element cap).
- The pinned MU/SD must travel with the model; serving with a refit
  normalizer is silent train/serve skew.

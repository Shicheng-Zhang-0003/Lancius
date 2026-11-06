# generation_1 — six generations, one lineage (1I1 through 1I6)

Parameter counts and architecture claims below are owned by each
generation's model card; this README does not restate them as authority.
(It previously claimed a "58M-parameter verifier" for 1I1; the I1 card owns
the count — **10,771,461 weights** — and this README had drifted from it.)

## Layout
```
generation_1/
  I1/                 <- 1I1 heritage (original generation-1 build)
    run_big.py        100-algo randomized benchmark (master seed 20261007)
    run_tests_gen1.py DETAIL + 3-tier sweep harness
    gen_data_gen1.py  corpus builder (level-balanced bins)
    algos_gen1_ext.py / algos_gen1_big.py   30 extension algorithms
    MODEL_CARD_GEN1.md  README.md  BIG_SUITE.md  gen1_model.txt
    corpus/  results/  results_big/          (generated; gitignored)
  I2/                 <- 1I2 generation-2 (model conglomeration)
    chat.py           ENTRY POINT: the full chatbot
    paths.py          repo/dir resolution + lancius locator
    bus/              real-runtime dispatch (lancius run subprocesses)
    isa/              LAIS-F op encoders + cu_manual.txt (CU instruction set)
    models/           Lancius v2 constructors + 3344-dim demo model
    alu_pool/         8 gen-1 verifier ALUs (clones of gen1-lr002)
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat (chatbot core) / fabric_test_chat (24-case suite)
  I3/                 <- 1I3 generation-3 (the warm-start generation)
    algos_i3.py       50 NEW algorithms (14 algebra / 12 calculus / 12 LA / 12 NT)
    gen_data_i3.py    corpus builder -> corpus/ (101,549 rows over 150 algos)
    run_big_i3.py     150-algo head-to-head verification (i3 vs the I2 basis)
    gen1_mu_sd.txt    PINNED normalizer: the stats the I3 weights were fitted under
    MODEL_CARD_I3.md  the I3 verifier's own card (owns its numbers)
    chat.py           ENTRY POINT (I2-style chatbot with the I3 ALU pool)
    paths.py          repo resolution + pinned MU/SD loader + lancius locator
    alu_pool/         12 clones of the I3 verifier (warm-started; 150-algo trained)
    models/           Lancius v2 constructors + 3344-dim demo model (I3 seed)
    bus/ isa/         same bare-metal dispatch + LAIS-CU manual as I2
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat / fabric_test_chat (phi4-mini:3.8b CU)
    corpus/ results/ results_big/             (generated; gitignored)
  I4/                 <- 1I4 research prototype: next-year fire risk
    MODEL_CARD.md     owns the I4 numbers (AP 0.03668 calibrated test AP, ...)
    targets/          i4_fire_next_year_v0.1.json (frozen task spec)
    tools/            feature builders + corpus/QC + baselines + calibrate/evaluate
    models/           i4_mlp.lancius (export; release asset, gitignored) + checkpoints
    results/          calibrated/raw probability dumps + metrics/calibration JSON
    figures/          PR/ROC + calibration plots
    corpus/  data/                            (generated; gitignored)
    manuscript/  specs/                       (work in progress; empty)
  I5/                 <- 1I5 Lancius as rules engine (generic decision surface)
    paths.py          repo resolution, corpus/trainer locators
    run_big_i5.py     binary eval: accuracy, MAE, MSE, ROC-AUC, threshold sweep
    gen1_mu_sd.txt    PINNED normalizer (synthetic corpus, FEAT=10)
    README.md         I5 rules-engine framing
    corpus/           symlinks: synthetic -> data_synthetic/, real -> data_i5_real/
    tools/            symlinks: train_i5.c, acquire_and_train_i5.py,
                      fire_data_acquire.py, make_synthetic_corpus.py
    models/           exported .lancius + checkpoints
    results_big/      eval outputs
    (also cloned at 243-BDC/I5/)
  I4/                 <- 1I4 research prototype: next-year fire risk
    (also cloned at 243-BDC/I4/)
  I6/                 <- 1I6 flagship verifier (MAE/MSE < 1e-3 target)
    paths.py          repo resolution, pinned MU/SD loader, lancius locator
    gen_data_i6.py    expanded corpus builder + hard-negative mining vs I3 ALU pool
    run_big_i6.py     head-to-head vs all 12 I3 ALUs on identical cells
    gen1_mu_sd.txt    PINNED normalizer (shared with I3, FEAT=65)
    README.md         full reproduce instructions: build -> corpus -> train -> eval
    corpus/           158K train / 46K test rows, FEAT=65 NCLASS=5
    models/           i6_alu_00.lancius + best.bin / last.bin checkpoints
    results_big/      big_summary.txt + per-epoch prelims
    train_i6.c        (in ../examples/) CE loss, persistent graph, warm start,
                      hard-negative mining, checkpoint/resume
```

## Lineage

I1 trained the gen-1 verifier on the big suite: 65-dim step-features ->
4096 -> 2560 -> 5 classes, **10,771,461 weights**, 0.9356 overall-exact on
the 100-algorithm randomized suite (numbers owned by `I1/MODEL_CARD_GEN1.md`).
I2 clones that single model into 8 bare-metal ALUs, gives them a bus and an
external control unit (qwen2.5-coder:3b via Ollama), and wraps the ensemble
in a chatbot whose claims are gate-checked by the silicon: the CU may never
quote a verdict, margin, or vote that no ALU packet actually delivered.
I3 warm-starts from the I2 basis and trains the SAME verifier shape on a
150-algorithm corpus (100 gen1 + 50 new I3 families): held-out test exact
0.9741 / per-step 0.9940, and 0.9891 overall-exact head-to-head on 130,197
randomized cells, 4,453:89 one-sided wins vs the basis
(numbers owned by `I3/MODEL_CARD_I3.md`). The I3 fabric is the I2 combined
model over a 12-clone pool, with the CU swapped to phi4-mini:3.8b and MU/SD
pinned to the training-time fit — serving with a refit normalizer is a
silent train/serve skew, so `I3/paths.read_musd()` is the only loader.
I4 leaves the verifier family for a prediction task: next-year rangeland
fire risk in the southern BC/Alberta dry interior (task spec frozen in
`I4/targets/i4_fire_next_year_v0.1.json`), a 77→512→256→2 Lancius MLP
(171,778 parameters) trained through `examples/train_i4.c`, Platt-calibrated
on validation only, and reported against logistic / LightGBM /
cell-climatology baselines — where it beats prevalence, climatology and
logistic but trails LightGBM on test AP (0.0367 vs 0.0461), reported
without test-set tuning. It is a research prototype, not operational.

I5 (1I5) generalizes the 1I4 fire-risk pilot into "Lancius as rules engine":
a generic decision-surface trainer that reads any FEAT/NCLASS corpus from
`corpus.hdr`, trains with fused cross-entropy + persistent graph, and exports
a `.lancius` model whose output is a score for external probing — not an
attribution to be trusted. Synthetic smoke test (FEAT=10, NCLASS=2) in
`data_synthetic/`; real fire-risk corpus (FEAT=77, NCLASS=2) in
`data_i5_real/`.

I6 (1I6) returns to the verifier flagship: 65→4096→2560→5, warm-started
from the I3 ALU pool, trained with CE loss + hard-negative mining, targeting
MAE/MSE < 1e-3. Training in progress — see `I6_TRAINING_STATUS.md` and
`I6/models/train_i6.log`.

## Run
```
# I2 chatbot (interactive / scripted / 24-case suite)
python3 generation_1/I2/chat.py --alus 4
python3 generation_1/I2/chat.py --alus 4 --demo "7*8?" \
  "Check: derivation of euclid_gcd reported +1.0. Trustworthy?"
python3 generation_1/I2/orchestrator/fabric_test_chat.py --part all

# I3 (warm-start model, 150 algos, phi4-mini CU of microsoft/Phi-4-mini)
python3 generation_1/I3/chat.py --alus 4
python3 generation_1/I3/orchestrator/fabric_net.py 6 4
python3 generation_1/I3/orchestrator/fabric_test_chat.py --part all
python3 generation_1/I3/run_big_i3.py 1000

# batch + live fabric demos
python3 generation_1/I2/orchestrator/fabric_net.py 6 4
python3 generation_1/I2/orchestrator/fabric_live.py --rows 5 --alus 4

# I4 (research prototype; needs temp/i4venv + generated corpus/data)
temp/i4venv/bin/python generation_1/I4/tools/check_corpus.py
make build/bin/train_i4
build/bin/train_i4 generation_1/I4/corpus 512 256 30 512 \
  --pos-repeat 20 --save-dir generation_1/I4/models/i4 \
  --export-lancius generation_1/I4/models/i4_mlp.lancius --verify-lancius
temp/i4venv/bin/python generation_1/I4/tools/calibrate_i4.py
temp/i4venv/bin/python generation_1/I4/tools/evaluate_i4.py --name test

# I5 (rules engine): synthetic smoke test
make build/bin/train_i5
python3 tools/make_synthetic_corpus.py
build/bin/train_i5 data_synthetic 64 32 20 64 \
  --export-lancius generation_1/I5/models/i5_synthetic.lancius \
  --verify-lancius --save-dir generation_1/I5/models
python3 generation_1/I5/run_big_i5.py generation_1/I5/models/i5_synthetic.lancius data_synthetic

# I5 (real fire-risk corpus, if data_i5_real/ is staged)
build/bin/train_i5 data_i5_real 64 32 20 64 \
  --export-lancius generation_1/I5/models/i5_fire.lancius \
  --verify-lancius --save-dir generation_1/I5/models
python3 generation_1/I5/run_big_i5.py generation_1/I5/models/i5_fire.lancius data_i5_real

# heritage (I1)
python3 generation_1/I1/run_tests_gen1.py 6
python3 generation_1/I1/run_big.py 200
```
I2 depends on I1: the ALU weights, the algorithm corpus, and the feature
builder all live there. Keep `generation_1/I1/` stable. I3 warm-starts from
the I2 basis and owns the preferred lineage head going into gen 4.

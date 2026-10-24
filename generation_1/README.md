# generation_1 — four generations, one lineage

Parameter counts and architecture claims below are owned by each
generation's model card; this README does not restate them as authority.
(It previously claimed a "58M-parameter verifier" for I1; the I1 card owns
the count — **10,771,461 weights** — and this README had drifted from it.)

## Layout
```
generation_1/
  I1/                 <- heritage (original generation-1 build)
    run_big.py        100-algo randomized benchmark (master seed 20261007)
    run_tests_gen1.py DETAIL + 3-tier sweep harness
    gen_data_gen1.py  corpus builder (level-balanced bins)
    algos_gen1_ext.py / algos_gen1_big.py   30 extension algorithms
    MODEL_CARD_GEN1.md  README.md  BIG_SUITE.md  gen1_model.txt
    corpus/  results/  results_big/          (generated; gitignored)
  I2/                 <- generation-2 (model conglomeration)
    chat.py           ENTRY POINT: the full chatbot
    paths.py          repo/dir resolution + lancius locator
    bus/              real-runtime dispatch (lancius run subprocesses)
    isa/              LAIS-F op encoders + cu_manual.txt (CU instruction set)
    models/           Lancius v2 constructors + 3344-dim demo model
    alu_pool/         8 gen-1 verifier ALUs (clones of gen1-lr002)
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat (chatbot core) / fabric_test_chat (24-case suite)
  I3/                 <- generation-3 (the warm-start generation)
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
  I4/                 <- generation-4 (research prototype: next-year fire risk)
    MODEL_CARD.md     owns the I4 numbers (AP 0.03668 calibrated test AP, ...)
    targets/          i4_fire_next_year_v0.1.json (frozen task spec)
    tools/            feature builders + corpus/QC + baselines + calibrate/evaluate
    models/           i4_mlp.lancius (export; release asset, gitignored) + checkpoints
    results/          calibrated/raw probability dumps + metrics/calibration JSON
    figures/          PR/ROC + calibration plots
    corpus/  data/                            (generated; gitignored)
    manuscript/  specs/                       (work in progress; empty)
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

# heritage (I1)
python3 generation_1/I1/run_tests_gen1.py 6
python3 generation_1/I1/run_big.py 200
```
I2 depends on I1: the ALU weights, the algorithm corpus, and the feature
builder all live there. Keep `generation_1/I1/` stable. I3 warm-starts from
the I2 basis and owns the preferred lineage head going into gen 4.

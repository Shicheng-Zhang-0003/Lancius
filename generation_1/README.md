# generation_1 — two generations, one lineage

## Layout
```
generation_1/
  I1/                 <- previous work (original generation-1 build)
    run_big.py        100-algo randomized benchmark (master 20261007)
    run_tests_gen1.py DETAIL + 3-tier sweep harness
    gen_data_gen1.py  corpus builder (level-balanced bins)
    algos_gen1_ext.py / algos_gen1_big.py   30 extension algorithms
    MODEL_CARD_GEN1.md  README.md  BIG_SUITE.md  gen1_model.txt
    corpus/  results/  results_big/
  I2/                 <- generation-2 work (model conglomeration)
    chat.py           ENTRY POINT: the full chatbot
    paths.py          repo/dir resolution + sys.path enforcement
    models/           Lancius v2 constructors + 3344-dim demo model
    alu_pool/         8 gen-1 verifier ALUs (clones of gen1-lr002)
    bus/              real-runtime dispatch (lancius run subprocesses)
    isa/              LAIS-F op encoders + cu_manual.txt (CU instruction set)
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat (chatbot core) / fabric_test_chat (24-case suite)
  I3/                 <- generation-3 work (the warm-start generation)
    algos_i3.py       50 NEW algorithms (14 algebra / 12 calculus / 12 LA / 12 NT)
    gen_data_i3.py    corpus builder -> corpus/ (101,549 rows over 150 algos)
    run_big_i3.py     150-algo head-to-head verification (i3 vs the I2 basis)
    gen1_mu_sd.txt    PINNED normalizer: the stats the I3 weights were fitted under
    chat.py           ENTRY POINT (I2-style chatbot with the I3 ALU pool)
    paths.py          repo resolution + pinned MU/SD loader + lancius locator
    alu_pool/         8 clones of the I3 verifier (warm-started; 150-algo trained)
    models/           Lancius v2 constructors + 3344-dim demo model (I3 seed)
    bus/ isa/         same bare-metal dispatch + LAIS-CU manual as I2
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat / fabric_test_chat (phi4-mini:3.8b CU)
    corpus/ results/ results_big/  training + verification artifacts
```

## Lineage
I1 trained the 58M-parameter verifier (65-dim step-features -> 5 classes,
93.5% overall-exact on the 100-algorithm suite). I2 clones that single model
into N bare-metal ALUs, gives them a bus and an external control unit
(qwen2.5-coder:3b via Ollama), and wraps the ensemble in a chatbot whose
claims are gate-checked by the silicon: the CU may never quote a verdict,
margin, or vote that no ALU packet actually delivered.
I3 warm-starts from the I2 basis and trains the SAME verifier shape on a
150-algorithm corpus (100 gen1 + 50 new I3 families): 0.9891 overall-exact
head-to-head on 130,197 randomized cells, 4,453:89 one-sided wins vs the
basis. The I3 fabric is the I2 combined model over that new pool, with the
CU swapped to phi4-mini:3.8b and MU/SD pinned to the training-time fit.

## Run
```
# interactive chatbot
python3 generation_1/I2/chat.py --alus 4

# scripted demo
python3 generation_1/I2/chat.py --alus 4 --demo "7*8?" \
  "Check: derivation of euclid_gcd reported +1.0. Trustworthy?"

# end-to-end chatbot test suite (24 cases)
python3 generation_1/I2/orchestrator/fabric_test_chat.py --part all

# I3 (warm-start model, 150 algos, phi4-mini CU of microsoft/Phi-4-mini)
python3 generation_1/I3/chat.py --alus 4
python3 generation_1/I3/orchestrator/fabric_net.py 6 4
python3 generation_1/I3/orchestrator/fabric_test_chat.py --part all
python3 generation_1/I3/run_big_i3.py 1000

# batch + live fabric demos
python3 generation_1/I2/orchestrator/fabric_net.py 6 4
python3 generation_1/I2/orchestrator/fabric_live.py --rows 5 --alus 4

# heritage (I1)
python3 generation_1/I1/run_tests_gen1.py 6
python3 generation_1/I1/run_big.py 200
```
I2 depends on I1: the ALU weights, the algorithm corpus, and the feature
builder all live there. Keep `generation_1/I1/` stable.

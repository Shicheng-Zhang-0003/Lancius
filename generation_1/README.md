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
  I2/                 <- new work (model conglomeration)
    chat.py           ENTRY POINT: the full chatbot
    paths.py          repo/dir resolution + sys.path enforcement
    models/           Lancius v2 constructors + 3344-dim demo model
    alu_pool/         8 gen-1 verifier ALUs (clones of gen1-lr002)
    bus/              real-runtime dispatch (lancius run subprocesses)
    isa/              LAIS-F op encoders + cu_manual.txt (CU instruction set)
    orchestrator/     fabric_net / fabric_live / fabric_coherent /
                      fabric_chat (chatbot core) / fabric_test_chat (24-case suite)
```

## Lineage
I1 trained the 58M-parameter verifier (65-dim step-features -> 5 classes,
93.5% overall-exact on the 100-algorithm suite). I2 clones that single model
into N bare-metal ALUs, gives them a bus and an external control unit
(qwen2.5-coder:3b via Ollama), and wraps the ensemble in a chatbot whose
claims are gate-checked by the silicon: the CU may never quote a verdict,
margin, or vote that no ALU packet actually delivered.

## Run
```
# interactive chatbot
python3 generation_1/I2/chat.py --alus 4

# scripted demo
python3 generation_1/I2/chat.py --alus 4 --demo "7*8?" \
  "Check: derivation of euclid_gcd reported +1.0. Trustworthy?"

# end-to-end chatbot test suite (24 cases)
python3 generation_1/I2/orchestrator/fabric_test_chat.py --part all

# batch + live fabric demos
python3 generation_1/I2/orchestrator/fabric_net.py 6 4
python3 generation_1/I2/orchestrator/fabric_live.py --rows 5 --alus 4

# heritage (I1)
python3 generation_1/I1/run_tests_gen1.py 6
python3 generation_1/I1/run_big.py 200
```
I2 depends on I1: the ALU weights, the algorithm corpus, and the feature
builder all live there. Keep `generation_1/I1/` stable.

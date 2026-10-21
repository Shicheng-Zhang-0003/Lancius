# Generation 3 (I3) — the math model of everything: the "understand all of maths" warm-start

I3 is the first Lancius generation verified on a far wider algorithm scope than gen1: **150 algorithms**
(the generation-1 100 plus 50 new I3 families), one ALU-model, **warm-started from the I2 basis model** rather
than random. The premise being tested: a trunk that already verifies 100 algorithms should be able to fold in 50
unseen *new* algorithm families with a light, warm pass while *gaining accuracy on both old and new*. It does.

```
i3  =  basis (I2 alu_00, 100-algo trained)  --warm-->  I3 model (150-algo trained)
```

## Pipeline

| stage | command | guardrail |
|---|---|---|
| algorithms | `generation_1/I3/algos_i3.py` — 50 new Algorithm entries | 50/50 validated (300 seeded draws each), max_steps<=12, no float/int leaks, no name collisions with the existing 100 |
| corpus | `python3 generation_1/I3/gen_data_i3.py` (seed 20261019) | 101,549 rows over 150 algos, zero_coverage=[], edge rows on every algorithm |
| training | examples/train_i3.c (fork of train_gen1.c with warm-start) | 4 epochs, lr 0.0008 (eff 0.002, 1/3 of gen1's), gradcheck 6.2e-08 |
| verification | `python3 generation_1/I3/run_big_i3.py 1000` | head-to-head vs the basis on identical cells, randomized |

## Training result (temp/i3-runs/i3_alu_00.lancius)

```
warm start loaded from generation_1/I2/alu_pool/alu_00.lancius
  ||W1||^2=8153.08 ||b1||^2=0.00220557 ||W2||^2=5088.8 ||b2||^2=0.00345881 ||W3||^2=8.51902 ||b3||^2=0.00430351

epoch  val-exact   val-MAE
   1     0.9708     0.0267
   2     0.9740     0.0235
   3     0.9735     0.0237
   4     0.9740     0.0235
selected epoch 2 on validation (val exact=0.9740)
```

Held-out test (n=3,588 overall / 16,722 per-step rows):

* overall: exact 0.9741, within1 0.9797, macroF1 0.9736, MAE 0.0249, kappa 0.9675
* per-step: exact 0.9940, kappa 0.9909
* export: temp/i3-runs/i3_alu_00.lancius (86,173,352 bytes, fixed batch), round-trip reload max|diff| = 5.773e-15

## The head-to-head that matters — run_big_i3.py

150 algorithms x 1,000 randomized cells each (master seed printed in the log), both models score the
same cells: i3 = the trained I3 model, alu00 = the I2 basis it was warm-started from. Neither model has a
fair-advantage angle: identical feature extraction, identical pinned MU/SD normalization — the only
difference is the weights.

| | i3 (I3)  | alu00 (I2/I3-basis) | delta |
|---|---|---|---|
| overall n=130,197 | **0.9891** | 0.9556 | **+0.0335** |
| per-step n=479,233 | **0.9946** | 0.9741 | +0.0205 |
| tier base (n=48,097) | 0.9944 | 0.9641 | +0.0303 |
| tier big (n=45,281) | 0.9841 | 0.9506 | +0.0335 |
| tier neg (n=36,819) | 0.9882 | 0.9506 | +0.0376 |
| the 50 NEW I3 algos (n=38,177) | 0.9849 | 0.9642 | +0.0207 |
| the pre-existing 100 (n=92,020) | 0.9908 | 0.9520 | +0.0388 |

Paired verdicts on the identical 130,197 cells:

```
both right     124,325
i3 only          4,453
alu00 only          89
neither          1,330
```

A 50-to-1 asymmetry in the disputed cells: the warm-started model wins nearly all of the cells on which the
two verdicts disagree, and the cells both models miss (1,330) are largely the same hard ones. This is not a
reshuffle of errors: the basis's own blind spots (e.g. fermat_little, euler_phi, legendre_euler at 0.53-0.56)
are exactly the cells the warm model now resolves (0.87-0.98).

Per-level for the trained i3: -1.0 0.9847, -0.5 0.9771, +0.0 0.9997, +0.5 0.9847, +1.0 0.9997.
Per-level for the basis:            -1.0 0.9404, -0.5 0.9223, +0.0 0.9997, +0.5 0.9614, +1.0 0.9571.
"probably false" is the hardest band (a -0.5 verdict really is a boundary-sign band with truly thin
evidence), while +0.0 undecidable is essentially solved by both models.

The regression side is flat: the worst algorithm-level difference is -0.003 (second_diff_central,
0.9920 -> 0.9890), then -0.0015 (prob_urn) and -0.001 (quotient_rule); everything else is a tie or
sub-tenth of a point. No algorithm dropped by more than 0.3 percentage points.

The 8 biggest single-algorithm gains:

| algorithm | i3 | alu00 | delta |
|---|---|---|---|
| fermat_little | 0.9755 | 0.5285 | +0.4470 |
| legendre_euler | 0.9793 | 0.5646 | +0.4147 |
| euler_phi |  0.9217 | 0.5628 | +0.3589 |
| wilson_check | 0.8744 | 0.5419 | +0.3325 |
| divisor_count | 0.9149 | 0.5885 | +0.3264 |
| median_five | 0.9378 | 0.6667 | +0.2711 |
| nilpotent_sq | 0.9960 | 0.7450 | +0.2510 |
| inflection_cubic | 0.9930 | 0.7750 | +0.2180 |

Full per-algorithm table: `generation_1/I3/results_big/big_summary.txt`

## Harness accounting (the "don't inflate with easy cells" head)

* 16,363 cells skipped: no valid cell survived the 10 build() retries (mostly prime-heavy builds under
  the big/neg tiers). These are not model-scored on either side, so they cannot flatter either model.
* 3,440 cells dropped: the requested level (e.g. +0.5 on a single-step algorithm) is mathematically
  unexpressible for that derivation, and the drop applies equally to both models. Scoring them as-is
  would cap reachable accuracy at 0.9743 by harness arithmetic, not by the model.
* Standardization is the pinned MU/SD `generation_1/I3/gen1_mu_sd.txt` for both models. A refit MU/SD
  is easily produced but any drift there becomes a hidden divergence between the two models; the whole
  point of the comparison is an identical feature space.

## The I3 fabric — the I2-style combined model with I3 ALUs

I2 proved the "combined model as ONE" idea: an external control unit (a small
LLM via Ollama) holding conversation and protocol, backed by a pool of bare-metal
Lancius ALUs whose LAIS-CU packets are the only permitted source of verdict,
margin, and vote numbers — a designed-in anti-hallucination contract. The I3
fabric is exactly that structure re-based onto the warm-start generation, with
the changes that the I3 training and a repo reorganization force:

1. **ALU pool** — `generation_1/I3/alu_pool/` = **12 models** (11 champion
   clones + 1 diverse sibling `alu_01` trained at seed 20261021, test exact
   0.9744, fabric overall 0.9906 vs champion 0.9891). Every claim the CU
   routes now runs against weights that verify **150 algorithms** (the old
   100 + all 50 new I3 families) at up to 0.9906 overall fabric accuracy.
2. **Control unit** — `phi4-mini:3.8b` via Ollama (`paths.CU_MODEL`), taking
   over from `qwen2.5-coder:3b`.
3. **Pinned normalization** — the fabric normalizers go through
   `paths.read_musd()` → `generation_1/I3/gen1_mu_sd.txt`, the exact stats the
   I3 weights were fitted under. A refit from `data_vec/v5` no longer reproduces
   those stats (the directory drifted after gen1 training), so `compute_musd()`
   there is off-limits for this lineage. (I2's own fabric still refits; that is
   the same latent skew discovered here — to be fixed in I2 only if requested.)

Robustness the CU change forced (`fabric_chat.py` / `paths.py`):

* `paths.find_lancius()` locates the CLI in `build/bin/` — I2's bus assumed a
  repo-root `lancius` binary that the repo reorganization moved away.
* The CU protocol prompt (CUE) now says plainly: pure arithmetic is not a
  derivation claim (answer it yourself), and FEED is forbidden unless the
  caller literally handed over 65 floats — never pad invented features
  (phi4-mini tried to answer "quick check: what is 12*12?" with a 300-element
  array of ones, cut off by its token budget, in the raw smoke).
* Two deterministic clamps in the round loop: repeating the same reply twice
  breaks the turn; the same failing op issued twice retracts cleanly ("there is
  no valid hardware packet for this request") instead of burning rounds. Both
  keep the I2 honesty semantics — nothing invented ever reaches the bus.

* **24-case anti-hallucination suite** (`fabric_test_chat.py --part all`): **24 PASS / 0 FAIL**
  on the 12-ALU fabric with phi4-mini CU. Cases cover plain chat, identity,
  translation, verified claims (accept/doubt/reject with honest margin quoting),
  FEED edge cases (exact 65, short, non-numeric), fresh-bot unknown-algo
  confession, injected fake margin rejection, noise-wrapped claims, empty/nonexistent
  algo names, most-confident comparison, two-claims-one-turn packet routing,
  class-map quiz, repeat-claim idempotence, and "fabric does not compute"
  arithmetic self-answer guard.

Keep-alive note: on this CPU box, phi4-mini replies run 5–30 s; a spilled
protocol turn costs minutes. The clamps keep the worst case bounded.

## Files

```
generation_1/I3/
  algos_i3.py         50 new algorithms (14 algebra, 12 calculus, 12 linear_algebra, 12 number_theory)
  gen_data_i3.py      corpus builder -> corpus/ (101,549 rows; split 81,239 / 20,310)
  corpus/             {train,test}.{X,L,S,A}.bin + algos.txt + meta.json + corpus.hdr + model_i3.txt
  gen1_mu_sd.txt      pinned normalizer (use for any eval that claims results for this lineage)
  run_big_i3.py       150-algo two-model randomized verification (head-to-head)
  results_big/        big_summary.txt  <- the full head-to-head
  README.md           this file
  MODEL_CARD_I3.md    model card for temp/i3-runs/i3_alu_00.lancius

  alu_pool/           **12 models** (11 champion clones + 1 diverse `alu_01` sib2021)
  chat.py             ENTRY POINT: the combined-model chatbot (I3 pool + phi CU)
  paths.py            repo resolution, pinned MU/SD loader, lancius locator
  models/             v2 constructors + 3344-dim demo graph (I3 seed 20261020)
  bus/                real-runtime dispatch over the pool (lancius run)
  isa/                LAIS-F op encoders + cu_manual.txt (verbatim from I2)
  orchestrator/       fabric_chat / fabric_coherent / fabric_net / fabric_live /
                      fabric_test_chat (24-case suite)

temp/i3-runs/i3_alu_00.lancius   the trained model (exported; round-trip verify 5.773e-15)
```

## Reproduce

```bash
make build/bin/train_i3
# train from scratch (warm start, ~2h45m on 18 cores):
build/bin/train_i3 generation_1/I3/corpus 4096 2560 4 256 \
  --seed 20261019 --lr 0.0008 \
  --init-from generation_1/I2/alu_pool/alu_00.lancius \
  --mu-sd generation_1/I3/gen1_mu_sd.txt \
  --export-lancius temp/i3-runs/i3_alu_00.lancius --verify-lancius
# head-to-head (~20 min):
python3 generation_1/I3/run_big_i3.py 1000
# the combined model (I2-style fabric with I3 ALUs, phi4-mini CU):
ollama pull phi4-mini:3.8b
python3 generation_1/I3/chat.py --alus 12
python3 generation_1/I3/orchestrator/fabric_net.py 6 12
python3 generation_1/I3/orchestrator/fabric_test_chat.py --part all
```

## Known limits / what an I4 should probe

* `wilson_check` (0.874 in the sweep) trails the pack; its residual sits in the -0.5 boundary band,
  where the available evidence is genuinely thin — not obviously trainable away.
* `median_five` (12 steps, exactly at verifier5.MAX_STEPS=12) stays a special workload: row_features
  silently drops steps beyond 12, so this algorithm must self-cap (it does, at 12) — and still went
  0.667 -> 0.938 through the warm start.
* Debug-resistant families are effectively solved at +0.0 for both models (0.9997 / 0.9997); future
  corpora should keep undecidable coverage but add harder near-miss derivations instead of more
  indeterminate mass.
* The `second_diff_central` dip (0.9920 -> 0.9890 on n=1000) is small but repeatable-looking; worth a
  seed-sweep probe in I4 to separate noise from a real regression.
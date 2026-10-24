# Lancius V1.2-RC3.1

> **What this release is:** the first point release of the RC3 model line —
> **model, dataset and fabric iteration** cut from an unchanged `v12R2`
> codebase. Not `V1.2RC4` and not `V1.2`: no format break, no stable-API
> change, no numerical behaviour change, so no compatibility claim moved
> (see `MANIFEST.md`, which records this release line explicitly).

| Field | Value |
|---|---|
| Public tag | `V1.2-RC3.1` |
| Base | `V1.2-RC3` (internal `v12R2`, unchanged) |
| Date | 2026-10-24 |
| Type | Model-lineage point release (assets + composition-layer repairs) |
| Compatibility | v2 `.lancius` files written by v11S+ load unchanged; RC2/RC3 assets remain valid |

## What is new versus RC3

### Assets: the I3 verifier generation

The RC3 page shipped the gen-1 verifier family (the flagship 65→4096→2560→5
trainer artifacts and the I2 ALU pool). RC3.1 ships its warm-started
successor, trained on 150 algorithms (the 100-algo gen-1 suite plus 50 new
I3 families in algebra / calculus / linear algebra / number theory):

| Asset | Shape / params | Size | sha256 (first 16) |
|---|---|---:|---|
| `i3_alu_00.lancius` | 65→4096→2560→5, 10,771,461 | 86,173,352 | `0ca8a6a4…e45ce673` |
| `i3_alu_sib2021.lancius` | same architecture, sibling seed 20261021 | 86,173,352 | `922266cc…93a61d50` |

- Numbers are owned by `generation_1/I3/MODEL_CARD_I3.md`; the pool it
  describes is **11 byte-identical champion clones + 1 diverse sibling**
  (`alu_01`, verified by sha256 across all twelve pool files — earlier card
  revisions said "12 clones" and are corrected beside the original).
- Headline (card): held-out test exact **0.9741** / per-step **0.9940**;
  head-to-head 0.9891 overall-exact over 130,197 randomized cells, 4,453:89
  one-sided wins against the RC3 basis.
- **Serving contract, load-bearing:** the I3 weights were fitted under the
  pinned normalizer `generation_1/I3/gen1_mu_sd.txt`. Refitting MU/SD from
  the data load is a silent train/serve skew — use `I3/paths.read_musd()`.
- The champion clones `alu_02…alu_11` are byte-identical to
  `i3_alu_00.lancius` by construction; they are not separate assets. The
  sibling *is* shipped because it is the one pool slot with different bytes.
- Eval batch is **fixed in the file at 256 rows** (the v2 format serializes
  node shapes; patching `[rows]` after load does not propagate). Drive it
  through the CLI at that size: `lancius run i3_alu_00.lancius --mode static`.

### Assets: I4, the first prediction-era artifact

| Asset | Shape / params | Size | sha256 (first 16) |
|---|---|---:|---|
| `i4_mlp.lancius` | 77→512 tanh→256 tanh→2, 171,778 | 1,375,888 | `7d7481a0…e3f0ced6` |

- Task: probability a 2 km rangeland cell burns by NBAC in year t+1 in the
  southern BC/Alberta dry interior. Task spec frozen in
  `generation_1/I4/targets/i4_fire_next_year_v0.1.json`; numbers owned by
  `generation_1/I4/MODEL_CARD.md`.
- Calibrated test (Platt fit on validation only; constants in
  `generation_1/I4/results/calibration.json` — slope `0.5860785`,
  intercept `-4.3665925`): **AP 0.03668**, ROC-AUC 0.88444, Brier 0.006832,
  log loss 0.035182, ECE 0.002697. It beats prevalence, cell climatology
  and logistic regression on AP, and **trails LightGBM** (AP 0.04611) —
  reported without test-set tuning.
- **Status: research prototype, not operational.** Predictive, not causal;
  calibration from 97 validation positives; spatial folds provisional.
- Eval batch is **fixed at 1 row** (the trainer exports at
  `--lancius-rows` default 1, and the in-binary `--verify-lancius`
  round-trips at that size; export check max abs diff `2.22e-16`).

### Code: composition-layer repairs only (nothing in `src/`)

The RC3.x policy is that a point release does not develop the framework.
RC3.1 nonetheless carries four composition-layer repairs recorded in
`CHANGELOG.md`, each verified independently:

1. The I2 fabric unusable from any cwd but the repo root (removed
   `lancius` sentinel falls back to `getcwd()`; CLI hardcoded the same
   removed layout) — I3's locator backported unchanged.
2. `lancius doctor`/`status`/`train` probing stale root-relative binary
   paths and reporting `binaries: 0/5 present` against a built tree —
   `bin_path()` resolution (running-CLI sibling → `./name` →
   `build/bin/name`), three call sites.
3. Plain `make` never built `train_i3` or `train_i4` (a `LINK` rule
   without `BINS` membership). Both registered; `make` now produces 41
   binaries.
4. `generation_1/README.md` documented two generations and claimed a
   58M-parameter I1 verifier (the card owns 10,771,461); the I2
   `chat.py` docstring carried the same claim; `MODEL_CARD_I3.md` said
   "12 clones" against the README's 11 clones + 1 sibling. Corrected,
   with the corrections recorded rather than silently overwritten.

I4 also entered the tree as a first-class trainer/generation
(`examples/train_i4.c`, task spec, tool pipeline, evidence JSONs), with
its bulk outputs (157MB corpus, 172MB raw data, weight dumps, probability
dumps) declared reproducible and gitignored.

## Verified rather than asserted

- `make check` RC=0 with the release edits in the tree; `-Wall -Wextra
  -Werror` build clean throughout.
- No numeric pathway was touched: the six-gate CI set from RC3
  (`check`, `check-long`, `check-sanitizers`, `check-oracle`,
  `check-physics`, `check-ubstrict`; `check-mutation` sampled) is
  unchanged in content, and `check-oracle` stood at 284/284 with 0
  skipped this cycle.
- `lancius doctor` prints five OK and `status` prints `5/5 present` from
  the repo root and from a foreign cwd.
- The I2 bus executes a real dispatch → aggregate round trip
  (`lancius run` per ALU, row-0 CLI contract parse) from a foreign cwd;
  the I3 chat demo holds the provenance contract live: the CU quoted a
  margin it received from the bus packet (its 0.9605 is the packet
  precision; the stream line rounded the same value to 0.961), not one
  it could have invented.
- Export integrity: I4 round-trip `2.22e-16` inside `train_i4.c`
  (`--verify-lancius`); I3 carried the RC3 unified-trainer export checks.

## What this release does not establish

- The verifier numbers come from injected-defect labels and the
  generator's own feature families (`DESPOT_TRUTH_V2` §18.8): the scale
  and machinery are validated, not real-world accuracy. No
  human-written derivation has been scored by I3.
- The variant `+0.5` ("mostly true, something wrong") remains a judgement
  call with a chosen 5% defect floor.
- I4's head-to-head loss to LightGBM is published as-is; neither a
  tuning pass nor a causal claim is attached to it.
- Generation remains scrapped by design: the CU scores and reports; it
  never speaks for the framework.

## Upgrade notes

- Pull the tag, `make`, and run `lancius doctor` — it should show
  `OK ×5`. If you drive the I2 fabric, `lancius` now resolves through
  `paths.find_lancius()`; no environment is required.
- RC3 assets need no re-download: shapes are unchanged; MU/SD pinning and
  batch contracts above apply per family.
- The v2 format and the CRC-mandatory default carry over unchanged from
  RC3.

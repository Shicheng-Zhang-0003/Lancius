# I4 model card: next-year rangeland fire risk

**Status:** research prototype, not operational. **Version:** 0.1.0. **Target spec:** `targets/i4_fire_next_year_v0.1.json` (frozen).

## Task

Given public earth-observation and biodiversity features known by the end of year `t`, estimate the probability that a 2 km rangeland cell burns by NBAC in year `t+1` in the southern BC/Alberta dry interior. This is a predictive risk model; it is not a causal attribution.

## Data

- Study area: WGS84 bbox `-121.0, 49.0, -110.0, 51.2`; 2 km equal-area grid EPSG:5070.
- Grid rectangle: 83,400 cells, of which 48,147 centers fall inside the bbox. The projected rectangle also covers cells outside the bbox; those are excluded.
- Rangeland mask: ESA WorldCover 2021 classes 20, 30, 40, 60 with fraction >= 0.2; 29,022 cells.
- Label: NBAC burn fraction >= 1% in the following year, rasterized at 400 m and block-averaged to 2 km; prescribed fires excluded.
- Feature years 2015-2023; label years 2016-2024; 77 features:
  - static 16, climate 22, MODIS 17, invasion 15, fire history 7.
- Missing values: training-median imputation. Scaling: training mean/std, clipped to +/-8.
- Coverage: mean 0.9948, minimum 0.8787; MODIS group 0.9762; all other groups 1.0 inside the study mask.

### Splits

| Split | Label years | Rows | Positives | Prevalence |
|---|---|---:|---:|---:|
| train | 2016-2021 | 174,132 | 1,420 | 0.008155 |
| val | 2022 | 29,022 | 97 | 0.003342 |
| test | 2023-2024 | 58,044 | 405 | 0.006977 |

Total corpus: 261,198 rows, 1,922 positives, 29,022 cells. No test labels were used for training or calibration.

## Model

- Lancius binary MLP: `77 -> 512 tanh -> 256 tanh -> 2 logits`; 171,778 parameters.
- Cross-entropy, SGD momentum 0.9, weight decay 1e-4, batch 512, 30 epochs, warmup + cosine learning rate.
- Positive training rows oversampled 20x; no class weights.
- Checkpoint selection by validation AP (tiebreak validation log loss). Selected step 588 of 11,790; best validation AP 0.0168. Later epochs had lower validation AP under the 2022 domain shift.
- Export: `models/i4_mlp.lancius`; raw Lancius scores must be calibrated (see below). Round-trip export check: max absolute difference `2.22e-16`.

## Calibration

The 20x oversampling makes raw probabilities close to the oversampled prior. Platt scaling is fit on validation logits only (29,022 rows, 97 positives), then applied to test:

- slope `0.5860785`, intercept `-4.3665925` (`results/calibration.json`).
- Raw test mean probability `0.1233`, Brier `0.06080`, log loss `0.19445`, ECE `0.11630`.
- Calibrated test mean `0.004316`, Brier `0.006832`, log loss `0.035182`, ECE `0.002697`; test prevalence `0.006977`.
- AP and ROC-AUC are unchanged by this monotone transform.

## Test results

Calibrated test probabilities, temporal holdout 2023-2024:

| Metric | Lancius I4 | Prevalence | Cell climatology | Logistic | LightGBM |
|---|---:|---:|---:|---:|---:|
| AP (PR-AUC) | **0.03668** | 0.006977 | 0.009883 | 0.03445 | **0.04611** |
| ROC-AUC | 0.88444 | 0.50000 | 0.55638 | 0.86708 | **0.89388** |
| Brier | 0.006832 | 0.006930 | 0.008080 | 0.007262 | 0.006956 |
| Log loss | 0.035182 | 0.041687 | 0.173036 | 0.037624 | **0.033806** |
| ECE (10 bins) | 0.002697 | 0.001177 | 0.012927 | 0.007342 | 0.002350 |
| Precision top 1% | 0.04828 | 0.04828 | 0.01897 | 0.03793 | **0.07931** |
| Precision top 5% | 0.04411 | 0.03067 | 0.02412 | 0.04101 | **0.04583** |

Lancius beats prevalence and logistic regression on test AP but trails LightGBM. It has better Brier and log loss than the prevalence, cell-climatology, and logistic baselines, and lower ECE than the logistic baseline. Recall at 80% precision is 0 because no threshold reaches 80% precision at this prevalence.

Cell-cluster bootstrap 95% CIs (500 resamples): AP `0.0320-0.0433`; ROC-AUC `0.8746-0.8946`; Brier `0.006255-0.007547`; log loss `0.03246-0.03856`; ECE `0.002113-0.003401`.

By test year:

| Year | Positives | AP | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| 2023 | 287 | 0.04111 | 0.87178 | 0.009641 |
| 2024 | 118 | 0.02329 | 0.86781 | 0.004023 |

Spatial folds are provisional contiguous x-quantile blocks over study cells. Fire occurrence is spatially concentrated: test positives by fold are 270, 135, 0, 0, 0. Fold AP is reported only for folds 0 and 1 (`0.0400`, `0.0323`); AP is undefined on the other folds. These folds are not a substitute for ecodistrict holdouts.

## Limitations

- Climate features are sampled at roughly 0.1 degrees (~11 km) and bilinearly interpolated to 2 km, not native 1 km.
- MODIS has ~2.4% missing values in the feature rows; imputation is training-median.
- GBIF records are sparse and biased toward accessible/roaded areas.
- Fire history is left-censored at 2013 and caps years-since-fire at 3.
- NBAC may miss small or low-intensity fires.
- Validation has only 97 positives and test only 405; temporal and calibration uncertainty is substantial.
- Platt calibration is fit on 97 validation positives; calibration uncertainty is high.
- No ecodistrict layer is sourced; spatial folds are provisional.
- The model is predictive and associational; no causal interpretation.
- The test AP result is below LightGBM; this is reported without test-set tuning.

## Artifacts and reproduction

Artifacts:

- `corpus/` and `corpus/qc.json`
- `models/i4/{best,last}.bin`, `models/i4_mlp.lancius`, `models/i4/model_i4.txt`
- `results/test.prob.bin` (calibrated), `results/test.raw.prob.bin`, `results/calibration.json`
- `results/baselines.json`, `results/i4_test_metrics.json`
- `figures/i4_test_pr_roc.png`, `figures/i4_test_calibration.png`

Commands:

```bash
temp/i4venv/bin/python generation_1/I4/tools/build_corpus.py
temp/i4venv/bin/python generation_1/I4/tools/check_corpus.py
temp/i4venv/bin/python generation_1/I4/tools/baselines.py
make build/bin/train_i4
build/bin/train_i4 generation_1/I4/corpus 512 256 30 512 \
  --pos-repeat 20 --save-dir generation_1/I4/models/i4 \
  --export-lancius generation_1/I4/models/i4_mlp.lancius --verify-lancius
temp/i4venv/bin/python generation_1/I4/tools/calibrate_i4.py
temp/i4venv/bin/python generation_1/I4/tools/evaluate_i4.py --name test
```

The Lancius export emits raw scores; apply `sigmoid(0.5860785 * logit(p_raw) - 4.3665925)` for calibrated probabilities, or use `results/test.prob.bin` directly.

#!/usr/bin/env python3
"""Fit Platt scaling on validation probabilities and apply it to test.

The trainer oversamples positives 20x, so its raw probabilities are close to the
oversampled prior rather than the true fire prevalence.  Platt scaling is fit
on validation rows only; test labels are never used.  AP/ROC are invariant to
this monotone transform; Brier/log-loss/ECE are reported after calibration.
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
MODEL = os.path.join(I4, "models", "i4")
RESULTS = os.path.join(I4, "results")
CORPUS = os.path.join(I4, "corpus")
os.makedirs(RESULTS, exist_ok=True)

def probs(path):
    a = np.fromfile(path, dtype="<f8")
    if a.size % 2:
        raise SystemExit(f"{path}: not [n,2]")
    return a.reshape(-1, 2)[:, 1]

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def write_probs(path, p):
    out = np.empty((len(p), 2), dtype="<f8")
    out[:, 0] = 1.0 - p
    out[:, 1] = p
    out.tofile(path)

def main():
    pv = probs(os.path.join(MODEL, "val.prob.bin"))
    pt = probs(os.path.join(MODEL, "test.prob.bin"))
    yv = np.load(os.path.join(CORPUS, "val.meta.npz"))["label"].astype(np.int32)
    yt = np.load(os.path.join(CORPUS, "test.meta.npz"))["label"].astype(np.int32)
    if len(pv) != len(yv) or len(pt) != len(yt):
        raise SystemExit("probability/label length mismatch")
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(logit(pv).reshape(-1, 1), yv)
    a = float(cal.coef_[0, 0])
    b = float(cal.intercept_[0])
    pt_cal = sigmoid(a * logit(pt) + b)
    pv_cal_insample = sigmoid(a * logit(pv) + b)

    write_probs(os.path.join(RESULTS, "test.raw.prob.bin"), pt)
    write_probs(os.path.join(RESULTS, "test.prob.bin"), pt_cal)
    write_probs(os.path.join(RESULTS, "val.prob.bin"), pv)
    write_probs(os.path.join(RESULTS, "val.calibrated.insample.prob.bin"), pv_cal_insample)
    info = {
        "method": "Platt scaling on logit(validation probability)",
        "fit_split": "val 2022",
        "val_rows": int(len(yv)),
        "val_positives": int(yv.sum()),
        "slope": a,
        "intercept": b,
        "test_probabilities": "results/test.prob.bin (calibrated)",
        "test_raw_probabilities": "results/test.raw.prob.bin",
        "note": "AP and ROC-AUC are invariant to this monotone calibration; Brier, log-loss and ECE use calibrated test probabilities.",
    }
    with open(os.path.join(RESULTS, "calibration.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print("test mean raw", float(pt.mean()), "calibrated", float(pt_cal.mean()))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cell-cluster bootstrap confidence intervals for calibrated test metrics."""
import os, json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

HERE = os.path.dirname(os.path.abspath(__file__)); I4 = os.path.dirname(HERE)
RESULTS = os.path.join(I4, "results"); CORPUS = os.path.join(I4, "corpus")

def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        q = idx == b
        if q.any(): out += q.mean() * abs(p[q].mean() - y[q].mean())
    return float(out)

def met(y, p):
    return (float(average_precision_score(y, p)), float(roc_auc_score(y, p)),
            float(brier_score_loss(y, p)), float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))),
            ece(y, p))

def main():
    p = np.fromfile(os.path.join(RESULTS, "test.prob.bin"), dtype="<f8").reshape(-1, 2)[:, 1]
    m = np.load(os.path.join(CORPUS, "test.meta.npz"))
    y = m["label"].astype(np.int32); cell = m["cell"].astype(np.int64)
    uniq, inv = np.unique(cell, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    obs = met(y, p)
    rng = np.random.default_rng(20261022)
    boot = []
    for _ in range(500):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows[k] for k in pick])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        boot.append(met(yy, p[idx]))
    b = np.asarray(boot)
    lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
    out = {
        "method": "cell-cluster bootstrap, 500 resamples, percentiles 2.5/97.5",
        "n_cells": int(len(uniq)), "n_rows": int(len(y)), "n_boot": int(len(b)),
        "metrics": {
            "ap": {"estimate": obs[0], "lo": float(lo[0]), "hi": float(hi[0])},
            "roc_auc": {"estimate": obs[1], "lo": float(lo[1]), "hi": float(hi[1])},
            "brier": {"estimate": obs[2], "lo": float(lo[2]), "hi": float(hi[2])},
            "logloss": {"estimate": obs[3], "lo": float(lo[3]), "hi": float(hi[3])},
            "ece_10": {"estimate": obs[4], "lo": float(lo[4]), "hi": float(hi[4])},
        },
    }
    with open(os.path.join(RESULTS, "i4_test_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

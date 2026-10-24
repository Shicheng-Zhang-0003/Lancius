#!/usr/bin/env python3
"""I4 baselines: prevalence, cell climatology, logistic regression, LightGBM.

The corpus is already median-imputed; this script applies the same train-only
z-score/clip as examples/train_i4.c, then evaluates on the temporal validation
(2022) and test (2023-2024) splits.  No test rows are used for fitting or
early stopping.
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss, precision_recall_curve)
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
CORPUS = os.path.join(I4, "corpus")
RESULTS = os.path.join(I4, "results")
os.makedirs(RESULTS, exist_ok=True)

def read_hdr():
    d = {}
    with open(os.path.join(CORPUS, "corpus.hdr")) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                d[p[0]] = int(p[1])
    return d

def load_split(name, feat, n):
    X = np.fromfile(os.path.join(CORPUS, f"{name}.X.bin"), dtype="<f8", count=n*feat)
    L = np.fromfile(os.path.join(CORPUS, f"{name}.L.bin"), dtype="<i4", count=n)
    m = np.load(os.path.join(CORPUS, f"{name}.meta.npz"))
    if X.size != n*feat or L.size != n:
        raise SystemExit(f"{name}: size mismatch")
    return X.reshape(n, feat), L.astype(np.int32), m

def ece(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins+1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins-1)
    out = 0.0
    for b in range(bins):
        q = idx == b
        if not q.any(): continue
        out += q.mean() * abs(p[q].mean() - y[q].mean())
    return float(out)

def at_precision(y, p, target=0.8):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec[:-1] >= target
    return float(rec[:-1][ok].max()) if ok.any() else 0.0

def top_fraction(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    order = np.argsort(-p, kind="stable")[:k]
    return float(y[order].mean())

def metrics(y, p):
    y = y.astype(np.int32); p = np.asarray(p, dtype=np.float64)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-12, 1-1e-12))),
        "ece_10": ece(y, p),
        "recall_at_80pct_precision": at_precision(y, p, 0.8),
        "precision_top_1pct": top_fraction(y, p, 0.01),
        "precision_top_5pct": top_fraction(y, p, 0.05),
        "mean_prob": float(p.mean()),
    }

def norm(X, mean, std):
    Z = (X - mean) / std
    return np.clip(Z, -8.0, 8.0)

def main():
    hdr = read_hdr()
    feat = hdr["FEAT"]
    Xtr, ytr, mtr = load_split("train", feat, hdr["NTRAIN"])
    Xva, yva, mva = load_split("val", feat, hdr["NVAL"])
    Xte, yte, mte = load_split("test", feat, hdr["NTEST"])
    st = np.load(os.path.join(CORPUS, "standardizer.npz"))
    mean, std = st["mean"], st["std"]
    Ztr, Zva, Zte = norm(Xtr, mean, std), norm(Xva, mean, std), norm(Xte, mean, std)
    print("train", Xtr.shape, "pos", int(ytr.sum()), "val pos", int(yva.sum()), "test pos", int(yte.sum()))

    out = {"features": feat, "splits": {}, "baselines": {}}

    # prevalence baseline
    prev = float(ytr.mean())
    for split, y in [("val", yva), ("test", yte)]:
        out["baselines"].setdefault("prevalence", {})[split] = metrics(y, np.full(len(y), prev))
    out["baselines"]["prevalence"]["train_prevalence"] = prev

    # cell climatology from train years only
    rate = np.zeros(hdr.get("NCELL", 417*200), dtype=np.float64)
    cnt = np.zeros_like(rate)
    np.add.at(rate, mtr["cell"], ytr)
    np.add.at(cnt, mtr["cell"], 1)
    rate = np.divide(rate, cnt, out=np.full_like(rate, prev), where=cnt > 0)
    for split, meta, y in [("val", mva, yva), ("test", mte, yte)]:
        p = rate[meta["cell"]]
        out["baselines"].setdefault("cell_climatology", {})[split] = metrics(y, p)
        np.save(os.path.join(RESULTS, f"cell_climatology_{split}_prob.npy"), p)

    # logistic regression
    t0 = time.time()
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, n_jobs=-1)
    lr.fit(Ztr, ytr)
    for split, Z, y in [("val", Zva, yva), ("test", Zte, yte)]:
        p = lr.predict_proba(Z)[:, 1]
        out["baselines"].setdefault("logistic", {})[split] = metrics(y, p)
        np.save(os.path.join(RESULTS, f"logistic_{split}_prob.npy"), p)
    out["baselines"]["logistic"]["seconds"] = time.time() - t0

    # LightGBM, early stopping on validation average precision
    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.03,
        num_leaves=31, min_child_samples=100, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8,
        random_state=20261022, verbosity=-1)
    model.fit(Ztr, ytr, eval_set=[(Zva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    for split, Z, y in [("val", Zva, yva), ("test", Zte, yte)]:
        p = model.predict_proba(Z)[:, 1]
        out["baselines"].setdefault("lightgbm", {})[split] = metrics(y, p)
        np.save(os.path.join(RESULTS, f"lightgbm_{split}_prob.npy"), p)
    out["baselines"]["lightgbm"]["best_iteration"] = int(model.best_iteration_ or model.n_estimators)
    out["baselines"]["lightgbm"]["seconds"] = time.time() - t0

    out["splits"] = {
        "train": {"rows": int(len(ytr)), "positives": int(ytr.sum()), "prevalence": float(ytr.mean())},
        "val": {"rows": int(len(yva)), "positives": int(yva.sum()), "prevalence": float(yva.mean())},
        "test": {"rows": int(len(yte)), "positives": int(yte.sum()), "prevalence": float(yte.mean())},
    }
    with open(os.path.join(RESULTS, "baselines.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["baselines"], indent=2))
    print("wrote", os.path.join(RESULTS, "baselines.json"))

if __name__ == "__main__":
    main()

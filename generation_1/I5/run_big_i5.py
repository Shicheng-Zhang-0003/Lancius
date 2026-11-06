#!/usr/bin/env python3
"""1I5 head-to-head evaluation: score an I5 .lancius model on its test corpus.

Binary classification (NCLASS=2). Parses the .lancius v2 file directly,
runs the forward pass in NumPy, reports accuracy, MAE, MSE, ROC-AUC,
precision/recall at thresholds, and confusion matrix.

USAGE
    python3 generation_1/I5/run_big_i5.py [model.lancius] [corpus_dir]
      defaults: newest .lancius in generation_1/I5/models/, data_i5_real
"""
import sys, os, glob, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools", "onnx"))
sys.path.insert(0, REPO)

import numpy as np
from export_lancius_onnx import parse_lancius

OP_INPUT = 1
OP_TANH = 38
OP_MATMUL = 6
OP_ADD = 3

def read_hdr(dir):
    hdr = {}
    with open(os.path.join(dir, "corpus.hdr")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                hdr[parts[0]] = int(parts[1])
    return hdr

def load_split(dir, split, feat):
    X = np.fromfile(os.path.join(dir, f"{split}.X.bin"), dtype=np.float64).reshape(-1, feat)
    L = np.fromfile(os.path.join(dir, f"{split}.L.bin"), dtype=np.int32)
    S = np.fromfile(os.path.join(dir, f"{split}.S.bin"), dtype=np.int32)
    A = np.fromfile(os.path.join(dir, f"{split}.A.bin"), dtype=np.int32)
    return X, L, S, A

def read_musd(path):
    mu, sd = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MU "):
                mu.append(float(line[3:]))
            elif line.startswith("SD "):
                sd.append(float(line[3:]))
    mu = np.asarray(mu); sd = np.asarray(sd)
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd

def roc_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return (greater + 0.5 * equal) / (len(pos) * len(neg))

def forward(model_nodes, X, FEAT):
    """Manual forward pass: FEAT->H1(tanh)->H2(tanh)->NCLASS(logits)."""
    # Extract weight/bias matrices by shape.
    Ws = {}
    bs = {}
    for n in model_nodes:
        if n.get("op") != OP_INPUT:
            continue
        if not n.get("has_w"):
            continue
        w = n.get("weights")
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float64)
        sh = tuple(n["shape"][:n["ndim"]])
        expected = 1
        for d in sh:
            expected *= d
        if w.size != expected:
            continue
        w = w.reshape(sh)
        if len(sh) == 2 and sh[0] > 1 and sh[1] > 1:
            Ws[sh] = w
        elif len(sh) == 2 and sh[0] == 1:
            bs[sh[1]] = w.reshape(-1)

    # Identify shapes: W1=[FEAT,H1], W2=[H1,H2], W3=[H2,NCLASS]
    w1_sh = None; w2_sh = None; w3_sh = None
    for sh in Ws:
        if len(sh) == 2 and sh[0] == FEAT:
            w1_sh = sh
            break
    if w1_sh:
        H1 = w1_sh[1]
        for sh in Ws:
            if len(sh) == 2 and sh[0] == H1:
                w2_sh = sh
                break
    if w2_sh:
        H2 = w2_sh[1]
        for sh in Ws:
            if len(sh) == 2 and sh[0] == H2:
                w3_sh = sh
                break
    # Fallback: sort matrices by total size
    if not (w1_sh and w2_sh and w3_sh):
        mats = sorted([(s, Ws[s]) for s in Ws], key=lambda t: t[0][0]*t[0][1])
        if len(mats) >= 3:
            w1_sh, W1 = mats[0]
            w2_sh, W2 = mats[1]
            w3_sh, W3 = mats[2]
        else:
            raise SystemExit(f"could not identify weight matrices from {len(mats)} found")
    else:
        W1, W2, W3 = Ws[w1_sh], Ws[w2_sh], Ws[w3_sh]

    b1 = bs.get(W1.shape[1], np.zeros(W1.shape[1]))
    b2 = bs.get(W2.shape[1], np.zeros(W2.shape[1]))
    b3 = bs.get(W3.shape[1], np.zeros(W3.shape[1]))

    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    z = h2 @ W3 + b3
    return z

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    corpus_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data_i5_real")

    if model_path is None:
        cands = sorted(glob.glob(os.path.join(HERE, "models", "*.lancius")),
                       key=os.path.getmtime)
        if cands:
            model_path = cands[-1]
        else:
            raise SystemExit("no .lancius model found in generation_1/I5/models/")

    if not os.path.exists(model_path):
        raise SystemExit(f"model not found: {model_path}")
    if not os.path.exists(os.path.join(corpus_dir, "corpus.hdr")):
        raise SystemExit(f"corpus.hdr not found in {corpus_dir}")

    hdr = read_hdr(corpus_dir)
    FEAT = hdr["FEAT"]; NCLASS = hdr["NCLASS"]
    if NCLASS != 2:
        raise SystemExit(f"run_big_i5 expects NCLASS=2, got {NCLASS}")

    Xte, Lte, Ste, Ate = load_split(corpus_dir, "test", FEAT)

    # pinned normalizer: prefer I5's own gen1_mu_sd.txt, fall back to train fit
    musd_path = os.path.join(HERE, "gen1_mu_sd.txt")
    if os.path.exists(musd_path):
        mu, sd = read_musd(musd_path)
        if len(mu) != FEAT:
            mu, sd = None, None
    else:
        mu, sd = None, None
    if mu is None:
        Xtr, _, _, _ = load_split(corpus_dir, "train", FEAT)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[~(sd > 1e-12)] = 1.0
        print("WARN: no pinned MU/SD; using train-fit stats")

    Xs = (Xte - mu) / sd

    d = parse_lancius(model_path)
    nodes = d["nodes"] if isinstance(d, dict) and "nodes" in d else d
    z = forward(nodes, Xs, FEAT)

    # softmax prob for class 1
    z = z - z.max(axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz[:, 1] / expz.sum(axis=1)

    preds = (probs >= 0.5).astype(int)
    y = Lte

    acc = (preds == y).mean()
    mae = np.abs(preds - y).mean()
    mse = ((preds - y) ** 2).mean()

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = roc_auc(y, probs)

    print("================================================")
    print(f"  1I5 evaluation: {os.path.basename(model_path)}")
    print(f"  corpus: {corpus_dir} (FEAT={FEAT}, NCLASS={NCLASS})")
    print(f"  test rows: {len(Xte)}  positives: {int(y.sum())}  prevalence: {y.mean():.6f}")
    print("================================================")
    print(f"  accuracy : {acc:.6f}")
    print(f"  MAE      : {mae:.6f}")
    print(f"  MSE      : {mse:.6f}")
    print(f"  precision: {prec:.6f}")
    print(f"  recall   : {rec:.6f}")
    print(f"  F1       : {f1:.6f}")
    print(f"  ROC-AUC  : {auc:.6f}")
    print("------------------------------------------------")
    print(f"  confusion: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("================================================")

    print("\n  threshold sweep (precision / recall / F1):")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (probs >= thr).astype(int)
        t2 = int(((p == 1) & (y == 1)).sum())
        f2 = int(((p == 1) & (y == 0)).sum())
        n2 = int(((p == 0) & (y == 1)).sum())
        pr2 = t2 / (t2 + f2) if (t2 + f2) > 0 else 0.0
        rc2 = t2 / (t2 + n2) if (t2 + n2) > 0 else 0.0
        f2_ = 2 * pr2 * rc2 / (pr2 + rc2) if (pr2 + rc2) > 0 else 0.0
        print(f"    {thr:.1f}   prec={pr2:.4f}  rec={rc2:.4f}  f1={f2_:.4f}")

    print(f"\n  model file: {model_path} ({os.path.getsize(model_path)} bytes)")
    print("Done.")

if __name__ == "__main__":
    main()

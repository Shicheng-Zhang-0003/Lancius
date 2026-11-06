#!/usr/bin/env python3
"""Generate a tiny synthetic corpus in the exact binary format train_i5 expects.

Output layout (all little-endian native struct):
  corpus.hdr       text: "KEY value" lines (FEAT, NCLASS, NTRAIN, NVAL, NTEST)
  {split}.X.bin    float64 array, shape (n, FEAT)
  {split}.L.bin    int32   array, shape (n,)          labels in [0, NCLASS)
  {split}.S.bin    int32   array, shape (n,)          split/group id (unused by trainer, must exist)
  {split}.A.bin    int32   array, shape (n,)          auxiliary id (unused by trainer, must exist)

The synthetic task: 2 classes, 10 features. Class 1 has a shifted mean on the
first 3 features so the MLP can actually learn something (not pure noise).
Small enough to train in seconds; large enough to show the full pipeline.
"""
import os
import struct
import sys

import random
random.seed(20261026)

OUTDIR = "data_synthetic"
FEAT = 10
NCLASS = 2
NTRAIN = 300
NVAL = 80
NTEST = 80
POS_FRAC = 0.15  # ~15% positives (rare-event-ish, like fire)

os.makedirs(OUTDIR, exist_ok=True)


def sigmoid(x):
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + __import__("math").exp(-x))


def gen(n, feat, pos_frac):
    X = []
    L = []
    S = []
    A = []
    for i in range(n):
        is_pos = random.random() < pos_frac
        row = [random.gauss(0, 1) for _ in range(feat)]
        if is_pos:
            # shift first 3 features upward so there's a learnable signal
            row[0] += 1.5
            row[1] += 1.5
            row[2] += 1.0
        label = 1 if is_pos else 0
        X.append(row)
        L.append(label)
        S.append(i % 5)   # 5 spatial-ish groups
        A.append(i)        # row id
    return X, L, S, A


def write_split(split, X, L, S, A, feat, nclass):
    n = len(X)
    with open(os.path.join(OUTDIR, f"{split}.X.bin"), "wb") as f:
        for row in X:
            f.write(struct.pack(f"<{feat}d", *row))
    with open(os.path.join(OUTDIR, f"{split}.L.bin"), "wb") as f:
        for v in L:
            f.write(struct.pack("<i", v))
    with open(os.path.join(OUTDIR, f"{split}.S.bin"), "wb") as f:
        for v in S:
            f.write(struct.pack("<i", v))
    with open(os.path.join(OUTDIR, f"{split}.A.bin"), "wb") as f:
        for v in A:
            f.write(struct.pack("<i", v))


Xtr, Ltr, Str, Atr = gen(NTRAIN, FEAT, POS_FRAC)
Xva, Lva, Sva, Ava = gen(NVAL, FEAT, POS_FRAC)
Xte, Lte, Ste, Ate = gen(NTEST, FEAT, POS_FRAC)

write_split("train", Xtr, Ltr, Str, Atr, FEAT, NCLASS)
write_split("val", Xva, Lva, Sva, Ava, FEAT, NCLASS)
write_split("test", Xte, Lte, Ste, Ate, FEAT, NCLASS)

with open(os.path.join(OUTDIR, "corpus.hdr"), "w") as f:
    f.write(f"FEAT {FEAT}\n")
    f.write(f"NCLASS {NCLASS}\n")
    f.write(f"NTRAIN {NTRAIN}\n")
    f.write(f"NVAL {NVAL}\n")
    f.write(f"NTEST {NTEST}\n")

npos_tr = sum(1 for v in Ltr if v == 1)
npos_va = sum(1 for v in Lva if v == 1)
npos_te = sum(1 for v in Lte if v == 1)

print(f"Wrote synthetic corpus to {OUTDIR}/")
print(f"  FEAT={FEAT} NCLASS={NCLASS}")
print(f"  train={NTRAIN} (pos={npos_tr})  val={NVAL} (pos={npos_va})  test={NTEST} (pos={npos_te})")
print()
print("Next:")
print(f"  ./build/bin/train_i5 {OUTDIR} 64 32 20 64")
print(f"  ./build/bin/train_i5 {OUTDIR} 64 32 20 64 --export-lancius {OUTDIR}/i5.lancius --verify-lancius")

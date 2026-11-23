#!/usr/bin/env python3
"""Split an I6 corpus into family-specific subdirectories for expert training.

Usage:
    python3 generation_1/I6/split_corpus_by_family.py [corpus_dir]

Reads {train,test}.{X,L,S,A}.bin from corpus_dir and splits rows by the
FAM_ID feature (index 58) into per-family subdirectories.  Each subdirectory
gets a corpus.hdr with the combined train/test counts, which is what
train_i6.c reads.

The family IDs match verifier5.FAMILIES:
    0 algebra, 1 calculus, 2 linear_algebra, 3 number_theory
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = 65
FAM_ID_INDEX = 58  # 0-indexed position of family ID in feature vector

FAMILIES = [
    (0, "algebra"),
    (1, "calculus"),
    (2, "linear_algebra"),
    (3, "number_theory"),
]


def load_split(corpus_dir, split_name):
    X = np.fromfile(os.path.join(corpus_dir, f"{split_name}.X.bin"),
                    dtype=np.float64).reshape(-1, FEAT)
    L = np.fromfile(os.path.join(corpus_dir, f"{split_name}.L.bin"),
                    dtype=np.int32)
    S = np.fromfile(os.path.join(corpus_dir, f"{split_name}.S.bin"),
                    dtype=np.int32)
    A = np.fromfile(os.path.join(corpus_dir, f"{split_name}.A.bin"),
                    dtype=np.int32)
    return X, L, S, A


def main():
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus")
    train = load_split(corpus_dir, "train")
    test = load_split(corpus_dir, "test")
    print(f"corpus: {corpus_dir}")
    print(f"train: {len(train[0])} rows, test: {len(test[0])} rows")

    for fam_idx, fam_name in FAMILIES:
        out_dir = os.path.join(corpus_dir, fam_name)
        os.makedirs(out_dir, exist_ok=True)
        ntr = nte = 0
        for split_name, (X, L, S, A) in (("train", train), ("test", test)):
            mask = X[:, FAM_ID_INDEX] == fam_idx
            n = int(mask.sum())
            X[mask].tofile(os.path.join(out_dir, f"{split_name}.X.bin"))
            L[mask].tofile(os.path.join(out_dir, f"{split_name}.L.bin"))
            S[mask].tofile(os.path.join(out_dir, f"{split_name}.S.bin"))
            A[mask].tofile(os.path.join(out_dir, f"{split_name}.A.bin"))
            if split_name == "train":
                ntr = n
            else:
                nte = n

        with open(os.path.join(out_dir, "corpus.hdr"), "w") as f:
            f.write(f"FEAT {FEAT}\n")
            f.write("NCLASS 5\n")
            f.write(f"NTRAIN {ntr}\n")
            f.write(f"NTEST {nte}\n")

        print(f"  {fam_name}: train {ntr}, test {nte}")

    print("Corpus split complete. Subdirectories ready for per-family training.")


if __name__ == "__main__":
    main()

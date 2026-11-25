#!/usr/bin/env python3
"""Split an I6 corpus into family-specific subdirectories for expert training.

Usage:
    python3 generation_1/I6/split_corpus_by_family.py [corpus_dir]

Reads {train,val,test}.{X,L,S,A}.bin from corpus_dir and splits rows by the
algorithm ID in A, using gen_data_i6's canonical algorithm-to-family mapping.
The sparse I6 feature schema intentionally omits the old FAM_ID input. The
source corpus must hold out whole algorithms and carry its train-only
mu_sd.txt normalizer.

The family IDs match verifier5.FAMILIES:
    0 algebra, 1 calculus, 2 linear_algebra, 3 number_theory
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FEAT = 65
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
from gen_data_i6 import ALL

FAMILIES = ["algebra", "calculus", "linear_algebra", "number_theory"]
ALGO_IDS_BY_FAMILY = {
    family: np.asarray([i for i, algo in enumerate(ALL)
                        if algo.family == family], dtype=np.int32)
    for family in FAMILIES
}


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
    splits = {name: load_split(corpus_dir, name)
              for name in ("train", "val", "test")}
    print(f"corpus: {corpus_dir}")
    print("rows: " + ", ".join(
        f"{name} {len(data[0])}" for name, data in splits.items()))

    for fam_name in FAMILIES:
        out_dir = os.path.join(corpus_dir, fam_name)
        os.makedirs(out_dir, exist_ok=True)
        counts = {}
        for split_name, (X, L, S, A) in splits.items():
            mask = np.isin(A, ALGO_IDS_BY_FAMILY[fam_name])
            n = int(mask.sum())
            X[mask].tofile(os.path.join(out_dir, f"{split_name}.X.bin"))
            L[mask].tofile(os.path.join(out_dir, f"{split_name}.L.bin"))
            S[mask].tofile(os.path.join(out_dir, f"{split_name}.S.bin"))
            A[mask].tofile(os.path.join(out_dir, f"{split_name}.A.bin"))
            counts[split_name] = n

        with open(os.path.join(out_dir, "corpus.hdr"), "w") as f:
            f.write(f"FEAT {FEAT}\n")
            f.write("NCLASS 5\n")
            f.write(f"NTRAIN {counts['train']}\n")
            f.write(f"NVAL {counts['val']}\n")
            f.write(f"NTEST {counts['test']}\n")

        print(f"  {fam_name}: train {counts['train']}, val {counts['val']}, "
              f"test {counts['test']}")

    print("Corpus split complete. Subdirectories ready for per-family training.")


if __name__ == "__main__":
    main()

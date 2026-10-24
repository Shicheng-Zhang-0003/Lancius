#!/usr/bin/env python3
"""Build the I4 2 km cell-year corpus from the frozen feature stack.

Layout (Lancius convention):
  corpus.hdr
  <split>.X.bin   float64 [n, FEAT]
  <split>.L.bin   int32   [n]       binary label
  <split>.S.bin   int32   [n]       per-step flag (always 0 here)
  <split>.A.bin   int32   [n]       auxiliary id (always 0 here)
  <split>.meta.npz year, cell, label, fold
  feature_manifest.json
  standardizer.npz mean, std, median
  spatial_folds.npz fold [height,width]
  metadata.json

Missing values are filled with training medians only.  The trainer computes
its own z-score mean/std from the train split; standardizer.npz is for Python
baselines and auditability.
"""
import os, sys, json, hashlib, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
DATA = os.path.join(I4, "data")
OUT = os.path.join(I4, "corpus")
SPEC = os.path.join(I4, "targets", "i4_fire_next_year_v0.1.json")
LABEL_THRESH = 0.01
TRAIN_YEARS = list(range(2016, 2022))
VAL_YEARS = [2022]
TEST_YEARS = [2023, 2024]

GROUPS = [
    ("static", os.path.join(DATA, "static_features_2km.npz"), None),
    ("climate", os.path.join(DATA, "climate_features_2km.npz"), "year"),
    ("modis", os.path.join(DATA, "modis_features_2km.npz"), "year"),
    ("invasion", os.path.join(DATA, "gbif_features_2km.npz"), "year"),
    ("fire_history", os.path.join(DATA, "fire_history_features_2km.npz"), "year"),
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def group_keys(path, year=None):
    z = np.load(path)
    if year is None:
        return sorted(z.files)
    suffix = f"_{year}"
    ks = sorted(k for k in z.files if k.endswith(suffix))
    return [k[:-len(suffix)] for k in ks]

def load_group(path, names, year):
    z = np.load(path)
    if year is None:
        return [z[n] for n in names]
    return [z[f"{n}_{year}"] for n in names]

def row_for_group(path, names, year, cell_idx):
    return np.column_stack([
        np.asarray(a).reshape(-1)[cell_idx] for a in load_group(path, names, year)
    ]).astype("float32")

def main():
    os.makedirs(OUT, exist_ok=True)
    g = C.grid()
    h, w = g["height"], g["width"]
    ncell = h * w
    zlab = np.load(os.path.join(DATA, "labels_2km.npz"))
    lab = {int(k): np.asarray(zlab[k]).reshape(-1) for k in zlab.files}
    wc = np.load(os.path.join(DATA, "worldcover_frac_2km.npz"))
    rangeland = sum(wc[str(c)] for c in C.RANGELAND)
    inside = C.study_mask(g)
    mask = (rangeland >= 0.2) & inside
    cell_idx = np.flatnonzero(mask.ravel())
    n = len(cell_idx)
    print("mask cells", n, "of", ncell)

    names_by_group = {}
    for gname, path, mode in GROUPS:
        if not os.path.exists(path):
            raise SystemExit(f"missing feature file: {path} (finish extraction first)")
        names = group_keys(path, None)
        if mode == "year":
            names = group_keys(path, C.FEATURE_YEARS[0])
            for y in C.FEATURE_YEARS:
                got = group_keys(path, y)
                if got != names:
                    raise SystemExit(f"{path}: feature names differ in {y}")
        names_by_group[gname] = names
        print(gname, len(names), "features")

    feat_names = []
    feat_group = []
    for gname, _, _ in GROUPS:
        for nm in names_by_group[gname]:
            feat_names.append(f"{gname}/{nm}")
            feat_group.append(gname)
    FEAT = len(feat_names)
    print("FEAT", FEAT)
    if FEAT != 77:
        print("WARNING: target spec expects 77 features", flush=True)

    all_years = TRAIN_YEARS + VAL_YEARS + TEST_YEARS
    X_parts, meta_parts = [], []
    for y in all_years:
        t = y - 1
        cols = []
        for gname, path, mode in GROUPS:
            names = names_by_group[gname]
            if mode is None:
                cols.append(row_for_group(path, names, None, cell_idx))
            else:
                cols.append(row_for_group(path, names, t, cell_idx))
        F = np.concatenate(cols, axis=1)
        if F.shape != (n, FEAT):
            raise SystemExit(f"year {y}: shape {F.shape}, expected {(n, FEAT)}")
        X_parts.append(F)
        meta_parts.append((y, cell_idx, (lab[y][cell_idx] >= LABEL_THRESH).astype("int8")))
        print("year", y, "feature_year", t, "positives", int(meta_parts[-1][2].sum()), flush=True)
    X = np.vstack(X_parts)
    years = np.concatenate([np.full(n, y, dtype=np.int16) for y in all_years])
    cells = np.concatenate([m[1] for m in meta_parts]).astype(np.int32)
    labels = np.concatenate([m[2] for m in meta_parts]).astype(np.int32)
    split_idx = {}
    pos = 0
    for y in all_years:
        split_idx[y] = slice(pos, pos + n)
        pos += n
    tr_sel = np.concatenate([np.arange(split_idx[y].start, split_idx[y].stop) for y in TRAIN_YEARS])
    va_sel = np.concatenate([np.arange(split_idx[y].start, split_idx[y].stop) for y in VAL_YEARS])
    te_sel = np.concatenate([np.arange(split_idx[y].start, split_idx[y].stop) for y in TEST_YEARS])

    # medians from train rows only, before imputation
    finite = np.isfinite(X)
    med = np.zeros(FEAT, dtype="float64")
    for j in range(FEAT):
        v = X[tr_sel, j]
        v = v[np.isfinite(v)]
        med[j] = float(np.median(v)) if v.size else 0.0
    for j in range(FEAT):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = med[j]
    if not np.isfinite(X).all():
        raise SystemExit("non-finite values remain after imputation")

    coverage = finite.mean(axis=0)
    group_coverage = {}
    for gname, _, _ in GROUPS:
        js = [i for i, gg in enumerate(feat_group) if gg == gname]
        group_coverage[gname] = float(coverage[js].mean())
    print("coverage by group", group_coverage)

    mean = X[tr_sel].mean(axis=0, dtype="float64")
    std = X[tr_sel].std(axis=0, dtype="float64")
    std[std < 1e-12] = 1.0

    # provisional contiguous 5-block spatial split by x quantile over study cells
    t = g["transform"]
    cols_ = np.arange(w)
    x = t.c + (cols_ + 0.5) * t.a
    x2 = np.broadcast_to(x, (h, w)).ravel()
    order = np.argsort(x2[cell_idx], kind="stable")
    fold_flat = np.full(ncell, -1, dtype=np.int8)
    fold_cell = np.empty(n, dtype=np.int8)
    fold_cell[order] = np.floor(5.0 * np.arange(n) / n).astype(np.int8)
    fold_flat[cell_idx] = fold_cell
    fold = fold_flat.reshape(h, w)
    cell_fold = fold_flat[cell_idx]

    rows = {
        "train": (tr_sel, TRAIN_YEARS),
        "val": (va_sel, VAL_YEARS),
        "test": (te_sel, TEST_YEARS),
    }
    hdr = [f"FEAT {FEAT}", "NCLASS 2"]
    meta = {
        "id": "i4_fire_next_year_v0.1",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "target_spec": os.path.relpath(SPEC, I4),
        "target_spec_sha256": sha256(SPEC),
        "feature_count": FEAT,
        "mask_cells": int(n),
        "study_cells_inside_bbox": int(inside.sum()),
        "grid_cells": int(ncell),
        "label_threshold": LABEL_THRESH,
        "feature_years": [int(C.FEATURE_YEARS[0]), int(C.FEATURE_YEARS[-1])],
        "label_years": [2016, 2024],
        "group_counts": {gname: len(names_by_group[gname]) for gname, _, _ in GROUPS},
        "group_coverage": group_coverage,
        "splits": {},
        "spatial_folds": {"n": 5, "method": "contiguous x-quantile blocks", "status": "provisional"},
    }
    for split, (sel, yrs) in rows.items():
        Xs = X[sel].astype("<f8", copy=False)
        Ls = labels[sel].astype("<i4", copy=False)
        Ss = np.zeros(len(sel), dtype="<i4")
        As = np.zeros(len(sel), dtype="<i4")
        Xs.tofile(os.path.join(OUT, f"{split}.X.bin"))
        Ls.tofile(os.path.join(OUT, f"{split}.L.bin"))
        Ss.tofile(os.path.join(OUT, f"{split}.S.bin"))
        As.tofile(os.path.join(OUT, f"{split}.A.bin"))
        fold_sel = fold_flat[cells[sel]]
        np.savez_compressed(os.path.join(OUT, f"{split}.meta.npz"),
                            year=years[sel], cell=cells[sel], label=labels[sel], fold=fold_sel)
        p = int(labels[sel].sum())
        meta["splits"][split] = {
            "years": [int(yrs[0]), int(yrs[-1])],
            "rows": int(len(sel)),
            "positives": p,
            "prevalence": float(p / len(sel)),
            "fold_counts": {str(f): int((fold_flat[cells[sel]] == f).sum()) for f in range(5)},
        }
        hdr.append(f"N{ {'train':'TRAIN','val':'VAL','test':'TEST'}[split] } {len(sel)}")
        print(split, len(sel), "rows", p, "positives", flush=True)

    with open(os.path.join(OUT, "corpus.hdr"), "w") as f:
        f.write("\n".join(hdr) + "\n")

    manifest = {
        "total": FEAT,
        "features": [
            {"index": i, "name": feat_names[i], "group": feat_group[i],
             "median": float(med[i]), "mean": float(mean[i]), "std": float(std[i]),
             "coverage": float(coverage[i])}
            for i in range(FEAT)
        ],
        "imputation": "training median",
        "stats_source": "train label years 2016-2021",
    }
    with open(os.path.join(OUT, "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "standardizer.npz"), mean=mean, std=std, median=med)
    np.savez_compressed(os.path.join(OUT, "spatial_folds.npz"), fold=fold, cell_fold=cell_fold)
    with open(os.path.join(OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote corpus ->", OUT)
    print(json.dumps(meta["splits"], indent=2))

if __name__ == "__main__":
    main()

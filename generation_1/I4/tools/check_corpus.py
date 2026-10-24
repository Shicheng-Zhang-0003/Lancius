#!/usr/bin/env python3
"""Fail-loud checks for the I4 corpus before training."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE=os.path.dirname(os.path.abspath(__file__)); I4=os.path.dirname(HERE)
C=os.path.join(I4,"corpus")
EXPECT_TRAIN=set(range(2016,2022)); EXPECT_VAL={2022}; EXPECT_TEST={2023,2024}

def hdr():
    d={}
    for line in open(os.path.join(C,"corpus.hdr")):
        p=line.split()
        if len(p)==2: d[p[0]]=int(p[1])
    return d

def main():
    h=hdr()
    feat=h["FEAT"]; nclass=h["NCLASS"]
    assert nclass==2, nclass
    qc={"feature_count":feat,"nclass":nclass,"splits":{},"errors":[]}
    man=json.load(open(os.path.join(C,"feature_manifest.json")))
    assert man["total"]==feat and len(man["features"])==feat
    names=[x["name"] for x in man["features"]]
    assert len(set(names))==feat, "duplicate feature names"
    st=np.load(os.path.join(C,"standardizer.npz"))
    for k in ("mean","std","median"):
        assert np.isfinite(st[k]).all(), f"{k} non-finite"
    assert (st["std"]>0).all(), "zero std"
    qc["group_coverage"]={}
    for g in ("static","climate","modis","invasion","fire_history"):
        vals=[x["coverage"] for x in man["features"] if x["group"]==g]
        qc["group_coverage"][g]=float(np.mean(vals)) if vals else 0.0
    qc["coverage_min"]=float(min(x["coverage"] for x in man["features"]))
    qc["coverage_mean"]=float(np.mean([x["coverage"] for x in man["features"]]))

    all_years=[]; all_cells=[]; all_labels=[]
    for split,nkey,years in [("train","NTRAIN",EXPECT_TRAIN),("val","NVAL",EXPECT_VAL),("test","NTEST",EXPECT_TEST)]:
        n=h[nkey]
        def rd(ext,dtype):
            p=os.path.join(C,f"{split}.{ext}.bin")
            sz=os.path.getsize(p)
            return np.fromfile(p,dtype=dtype)
        X=rd("X","<f8"); L=rd("L","<i4"); S=rd("S","<i4"); A=rd("A","<i4")
        assert X.size==n*feat, f"{split} X size {X.size}"
        assert L.size==n and S.size==n and A.size==n
        X=X.reshape(n,feat)
        assert np.isfinite(X).all(), f"{split} non-finite X"
        assert set(np.unique(L).tolist()) <= {0,1}, f"{split} labels"
        assert (S==0).all() and (A==0).all()
        meta=np.load(os.path.join(C,f"{split}.meta.npz"))
        assert len(meta["year"])==n and len(meta["cell"])==n and len(meta["label"])==n
        assert np.array_equal(meta["label"],L), f"{split} meta label mismatch"
        assert set(int(v) for v in meta["year"])==years, (split,set(meta["year"]))
        assert meta["cell"].min()>=0 and meta["cell"].max()<417*200
        assert meta["fold"].min()>=0 and meta["fold"].max()<5
        p=int(L.sum())
        qc["splits"][split]={
            "rows":int(n),"positives":p,"prevalence":float(p/n),
            "years":sorted(set(int(v) for v in meta["year"])),
            "x_min":float(X.min()),"x_max":float(X.max()),
            "folds":{str(f):{"rows":int((meta["fold"]==f).sum()),
                             "positives":int(L[meta["fold"]==f].sum())} for f in range(5)},
        }
        all_years.append(meta["year"]); all_cells.append(meta["cell"]); all_labels.append(L)
    years=np.concatenate(all_years); cells=np.concatenate(all_cells); labels=np.concatenate(all_labels)
    spec=json.load(open(os.path.join(I4,"targets","i4_fire_next_year_v0.1.json")))
    exp_rows=spec["splits"]["train_rows"]+spec["splits"]["val_rows"]+spec["splits"]["test_rows"]
    assert len(years)==exp_rows, (len(years),exp_rows)
    key=years.astype(np.int64)*83400+cells
    assert len(np.unique(key))==len(key), "duplicate cell-year rows"
    assert len(np.unique(cells))==spec["mask"]["cells"], len(np.unique(cells))
    qc["total_rows"]=int(len(years)); qc["total_positives"]=int(labels.sum())
    qc["cell_count"]=int(len(np.unique(cells)))
    qc["gate"]={
        "coverage_ge_70pct": bool(qc["coverage_min"]>=0.70),
        "prevalence_in_range": bool(0.005<=qc["total_positives"]/len(labels)<=0.05),
    }
    with open(os.path.join(C,"qc.json"),"w") as f: json.dump(qc,f,indent=2)
    print(json.dumps(qc,indent=2))

if __name__=="__main__": main()

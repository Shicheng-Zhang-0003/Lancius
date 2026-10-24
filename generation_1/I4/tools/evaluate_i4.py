#!/usr/bin/env python3
"""Evaluate the Lancius I4 probability dump and compare it to baselines."""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss, precision_recall_curve,
                             roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
CORPUS = os.path.join(I4, "corpus")
RESULTS = os.path.join(I4, "results")
FIG = os.path.join(I4, "figures")
os.makedirs(FIG, exist_ok=True)

def ece(y, p, bins=10):
    edges = np.linspace(0,1,bins+1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins-1)
    out=0.0; rows=[]
    for b in range(bins):
        q=idx==b
        if not q.any(): continue
        conf=float(p[q].mean()); acc=float(y[q].mean()); n=int(q.sum())
        out += q.mean()*abs(conf-acc)
        rows.append({"bin":b,"n":n,"mean_prob":conf,"fire_rate":acc})
    return float(out), rows

def at_precision(y,p,target=0.8):
    prec,rec,_=precision_recall_curve(y,p)
    ok=prec[:-1]>=target
    return float(rec[:-1][ok].max()) if ok.any() else 0.0

def top_fraction(y,p,frac):
    k=max(1,int(round(frac*len(y))))
    o=np.argsort(-p,kind="stable")[:k]
    return float(y[o].mean())

def metrics(y,p):
    y=y.astype(np.int32); p=np.asarray(p,dtype=np.float64)
    e,rows=ece(y,p)
    return {
        "n":int(len(y)),"positives":int(y.sum()),"prevalence":float(y.mean()),
        "ap":float(average_precision_score(y,p)),
        "roc_auc":float(roc_auc_score(y,p)),
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,np.clip(p,1e-12,1-1e-12))),
        "ece_10":e,
        "recall_at_80pct_precision":at_precision(y,p,0.8),
        "precision_top_1pct":top_fraction(y,p,0.01),
        "precision_top_5pct":top_fraction(y,p,0.05),
        "mean_prob":float(p.mean()),
        "calibration":rows,
    }

def load_probs(name):
    p=np.fromfile(os.path.join(RESULTS,f"{name}.prob.bin"),dtype="<f8")
    if p.size%2: raise SystemExit(f"{name}.prob.bin not [n,2]")
    return p.reshape(-1,2)[:,1]

def load_meta(name):
    return np.load(os.path.join(CORPUS,f"{name}.meta.npz"))

def plot_curves(y,p,out_prefix):
    prec,rec,_=precision_recall_curve(y,p)
    fpr,tpr,_=roc_curve(y,p)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].plot(rec,prec,lw=1.6,label=f"AP={average_precision_score(y,p):.4f}")
    ax[0].axhline(y.mean(),color="grey",ls="--",lw=0.8,label=f"prevalence={y.mean():.4f}")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("Precision-recall (test)")
    ax[0].set_xlim(0,1); ax[0].set_ylim(0,1); ax[0].legend(loc="upper right")
    ax[1].plot(fpr,tpr,lw=1.6,label=f"AUC={roc_auc_score(y,p):.4f}")
    ax[1].plot([0,1],[0,1],color="grey",ls="--",lw=0.8)
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
    ax[1].set_title("ROC (test)"); ax[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_prefix+"_pr_roc.png",dpi=160); plt.close(fig)

    edges=np.linspace(0,1,11); idx=np.clip(np.digitize(p,edges[1:-1]),0,9)
    xs=[]; ys=[]; ns=[]
    for b in range(10):
        q=idx==b
        if q.any(): xs.append(p[q].mean()); ys.append(y[q].mean()); ns.append(q.sum())
    fig,ax=plt.subplots(figsize=(4.8,4.4))
    ax.plot([0,1],[0,1],color="grey",ls="--",lw=0.8)
    ax.plot(xs,ys,"o-",lw=1.4)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed fire rate")
    ax.set_title("Calibration (test, 10 bins)")
    fig.tight_layout(); fig.savefig(out_prefix+"_calibration.png",dpi=160); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",default="test")
    args=ap.parse_args()
    os.makedirs(RESULTS,exist_ok=True)
    p=load_probs(args.name)
    meta=load_meta(args.name)
    L=meta["label"].astype(np.int32)
    if len(p)!=len(L): raise SystemExit("prob/meta length mismatch")
    m=metrics(L,p)
    m["by_year"]={}
    for y in sorted(set(int(v) for v in meta["year"])):
        q=meta["year"]==y
        yq=L[q]
        m["by_year"][str(y)]=(metrics(yq,p[q]) if 0<yq.sum()<len(yq)
                              else {"n":int(len(yq)),"positives":int(yq.sum()),
                                    "note":"single-class; AP undefined"})
    m["by_fold"]={}
    for f in sorted(set(int(v) for v in meta["fold"])):
        q=meta["fold"]==f
        yq=L[q]
        m["by_fold"][str(f)]=(metrics(yq,p[q]) if 0<yq.sum()<len(yq)
                              else {"n":int(len(yq)),"positives":int(yq.sum()),
                                    "note":"single-class; AP undefined"})
    out={"model":"lancius_i4","split":args.name,"metrics":m}
    base=os.path.join(RESULTS,"baselines.json")
    if os.path.exists(base):
        b=json.load(open(base))
        comp={"lancius_i4":m}
        for name,d in b.get("baselines",{}).items():
            if "test" in d: comp[name]=d["test"]
        out["comparison_test"]=comp
    with open(os.path.join(RESULTS,f"i4_{args.name}_metrics.json"),"w") as f:
        json.dump(out,f,indent=2,default=float)
    plot_curves(L,p,os.path.join(FIG,f"i4_{args.name}"))
    print(json.dumps(out["metrics"],indent=2,default=float))
    print("wrote",os.path.join(RESULTS,f"i4_{args.name}_metrics.json"))

if __name__=="__main__": main()

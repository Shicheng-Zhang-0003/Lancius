#!/usr/bin/env python3
"""
ALU bus — executes LAIS-F VERDICT/AGGREGATE against the real Lancius C runtime.

Every verdict is a real `lancius run <alu.lancius> --input <f64 batch>` process
mapped by the bare-metal runtime (static scheduler, arena/flat buffers). Verdict
parsing is from the CLI contract line:
    row 0: argmax=<i> (<v>) | top5: a:v b:v ...
The harness's own numpy forward is never used here.
"""
import os, re, subprocess, struct, tempfile, sys, json
from concurrent.futures import ThreadPoolExecutor

_sd = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_sd, 'paths.py')):
    _n = os.path.dirname(_sd)
    assert _n != _sd, 'paths.py not found above %s' % __file__
    _sd = _n
sys.path.insert(0, _sd)
import paths as _paths
_paths.enforce([_paths.BUS])
REPO = _paths.REPO
LANCIUS = _paths.LANCIUS
SCRATCH = "/tmp/opencode"
ROW0 = re.compile(
    r"row 0: argmax=(\d+) \((-?[\d.]+)\) \| top5: ((?:\d+:-?[\d.]+\s?)+)")

class Verdict(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    @property
    def argmax(self): return self["argmax"]
    @property
    def logits(self): return self["logits"]
    @property
    def alu(self): return self["alu"]

def _vallogits(vals):
    """top5 is printed sorted by value ('4:1.019 0:-0.015 ...'); map by class id
    back into class order, so the logits vector is index-aligned with classes."""
    d = {}
    for t in vals.split():
        cid, v = t.split(":")
        d[int(cid)] = float(v)
    out = [d.get(i, 0.0) for i in range(5)]  # 5-class model; topk=5 covers all
    return out

def verdict(alu_path, x65norm, serial=False):
    """One LAIS-F VERDICT. x65norm: normalized 65-dim feature vector."""
    if not os.path.isdir(SCRATCH): os.makedirs(SCRATCH, exist_ok=True)
    fb = os.path.join(SCRATCH, "bus_f%d_%d.bin" % (abs(hash(alu_path)) % 10**8, os.getpid()))
    xb = np_repeat(x65norm)
    with open(fb, "wb") as f:
        f.write(xb.tobytes())
    try:
        cmd = [LANCIUS, "run", alu_path, "--input", fb, "--mode", "static",
               "--topk", "5", "--show", "8"]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=180, cwd=REPO, check=True)
        m = ROW0.search(r.stdout)
        if not m:
            raise RuntimeError("no verdict line from Lancius:\n" + r.stdout[-1200:])
        argmax = int(m.group(1))
        logits = _vallogits(m.group(3))
        return Verdict(alu=os.path.basename(alu_path), argmax=argmax,
                       top=float(m.group(2)), logits=logits)
    finally:
        try: os.unlink(fb)
        except OSError: pass

def np_repeat(x):
    import numpy as np
    return np.repeat(np.asarray(x, dtype=np.float64).reshape(1, -1), 256, axis=0).reshape(-1)

def dispatch(alus, x65norm, workers=None):
    """Parallel broadcast of one feature row to N ALUs -> list[Verdict]."""
    with ThreadPoolExecutor(max_workers=workers or min(len(alus), 8)) as ex:
        return list(ex.map(lambda p: verdict(p, x65norm), alus))

def aggregate(vs, strategy="mean"):
    """LAIS-F AGGREGATE over list[Verdict]."""
    assert strategy in ("mean", "majority", "unanimous"), strategy
    import numpy as np
    L = np.mean([v.logits for v in vs], axis=0)
    argmax = int(np.argmax(L))
    if strategy == "majority":
        import collections
        c = collections.Counter(v.argmax for v in vs).most_common(1)[0]
        if c[1] <= len(vs) // 2: argmax = int(np.argmax(L))
        else: argmax = int(c[0])
    elif strategy == "unanimous":
        if len(set(v.argmax for v in vs)) != 1: argmax = -1
    votes = [v.argmax for v in vs]
    margin = float(L[argmax] - (max(np.delete(L, argmax)) if argmax >= 0 else 0.0))
    return Verdict(op="AGGREGATE", strategy=strategy, argmax=argmax,
                   logits=[round(float(v), 6) for v in L], votes=votes,
                   margin=round(margin, 6))

def load_pool(dirpool, n=8):
    return [os.path.join(dirpool, "alu_%02d.lancius" % i) for i in range(n)]

if __name__ == "__main__":
    import numpy as np
    import paths
    paths.enforce([paths.BUS])
    x = np.random.default_rng(1).normal(size=65).astype(np.float64)
    pool = load_pool(paths.ALU_POOL, 2)
    print(json.dumps(pool))
    vs = dispatch(pool[:2], x)
    for v in vs: print(v)
    print(aggregate(vs))

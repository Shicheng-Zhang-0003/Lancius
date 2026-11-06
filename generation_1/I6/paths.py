"""I6 paths — one place for repo resolution and import paths.

I6 is the flagship verifier lineage (I1 -> I2 -> I3 -> I6), warm-started
from the I3 ALU pool and trained to MAE/MSE < 1e-3 on the 150-algorithm
corpus.

I6 shares the I3 pinned normalizer (gen1_mu_sd.txt) — the exact stats the
I3 weights were fitted under. A refit from data_vec/v5 no longer reproduces
those stats, so I6 MUST NOT refit from disk.
"""
import os, sys

def _find_repo():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "makefile")):
            return cur
        up = os.path.dirname(cur)
        if up == cur:
            break
        cur = up
    return os.getcwd()

REPO = _find_repo()
GEN1 = os.path.join(REPO, "generation_1")
I6 = os.path.join(GEN1, "I6")
I3 = os.path.join(GEN1, "I3")
I2 = os.path.join(GEN1, "I2")
I1 = os.path.join(GEN1, "I1")
CORPUS = os.path.join(I6, "corpus")
MODELS = os.path.join(I6, "models")
RESULTS = os.path.join(I6, "results_big")
ALU_POOL = os.path.join(I3, "alu_pool")   # I3 ALU models: warm-start substrate
MU_SD = os.path.join(I3, "gen1_mu_sd.txt") # pinned normalizer (shared with I3)
TOOLS_PRM = os.path.join(REPO, "tools", "prm")

def find_lancius():
    for cand in (os.path.join(REPO, "lancius"),
                 os.path.join(REPO, "build", "bin", "lancius"),
                 os.path.join(REPO, "build", "lancius")):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(
        "lancius CLI not found (looked in %s) -- build it first: `make lancius`"
        % ", ".join(os.path.join(REPO, p, "") for p in ("", "build/bin/", "build/")))

LANCIUS = find_lancius()

def enforce(paths=None):
    for p in (paths or [I6, GEN1, TOOLS_PRM, REPO]):
        if p not in sys.path:
            sys.path.insert(0, p)

enforce([I6])

def alu_pool(n=12):
    return [os.path.join(ALU_POOL, "alu_%02d.lancius" % i) for i in range(n)]

def read_musd(path=MU_SD):
    """The pinned normalizer: MU/SD lines exactly as the trainers write them."""
    mu, sd = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MU "):
                mu.append(float(line[3:]))
            elif line.startswith("SD "):
                sd.append(float(line[3:]))
    if len(mu) < 65 or len(sd) < 65:
        raise SystemExit("%s: only %d MU / %d SD lines, want 65" % (path, len(mu), len(sd)))
    import numpy as np
    mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd

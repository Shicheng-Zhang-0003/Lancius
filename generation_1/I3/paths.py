"""I3 paths — one place for repo resolution and import paths.

Every I3 module derives REPO from this rather than from its own relative
depth, because the package now lives at generation_1/I3/.

I3 fabric specifics (vs I2):
  * ALU_POOL points at generation_1/I3/alu_pool — 12 clones of the I3 verifier
    (150-algorithm trained, warm-started from the I2 basis).
  * MU_SD pins the normalizer the I3 ALUs were trained under
    (generation_1/I3/gen1_mu_sd.txt, the gen1 fit). compute_musd(data_vec/v5)
    no longer reproduces these stats (that dir has drifted since gen1
    training), so fabrics over I3 weights MUST NOT refit from disk.
  * CU_MODEL is phi4-mini:3.8b via Ollama (I2 used qwen2.5-coder:3b).
"""
import os, sys

def _find_repo():
    # The old sentinel was a repo-root `lancius` binary; the reorg moved that
    # to build/bin, so walk up on the root `makefile` instead.
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
I3 = os.path.join(GEN1, "I3")
I2 = os.path.join(GEN1, "I2")
I1 = os.path.join(GEN1, "I1")
ALU_POOL = os.path.join(I3, "alu_pool")
ISA = os.path.join(I3, "isa")
BUS = os.path.join(I3, "bus")
MODELS = os.path.join(I3, "models")
ORCH = os.path.join(I3, "orchestrator")
TOOLS_PRM = os.path.join(REPO, "tools", "prm")
DATA_DIR = os.path.join(REPO, "data_vec", "v5")
MU_SD = os.path.join(I3, "gen1_mu_sd.txt")
CU_MODEL = "phi4-mini:3.8b"


def find_lancius():
    """Locator for the Lancius CLI binary.

    The repo reorganization moved it to build/bin; older layouts kept it at
    the repo root. Try the likely spots and fail with a pointed message."""
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
    for p in (paths or [I1, GEN1, TOOLS_PRM, REPO, BUS, ISA, ORCH]):
        if p not in sys.path:
            sys.path.insert(0, p)

enforce([I3])

def read_musd(path=None):
    """The pinned normalizer (MU/SD lines as the trainers write them).

    Never call this with a corpus dir: the I3 weights were fitted under these
    exact stats and any refit is a train/serve skew, silently."""
    path = path or MU_SD
    mu, sd = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MU "):
                mu.append(float(line[3:]))
            elif line.startswith("SD "):
                sd.append(float(line[3:]))
    if len(mu) < 65 or len(sd) < 65:
        raise SystemExit(f"{path}: only {len(mu)} MU / {len(sd)} SD lines, want 65")
    import numpy as np
    mu = np.asarray(mu[:65]); sd = np.asarray(sd[:65])
    sd[~(sd > 1e-12)] = 1.0
    return mu, sd

def pool(n=8):
    return [os.path.join(ALU_POOL, "alu_%02d.lancius" % i) for i in range(n)]

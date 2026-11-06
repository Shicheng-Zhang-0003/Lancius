"""I5 paths — one place for repo resolution and import paths.

I5 (1I5) is the "Lancius as rules engine" generic decision-surface trainer:
Lancius learns P(out | features) as a deterministic decision surface that an
external exploration loop can query repeatedly — perturb inputs, dump scores,
compare equal and different situations, regress deviations, plot association
graphs. Causation lives outside Lancius; Lancius is the mute mathematician
that scores, never speaks.

I5 sits in the 1IX lineage (1I1 -> 1I2 -> 1I3 -> 1I4 -> 1I5 -> 1I6).
I4 was the fire-risk pilot; I5 generalizes the same persistent-graph,
fused-CE trainer to any FEAT/NCLASS corpus read from corpus.hdr.
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
I5 = os.path.join(GEN1, "I5")
I4 = os.path.join(GEN1, "I4")
I6 = os.path.join(GEN1, "I6")
CORPUS = os.path.join(I5, "corpus")
MODELS = os.path.join(I5, "models")
RESULTS = os.path.join(I5, "results_big")
TOOLS = os.path.join(I5, "tools")
DATA_SYNTHETIC = os.path.join(REPO, "data_synthetic")
DATA_I5_REAL = os.path.join(REPO, "data_i5_real")
EXAMPLES = os.path.join(REPO, "examples")
TOOLS_PRM = os.path.join(REPO, "tools", "prm")
TOOLS_ROOT = os.path.join(REPO, "tools")

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
    for p in (paths or [I5, GEN1, TOOLS_PRM, REPO]):
        if p not in sys.path:
            sys.path.insert(0, p)

enforce([I5])

def synthetic_corpus():
    return DATA_SYNTHETIC

def real_corpus():
    return DATA_I5_REAL

def trainer():
    return os.path.join(EXAMPLES, "train_i5.c")

def acquire_script():
    return os.path.join(TOOLS_ROOT, "acquire_and_train_i5.py")

def fire_acquire_script():
    return os.path.join(TOOLS_ROOT, "fire_data_acquire.py")

def make_synthetic_script():
    return os.path.join(TOOLS_ROOT, "make_synthetic_corpus.py")

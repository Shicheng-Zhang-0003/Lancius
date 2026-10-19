"""I2 paths — one place for repo resolution and import paths.

Every I2 module derives REPO from this rather than from its own relative
depth, because the package now lives at generation_1/I2/.
"""
import os, sys

def _find_repo():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "lancius")):
            return cur
        up = os.path.dirname(cur)
        if up == cur:
            break
        cur = up
    return os.getcwd()

REPO = _find_repo()
GEN1 = os.path.join(REPO, "generation_1")
I2 = os.path.join(GEN1, "I2")
I1 = os.path.join(GEN1, "I1")
ALU_POOL = os.path.join(I2, "alu_pool")
ISA = os.path.join(I2, "isa")
BUS = os.path.join(I2, "bus")
MODELS = os.path.join(I2, "models")
ORCH = os.path.join(I2, "orchestrator")
TOOLS_PRM = os.path.join(REPO, "tools", "prm")
DATA_DIR = os.path.join(REPO, "data_vec", "v5")
CU_MODEL = "qwen2.5-coder:3b"

def enforce(paths=None):
    for p in (paths or [I1, GEN1, TOOLS_PRM, REPO, BUS, ISA, ORCH]):
        if p not in sys.path:
            sys.path.insert(0, p)

enforce([I2])

def pool(n=8):
    return [os.path.join(ALU_POOL, "alu_%02d.lancius" % i) for i in range(n)]

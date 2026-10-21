#!/usr/bin/env python3
"""
FABRIC LIVE — real-time hook: Lancius ALU verdicts stream into the CU LLM.

DATAPLANE (every few seconds, forever):
    gen1 derivation -> 4 ALUs (real lancius run, 1.2s each, parallel)
                    -> AGGREGATE + CONFIDENCE
CONTROL PLANE:
    the CU (Ollama phi4-mini:3.8b) sees ONLY the verdict packets, keeps a
    rolling memory of the last --mem turns, and answers with:
        DIRECTIVE: CONTINUE|HALT       <- it literally gates the fabric clock
        <one analysis sentence>
    The CU does NOT see the math, does NOT see the truth label, and never
    computes anything. It reads packets, watches margins drift, and can halt.

usage:
    python3 fabric/orchestrator/fabric_live.py [--rows N] [--alus K] [--mem K]
        [--master-seed S] [--cu phi4-mini:3.8b] [--speed-fast]
    default: endless (--rows 0). Ctrl+C prints session stats.
"""
import os, sys, time, json, random, urllib.request, argparse, threading

_sd = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_sd, 'paths.py')):
    _n = os.path.dirname(_sd)
    assert _n != _sd, 'paths.py not found above %s' % __file__
    _sd = _n
sys.path.insert(0, _sd)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import paths as _paths
_paths.enforce()
REPO = _paths.REPO
ALU_POOL = _paths.ALU_POOL
DATA_DIR = _paths.DATA_DIR

from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
from algos_i3 import EXT3_ALGOS
import verifier5 as V
import bench_gen1 as B
import bus as FABRIC
import isa as LAIS

ALL150 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS) + list(EXT3_ALGOS)
DATA_DIR = DATA_DIR

CU_SYSTEM = (
    "You are the control unit of the Moore-style fabric. The Lancius ALUs "
    "(bare-metal C runtime) verified math derivations and stream you packets.\n"
    "Class map: 4=accept(+1.0) 3=accept(+0.5) 2=neutral(0.0) 1=doubt(-0.5) 0=reject(-1.0).\n"
    "Reply STRICTLY as exactly two lines, BOTH mandatory:\n"
    "DIRECTIVE: CONTINUE\n"
    "<one short analysis sentence: what the packet says, or margin drift across "
    "recent packets from your memory>\n"
    "Example:\n"
    "DIRECTIVE: CONTINUE\n"
    "Verdict accept with margin 0.83; confidence is stable.\n"
    "HALT RULE: only when the SESSION line shows exact_match under 50% or a "
    "majority of packets vote class 2 (undecidable) for 5+ consecutive rows.\n"
    "You never see the math, only packets. You are the heartbeat supervisor."
)

class OllamaCU:
    def __init__(self, model, mem=4):
        self.model, self.mem = model, mem
        self.history = []

    def _post(self, path, body):
        req = urllib.request.Request(
            "http://127.0.0.1:11434" + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def analyze(self, packet, session_line):
        msgs = [{"role": "system", "content": CU_SYSTEM}]
        for pkt, ans in self.history[-2 * self.mem:]:
            msgs.append({"role": "user", "content": pkt})
            msgs.append({"role": "assistant", "content": ans})
        msgs.append({"role": "user",
                     "content": packet + "\n\n" + session_line})
        out = self._post("/api/chat", {"model": self.model, "messages": msgs,
                                       "stream": False, "keep_alive": "10m",
                                       "options": {"num_predict": 90,
                                                   "temperature": 0.3}})
        text = (out.get("message", {}).get("content") or "").strip()
        directive, analysis = "CONTINUE", text
        for line in text.splitlines():
            if line.upper().startswith("DIRECTIVE"):
                if "HALT" in line.upper(): directive = "HALT"
                analysis = "\n".join(l for l in text.splitlines()
                                     if not l.upper().startswith("DIRECTIVE")).strip()
        if not analysis:
            analysis = "(no commentary)"
        turn = (packet, "DIRECTIVE: %s\n%s" % (directive, analysis))
        self.history.append(turn)
        return directive, analysis

def now():
    return time.strftime("%H:%M:%S")

def run_live(args):
    mu, sd = _init_axes()
    pool = FABRIC.load_pool(ALU_POOL, args.alus)
    master = random.Random(args.master_seed if args.master_seed >= 0
                           else random.randrange(2**30))
    cu = OllamaCU(args.cu, args.mem)
    print("== FABRIC LIVE == pool: %d ALUs | CU: %s | rows: %s ==" %
          (args.alus, args.cu, "endless" if args.rows <= 0 else args.rows))
    n = 0
    ok = 0
    margins = []
    halted = False
    try:
        while args.rows <= 0 or n < args.rows:
            seed = random.randrange(2**30)
            row = None
            for _ in range(8):
                row = _make_row(seed, mu, sd, master)
                if row: break
            if not row:
                continue
            t0 = time.time()
            vs = FABRIC.dispatch(pool[:args.alus], row["feats"])
            agg = FABRIC.aggregate(vs, strategy="mean")
            conf = LAIS.packet_confidence(agg)
            margins.append(agg.margin)
            n += 1
            match = agg.argmax >= 0 and agg.argmax == row["label"]
            ok += int(match)
            truth = B.LVLNAME[row["level"]].split()[0:1][0]
            packet = json.dumps({
                "i": n, "algo": row["algo"], "kind": row["kind"][:24],
                "votes": [v.argmax for v in vs],
                "argmax": agg.argmax, "verdict": LAIS.CLASS_TEXT[max(agg.argmax,0)],
                "margin": round(agg.margin, 4),
                "logits": agg.logits}, sort_keys=True)
            session = ("SESSION STATS: derivations=%d exact_match=%d/%d "
                       "mean_margin=%.3f total_wall=%.0fs" %
                       (n, ok, n, sum(margins) / len(margins), time.time() - t00))
            directive, analysis = cu.analyze(packet, session)
            marks = "".join("=%d" % v.argmax for v in vs)
            CU_GATES_CLK = ""
            if directive == "HALT":
                halted = True
                CU_GATES_CLK = "  [CLOCK HALTED BY CU]"
            print("%s | d%-4d %s truth=%s | ALUs%s | agg=%d m=%.3f | %s | %s%s"
                  % (now(), n, row["algo"][:18], truth, marks, agg.argmax,
                     agg.margin, match and "OK " or "MISS", analysis,
                     CU_GATES_CLK))
            if halted:
                break
    except KeyboardInterrupt:
        pass
    print("\n== SESSION END ==")
    if n:
        print("derivations=%d  ALU exact=%d/%d (%.1f%%)  mean_margin=%.3f  "
              "halted_by_cu=%s"
              % (n, ok, n, 100.0 * ok / n, sum(margins) / len(margins),
                 str(halted)))

def _init_axes():
    p0 = FABRIC.load_pool(ALU_POOL, 1)[0]
    d0 = B.parse_lancius(p0)
    nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
    shapes = {tuple(nn["shape"][:nn["ndim"]]) for nn in nodes if nn.get("has_w")}
    B.H1G = [s[1] for s in shapes if len(s) == 2 and s[0] == 65][0]
    B.H2G = [s[0] for s in shapes if len(s) == 2 and s[1] == 5][0]
    # pinned gen1-fit MU/SD (see paths.read_musd)
    return _paths.read_musd()

def _make_row(seed, mu, sd, master):
    import bench_gen1 as B
    rng = B.PyRng(seed)
    k = master.randrange(len(ALL150))
    algo = ALL150[k]
    params0, _ = algo.build(rng)
    params = dict(params0)
    level = master.choice(B.LEVELS)
    steps = algo._steps(params)
    if not steps:
        return None
    rep, stc, kind = V.make_candidate(algo, params, steps, rng, level)
    chk = V.check(algo, params, steps, rep, stc)
    f = V.row_features(algo, chk, -1)
    x = (f - mu) / sd
    return dict(feats=x, label=B.LEVELS.index(level), level=level,
                algo=algo.name, kind=kind)

t00 = time.time()

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=0,
                    help="0/omitted = endless stream")
    ap.add_argument("--alus", type=int, default=12)
    ap.add_argument("--mem", type=int, default=4, help="CU rolling memory turns")
    ap.add_argument("--master-seed", type=int, default=-1)
    ap.add_argument("--cu", default="phi4-mini:3.8b")
    return ap.parse_args()

if __name__ == "__main__":
    run_live(parse())

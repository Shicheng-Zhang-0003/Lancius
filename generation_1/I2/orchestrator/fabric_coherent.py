#!/usr/bin/env python3
"""
FABRIC COHERENT — the combined model set running as ONE model.

The CU (Ollama qwen2.5-coder:3b) is given LAIS-CU MANUAL (fabric/isa/cu_manual.txt)
verbatim as its instruction set. It then DRIVES the fabric:

    CU json command -> orchestrator executes against real Lancius ALUs
                    -> OBSERVATION packet returned to CU
                    -> repeat until ops REPORT (cu's own English verdict)

Teaches two things literally:
  INPUT encoding : VERDICT (derivation by name/level) and FEED (raw 65-dim
                   feature layout; bus normalises with training mu/sd).
  OUTPUT         : packet schema, class-order logits, votes, margin semantics.

usage:
    python3 fabric/orchestrator/fabric_coherent.py [--sessions 3] [--alus 4]
        [--task-algo NAME] [--task-level -1.0] [--master-seed S]
"""
import os, sys, re, json, random, urllib.request, argparse

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
import verifier5 as V
import bench_gen1 as B
import bus as FABRIC
import isa as LAIS

ALL100 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS)
NAMEdict = {a.name: a for a in ALL100}
DATA_DIR = DATA_DIR
MANUAL = os.path.join(_paths.ISA, "cu_manual.txt")
CU_MODEL = "qwen2.5-coder:3b"
LEVELS_ = ["-1.0", "-0.5", "0.0", "+0.5", "+1.0"]

LEVEL_MAP = {"-1.0": -1.0, "-0.5": -0.5, "0.0": 0.0, "-1": -1.0, "-0.5": -0.5,
             "0": 0.0, "0.5": 0.5, "+0.5": 0.5, "1.0": 1.0, "1": 1.0, "+1.0": 1.0}


class CU:
    def __init__(self, model=CU_MODEL):
        self.model = model
        with open(MANUAL) as f:
            self.system = f.read()
        self.msgs = [{"role": "system", "content": self.system}]

    def ask(self, user_line, tries=2, timeout=300):
        self.msgs.append({"role": "user", "content": user_line})
        body = {"model": self.model, "messages": self.msgs, "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 600, "temperature": 0.1,
                            "num_ctx": 8192}}
        err = None
        for _ in range(tries):
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out = json.loads(r.read().decode())
                break
            except Exception as e:
                err = e
        else:
            raise err
        text = (out.get("message", {}).get("content") or "").strip()
        self.msgs.append({"role": "assistant", "content": text})
        return text


class Fabric:
    """Executes CU ops against the real Lancius ALU pool."""

    def __init__(self, alus):
        self.alus = alus
        self.last = None
        # mu/sd
        p0 = self.alus[0]
        d0 = B.parse_lancius(p0)
        nodes = d0["nodes"] if isinstance(d0, dict) and "nodes" in d0 else d0
        shapes = {tuple(nn["shape"][:nn["ndim"]]) for nn in nodes if nn.get("has_w")}
        B.H1G = [s[1] for s in shapes if len(s) == 2 and s[0] == 65][0]
        B.H2G = [s[0] for s in shapes if len(s) == 2 and s[1] == 5][0]
        self.mu, self.sd = B.compute_musd(DATA_DIR)

    def _run(self, x_norm):
        vs = FABRIC.dispatch(self.alus, x_norm)
        agg = FABRIC.aggregate(vs, strategy="mean")
        self.last = agg
        return dict(votes=[v.argmax for v in vs], argmax=agg.argmax,
                    verdict=LAIS.CLASS_TEXT[max(agg.argmax, 0)], margin=agg.margin,
                    logits=agg.logits)

    def op_list(self):
        names = sorted(NAMEdict)[:30]
        return dict(pool=["alu_%02d.lancius" % i for i in range(len(self.alus))],
                    count_algos=len(ALL100), algos_head=names,
                    classes=dict(zip(range(4, -1, -1), LAIS.CLASS_TEXT[::-1])))

    def op_verdict(self, algo, level):
        if not isinstance(algo, str) or algo not in NAMEdict:
            return dict(error="unknown algo '%s'; op LIST gives names" % str(algo))
        lv = LEVEL_MAP.get(str(level), None)
        if lv is None: lv = float(level) if not isinstance(level, float) else level
        if lv is None or not any(abs(lv - x) < 1e-9 for x in B.LEVELS):
            return dict(error="level must be one of %s" % LEVELS_)
        algo_obj = NAMEdict[algo]
        rng = B.PyRng(random.randrange(2**30))
        try:
            params0, _ = algo_obj.build(rng)
            steps = algo_obj._steps(params0)
            rep, stc, kind = V.make_candidate(algo_obj, params0, steps, rng, lv)
            chk = V.check(algo_obj, params0, steps, rep, stc)
            f = V.row_features(algo_obj, chk, -1)
        except Exception as e:
            return dict(error="feeder failed for %s: %s" % (algo, str(e)[:60]))
        x = (f - self.mu) / self.sd
        pkt = self._run(x)
        pkt.update(i="-", algo=algo, kind=kind[:20], truth_of_row=B.LVLNAME[lv])
        return pkt

    def op_feed(self, features):
        if not isinstance(features, list) or len(features) != 65:
            n = len(features) if isinstance(features, list) else "no-list"
            return dict(error="FEED needs exactly 65 raw floats, got %s" % n)
        try:
            import numpy as np
            x = (np.asarray(features, dtype=np.float64) - self.mu) / self.sd
        except Exception as e:
            return dict(error="FEED values must be numbers: %s" % str(e)[:60])
        pkt = self._run(x)
        pkt.update(i="-", algo="(cu-supplied)", kind="feed",
                   normalized=True)
        return pkt

    def op_confidence(self):
        if self.last is None:
            return dict(error="no prior packet")
        conf = LAIS.packet_confidence(self.last)
        return dict(confident=conf["margin"], span=conf["uniform"],
                    last_argmax=self.last.argmax)

    def exec_cmd(self, cmd):
        op = cmd.get("op")
        if op == "LIST": return self.op_list()
        if op == "VERDICT": return self.op_verdict(cmd.get("algo"), cmd.get("level", "0.0"))
        if op == "FEED": return self.op_feed(cmd.get("features"))
        if op == "CONFIDENCE": return self.op_confidence()
        if op == "REPORT": return None
        return dict(error="unknown op '%s'; allowed: LIST VERDICT FEED CONFIDENCE REPORT" % op)


JSON_RE = re.compile(r"\{.*?\}", re.S)

def extract_json(text):
    """Find the first top-level json object in the CU reply (fence- and
    over-escape-tolerant; 3b models sometimes emit {\\\"op\\\":...})."""
    text = text.replace("```json", "\n").replace("```", "\n")
    cands = JSON_RE.findall(text)
    if text.lstrip().startswith("{") and not cands:
        cands = [text]
    for cand in cands:
        for t in (cand, cand.replace('\\"', '"')):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


VERDICT_WORDS = ("accept", "reject", "doubt", "neutral", "undecidable",
                 "verdict", "probably false", "plain true", "plain false")

def session(cu, fabric, task_heading, task_lines, max_turns=6):
    print("\n" + "=" * 72)
    print(task_heading)
    cu.msgs = [dict(cu.msgs[0])]
    cu.msgs.append({"role": "user",
                    "content": "FABRIC ONLINE. pool=%d alus. TASK: %s\nBegin "
                               "with one command per turn (json line). When "
                               "you have enough: reply ONLY a single line "
                               '{"op":"REPORT","text":"<your English verdict>"}'
                               % (len(fabric.alus), task_lines)})
    for turn in range(max_turns):
        reply = cu.ask("")
        wait = extract_json(reply) or {}
        if wait.get("op") == "REPORT":
            print("  CU> REPORT: %s" % (wait.get("text") or reply.strip()))
            return wait.get("text") or reply.strip()
        if wait is not None and wait.get("op"):
            print("  CU> %s" % json.dumps(wait, sort_keys=True))
            obs = fabric.exec_cmd(wait)
            if obs is None:
                obs = dict(info="reported; session done")
            cu.msgs.append({"role": "user",
                            "content": "OBSERVATION:\n" + json.dumps(obs) +
                                       "\nWhen ready reply ONLY the REPORT json."})
            if isinstance(obs, dict) and obs.get("argmax") is not None:
                print("  BUS> argmax=%s verdict=%s margin=%.3f votes=%s" %
                      (obs.get("argmax"), obs.get("verdict"),
                       obs.get("margin") or -1, obs.get("votes")))
            elif isinstance(obs, dict) and "error" in obs:
                print("  BUS> error: %s" % obs["error"])
            continue
        # natural language but no json: treat as an implicit Report if it
        # reads like one (small models slip out of json mid-report).
        low = reply.lower()
        if any(w in low for w in VERDICT_WORDS):
            print("  CU> REPORT (implicit): %s" % reply.strip())
            return reply.strip()
        print("  CU> (off-protocol) %-60s" % reply[:60])
        cu.msgs.append({"role": "user",
                        "content": "ERROR OFF-PROTOCOL: reply with exactly ONE "
                                   "json line: LIST, VERDICT, FEED, CONFIDENCE, REPORT."})
    return "(max turns reached without REPORT)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--alus", type=int, default=4)
    ap.add_argument("--master-seed", type=int, default=-1)
    args = ap.parse_args()
    master = random.Random(args.master_seed if args.master_seed >= 0
                           else random.randrange(2**30))
    fab = Fabric(FABRIC.load_pool(ALU_POOL, args.alus))
    cu = CU()
    for si in range(args.sessions):
        algo = master.choice(list(NAMEdict))
        lv = master.choice(LEVELS_)
        session(cu, fab,
                "SESSION %d: claimed derivation: %-18s claimed honesty: %s"
                % (si + 1, algo, lv),
                "The caller says: '%s' claims the derivation %s at level "
                "%s. Drive the fabric (VERDICT first), then REPORT what "
                "the ALUs said in plain English." % (algo, algo, lv))
    print("\n== COHERENT MODEL SESSIONS END ==")

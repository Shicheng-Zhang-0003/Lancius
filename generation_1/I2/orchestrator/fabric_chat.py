#!/usr/bin/env python3
"""
FABRIC CHAT — full chatbot: conversational CU (qwen2.5-coder:3b) backed by the
Lancius ALU fabric.

The CU keeps chat memory (rolling window) and the LAIS-CU manual.
Inside ONE user turn it may chain ops transparently:
    reply {"op":"VERDICT"...} -> bus executes on real Lancius ALUs
                              -> OBSERVATION fed back -> CU continues talking
It decides itself when the claim needs hardware verification; if the math is
ordinary conversation, it just answers (fabric untouched).

Interactive REPL by default. Scripted: --demo "turn 1" "turn 2" ...
Commands: /help /reset /status /quit
"""
import os, sys, json, urllib.request, argparse, re, time

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

import bus as FABRIC
import isa as LAIS
import bench_gen1 as B
import verifier5 as V
import random
from fabric_coherent import Fabric, CU, VERDICT_WORDS, NAMEdict  # reuse executor + CU core

OPS = ("VERDICT", "FEED", "LIST", "CONFIDENCE", "NONE")
EVIDENCE_OPS = ("VERDICT", "FEED")
CLAIM_WORDS = ("accept", "reject", "doubt", "undecidable", "verdict",
               "margin", "logit", "honest", "dishonest", "claimed",
               "reported", "trustworthy")
NUMERIC_CLAIM_RE = re.compile(r"[+-]?\d+\.\d")

def trace(tag, s):
    try:
        with open("/tmp/opencode/chat_api.log", "a") as lg:
            lg.write("%s: %r\n" % (tag, str(s)[:100]))
    except Exception:
        pass


def chat_api(body, timeout=300, tries=2):
    """POST /api/chat with retry — CPU ollama can exceed short timeouts."""
    last = None
    trace(" req", "n=%d last=%r" % (
        len(body["messages"]),
        (body["messages"][-1].get("content") or "")[:50]))
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            last = e
            trace("fail", e)
            time.sleep(3)
    raise last

from fabric_coherent import extract_json

class CUChat(CU):
    """Persistent chat memory + transparent op mining inside one turn."""

    KEEP = 26  # rolling window (system + ~12 turns)

    def _trim(self):
        keep = dict(sys=self.msgs[0])
        tail = [m for m in self.msgs[1:] if m["role"] != "system"]
        self.msgs = [keep["sys"]] + tail[-self.KEEP:]

    def stream_answer(self):
        """Stream final (op-free) answer to stdout live."""
        body = {"model": self.model, "messages": self.msgs, "stream": True,
                "keep_alive": "10m",
                "options": {"num_predict": 350, "temperature": 0.5, "num_ctx": 8192}}
        chunks = []
        with chat_api(body) as r:
            for line in r:
                try:
                    obj = json.loads(line.decode())
                except Exception:
                    continue
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    print(piece, end="", flush=True)
                    chunks.append(piece)
                if obj.get("done"):
                    break
        text = "".join(chunks).strip()
        trace("stream", text)
        self.msgs.append({"role": "assistant", "content": text})
        self._trim()
        print()
        return text

    def _plain_turn(self, user_text):
        """No claim words, no claim-level tokens, no big float payload ->
        pure chat turn: skip the op protocol entirely."""
        low = user_text.lower()
        if re.findall(r"-?\d+\.\d+", user_text):
            return False
        kws = ("verify", "check", "claims", "claimed", "reported",
               "trustworthy", "honest", "dishonest", "undecidable",
               "verdict", "margin", "derivation", "derivation of")
        for k in kws:
            if k in low:
                return False
        for t in ("+1.0", "+0.5", "-1.0", "-0.5", "0.0"):
            if t in user_text:
                return False
        return True

    def _auto(self, user_text):
        """Deterministic CU routes: exactly-65 dotted floats -> FEED;
        known algo + level token with claim words -> VERDICT.
        Code executes the op; the model only renders the packet."""
        if not getattr(self, "_fabric", None):
            return None
        txt, low = user_text, user_text.lower()
        nums = re.findall(r"-?\d+\.\d+", txt)
        if len(nums) == 65:
            return dict(op="FEED", features=[float(n) for n in nums])
        toks = [t for t in ("+1.0", "+0.5", "-1.0", "-0.5", "0.0") if t in txt]
        ctx = any(k in low for k in ("verify", "check", "claims", "claimed",
                                     "reported", "trustworthy", "honest",
                                     "dishonest", "undecidable", "confirm"))
        if toks and ctx:
            for t in toks:
                for name in sorted(NAMEdict, key=len, reverse=True):
                    if name in low:
                        return dict(op="VERDICT", algo=name, level=t)
        return None

    def _print_obs(self, obs):
        if isinstance(obs, dict) and obs.get("argmax") is not None:
            self._printer("  [bus   ] argmax=%s verdict=%s margin=%.3f votes=%s"
                          % (obs.get("argmax"), obs.get("verdict"),
                             obs.get("margin") or -1, obs.get("votes")))
        elif isinstance(obs, dict) and "error" in obs:
            self._printer("  [bus   ] error: %s" % obs["error"])

    def finish_stream(self, strict):
        """Stream the final answer behind a plain-English nudge; re-stream
        if the model leaked json; then run the honesty filter."""
        last_c = self.msgs[-1].get("content") or ""
        if not (last_c.endswith("unless needed.") or
                last_c.endswith("no op.") or last_c.endswith("in plain English.")):
            self.msgs.append({"role": "user", "content":
                              "Now answer me in plain English; no json, no ops."})
            self._trim()
        res = self.stream_answer()
        if re.search(r"```json|\{\"op\"", res or ""):
            self.msgs.append({"role": "user", "content":
                              "Output PLAIN PROSE only. The op phase ended; do "
                              "not print json in the answer."})
            self._trim()
            res = self.stream_answer()
        return self._filter_stream(res, strict)

    def turn(self, user_text, max_rounds=3, verbose=False):
        """One chatbot turn. ROUTES (deterministic first, model fallback):
        1. plain chat turn -> no ops at all
        2. deterministic auto-op (65-float FEED / known algo+level VERDICT)
        3. model-run op protocol with the CUE, guarded by the
           hallucination guard + retraction filter"""
        CUE = ("[fabric protocol] For this turn reply with EXACTLY one json "
               "line, then stop:\n"
               "* claim about a derivation's verdict/honesty -> "
               "{\"op\":\"VERDICT\",\"algo\":\"<name from the claim>\","
               "\"level\":\"<level from the claim>\"} (e.g. 'euclid_gcd was "
               "reported +1.0' -> {\"op\":\"VERDICT\",\"algo\":\"euclid_gcd\","
               "\"level\":\"+1.0\"}; use 0.0 if the claim says undecidable)\n"
               "* turn provides 65 floats -> {\"op\":\"FEED\",\"features\":[...]} "
               "copy the caller's 65 floats verbatim (use scratch lines of 5 "
               "prefixed with '#' to count; never inside features)\n"
               "* no derivation claim, no floats -> {\"op\":\"NONE\"}\n"
               "Only ops may produce verdict/margin facts; paraphrase OBSERVATION "
               "packets, never invent values.")
        user_low = user_text.lower()
        strict = (any(k in user_low for k in
                      ("verify", "check", "claims", "claimed", "reported",
                       "trustworthy", "honest", "told me", "you said",
                       "earlier", "remember"))
                  and bool(re.search(r"margin\s*(?:of|is|was)?\s*(?:exactly|about|roughly|around|just|near)?\s*-?\d", user_low)))
        plain = self._plain_turn(user_text)
        auto = None if plain else self._auto(user_text)
        self.msgs.append({"role": "user",
                          "content": user_text + ("" if plain else "\n" + CUE)})
        self._trim()
        if plain:
            self._evidence_ran = False
            return self.finish_stream(strict)
        ran_evidence = getattr(self, "_evidence_ran", False)
        if auto is not None:
            obs = self._fabric.exec_cmd(auto)
            if obs is None:
                obs = dict(info="reported")
            ran_evidence = True
            self._evidence_ran = ran_evidence
            if verbose:
                self._printer("  [fabric] " + json.dumps(auto, sort_keys=True))
            self._print_obs(obs)
            self.msgs.append({"role": "assistant",
                              "content": json.dumps(auto, sort_keys=True)})
            self.msgs.append({"role": "user",
                              "content": "OBSERVATION:\n" + json.dumps(obs) +
                                         "\nNow answer me in plain English."})
            self._trim()
            return self.finish_stream(strict)
        for _round in range(max_rounds):
            body = {"model": self.model, "messages": self.msgs, "stream": False,
                    "keep_alive": "10m",
                    "options": {"num_predict": 600, "temperature": 0.0,
                                "num_ctx": 8192}}
            with chat_api(body) as r:
                out = json.loads(r.read().decode())
            text = (out.get("message", {}).get("content") or "").strip()
            trace(" rep", text)
            cmd = extract_json(text)
            if cmd and cmd.get("op") in OPS:
                if cmd.get("op") == "NONE" and not (strict and not ran_evidence):
                    self.msgs.append({"role": "assistant", "content": text})
                    self.msgs.append({"role": "user",
                                      "content": "Answer plainly now; no op."})
                    self._trim()
                    break
                if cmd.get("op") == "NONE":
                    guard = ("WRONG. This turn asserts a verdict/margin. Call "
                             "VERDICT now with the algorithm name and level "
                             "from the user message above, like: "
                             "{\"op\":\"VERDICT\",\"algo\":\"<name from the "
                             "user message>\",\"level\":\"<level from the user "
                             "message>\"}.")
                    self.msgs.append({"role": "user", "content": guard})
                    self._trim()
                    continue
                if verbose:
                    self._printer("  [fabric] " + json.dumps(cmd, sort_keys=True))
                obs = self._fabric.exec_cmd(cmd)
                if obs is None:
                    obs = dict(info="reported")
                ran_evidence = ran_evidence or cmd.get("op") in EVIDENCE_OPS
                self._evidence_ran = ran_evidence
                self.msgs.append({"role": "assistant", "content": text})
                packet = "OBSERVATION:\n" + json.dumps(obs)
                self._print_obs(obs)
                self.msgs.append({"role": "user",
                                  "content": packet + "\nNow answer me in plain "
                                             "English (no more ops unless "
                                             "needed)."})
                self._trim()
                continue
            low = text.lower()
            cites = (any(w in low for w in CLAIM_WORDS)
                     and bool(NUMERIC_CLAIM_RE.search(low)))
            cites = cites or ("op" in text and cmd is None)
            if cites and (not ran_evidence or strict):
                guard = ("HALLUCINATION GUARD: you quoted a verdict/margin but "
                         "no VERDICT or FEED ran in this chat. Issue ONE json op "
                         "now (VERDICT/FEED) or explicitly say there is no "
                         "hardware packet for that claim.")
                self.msgs.append({"role": "user", "content": guard})
                self._trim()
                continue
            self._evidence_ran = ran_evidence
            break
        else:
            if strict and not ran_evidence:
                r = ("No hardware packet supports that claim; I will not "
                     "confirm it.")
            else:
                r = "(round limit reached)"
            print(r)
            return r
        return self.finish_stream(strict)

    def _observed_margins(self):
        out = []
        for m in self.msgs:
            c = m.get("content") or ""
            if m["role"] == "user" and "OBSERVATION" in c[:12]:
                for mm in re.finditer(r"margin[\":\s]+(-?\d+\.?\d*)", c):
                    out.append(float(mm.group(1)))
        return out

    def _filter_stream(self, res, strict):
        """Last line of honesty: a streamed answer quoting margins/verdict
        numbers that no bus packet supports gets visibly retracted."""
        low = (res or "").lower()
        quoted = [float(x) for x in re.findall(
            r"margin(?:[^0-9\n]{0,20}?)(-?\d+\.?\d*)", low, re.I)]
        packets = self._observed_margins()
        bad_n = [q for q in quoted if not any(abs(q - p) < 0.06 for p in packets)]
        verdict_speak = any(w in low for w in
                            ("accept", "reject", "doubt", "undecidable"))
        if not bad_n:
            return res
        if not (strict or verdict_speak):
            return res
        note = ("[fabric-note] That reply quoted numbers no bus packet supports. "
                "Quoting only verified packets from now on.")
        self._printer(note)
        self.msgs.append({"role": "user", "content": note})
        honest = ("Retraction: I have no verified packet supporting those "
                  "numbers. The fabric only backs numbers from VERDICT/FEED "
                  "runs (OBSERVATION packets).")
        self.msgs.append({"role": "assistant", "content": honest})
        self._trim()
        return honest + "\n(replaced unverified quote: %s)" % bad_n[:3]

    def _printer(self, s):
        print(s)

    def attach_fabric(self, fabric, printer=None):
        self._fabric = fabric
        if printer:
            self._printer = printer


HELP = """commands:
  type naturally. When your message states a math claim you want checked,
  the CU may run fabric ops (VERDICT/FEED/CONFIDENCE) automatically.
  /status   fabric + memory state
  /reset    clear chat memory
  /help     this
  /quit     leave
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", nargs="*", help="scripted turns; else interactive")
    ap.add_argument("--alus", type=int, default=4)
    args = ap.parse_args()

    fab = Fabric(FABRIC.load_pool(ALU_POOL, args.alus))
    cu = CUChat()
    cu.attach_fabric(fab)
    print("== FABRIC CHAT == CU=%s | %d Lancius ALUs online" % (cu.model, args.alus))

    if args.demo:
        opening = ("You are a general math assistant. You also have access to "
                   " hardware verification ops from your manual (VERDICT/FEED/"
                   "LIST/CONFIDENCE). Use them when a claim's honesty matters; "
                   "otherwise answer normally.")
        cu.msgs.append({"role": "user", "content": opening})
        for t in args.demo:
            print("\nUSER: %s" % t)
            print("CU  :", end=" ", flush=True)
            cu.turn(t, verbose=True)
        print("\n== DEMO END ==")
        return

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            print("bye")
            break
        if line == "/help":
            print(HELP); continue
        if line == "/reset":
            cu.msgs = [dict(cu.msgs[0])]; print("memory wiped"); continue
        if line == "/status":
            last = fab.last
            print("pool: %d ALUs | last fabric packet: %s"
                  % (len(fab.alus), last.argmax if last else "none"))
            continue
        cu.turn(line, verbose=True)

if __name__ == "__main__":
    main()

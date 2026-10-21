#!/usr/bin/env python3
"""
MODEL CONGLOMERATION v1 — the combined model set as one chatbot.

    +----------------------------------------------------------+
    |  CONTROLLER UNIT (soft)   phi4-mini:3.8b via Ollama     |
    |   - owns conversation, memory, protocol                    |
    |   - packs inputs / reads packets per LAIS-CU MANUAL        |
    |                    ^                ^                      |
    |                    | packets        | ops                  |
    |                    v                v                      |
    |  ALU BUS (hard)          N x Lancius verifier ALUs         |
    |   - real bare-metal C runtime (lancius run)                |
    |   - VERDICT / FEED / LIST / CONFIDENCE ops                 |
    |   - anti-hallucination guard + retraction filter           |
    +----------------------------------------------------------+

I3 heritage: the 10.77M-parameter verifier (150 algorithms, I2-basis warm
start) came from the I3 training pipeline (0.9891 exact head-to-head).
Generation-2 (I2) fabric: construction, bus, ISA, orchestration, chat.

usage:  python3 generation_1/I3/chat.py [--alus K] [--demo t1 "t2" ...]
        interactive default; /help /status /reset /quit
tests:  python3 generation_1/I3/orchestrator/fabric_test_chat.py --part all
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orchestrator"))
import paths as _p
_p.enforce()

import bus as FABRIC
from fabric_chat import CUChat, HELP
from fabric_coherent import Fabric
import isa as LAIS


def boot(alus=4):
    fab = Fabric(FABRIC.load_pool(_p.ALU_POOL, alus))
    cu = CUChat()
    cu.attach_fabric(fab)
    return cu, fab


def interactive(cu, fab, n_alus):
    print("== MODEL CONGLOMERATION ==")
    print("  CU (control, soft) : %s via Ollama" % cu.model)
    print("  ALUs (compute hard): %d x Lancius verifier (bare-metal C)" % n_alus)
    print("  ISA                : LAIS-CU manual (fetch: ask 'what can you do')")
    print("  heritage           : i3 verifier, 0.9891%% exact on 150-algo suite (warm from I2)")
    print(HELP)
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
            print(HELP)
            continue
        if line == "/reset":
            cu.msgs = [dict(cu.msgs[0])]
            print("memory wiped")
            continue
        if line == "/status":
            last = fab.last
            print("pool: %d ALUs | last fabric packet: %s"
                  % (n_alus, last.argmax if last else "none"))
            continue
        cu.turn(line, verbose=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alus", type=int, default=12)
    ap.add_argument("--demo", nargs="*", help="scripted turns; else interactive")
    args = ap.parse_args()
    cu, fab = boot(args.alus)
    if args.demo:
        cu.msgs.append({"role": "user", "content":
            "Demo session on the model conglomeration. Claims about "
            "derivations will be verified against the ALU bus; plain chat and "
            "plain arithmetic get answered directly, no ops."})
        for t in args.demo:
            print("\nUSER: %s" % t)
            print("CU  :", end=" ", flush=True)
            cu.turn(t, verbose=True)
        print("\n== DEMO END ==")
        return
    interactive(cu, fab, args.alus)


if __name__ == "__main__":
    main()

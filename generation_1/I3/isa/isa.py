#!/usr/bin/env python3
"""
LAIS-F — Lancaster ALU Instruction Set, Fabric tier.

The ALU pool is N copies of the Lancius verifier model (bare-metal C runtime).
Each op below is a real executable instruction against that pool:

  0x01 VERDICT    ale, feature-file       -> peer argv: lancius run <alu> --input <f>
                          one 5-class verdict (argmax over honesty classes)
  0x02 AGGREGATE  id... strategy          -> combine verdicts (mean logits +
                          majority argmax + top1-top2 margin)
  0x03 ROUTE      packet, target          -> attach routing header (bus/cu/ store)
  0x04 CONFIDENCE verdict-id              -> margin + absolute logit span gate
  0x05 EXPLAIN    packet                  -> CU (external LLM) renders English

Classes (from training): 4:+1.0 honest | 3:+0.5 honest-stretch | 2:0.0 neutral
                          1:-0.5 dishonest | 0:-1.0 dishonest
"""
import json

OP_VERDICT, OP_AGG, OP_ROUTE, OP_CONF, OP_EXPLAIN = 0x01, 0x02, 0x03, 0x04, 0x05

CLASS_TEXT = ["reject (-1.0)", "doubt (-0.5)", "neutral (0.0)",
              "accept (stretched +0.5)", "accept (+1.0)"]

STRATEGIES = ("mean", "majority", "unanimous")


def encode(op, **kw):
    p = {"op": op, "opname": {OP_VERDICT: "VERDICT", OP_AGG: "AGGREGATE",
                              OP_ROUTE: "ROUTE", OP_CONF: "CONFIDENCE",
                              OP_EXPLAIN: "EXPLAIN"}[op]}
    p.update(kw)
    return p


def packet_verdict(alu, feature_file):
    return encode(OP_VERDICT, alu=alu, input=feature_file)


def packet_aggregate(ids, strategy="mean"):
    assert strategy in STRATEGIES, strategy
    return encode(OP_AGG, ids=ids, strategy=strategy)


def packet_route(packet, target):
    return encode(OP_ROUTE, target=target, payload=packet)


def packet_confidence(aggregated):
    d = aggregated if isinstance(aggregated, dict) else aggregated.__dict__
    m = (d["logits"][d["argmax"]] - max(
        (v for i, v in enumerate(d["logits"]) if i != d["argmax"]), default=0.0))
    return encode(OP_CONF, argmax=d["argmax"], margin=round(m, 6),
                  uniform=round(max(d["logits"]) - min(d["logits"]), 6))


def packet_explain(packet, target="cu"):
    return packet_route(packet, target)


def show(p):
    return json.dumps(p, sort_keys=False)

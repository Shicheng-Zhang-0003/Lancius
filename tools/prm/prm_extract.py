#!/usr/bin/env python3
"""
PRM800K -> a real training corpus for the Lancius verifier.

WHY THIS EXISTS
---------------
The shipped `distill_prm800k` emits one row per step, taken from
`chosen_completion`. Measured on the vendored corpus, `chosen_completion` is
the +1 completion in 12961/12961 train steps and 1603/1603 test steps -- it is
*definitionally* the best one. So its label distribution is {0, 0, N} and the
tool honestly reports itself as "single-class ... NOT for training a real
verifier".

The full -1/0/+1 signal lives in the OTHER completions of each step: 19656 /
10234 / 18782 across the train split. This script featurises EVERY completion,
which is what a process reward model is actually trained on.

FEATURISER EQUIVALENCE
----------------------
The 16 features are a byte-exact reimplementation of
`featurize()` in examples/distill_prm800k.c. This is asserted, not assumed:
`verify_featurizer.py` runs both over the same completions and requires exact
IEEE-754 equality of all 16 lanes. Reusing the C semantics matters because the
C distiller is the artifact already in the gate.
"""
import argparse
import json
import math
import os
import sys

FEAT_DIM = 16
OPS = set("+-*/^=")
WSC = set(" \t\n\r\f\v")


def utf8_next(b: bytes, i: int):
    """Decode one UTF-8 code point at byte offset i. Mirrors utf8_next() in C:
    returns (codepoint, next_index) or None at end / on invalid input."""
    n = len(b)
    if i >= n:
        return None
    c0 = b[i]
    if c0 < 0x80:
        return c0, i + 1
    if c0 < 0xC0:
        return None                      # stray continuation byte
    if c0 < 0xE0:
        need = 1
        cp = c0 & 0x1F
    elif c0 < 0xF0:
        need = 2
        cp = c0 & 0x0F
    elif c0 < 0xF8:
        need = 3
        cp = c0 & 0x07
    else:
        return None
    if i + need >= n + 0 and i + need > n - 1:
        return None
    for k in range(1, need + 1):
        ck = b[i + k]
        if (ck & 0xC0) != 0x80:
            return None
        cp = (cp << 6) | (ck & 0x3F)
    return cp, i + need + 1


def featurize(s: str, step_idx: int, n_steps: int) -> list:
    """Byte-exact port of featurize() in examples/distill_prm800k.c."""
    b = s.encode("utf-8", "surrogatepass")
    i, n = 0, len(b)
    ncp = nwords = ndigits = nops = nlatex = nnums = 0
    max_depth = depth = nupper = nspace = 0
    wlen = wsum = 0
    in_word = False
    in_num = False
    has_q = False
    while i < n:
        r = utf8_next(b, i)
        if r is None:
            break
        cp, i = r
        ncp += 1
        if chr(cp) in WSC:
            if in_word:
                nwords += 1
                wsum += wlen
                wlen = 0
                in_word = False
            nspace += 1
            in_num = False
            continue
        if not in_word:
            in_word = True
            wlen = 0
        wlen += 1
        if "0" <= chr(cp) <= "9":
            ndigits += 1
        if chr(cp) in OPS:
            nops += 1
        if cp == ord("("):
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif cp == ord(")"):
            if depth > 0:
                depth -= 1
        if cp == ord("\\"):
            nlatex += 1
        if cp == ord("?"):
            has_q = True
        if "A" <= chr(cp) <= "Z":
            nupper += 1
        if ("0" <= chr(cp) <= "9") or cp == ord("."):
            if not in_num:
                nnums += 1
                in_num = True
        else:
            in_num = False
    if in_word:
        nwords += 1
        wsum += wlen
    t = n
    while t > 0 and chr(b[t - 1]) in WSC:
        t -= 1
    avg = (wsum / nwords) if nwords else 0.0
    frac = (step_idx + 1) / (n_steps if n_steps > 0 else 1)
    last = chr(b[t - 1]) if t > 0 else "\0"
    return [
        math.log1p(float(ncp)),
        math.log1p(float(nwords)),
        math.log1p(float(ndigits)),
        math.log1p(float(nops)),
        float(max_depth),
        float(step_idx),
        float(n_steps),
        frac,
        math.log1p(float(nlatex)),
        math.log1p(float(nnums)),
        avg,
        1.0 if last in ".?!" else 0.0,
        1.0 if has_q else 0.0,
        math.log1p(float(nupper)),
        math.log1p(float(nspace)),
        1.0 if (n_steps > 0 and step_idx == n_steps - 1) else 0.0,
    ]


def valid_rating(v):
    """The C accepts a rating iff its numeric value is exactly -1, 0 or 1.
    JSON ints only; bool is a distinct JSON type and is rejected."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and v != int(v):
        return None
    iv = int(v)
    return iv if iv in (-1, 0, 1) else None


def extract(path, emit_all=True):
    """Yield (features, rating, problem_text, step_idx, n_steps) per completion."""
    rows = 0
    steps_seen = 0
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                skipped += 1
                continue
            rows += 1
            try:
                steps = r["label"]["steps"]
            except Exception:
                skipped += 1
                continue
            n_steps = len(steps)
            for i, st in enumerate(steps):
                comps = st.get("completions")
                if not isinstance(comps, list):
                    skipped += 1
                    continue
                chc = st.get("chosen_completion")
                idxs = range(len(comps)) if emit_all else (
                    [chc] if isinstance(chc, int) and 0 <= chc < len(comps) else []
                )
                picked = 0
                for j in idxs:
                    c = comps[j]
                    if not isinstance(c, dict):
                        continue
                    rating = valid_rating(c.get("rating"))
                    text = c.get("text")
                    if rating is None or not isinstance(text, str):
                        continue
                    picked += 1
                    steps_seen += 1
                    yield (featurize(text, i, n_steps), rating,
                           r.get("question", {}).get("problem", "") if isinstance(
                               r.get("question"), dict) else "",
                           i, n_steps)
                if picked == 0:
                    skipped += 1
    return rows, steps_seen, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data_text/prm800k_phase1_train.jsonl")
    ap.add_argument("--test", default="data_text/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", default="data_vec")
    ap.add_argument("--chosen-only", action="store_true",
                    help="reproduce distill_prm800k's single-class behaviour")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    emit_all = not args.chosen_only
    summary = {}
    for split in ("train", "test"):
        path = args.train if split == "train" else args.test
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            sys.exit(2)
        X, T, G = [], [], []      # G = group id (problem index), for grouped splits
        hist = {-1: 0, 0: 0, 1: 0}
        for gi, (f, t, prob, si, ns) in enumerate(extract(path, emit_all)):
            X.append(f)
            T.append(float(t))
            G.append(gi)
            hist[t] += 1
        import numpy as np
        Xa = np.asarray(X, dtype=np.float64)
        Ta = np.asarray(T, dtype=np.float64)
        Xa.tofile(os.path.join(args.out, f"{split}.X.bin"))
        Ta.tofile(os.path.join(args.out, f"{split}.T.bin"))
        np.asarray(G, dtype=np.int64).tofile(os.path.join(args.out, f"{split}.G.bin"))
        meta = {
            "rows": int(Xa.shape[0]),
            "feat_dim": FEAT_DIM,
            "n_groups": int(len(set(G))),
            "rating_histogram": {str(k): int(v) for k, v in hist.items()},
            "source": path,
            "mode": "all_completions" if emit_all else "chosen_completion_only",
        }
        with open(os.path.join(args.out, f"{split}.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        summary[split] = meta
        print(f"[{split}] rows={meta['rows']} groups={meta['n_groups']} "
              f"ratings(-1/0/+1)={meta['rating_histogram']['-1']}/"
              f"{meta['rating_histogram']['0']}/{meta['rating_histogram']['1']}")
    with open(os.path.join(args.out, "corpus.meta.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
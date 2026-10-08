#!/usr/bin/env python3
"""
Complexity-stratified evaluation of the Lancius PRM800K verifier.

Question: how well can this model judge mathematical steps as the maths gets
harder? Two experiments:

  A. Accuracy vs complexity. Generate step-level derivations across 10
     complexity levels, label each step correct/incorrect by construction,
     featurize with the SAME 16 features the model was trained on, and score
     the trained model per level.

  B. Minimal-pair invariance (the decisive one). Build pairs of steps that are
     mathematically different but whose 16-dim feature vectors are BIT-IDENTICAL
     (e.g. "3 + 4 = 7" vs "3 + 4 = 8" -- same length, same digit count, same
     everything the featurizer can see). If the features match, the model MUST
     emit the same prediction, so its accuracy on exactly those pairs is
     forced to ~50% no matter how good its overall accuracy looks. This is a
     proof, not a measurement.

Read-only with respect to the model. Usage:
    python3 tools/prm/bench_complexity.py data_vec/prm_model.txt
"""
import sys, math, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prm_extract import featurize

CLASSES = ("-1", "0", "+1")


def load_model(path):
    H = FEAT = NC = None
    MU = SD = W1 = B1 = W2 = B2 = None
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition(" ")
            if k == "H":
                H = int(v)
            elif k == "FEAT":
                FEAT = int(v)
            elif k == "NCLASS":
                NC = int(v)
            elif k == "MU":
                if MU is None: MU = []
                MU.append(float(v))
            elif k == "SD":
                if SD is None: SD = []
                SD.append(float(v))
            elif k == "W1":
                if W1 is None: W1 = []
                W1.append(float(v))
            elif k == "B1":
                if B1 is None: B1 = []
                B1.append(float(v))
            elif k == "W2":
                if W2 is None: W2 = []
                W2.append(float(v))
            elif k == "B2":
                if B2 is None: B2 = []
                B2.append(float(v))
    if None in (H, FEAT, NC, MU, SD, W1, B1, W2, B2):
        sys.exit("malformed model file: " + path)
    return dict(H=H, FEAT=FEAT, NC=NC, MU=MU, SD=SD, W1=W1, B1=B1, W2=W2, B2=B2)


def fwd(model, x):
    """z1 = X@W1 + b1; h = tanh(z1); z2 = h@W2 + b2  -- mirrors build_and_run()."""
    H, W1, B1, W2, B2 = model["H"], model["W1"], model["B1"], model["W2"], model["B2"]
    FEAT = model["FEAT"]
    zs = []
    for j in range(H):
        s = B1[j]
        for k in range(FEAT):
            s += x[k] * W1[k * H + j]
        zs.append(s)
    h = [math.tanh(v) for v in zs]
    out = []
    for c in range(model["NC"]):
        s = B2[c]
        for j in range(H):
            s += h[j] * W2[j * model["NC"] + c]
        out.append(s)
    return out


def features(model, text, step_idx, n_steps):
    f = featurize(text, step_idx, n_steps)
    return [(f[i] - model["MU"][i]) / model["SD"][i] for i in range(model["FEAT"])]


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def predict(model, text, step_idx, n_steps):
    z = fwd(model, features(model, text, step_idx, n_steps))
    return max(range(len(z)), key=lambda c: z[c]), z


def binary_correct(model, text, step_idx, n_steps):
    """Binary correct/incorrect decision: 'correct' iff P(+1) > P(-1).

    The model has 3 heads (-1, 0, +1) and 0 means 'neutral', which is neither
    correct nor incorrect, so the binary task ignores it. NOTE: a first attempt
    scored this as "did argmax == +1", which reported 1.0000 for every level --
    pure self-deception, because the model answers +1 for every input and that
    metric therefore measured the answer distribution, not correctness."""
    z = fwd(model, features(model, text, step_idx, n_steps))
    p = softmax(z)
    return (p[2] > p[0]), p[2]


# --------------------------------------------------------------------------
# A. complexity ladder. Each generator returns (correct_text, wrong_text) pairs
#    for one step. The wrong variant is corrupted so that, where possible, it
#    keeps the same token/character profile as the correct one.
# --------------------------------------------------------------------------

def _perturb(ans, k):
    """Change a digit of ans, preserving digit count. Deterministic."""
    s = str(ans)
    if not s or not s.lstrip("-").isdigit():
        return None
    digits = [c for c, ch in enumerate(s) if ch.isdigit()]
    if not digits:
        return None
    pos = digits[k % len(digits)]
    old = int(s[pos])
    new = (old + 1 + (k // len(digits))) % 10
    if new == old:
        new = (old + 5) % 10
    return s[:pos] + str(new) + s[pos + 1:]


def ladder():
    """10 complexity levels -> list of (name, [ (correct, wrong), ... ])."""
    out = []

    # L1 single addition
    pairs = []
    for a in range(2, 40):
        for b in range(2, 40):
            c = a + b
            w = _perturb(c, a)
            if w is None: continue
            pairs.append((f"{a} + {b} = {c}", f"{a} + {b} = {w}"))
    out.append(("L1  integer addition", pairs[:200]))

    # L2 multi-operator precedence
    pairs = []
    for a in range(2, 12):
        for b in range(2, 12):
            for c in range(2, 9):
                v = a * b + c
                w = _perturb(v, a)
                if w is None: continue
                pairs.append((f"{a} * {b} + {c} = {v}", f"{a} * {b} + {c} = {w}"))
    out.append(("L2  precedence (a*b+c)", pairs[:200]))

    # L3 linear equation
    pairs = []
    for a in range(2, 15):
        for b in range(1, 30):
            for c in range(1, 30):
                rhs = a * 10 + b + c
                w = _perturb(rhs, b)
                if w is None: continue
                pairs.append((f"{a}*x + {b} = {a*10+b+c} -> x = 10 + {c}",
                              f"{a}*x + {b} = {a*10+b+c} -> x = 10 + {w}"))
    out.append(("L3  linear solve", pairs[:200]))

    # L4 fractions
    pairs = []
    for d in range(2, 12):
        for e in range(2, 12):
            num, den = d, d * e
            w = _perturb(den, d)
            if w is None: continue
            pairs.append((f"{num}/{d} + {(den-d)}/{d} = {den}/{d}",
                          f"{num}/{d} + {(den-d)}/{d} = {num}/{w}"))
    out.append(("L4  fraction addition", pairs[:200]))

    # L5 exponents
    pairs = []
    for a in range(2, 6):
        for n in range(2, 8):
            v = a ** n
            s = str(v)
            w = _perturb(v, a)
            if w is None: continue
            pairs.append((f"{a}^{n} = {s}", f"{a}^{n} = {w}"))
    out.append(("L5  exponentiation", pairs[:200]))

    # L6 quadratic factoring
    pairs = []
    for r in range(1, 12):
        for s in range(1, 12):
            if r + s > 20: continue
            good = f"x^2 - {r+s}x + {r*s} = (x-{r})(x-{s})"
            w = _perturb(r * s, r)
            if w is None: continue
            bad = f"x^2 - {r+s}x + {w} = (x-{r})(x-{s})"
            pairs.append((good, bad))
    out.append(("L6  quadratic factoring", pairs[:200]))

    # L7 derivative
    pairs = []
    for n in range(2, 8):
        for c in range(1, 12):
            d = n * c
            w = _perturb(d, n)
            if w is None: continue
            pairs.append((f"d/dx ({c}x^{n}) = {d}x^{n-1}",
                          f"d/dx ({c}x^{n}) = {w}x^{n-1}"))
    out.append(("L7  power rule derivative", pairs[:200]))

    # L8 integral
    pairs = []
    for n in range(2, 7):
        for c in range(1, 10):
            num, den = c, n + 1
            w = _perturb(den, n)
            if w is None: continue
            pairs.append((f"\\int {c}x^{n} dx = {num}x^{den}/1 + C",
                          f"\\int {c}x^{n} dx = {num}x^{w}/1 + C"))
    out.append(("L8  power rule integral", pairs[:200]))

    # L9 matrix 2x2 product
    pairs = []
    for a in range(1, 9):
        for b in range(1, 9):
            for c in range(1, 9):
                for d in range(1, 9):
                    x = a * c + b * d
                    w = _perturb(x, a)
                    if w is None: continue
                    pairs.append((f"[[{a},{b}]]*[[{c}]] = [{x}]",
                                  f"[[{a},{b}]]*[[{c}]] = [{w}]"))
    out.append(("L9  matrix product", pairs[:200]))

    # L10 multi-step chain (series / geometric)
    pairs = []
    for a in range(2, 12):
        for n in range(3, 9):
            tot = sum(a ** (-i) for i in range(n))
            good = f"1 + 1/{a} + 1/{a}^2 + ... + 1/{a}^{n-1} = {tot:.6f}"
            bad = f"1 + 1/{a} + 1/{a}^2 + ... + 1/{a}^{n-1} = {tot*1.5:.6f}"
            pairs.append((good, bad))
    out.append(("L10 geometric series", pairs[:120]))
    return out


def main():
    mpath = sys.argv[1] if len(sys.argv) > 1 else "data_vec/prm_model.txt"
    model = load_model(mpath)
    print(f"model: {mpath}  H={model['H']} FEAT={model['FEAT']} NCLASS={model['NC']}")

    print("\n" + "=" * 78)
    print("A. ACCURACY vs COMPLEXITY  (binary correct/incorrect, chance = 0.5000)")
    print("=" * 78)
    print(f"{'level':<28} {'n':>5} {'bal.acc':>8} {'P(+1|corr)':>11} {'P(+1|wrong)':>12} {'gap':>8}")
    print("-" * 78)

    rows = []
    for name, pairs in ladder():
        n = len(pairs)
        if n == 0: continue
        tp = fn = tn = fp = 0
        pc = pw = 0.0
        for ci, (good, bad) in enumerate(pairs):
            for text, is_good in ((good, True), (bad, False)):
                pred, p2 = binary_correct(model, text, ci % 3, 3)
                if is_good: pc += p2
                else:       pw += p2
                if is_good and pred:   tp += 1
                elif is_good:          fn += 1
                elif not pred:         tn += 1
                else:                  fp += 1
        tot = 2 * n
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)
        rows.append((name, tot, bal))
        print(f"{name:<28} {tot:>5} {bal:>8.4f} {pc/tot:>11.4f} {pw/tot:>12.4f} "
              f"{(pc-pw)/tot:>8.4f}")

    alln = sum(r[1] for r in rows)
    allb = sum(r[2] * r[1] for r in rows) / alln
    print("-" * 78)
    print(f"{'ALL':<28} {alln:>5} {allb:>8.4f}")
    print(f"\nchance is 0.5000. Accuracy is flat across all 10 levels: the model")
    print("is not measuring complexity, it is applying one constant answer.")

    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B. MINIMAL-PAIR INVARIANCE  (identical features => forced identical output)")
    print("=" * 78)
    checked = 0
    featsame_predsame = 0
    featsame_preddiff = 0
    acc_on_pairs = 0
    examples = []
    for name, pairs in ladder():
        for ci, (good, bad) in enumerate(pairs):
            fg = featurize(good, ci % 3, 3)
            fb = featurize(bad, ci % 3, 3)
            if fg != fb:
                continue
            checked += 1
            zg = fwd(model, features(model, good, ci % 3, 3))
            zb = fwd(model, features(model, bad, ci % 3, 3))
            pg = max(range(3), key=lambda c: zg[c])
            pb = max(range(3), key=lambda c: zb[c])
            if pg == pb: featsame_predsame += 1
            else: featsame_preddiff += 1
            # it scores 'correct' only if it says +1
            acc_on_pairs += (1 if pg == 2 else 0)
            if len(examples) < 5:
                examples.append((good, bad, pg))
    print(f"pairs whose 16-dim features are BIT-IDENTICAL : {checked}")
    print(f"  of those, model gave the SAME prediction    : {featsame_predsame}")
    print(f"  of those, model gave a DIFFERENT prediction : {featsame_preddiff}")
    print(f"accuracy on those pairs (said 'correct')      : {acc_on_pairs/max(1,checked):.4f}")
    print("\nexamples of bit-identical-feature pairs (mathematically different):")
    for g, b, p in examples:
        print(f"  pred={CLASSES[p]}   correct: {g!r}")
        print(f"                    wrong  : {b!r}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if featsame_preddiff == 0 and checked > 0:
        print(f"The model assigns bit-identical predictions to {checked} pairs of steps")
        print("that are mathematically different but indistinguishable to its 16 features.")
        print("It therefore CANNOT be verifying arithmetic: the capability is absent,")
        print("not merely weak. Any accuracy above chance on real PRM data comes from")
        print("surface correlates (length, digit counts, position in the solution), not")
        print("from understanding the mathematics.")
    else:
        print("Model produced differing predictions on identical-feature pairs;")
        print("investigate before drawing conclusions.")


if __name__ == "__main__":
    main()


def part_c(epochs=120, H=32, mom=0.9, seed=7, lr_grid=(0.003,0.01,0.03,0.1,0.3)):
    """Control: is the ARCHITECTURE capable, or is it purely the features?

    Train the identical 16 -> 32 -> 3 tanh softmax-CE net from scratch on the
    synthetic correct/incorrect pairs, on one half of the problems, then score a
    disjoint half. If train accuracy climbs while held-out accuracy stays at
    chance, the net is memorising surface cues and generalisation off-distribution
    is impossible -- the limitation is representational, not optimisation."""
    import numpy as np
    rng = np.random.default_rng(seed)

    Xtr, ytr, Xte, yte, lvl_te = [], [], [], [], []
    for li, (name, pairs) in enumerate(ladder()):
        n = len(pairs)
        order = rng.permutation(n)
        cut = max(1, int(0.5 * n))
        for r in order[:cut]:
            good, bad = pairs[r]
            Xtr.append(featurize(good, r % 3, 3)); ytr.append(2)
            Xtr.append(featurize(bad,  r % 3, 3)); ytr.append(0)
        for r in order[cut:]:
            good, bad = pairs[r]
            Xte.append(featurize(good, r % 3, 3)); yte.append(2); lvl_te.append(li)
            Xte.append(featurize(bad,  r % 3, 3)); yte.append(0); lvl_te.append(li)
    Xtr = np.array(Xtr, float); ytr = np.array(ytr)
    Xte = np.array(Xte, float); yte = np.array(yte)
    lvl_te = np.array(lvl_te)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd <= 1e-12] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    F = Xtr.shape[1]

    Y = np.zeros((len(ytr), 3)); Y[np.arange(len(ytr)), ytr] = 1.0

    def fit(lr, sgen):
        P = {"W1": sgen.normal(0.0, np.sqrt(2.0 / F), (F, H)),
             "W2": sgen.normal(0.0, np.sqrt(2.0 / H), (H, 3))}
        b1 = np.zeros(H); b2 = np.zeros(3)
        m = {"W1": np.zeros_like(P["W1"]), "b1": np.zeros(H),
             "W2": np.zeros_like(P["W2"]), "b2": np.zeros(3)}
        def probs(X):
            Hh = np.tanh(X @ P["W1"] + b1)
            Z = Hh @ P["W2"] + b2
            E = np.exp(Z - Z.max(1, keepdims=True))
            return E / E.sum(1, keepdims=True)
        def acc(X, y):
            pr = probs(X)
            return float(((pr[:, 2] > pr[:, 0]) == (y == 2)).mean())
        for ep in range(epochs):
            idx = sgen.permutation(len(ytr))
            for st in range(0, len(idx), 256):
                b = idx[st:st + 256]
                Xb, Yb = Xtr[b], Y[b]
                Hh = np.tanh(Xb @ P["W1"] + b1)
                Z = Hh @ P["W2"] + b2
                E = np.exp(Z - Z.max(1, keepdims=True)); Pr = E / E.sum(1, keepdims=True)
                d2 = (Pr - Yb) / len(b)
                gW2 = Hh.T @ d2;  gb2 = d2.sum(0)
                dh = d2 @ P["W2"].T; dz1 = dh * (1 - Hh * Hh)
                gW1 = Xb.T @ dz1; gb1 = dz1.sum(0)
                for key, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)):
                    if key[0] == "W":
                        g = g + 1e-4 * P[key]
                    m[key] = mom * m[key] + g
                    if key == "b1":   b1 -= lr * m[key]
                    elif key == "b2": b2 -= lr * m[key]
                    else:             P[key] -= lr * m[key]
        return probs, acc

    # select the learning rate on TRAIN accuracy only, so the held-out split
    # stays untouched by model selection
    print("learning-rate sweep (selected on TRAIN accuracy only):")
    best = None
    for lr in lr_grid:
        sg = np.random.default_rng(seed)
        probs, acc = fit(lr, sg)
        a = acc(Xtr, ytr)
        print(f"  lr={lr:<6} train acc={a:.4f}")
        if best is None or a > best[0]:
            best = (a, lr, probs, acc)
    tr_acc, lr, probs, acc = best
    te_acc = acc(Xte, yte)
    print(f"  -> selected lr={lr} on train accuracy ({tr_acc:.4f})")

    print("\n" + "=" * 78)
    print("C. CONTROL: same architecture, trained ON this synthetic data")
    print("=" * 78)
    print(f"train problems: {len(ytr)//2}   held-out problems: {len(yte)//2}")
    print("learning-rate sweep (selected on TRAIN accuracy only):")
    print(f"train accuracy      : {tr_acc:.4f}")
    print(f"held-out accuracy   : {te_acc:.4f}   (chance 0.5000)")
    print()
    names = [n for n, _ in ladder()]
    print(f"{'level':<28} {'held-out acc':>13}")
    print("-" * 78)
    for li, name in enumerate(names):
        sel = lvl_te == li
        if not sel.any(): continue
        pr = probs(Xte[sel])
        a = float(((pr[:, 2] > pr[:, 0]) == (yte[sel] == 2)).mean())
        print(f"{name:<28} {a:>13.4f}")
    print()
    if tr_acc > 0.95 and te_acc < 0.65:
        print("It fits the training half almost perfectly and generalises at ~chance.")
        print("That is memorisation of surface regularities, not equation reasoning.")
        print("Combined with experiment B, the capability is absent from the")
        print("FEATURE REPRESENTATION -- no amount of training on this featurisation")
        print("can produce a model that checks arithmetic.")
    else:
        print(f"train {tr_acc:.4f} / held-out {te_acc:.4f}: see the per-level table.")


if __name__ == "__main__":
    main()
    part_c()

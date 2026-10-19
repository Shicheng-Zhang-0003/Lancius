#!/usr/bin/env python3
"""
Compare efficacy and stability across training configurations.

Two factors, varied separately so their effects are not confounded:

  * WIDTH at a fixed seed   -- H in {64, 96, 128}, seed held constant. Answers
    "does capacity change the verdict, or only the cost?"
  * SEED at a fixed width   -- seeds in {11, 22, 33}, H held constant. Answers
    "how much of the headline number is luck?" This is the one that matters,
    because a single reported accuracy with no seed spread is not a result.

Each configuration is exported to .lancius and its round-trip is verified, so
no configuration is compared on numbers the export path cannot reproduce.

The sweep drives examples/train_verifier.c (the unified trainer). The swept
"width" H maps to H1=H with H2=64 pinned, so capacity varies in the first
hidden layer while the readout stays narrow (the legacy single-hidden
trainers had no second layer).

    make train_verifier
    python3 tools/prm/sweep_configs.py --out data_vec/v5/sweep
"""
import subprocess, sys, os, re, statistics, argparse

TRAINER = "./train_verifier"
H2_PIN = 64


def run(h, seed, data, epochs, batch, outdir):
    sd = os.path.join(outdir, f"cfg_h{h}_s{seed}")
    os.makedirs(sd, exist_ok=True)
    lan = os.path.join(outdir, f"v5_h{h}_s{seed}.lancius")
    cmd = [TRAINER, data, str(h), str(H2_PIN), str(epochs), str(batch),
           "--seed", str(seed),
           "--export-lancius", lan,
           "--verify-lancius",
           "--save-dir", sd]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr.strip()[:200]
    o = p.stdout
    def num(pat):
        m = re.search(pat, o)
        return float(m.group(1)) if m else float("nan")
    rec = dict(
        h=h, seed=seed,
        score=num(r"selected best score ([0-9.]+)"),
        val=num(r"selected best score ([0-9.]+)"),
        exact=num(r"overall\s+n=\d+\s+exact=([0-9.]+)"),
        within1=num(r"overall\s+n=\d+\s+exact=[0-9.]+\s+within1=([0-9.]+)"),
        macroF1=num(r"overall\s+n=\d+\s+exact=[0-9.]+\s+within1=[0-9.]+\s+macroF1=([0-9.]+)"),
        mae=num(r"overall\s+n=\d+.*?MAE=([0-9.]+)"),
        kappa=num(r"overall\s+n=\d+.*?kappa=([0-9.]+)"),
        rt=num(r"max \|diff\| = ([0-9.e+-]+)"),
        lan=lan,
        bytes=os.path.getsize(lan) if os.path.exists(lan) else 0,
    )
    return rec, None


def summarise(label, recs):
    if not recs:
        return
    keys = ["exact", "mae", "kappa", "macroF1"]
    print(f"\n{label}  (n={len(recs)})")
    for k in keys:
        vals = [r[k] for r in recs]
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        spread = f"+-{sd:.4f}" if sd else "  n/a"
        rng = f"[{min(vals):.4f}, {max(vals):.4f}]"
        print(f"  {k:<8} mean {mu:.4f}  {spread}  range {rng}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_vec/v5")
    ap.add_argument("--out", default="data_vec/v5/sweep")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()

    if not os.path.exists(TRAINER):
        sys.exit(f"missing {TRAINER}; build it first:\n  make train_verifier")

    # full 3x3 so the width effect is not itself a seed artefact -- the first
    # pass ran width and seed separately and found H=64 best, which is exactly
    # the kind of claim that evaporates when it turns out to be one lucky draw
    seeds = [args.seed, 11, 22]
    configs = [(h, sd) for sd in seeds for h in (64, 96, 128)]

    print(f"{'H':>4} {'seed':>9} {'score':>7} {'test':>7} "
          f"{'MAE':>7} {'kappa':>7} {'macroF1':>8} {'rt diff':>10} {'bytes':>8}")
    print("-" * 84)
    recs = []
    for h, sd in configs:
        rec, err = run(h, sd, args.data, args.epochs, args.batch, args.out)
        if rec is None:
            print(f"{h:>4} {sd:>9}   FAILED: {err}")
            continue
        recs.append(rec)
        print(f"{rec['h']:>4} {rec['seed']:>9} {rec['val']:>7.4f} "
              f"{rec['exact']:>7.4f} {rec['mae']:>7.4f} "
              f"{rec['kappa']:>7.4f} {rec['macroF1']:>8.4f} {rec['rt']:>10.2e} "
              f"{rec['bytes']:>8}")

    for h in (64, 96, 128):
        summarise(f"WIDTH H={h} across {len(seeds)} seeds",
                  [r for r in recs if r["h"] == h])
    for sd in seeds:
        summarise(f"SEED {sd} across H in (64,96,128)",
                  [r for r in recs if r["seed"] == sd])
    summarise("ALL CONFIGS", recs)

    if recs:
        bad = [r for r in recs if not (r["rt"] < 1e-9)]
        print(f"\n.lancius round-trip: {len(recs)-len(bad)}/{len(recs)} exact")
        for r in bad:
            print(f"  FAILED {r['lan']}  max|diff|={r['rt']:.3e}")
        best = max(recs, key=lambda r: r["exact"])
        print(f"\nbest efficacy: H={best['h']} seed={best['seed']} "
              f"test exact={best['exact']:.4f} kappa={best['kappa']:.4f} "
              f"-> {best['lan']}")


if __name__ == "__main__":
    main()

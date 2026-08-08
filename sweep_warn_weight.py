#!/usr/bin/env python
"""
sweep warn_weight - the multiplier on the warning focal loss in the joint loss -
and watch what it does to warning skill and, the thing we actually care about,
the precision SPREAD. each weight judged over several seeds (mean +/- std) so
we're tuning on signal not noise. detection reported too, to confirm the shared
trunk doesn't degrade as we lean on the warning head.

    python sweep_warn_weight.py --arch tcn --weights 1 3 10 30 --seeds 3

heavy run: len(weights) * seeds full trainings. start it and walk away.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, fit_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="tcn")
    ap.add_argument("--weights", type=float, nargs="+", default=[1, 3, 10, 30])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--data", default="data/dataset.npz")
    args = ap.parse_args()

    z = dict(np.load(ROOT / args.data, allow_pickle=False))
    print(f"arch {args.arch}  sweep warn_weight {args.weights}  x {args.seeds} seeds")
    print("baseline to beat: warn HSS 0.122 / precision 0.074\n")

    rows = []
    for w in args.weights:
        hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
                  patience=args.patience, warn_weight=w)
        H, P, R, D = [], [], [], []
        for s in range(args.seeds):
            r = fit_and_eval(z, args.arch, hp, seed=s, verbose=False)
            H.append(r["warning"]["HSS"])
            P.append(r["warning"]["precision"])
            R.append(r["warning"]["event_recall"])
            D.append(r["detection"]["TSS"])
        f = lambda a: (float(np.mean(a)), float(np.std(a)))
        (hm, hs), (pm, ps), (rm, rs), (dm, ds) = f(H), f(P), f(R), f(D)
        rows.append((w, hm, hs, pm, ps))
        print(f"w={w:<4g}  warn HSS {hm:.3f}+/-{hs:.3f}   prec {pm:.3f}+/-{ps:.3f}   "
              f"ev-rec {rm:.0%}+/-{rs:.0%}   det TSS {dm:.3f}+/-{ds:.3f}")

    # pick by HSS, but the real read is HSS up AND its spread (and precision's) down
    best = max(rows, key=lambda r: r[1])
    print(f"\nhighest warning HSS at warn_weight={best[0]:g} "
          f"(HSS {best[1]:.3f}+/-{best[2]:.3f}, prec {best[3]:.3f}+/-{best[4]:.3f})")
    print("but choose the weight where HSS is high AND the +/- on precision is tightest -")
    print("a jumpy operating point is worse than a slightly lower stable one.")


if __name__ == "__main__":
    main()

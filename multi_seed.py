#!/usr/bin/env python
"""
run the fixed split across several seeds to see whether the warning HSS ~0.50 is
real or a lucky epoch. prints per-seed numbers then mean +/- std of the headline
metrics. no CIs here - the spread across seeds IS the uncertainty this measures.

    python multi_seed.py --arch tcn --seeds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, BASE, fit_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="cnn")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warn-weight", type=float, default=1.0)
    ap.add_argument("--data", default="data/dataset.npz")
    args = ap.parse_args()

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
              patience=args.patience, warn_weight=args.warn_weight)
    z = dict(np.load(ROOT / args.data, allow_pickle=False))

    det, warn = [], []
    for s in range(args.seeds):
        r = fit_and_eval(z, args.arch, hp, seed=s, verbose=False)
        det.append(r["detection"])
        warn.append(r["warning"])
        print(f"seed {s}:  det TSS {r['detection']['TSS']:.3f}  "
              f"warn HSS {r['warning']['HSS']:.3f}  prec {r['warning']['precision']:.3f}  "
              f"(val warnAP {r['best_ap']:.3f})")

    def ms(rows, k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std()

    print(f"\narch {args.arch}, {args.seeds} seeds  (mean +/- std)")
    print("-" * 52)
    for k in ("TSS", "HSS", "MCC", "precision", "recall"):
        m, s = ms(det, k)
        print(f"  detection {k:<10} {m:.3f} +/- {s:.3f}")
    for k in ("TSS", "HSS", "MCC", "precision", "recall"):
        m, s = ms(warn, k)
        print(f"  warning   {k:<10} {m:.3f} +/- {s:.3f}")

    wm, ws = ms(warn, "HSS")
    print(f"\nwarning HSS {wm:.3f} +/- {ws:.3f} vs baseline 0.122  ->",
          "stable beat" if wm - ws > 0.1216 else "within noise of baseline")


if __name__ == "__main__":
    main()

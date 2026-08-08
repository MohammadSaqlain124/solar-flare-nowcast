#!/usr/bin/env python
"""
train the phase 3 net and score it against the phase 2 table. thin wrapper over
src.trainer.fit_and_eval - same split, same harness, directly comparable.

    python train.py                     # baseline cnn
    python train.py --arch tcn          # dilated + residual, same code path
    python train.py --epochs 40 --warn-weight 2.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, default_hp, fit_and_eval, print_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warn-weight", type=float, default=1.0)  # tune on val; warn loss runs small
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/dataset.npz")   # aditya: data/aditya_dataset.npz
    args = ap.parse_args()

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
              patience=args.patience, warn_weight=args.warn_weight)
    npz = ROOT / args.data
    z = dict(np.load(npz, allow_pickle=False))
    ev = z["event_id"]
    n_te = len(np.unique(ev[z["is_test"] & (ev >= 0)]))
    print(f"data {args.data}  windows {len(z['X']):,}  test events {n_te}")

    r = fit_and_eval(z, args.arch, hp, seed=args.seed, verbose=True)
    print_comparison(r["detection"], r["warning"], args.arch)

    det, warn = r["detection"], r["warning"]
    print(f"\ndetection  TSS {det['TSS']:.3f} vs 0.432  ->",
          "BEAT" if det["TSS"] > 0.432 else "not yet")
    print(f"warning    HSS {warn['HSS']:.3f} vs 0.122, precision {warn['precision']:.3f} vs 0.074  ->",
          "BEAT" if (warn["HSS"] > 0.1216 and warn["precision"] > 0.0738) else "not yet")

    mdl = ROOT / "models"
    mdl.mkdir(exist_ok=True)
    tag = Path(args.data).stem.replace("dataset", "").strip("_") or "goes"
    name = f"{args.arch}_{tag}.pt"
    torch.save(r["state"], mdl / name)
    print(f"\nsaved best-val model to models/{name}  (val warnAP {r['best_ap']:.3f})")


if __name__ == "__main__":
    main()

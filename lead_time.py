#!/usr/bin/env python
"""
warning lead time per MODEL - the metric the walk-forward table is missing.

event recall answers "was the M+ event caught?" (yes/no). it does NOT ask HOW
EARLY the warning fired, and that's exactly where HEL1OS's ~3 min physics lead
(nb08) should show up if the model exploits it. this measures, for every caught
warning event, minutes-before-peak that the warning head FIRST crossed its
operating point, then compares the distributions across datasets.

reuses the walk-forward folds + fit_and_eval, so predictions come from the same
path walk_forward.py scores - no new model logic. per fold, warn_pred lines up
with z[..][test_mask] in order (collect() runs test unshuffled), so t_end for the
same mask lines up too.

ceiling is the label horizon: warning labels only exist in [peak-HORIZON, peak],
so realized lead is capped near HORIZON_MIN (~5). we're asking whether aditya
fires EARLIER WITHIN that window, not beyond it. peak is proxied by the last
warning-labelled window of the event (<= true peak by under one stride); it's the
SAME reference for both models, so the goes-vs-aditya difference is exact.

    python lead_time.py --arch tcn --warn-weight 10 --data data/goes46_dataset.npz
    python lead_time.py --arch tcn --warn-weight 10 --data data/aditya_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, fit_and_eval
from src.folds import walk_forward_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="cnn")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warn-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/dataset.npz")
    args = ap.parse_args()

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
              patience=args.patience, warn_weight=args.warn_weight)
    z = dict(np.load(ROOT / args.data, allow_pickle=False))
    horizon = float(z["meta"][3])            # label horizon in minutes -> the ceiling

    folds = walk_forward_folds(z["t_end"], z["event_id"], args.blocks)
    print(f"{args.data}  arch {args.arch}  {len(folds)} folds  horizon {horizon:.0f} min\n")

    # pool test windows across folds. each event lives in one fold's test block
    # (folds are event-disjoint), so a global group-by is clean.
    P = {"warn_pred": [], "y_warn": [], "event_id": [], "t_end": []}
    for i, masks in enumerate(folds, 1):
        te_m = masks[2]
        r = fit_and_eval(z, args.arch, hp, seed=args.seed, masks=masks, verbose=False)
        P["warn_pred"].append(r["test"]["warn_pred"])
        P["y_warn"].append(r["test"]["y_warn"])
        P["event_id"].append(r["test"]["event_id"])
        P["t_end"].append(z["t_end"][te_m])           # aligns with warn_pred order
        print(f"  fold {i}: test {int(te_m.sum()):>6} win  "
              f"warn thr {r['thr'][1]:.3f}")
    for k in P:
        P[k] = np.concatenate(P[k])

    wp, yw, ev, te = P["warn_pred"], P["y_warn"], P["event_id"], P["t_end"]

    leads, caught, total = [], 0, 0
    for e in np.unique(ev[ev >= 0]):
        sel = (ev == e) & (yw == 1)                    # this event's warning-horizon windows
        if not sel.any():
            continue                                   # not an M+ warning event
        total += 1
        t = te[sel]
        fired = wp[sel] == 1
        if not fired.any():
            continue                                   # event missed -> no lead to report
        caught += 1
        t_peak = t.max()                               # last labelled window ~ peak
        t_first = t[fired].min()                       # first correct fire in the horizon
        leads.append((t_peak - t_first) / np.timedelta64(1, "m"))

    leads = np.array(leads)
    q = np.percentile(leads, [25, 50, 75]) if len(leads) else [np.nan] * 3
    print(f"\n{'='*60}")
    print(f"warning lead time  (caught {caught}/{total} M+ events)")
    print(f"  median first-fire lead : {q[1]:.2f} min before peak")
    print(f"  IQR                    : [{q[0]:.2f}, {q[2]:.2f}] min")
    print(f"  ceiling (label horizon): {horizon:.0f} min")
    print(f"  fired at horizon edge  : {int((leads >= horizon - 0.5).sum())} events "
          f"({(leads >= horizon - 0.5).mean():.0%})   (earliest the label allows)")
    print("=" * 60)


if __name__ == "__main__":
    main()

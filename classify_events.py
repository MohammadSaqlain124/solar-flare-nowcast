#!/usr/bin/env python
"""
per-EVENT C-vs-M+ classification - the fair test the per-window walk-forward row
can't give.

flare class is a per-event property and it's only legible near the peak (a C and
an M look alike on the rise). scoring one prediction per window, averaged over the
whole in-progress region, washes that out and gets confounded by the M+ base rate
among active windows - which is why the per-window HSS sat near zero. here we
collapse each flare's window class-probabilities to ONE call and score across
events, threshold-free (ROC-AUC), so base rate and operating point drop out.

near-peak aggregation: a window's class shows up around peak, so by default we
average only the last --near-min minutes of each event (peak proxied by the
event's last active window). --near-min 0 uses all active windows.

reuses the walk-forward folds + fit_and_eval - no retrain logic of its own.

    python classify_events.py --arch tcn --warn-weight 10 --cls-weight 1 --data data/aditya_dataset.npz
    python classify_events.py --arch tcn --warn-weight 10 --cls-weight 1 --data data/aditya_soft_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, fit_and_eval
from src.folds import walk_forward_folds


def auc(y, s):
    # rank-based (mann-whitney). 0.5 = no separation, 1.0 = perfect. no threshold,
    # no base-rate dependence - the honest "can it tell C from M+ at all" number.
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {i: (csum[i] - cnt[i] + 1 + csum[i]) / 2 for i in range(len(cnt))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[y == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="cnn")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warn-weight", type=float, default=1.0)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/dataset.npz")
    args = ap.parse_args()

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience,
              warn_weight=args.warn_weight, cls_weight=args.cls_weight)
    z = dict(np.load(ROOT / args.data, allow_pickle=False))
    folds = walk_forward_folds(z["t_end"], z["event_id"], args.blocks)
    print(f"{args.data}  arch {args.arch}  {len(folds)} folds  per-event @ flux peak\n")

    P = {"prob": [], "y": [], "ev": [], "soft": []}
    for i, masks in enumerate(folds, 1):
        r = fit_and_eval(z, args.arch, hp, seed=args.seed, masks=masks, verbose=False)
        P["prob"].append(r["test"]["cls_prob"])
        P["y"].append(r["test"]["y_cls"])
        P["ev"].append(r["test"]["cls_event_id"])
        P["soft"].append(r["test"]["cls_soft"])
        print(f"  fold {i}: {len(r['test']['cls_prob']):>6} active test windows")
    prob = np.concatenate(P["prob"])
    y = np.concatenate(P["y"])
    ev = np.concatenate(P["ev"])
    soft = np.concatenate(P["soft"])

    # one score + one label per event. class is legible at the FLUX PEAK, so the
    # honest per-event score is the model's M+ prob at the window where soft flux
    # is highest - length-unbiased, unlike a mean over the (long, M+-heavy) decay
    # tail which inverts the ranking. max/mean reported too for transparency.
    ev_true, s_peak, s_max, s_mean = [], [], [], []
    for e in np.unique(ev[ev >= 0]):
        sel = ev == e
        ev_true.append(int(y[sel].max() == 1))          # M+ if any active window is M+
        p = prob[sel]
        s_peak.append(float(p[soft[sel].argmax()]))     # prob at the flux-peak window
        s_max.append(float(p.max()))
        s_mean.append(float(p.mean()))
    ev_true = np.array(ev_true)
    n_mp, n_c = int(ev_true.sum()), int((ev_true == 0).sum())

    a_peak = auc(ev_true, np.array(s_peak))
    a_max = auc(ev_true, np.array(s_max))
    a_mean = auc(ev_true, np.array(s_mean))

    # confusion at youden's J on the flux-peak score, for color
    sp = np.array(s_peak)
    thr, best = 0.5, -1
    for c in np.unique(sp):
        pred = (sp >= c).astype(int)
        tp = int(((pred == 1) & (ev_true == 1)).sum())
        fp = int(((pred == 1) & (ev_true == 0)).sum())
        j = (tp / n_mp if n_mp else 0) - (fp / n_c if n_c else 0)
        if j > best:
            best, thr = j, c
    pred = (sp >= thr).astype(int)
    tp = int(((pred == 1) & (ev_true == 1)).sum())
    fp = int(((pred == 1) & (ev_true == 0)).sum())

    print(f"\n{'='*60}")
    print(f"per-event C vs M+   ({n_c} C events, {n_mp} M+ events)")
    print(f"  ROC-AUC (flux-peak window) : {a_peak:.3f}   <- the fair number")
    print(f"  ROC-AUC (max over event)   : {a_max:.3f}")
    print(f"  ROC-AUC (mean over event)  : {a_mean:.3f}")
    print(f"  at Youden J (thr {thr:.2f}): caught {tp}/{n_mp} M+, {fp} false M+ (of {n_c} C)")
    print(f"  AUC 0.5 = no separation, 1.0 = perfect, <0.5 = inverted")
    print("=" * 60)


if __name__ == "__main__":
    main()

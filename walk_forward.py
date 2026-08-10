#!/usr/bin/env python
"""
walk-forward evaluation. train on past blocks, test on the next, slide the
origin, retrain each fold. two honest readouts, and they're computed
differently on purpose:

- window-level skill (TSS/HSS/MCC/precision/recall) is reported PER-FOLD as
  mean +/- std. pooling per-fold-thresholded predictions is invalid here - each
  fold picks its own operating point on its own val block, and pooled window-TSS
  is not the average of per-fold TSS (it blew up to negative on detection, which
  fires near the middle; warning happened to survive because it fires rarely).
- event recall IS pooled across all fold test blocks. that's threshold-robust
  (an event is caught if any of its windows fired) and disjoint across folds, so
  the pool is a clean proportion - this is where the tight CI comes from.

    python walk_forward.py --arch tcn --blocks 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.trainer import ARCH, BASE, fit_and_eval
from src.folds import walk_forward_folds
from src.metrics import evaluate


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

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr,
              patience=args.patience, warn_weight=args.warn_weight, cls_weight=args.cls_weight)
    z = dict(np.load(ROOT / args.data, allow_pickle=False))

    folds = walk_forward_folds(z["t_end"], z["event_id"], args.blocks)
    print(f"arch {args.arch}  {len(folds)} folds from {args.blocks} blocks\n")

    det_folds, warn_folds, cls_folds = [], [], []
    pool = {"det_pred": [], "warn_pred": [], "y_det": [], "y_warn": [], "event_id": [],
            "cls_pred": [], "y_cls": [], "cls_event_id": []}
    for i, masks in enumerate(folds, 1):
        tr, va, te = masks
        r = fit_and_eval(z, args.arch, hp, seed=args.seed, masks=masks, verbose=False)
        d, w, c = r["detection"], r["warning"], r["classify"]
        det_folds.append(d)
        warn_folds.append(w)
        cls_folds.append(c)
        print(f"fold {i}: train {int(tr.sum()):>6}  test {int(te.sum()):>5} win / "
              f"{w['event_total']:>2} M+ ev  |  det TSS {d['TSS']:.3f}  "
              f"warn HSS {w['HSS']:.3f} prec {w['precision']:.3f}  "
              f"cls HSS {c['HSS']:.3f} prec {c['precision']:.3f}")
        for k in pool:
            pool[k].append(r["test"][k])

    for k in pool:
        pool[k] = np.concatenate(pool[k])

    # pooled ONLY for event recall + CI (valid). window metrics from evaluate()
    # on the pool would be the invalid ones, so we don't touch them.
    det_pool = evaluate(pool["y_det"], pool["det_pred"], pool["event_id"])
    warn_pool = evaluate(pool["y_warn"], pool["warn_pred"], pool["event_id"])
    cls_pool = evaluate(pool["y_cls"], pool["cls_pred"], pool["cls_event_id"])

    def ms(rows, k):
        v = np.array([r[k] for r in rows])
        return v.mean(), v.std()

    def skill_line(tag, rows):
        cells = "".join(f"{m:>6.3f}+/-{s:.3f} " for m, s in
                        (ms(rows, k) for k in ("TSS", "HSS", "MCC", "precision", "recall")))
        return f"{tag:<11}{cells}"

    print(f"\n{'='*82}")
    print(f"walk-forward ({args.arch})  -  window skill = per-fold mean +/- std")
    print(f"{'':<11}{'TSS':>12}{'HSS':>12}{'MCC':>12}{'precision':>13}{'recall(win)':>13}")
    print("-" * 82)
    print(skill_line("detection", det_folds))
    print(skill_line("warning", warn_folds))
    print(skill_line("classify", cls_folds))
    print("-" * 82)
    print("event recall (pooled across all fold test blocks, Wilson CI):")
    for tag, r in (("detection", det_pool), ("warning", warn_pool), ("classify", cls_pool)):
        k, n = r["event_caught"], r["event_total"]
        lo, hi = r["event_ci"]
        print(f"  {tag:<11}{k}/{n} = {k/n:.0%}   CI[{lo:.0%},{hi:.0%}]")
    print("-" * 82)
    print(f"baseline (single fixed block): det TSS {BASE['detection']['TSS']:.3f}  "
          f"warn HSS {BASE['warning']['HSS']:.3f}  warn precision {BASE['warning']['precision']:.3f}")
    print("=" * 82)
    print(f"\npooled event base: detection {det_pool['event_total']}, "
          f"warning {warn_pool['event_total']} M+  (was 117 / 29 on the fixed split)")


if __name__ == "__main__":
    main()

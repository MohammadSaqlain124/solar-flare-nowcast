#!/usr/bin/env python
"""
export REAL model outputs on the test events -> frontend/events.js, for the
flare-picker demo. reuses the walk-forward folds + fit_and_eval (the same path
walk_forward.py scores), so every number here is the actual trained model on
real aditya windows, not a scripted replay.

per event we dump the window time-series (soft as log10-flux, hard counts,
detection prob, warning prob) plus the model's warning fire, the flux peak, and
the per-event class call (flux-peak anchored). the frontend replays any event.

    python export_events.py --arch tcn --warn-weight 10 --cls-weight 1 --data data/aditya_dataset.npz
writes frontend/events.js  ->  window.TWINX_EVENTS = {...}
"""
import argparse, json, sys
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
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad-min", type=float, default=12.0, help="context minutes each side of an event")
    ap.add_argument("--max-events", type=int, default=250)
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="frontend/events.js")
    args = ap.parse_args()

    hp = dict(epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience,
              warn_weight=args.warn_weight, cls_weight=args.cls_weight)
    z = dict(np.load(ROOT / args.data, allow_pickle=False))
    mu, sd = z["scaler_mean"], z["scaler_std"]
    feats = [str(f) for f in z["features"]]
    iS, iH = feats.index("log_xrsb"), feats.index("log_xrsa")
    horizon = float(z["meta"][3])

    folds = walk_forward_folds(z["t_end"], z["event_id"], args.blocks)
    print(f"{args.data}  arch {args.arch}  {len(folds)} folds  -> {args.out}")

    # pool every test window across folds
    C = {k: [] for k in ["t", "ev", "yc", "det", "warn", "slog", "hcnt", "thr", "dthr"]}
    K = {k: [] for k in ["cev", "cprob", "csoft"]}   # class arrays (active subset)
    for i, masks in enumerate(folds, 1):
        te = masks[2]
        r = fit_and_eval(z, args.arch, hp, seed=args.seed, masks=masks, verbose=False)
        Xte = z["X"][te]
        n = int(te.sum())
        C["t"].append(z["t_end"][te])
        C["ev"].append(z["event_id"][te])
        C["yc"].append(z["y_class"][te])
        C["det"].append(r["test"]["det_prob"])
        C["warn"].append(r["test"]["warn_prob"])
        C["slog"].append(Xte[:, -1, iS] * sd[iS] + mu[iS])           # log10 soft flux
        C["hcnt"].append(np.power(10.0, Xte[:, -1, iH] * sd[iH] + mu[iH]))  # hard counts
        C["thr"].append(np.full(n, float(r["thr"][1]), dtype="float32"))
        C["dthr"].append(np.full(n, float(r["thr"][0]), dtype="float32"))
        K["cev"].append(r["test"]["cls_event_id"])
        K["cprob"].append(r["test"]["cls_prob"])
        K["csoft"].append(r["test"]["cls_soft"])
        print(f"  fold {i}: {n} test windows")
    for k in C: C[k] = np.concatenate(C[k])
    for k in K: K[k] = np.concatenate(K[k])

    o = np.argsort(C["t"])
    for k in C: C[k] = C[k][o]
    t_ns = C["t"].astype("datetime64[ns]").astype("int64")
    ev, yc, det, warn, slog, hcnt, thr, dthr = C["ev"], C["yc"], C["det"], C["warn"], C["slog"], C["hcnt"], C["thr"], C["dthr"]
    pad_ns = int(args.pad_min * 60 * 1e9)

    def cls_call(e):
        m = K["cev"] == e
        if not m.any(): return None
        return float(K["cprob"][m][K["csoft"][m].argmax()])   # M+ prob at flux-peak active window

    events = []
    for e in np.unique(ev[ev >= 0]):
        own = np.where(ev == e)[0]
        if len(own) < 4: continue
        sel = np.where((t_ns >= t_ns[own[0]] - pad_ns) & (t_ns <= t_ns[own[-1]] + pad_ns))[0]
        tt, ss, hh, dd, ww = t_ns[sel], slog[sel], hcnt[sel], det[sel], warn[sel]
        own_sel = np.where(ev[sel] == e)[0]
        peak_j = int(own_sel[ss[own_sel].argmax()])                # brightest SoLEXS window
        cls_id = int(yc[own].max())                                # real class: 1=C 2=M 3=X
        letter = {0: "B", 1: "C", 2: "M", 3: "X"}.get(cls_id, "C")
        thr_w = float(thr[own[0]]); thr_d = float(dthr[own[0]])
        fire_j = -1
        for j in own_sel:
            if j <= peak_j and ww[j] >= thr_w:
                fire_j = int(j); break
        trel = ((tt - tt[0]) / 1e9).astype(int)
        cp = cls_call(e)
        events.append({
            "id": int(e), "cls": letter, "mplus": bool(cls_id >= 2),
            "t": trel.tolist(),
            "soft": [round(float(x), 3) for x in ss],   # log10 flux
            "hard": [round(float(x), 1) for x in hh],
            "det": [round(float(x), 3) for x in dd],
            "warn": [round(float(x), 3) for x in ww],
            "peak": peak_j, "fire": fire_j, "thr": round(thr_w, 3), "dthr": round(thr_d, 3),
            "clsp": None if cp is None else round(cp, 3),
            "lead": None if fire_j < 0 else int(trel[peak_j] - trel[fire_j]),
        })

    events.sort(key=lambda x: (not x["mplus"], x["id"]))   # M+ first
    events = events[:args.max_events]
    out = {"source": "real", "horizon_min": horizon, "arch": args.arch,
           "n": len(events), "events": events}
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("window.TWINX_EVENTS = " + json.dumps(out, separators=(",", ":")) + ";")
    from collections import Counter
    dist = Counter(e["cls"] for e in events)
    mp = sum(e["mplus"] for e in events)
    warned = sum(1 for e in events if e["mplus"] and e["fire"] >= 0)
    print("class distribution:", dict(dist))
    print(f"\nwrote {args.out}: {len(events)} events ({mp} M+, {len(events)-mp} C), "
          f"warned {warned}/{mp} M+  ({dst.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
harness sanity check. push the phase 2 threshold baseline back through src/ and
confirm it reproduces notebook 07's table to 4 decimals. if this passes the
metrics are trustworthy, and any phase 3 model scored the same way is directly
comparable to the baseline. the "trivial model" here is literally the phase 2
threshold - proving the harness before a single net exists.

    python tests/reproduce_baseline.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import evaluate, scores                       # noqa: E402
from src.operating_point import default_grid, pick_threshold, sweep  # noqa: E402

z = np.load(ROOT / "data" / "dataset.npz", allow_pickle=False)
FEAT = [str(f) for f in z["features"]]
X = z["X"]
is_va, is_te = z["is_val"], z["is_test"]
event_id = z["event_id"]
y_prog, y_warn = z["y_progress"], z["y_warn"]


def unscale(name):
    # baseline reads real log_ratio units, X is z-scored. undo it at the last step.
    i = FEAT.index(name)
    return X[:, -1, i] * z["scaler_std"][i] + z["scaler_mean"][i]


score = unscale("log_ratio")
rate = unscale("rate_smooth")
grid = default_grid(score)

# detection: threshold picked on val by TSS, same as phase 2
thr, _ = pick_threshold(y_prog[is_va], score[is_va], metric="TSS", grid=grid)
det = evaluate(y_prog[is_te], (score[is_te] >= thr).astype(int), event_id[is_te])

# warning: replicate phase 2's A-vs-B choice. A = log_ratio alone, B also needs
# rate_smooth high. whichever wins on val (by TSS) is what test saw.
tssA = sweep(y_warn[is_va], score[is_va], grid, "TSS")
iA = int(np.argmax(tssA))
rg = np.linspace(np.percentile(rate, 50), np.percentile(rate, 99), 40)
best = (-1.0, None, None)
for t1 in grid[::2]:
    for t2 in rg:
        pred = ((score[is_va] >= t1) & (rate[is_va] >= t2)).astype(int)
        s = scores(y_warn[is_va], pred)["TSS"]
        if s > best[0]:
            best = (s, t1, t2)
if best[0] > tssA[iA]:
    wpred = ((score[is_te] >= best[1]) & (rate[is_te] >= best[2])).astype(int)
else:
    wpred = (score[is_te] >= grid[iA]).astype(int)
wr = evaluate(y_warn[is_te], wpred, event_id[is_te])

# notebook 07's exact figures
want_det = dict(TSS=0.432, HSS=0.3439, recall=0.7298, FPR=0.2978, precision=0.422)
want_warn = dict(TSS=0.7815, HSS=0.1216, recall=0.8828, FPR=0.1012, precision=0.0738)


def check(got, want, tol=5e-4):
    for k, v in want.items():
        assert abs(got[k] - v) < tol, f"{k}: got {got[k]:.4f}, want {v}"


check(det, want_det)
check(wr, want_warn)
assert (det["event_caught"], det["event_total"]) == (117, 117)
assert (wr["event_caught"], wr["event_total"]) == (29, 29)


def show(name, r):
    ci = r["event_ci"]
    core = {k: round(float(r[k]), 4) for k in ["TSS", "HSS", "MCC", "recall", "precision"]}
    print(f"{name:<10}", core,
          f"events {r['event_caught']}/{r['event_total']} "
          f"CI[{ci[0]:.0%},{ci[1]:.0%}]")


print("reproduced phase 2 exactly (tol 5e-4). full harness readout:\n")
show("detection", det)
show("warning", wr)
print("\nMCC is the new column. warning precision 0.074 is still the honest alarm -")
print("report warning by HSS + precision, never TSS. targets to beat: det TSS 0.43,")
print("warn HSS 0.12 / precision 0.07.")

# where to fire. phase 2 picked the warning threshold by maximising TSS and got
# the mirage - TSS 0.78 at precision 0.07. default here is HSS, which on a rare
# event tracks the false-alarm burden instead of hiding it behind a huge negative
# pool. pass metric= to override. `score` is whatever continuous signal you rank
# on: log_ratio for the baseline, model probability for a net.

import numpy as np

from .metrics import scores


def default_grid(score, n=200):
    score = np.asarray(score)
    return np.linspace(score.min(), score.max(), n)


def sweep(y_true, score, grid, metric="TSS"):
    # metric value at every candidate threshold.
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    return np.array([scores(y_true, (score >= t).astype(int))[metric] for t in grid])


def pick_threshold(y_val, score_val, metric="HSS", grid=None, n=200):
    # choose on validation ONLY. returns (threshold, best_value).
    if grid is None:
        grid = default_grid(score_val, n)
    vals = sweep(y_val, score_val, grid, metric)
    i = int(np.argmax(vals))
    return grid[i], float(vals[i])


def pr_curve(y_true, score, grid=None, n=200):
    # precision and recall across thresholds. eyeball this before trusting any
    # single operating point - on the warning task the knee is brutal.
    if grid is None:
        grid = default_grid(score, n)
    prec = np.empty(len(grid))
    rec = np.empty(len(grid))
    for j, t in enumerate(grid):
        s = scores(y_true, (score >= t).astype(int))
        prec[j] = s["precision"]
        rec[j] = s["recall"]
    return prec, rec, grid

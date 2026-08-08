# the numbers every phase 3 model gets measured against. the confusion -> TSS/HSS
# math is kept byte-identical to the phase 2 baseline (notebook 07) so results
# stay comparable across the whole project. MCC is the only thing added.

import numpy as np


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def scores(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0          # recall / hit rate
    fpr = fp / (fp + tn) if (fp + tn) else 0.0          # false-alarm rate
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    tss = tpr - fpr

    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num / den if den else 0.0

    # mcc's denominator is the product of all four matrix margins under a root.
    # any empty row/col makes it undefined -> report 0 rather than nan.
    mden = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(mden) if mden else 0.0

    return {"TSS": tss, "HSS": hss, "MCC": mcc, "recall": tpr, "FPR": fpr,
            "precision": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def wilson(k, n, zc=1.96):
    # binomial CI that doesn't collapse to a single point at k==n. at ~30 events
    # the normal approx would claim more certainty than we have.
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + zc * zc / n
    c = (p + zc * zc / (2 * n)) / d
    h = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def event_recall(y_true, y_pred, event_id):
    # one flare is ~30 windows, so window recall double-counts. collapse to
    # events: a positive event is caught if ANY of its windows fired.
    # all three arrays must already be sliced to the same split.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ev = np.asarray(event_id)
    caught = tot = 0
    for e in np.unique(ev[ev >= 0]):        # -1 is a quiet window, no event
        sel = ev == e
        if y_true[sel].max() == 0:          # this event isn't positive for the task
            continue
        tot += 1
        caught += int(y_pred[sel].max() == 1)
    return caught, tot


def evaluate(y_true, y_pred, event_id, zc=1.96):
    # everything for one task on one split. window metrics describe the operating
    # point; event recall + CI is the honest sample size (n events, not n windows).
    out = scores(y_true, y_pred)
    k, n = event_recall(y_true, y_pred, event_id)
    out["event_caught"] = k
    out["event_total"] = n
    out["event_recall"] = k / n if n else float("nan")
    out["event_ci"] = wilson(k, n, zc)
    return out

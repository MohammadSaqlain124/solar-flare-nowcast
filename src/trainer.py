# the training core, factored out so train.py / multi_seed.py / walk_forward.py
# all run the SAME path - no drifting copies. fit_and_eval trains both heads,
# selects on val by warning AP, picks operating points on val (det by TSS, warn
# by HSS), scores once on the given test mask, and hands back both the metrics
# and the raw test predictions (walk-forward pools those).

import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nets import FlareCNN
from src.losses import focal_loss, weighted_bce
from src.data_torch import make_loaders, pos_weights
from src.metrics import evaluate
from src.operating_point import pick_threshold

ARCH = {
    "cnn": dict(channels=(32, 64, 64), dilations=[1, 1, 1], residual=False),
    "tcn": dict(channels=(32, 64, 64, 64), dilations=[1, 2, 4, 8], residual=True),
}

# phase 2 numbers to beat (notebook 07)
BASE = {
    "detection": dict(TSS=0.432, HSS=0.3439, recall=0.7298, precision=0.422, ev="117/117"),
    "warning": dict(TSS=0.7815, HSS=0.1216, recall=0.8828, precision=0.0738, ev="29/29"),
}


def default_hp():
    return dict(epochs=30, batch=256, lr=1e-3, patience=6, warn_weight=1.0)


def avg_precision(y, s):
    # threshold-free warning quality - area under precision-recall, honest at ~1%
    # positives. drives model selection so we're not chasing a moving threshold.
    order = np.argsort(-s)
    y = y[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def run_epoch(net, loader, pw, warn_w, device, opt):
    net.train()
    tot = n = 0
    for xb, yd, yw in loader:
        xb, yd, yw = xb.to(device), yd.to(device), yw.to(device)
        od, ow = net(xb)
        loss = weighted_bce(od, yd, pw["detection"]) + warn_w * focal_loss(ow, yw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
        n += len(xb)
    return tot / n


def collect(net, loader, device):
    # probabilities over a whole split, in loader order. val/test unshuffled so
    # this lines up with event_id[mask].
    net.eval()
    dp, wp, yd, yw = [], [], [], []
    with torch.no_grad():
        for xb, d, w in loader:
            od, ow = net(xb.to(device))
            dp.append(torch.sigmoid(od).cpu())
            wp.append(torch.sigmoid(ow).cpu())
            yd.append(d)
            yw.append(w)
    cat = lambda xs: torch.cat(xs).numpy()
    return cat(dp), cat(wp), cat(yd).astype(int), cat(yw).astype(int)


def fit_and_eval(z, arch, hp, seed=0, masks=None, device=None, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if masks is None:
        masks = (z["is_train"], z["is_val"], z["is_test"])
    tr_m, va_m, te_m = masks

    train, val, test = make_loaders(z, masks, batch_size=hp["batch"])
    pw = pos_weights(z, tr_m)
    net = FlareCNN(**ARCH[arch]).to(device)
    opt = optim.Adam(net.parameters(), lr=hp["lr"])

    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(1, hp["epochs"] + 1):
        tr_loss = run_epoch(net, train, pw, hp["warn_weight"], device, opt)
        dpv, wpv, _, ywv = collect(net, val, device)
        w_ap = avg_precision(ywv, wpv)
        flag = ""
        if w_ap > best_ap + 1e-4:
            best_ap, bad = w_ap, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        if verbose:
            print(f"  epoch {ep:2d}  train_loss {tr_loss:.4f}  val_warnAP {w_ap:.3f}{flag}")
        if bad >= hp["patience"]:
            if verbose:
                print(f"  early stop (no val warnAP gain in {hp['patience']})")
            break

    net.load_state_dict(best_state)

    dpv, wpv, ydv, ywv = collect(net, val, device)
    dpt, wpt, ydt, ywt = collect(net, test, device)
    thr_d, _ = pick_threshold(ydv, dpv, "TSS")
    thr_w, _ = pick_threshold(ywv, wpv, "HSS")
    ev_te = z["event_id"][te_m]
    det_pred = (dpt >= thr_d).astype(int)
    warn_pred = (wpt >= thr_w).astype(int)

    return {
        "detection": evaluate(ydt, det_pred, ev_te),
        "warning": evaluate(ywt, warn_pred, ev_te),
        "best_ap": best_ap,
        "thr": (float(thr_d), float(thr_w)),
        "state": best_state,
        "test": {"det_pred": det_pred, "warn_pred": warn_pred,
                 "y_det": ydt, "y_warn": ywt, "event_id": ev_te},
    }


def print_comparison(det, warn, arch):
    def base_row(tag, b):
        return (f"{tag:<16}{b['TSS']:>7.3f}{b['HSS']:>7.3f}{'':>7}"
                f"{b['recall']:>8.3f}{b['precision']:>11.3f}   {b['ev']}")

    def model_row(tag, r):
        ci = r["event_ci"]
        return (f"{tag:<16}{r['TSS']:>7.3f}{r['HSS']:>7.3f}{r['MCC']:>7.3f}"
                f"{r['recall']:>8.3f}{r['precision']:>11.3f}"
                f"   {r['event_caught']}/{r['event_total']} [{ci[0]:.0%},{ci[1]:.0%}]")

    print("\n" + "=" * 74)
    print(f"{'':<16}{'TSS':>7}{'HSS':>7}{'MCC':>7}{'recall':>8}{'precision':>11}   events")
    print("-" * 74)
    print(base_row("detection base", BASE["detection"]))
    print(model_row(f"detection {arch}", det))
    print(base_row("warning base", BASE["warning"]))
    print(model_row(f"warning {arch}", warn))
    print("=" * 74)

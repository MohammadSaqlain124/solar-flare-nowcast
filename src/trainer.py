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
    return dict(epochs=30, batch=256, lr=1e-3, patience=6, warn_weight=1.0, cls_weight=1.0)


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


def run_epoch(net, loader, pw, warn_w, cls_w, device, opt):
    net.train()
    tot = n = 0
    for xb, yd, yw, yc in loader:
        xb, yd, yw, yc = xb.to(device), yd.to(device), yw.to(device), yc.to(device)
        od, ow, oc = net(xb)
        loss = weighted_bce(od, yd, pw["detection"]) + warn_w * focal_loss(ow, yw)
        # classify only where a flare is actually active (yc>=1); target is M+ (yc>=2).
        # cls_w=0 -> this term vanishes and det/warn training is byte-identical to before.
        act = yc >= 1
        if cls_w and act.any():
            loss = loss + cls_w * weighted_bce(oc[act], (yc[act] >= 2).float(), pw["classify"])
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
    dp, wp, cp, yd, yw, yc = [], [], [], [], [], []
    with torch.no_grad():
        for xb, d, w, c in loader:
            od, ow, oc = net(xb.to(device))
            dp.append(torch.sigmoid(od).cpu())
            wp.append(torch.sigmoid(ow).cpu())
            cp.append(torch.sigmoid(oc).cpu())
            yd.append(d)
            yw.append(w)
            yc.append(c)
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(dp), cat(wp), cat(cp),
            cat(yd).astype(int), cat(yw).astype(int), cat(yc).astype(int))


def fit_and_eval(z, arch, hp, seed=0, masks=None, device=None, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if masks is None:
        masks = (z["is_train"], z["is_val"], z["is_test"])
    tr_m, va_m, te_m = masks

    train, val, test = make_loaders(z, masks, batch_size=hp["batch"])
    pw = pos_weights(z, tr_m)
    net = FlareCNN(n_feat=int(z["X"].shape[2]), **ARCH[arch]).to(device)   # infer F from data - was hardcoded 6 via the default, breaks the soft-only ablation
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    cls_w = hp.get("cls_weight", 1.0)

    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(1, hp["epochs"] + 1):
        tr_loss = run_epoch(net, train, pw, hp["warn_weight"], cls_w, device, opt)
        _, wpv, _, _, ywv, _ = collect(net, val, device)
        w_ap = avg_precision(ywv, wpv)                # selection stays on warning AP - keeps det/warn identical at cls_w=0
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

    dpv, wpv, cpv, ydv, ywv, ycv = collect(net, val, device)
    dpt, wpt, cpt, ydt, ywt, yct = collect(net, test, device)
    thr_d, _ = pick_threshold(ydv, dpv, "TSS")
    thr_w, _ = pick_threshold(ywv, wpv, "HSS")
    ev_te = z["event_id"][te_m]
    det_pred = (dpt >= thr_d).astype(int)
    warn_pred = (wpt >= thr_w).astype(int)

    # classification: C vs M+, only on flare-active windows (yc>=1). threshold on
    # val actives by HSS, then score test actives. positive class = M+.
    av, at = ycv >= 1, yct >= 1
    thr_c, _ = pick_threshold((ycv[av] >= 2).astype(int), cpv[av], "HSS")
    y_cls = (yct[at] >= 2).astype(int)
    cls_pred = (cpt[at] >= thr_c).astype(int)
    ev_cls = ev_te[at]

    return {
        "detection": evaluate(ydt, det_pred, ev_te),
        "warning": evaluate(ywt, warn_pred, ev_te),
        "classify": evaluate(y_cls, cls_pred, ev_cls),
        "best_ap": best_ap,
        "thr": (float(thr_d), float(thr_w), float(thr_c)),
        "state": best_state,
        "test": {"det_pred": det_pred, "warn_pred": warn_pred,
                 "y_det": ydt, "y_warn": ywt, "event_id": ev_te,
                 "cls_pred": cls_pred, "y_cls": y_cls, "cls_event_id": ev_cls},
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

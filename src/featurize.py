# notebooks 05 + 06 factored into one callable, param-driven so GOES and aditya
# run byte-identical logic - the only difference between them is the input
# parquet. this is what keeps the soft-vs-fused comparison honest. reproduces
# data/dataset.npz exactly (see tests/reproduce_dataset.py).
#
# input parquet must carry: xrsb_clean, xrsa_clean, bg_causal, xrsb_valid,
# xrsa_valid at 1s on a DatetimeIndex. for aditya, soft->xrsb, hard->xrsa.

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURES = ["log_xrsb", "log_xrsa", "log_ratio", "hardness", "rate_smooth", "rate_a"]
CLS = {"A": 0, "B": 0, "C": 1, "M": 2, "X": 3}

# defaults from config.py (HORIZON=5 is what the real dataset.npz used, not the
# stale 10 hardcoded in notebook 05)
DEFAULTS = dict(cadence="10s", window_min=30, stride_s=60, horizon_min=5,
                efold_min=12.6, k_efold=2.0, smooth_min=2.0, floor=1e-9,
                min_warn_class=2, train_frac=0.70, val_frac=0.85)


def _load_flares(flares_path):
    fl = pd.read_parquet(flares_path)
    for c in ["event_starttime", "event_peaktime", "event_endtime"]:
        fl[c] = pd.to_datetime(fl[c])
    fl = fl.sort_values("event_starttime").reset_index(drop=True)
    fl["cls_id"] = fl["class_letter"].map(CLS).fillna(0).astype(int)
    return fl


def _resample(d, cad):
    m = d.resample(cad).agg({"xrsb_clean": "max", "xrsa_clean": "max",
                             "bg_causal": "last"})
    m["valid"] = (d["xrsb_valid"].resample(cad).min().astype(bool)
                  & d["xrsa_valid"].resample(cad).min().astype(bool))
    m.loc[m["bg_causal"].isna(), "valid"] = False
    return m


def _features(m, cad_s, smooth_min, floor):
    n_smooth = max(1, int(round(smooth_min * 60 / cad_s)))
    m["log_xrsb"] = np.log10(m["xrsb_clean"].clip(lower=floor))
    m["log_xrsa"] = np.log10(m["xrsa_clean"].clip(lower=floor))
    m["log_ratio"] = np.log10((m["xrsb_clean"] / m["bg_causal"]).clip(lower=1e-3))
    m["hardness"] = np.log10((m["xrsa_clean"] / m["xrsb_clean"]).clip(lower=1e-6))
    m["rate_smooth"] = m["log_xrsb"].diff().rolling(n_smooth, min_periods=1).mean()
    m["rate_a"] = m["log_xrsa"].diff().rolling(n_smooth, min_periods=1).mean()
    return m


def _in_progress(index, fl, k, efold):
    # flare-in-progress + class, efold scheme. time+catalogue only, no flux.
    y = np.zeros(len(index), dtype=np.int8)
    c = np.zeros(len(index), dtype=np.int8)
    idx = index.values
    for _, f in fl.iterrows():
        start = f["event_starttime"]
        end = f["event_peaktime"] + pd.Timedelta(minutes=k * efold)
        sel = (idx >= np.datetime64(start)) & (idx <= np.datetime64(end))
        y[sel] = 1
        c[sel] = np.maximum(c[sel], f["cls_id"])
    return y, c


def _warning(index, fl, horizon_min, min_cls):
    y = np.zeros(len(index), dtype=np.int8)
    started = np.zeros(len(index), dtype=np.int8)
    idx = index.values
    H = pd.Timedelta(minutes=horizon_min)
    for _, f in fl.iterrows():
        if f["cls_id"] < min_cls:
            continue
        peak, start = f["event_peaktime"], f["event_starttime"]
        sel = (idx > np.datetime64(peak - H)) & (idx <= np.datetime64(peak))
        y[sel] = 1
        started[sel & (idx >= np.datetime64(start))] = 1
    return y, started


def _label_end(f, k, efold):
    return f["event_peaktime"] + pd.Timedelta(minutes=k * efold)


def _free_intervals(fl, t0, t1, pad_min, k, efold):
    pad = pd.Timedelta(minutes=pad_min)
    busy = sorted((f["event_starttime"] - pad, _label_end(f, k, efold) + pad)
                  for _, f in fl.iterrows())
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    free, cur = [], t0
    for s, e in merged:
        if s > cur:
            free.append((cur, min(s, t1)))
        cur = max(cur, e)
        if cur >= t1:
            break
    if cur < t1:
        free.append((cur, t1))
    return [(a, b) for a, b in free if b > a]


def _pick_boundary(target, free):
    if not free:
        raise RuntimeError("no flare-free interval -- widen the date range")
    best = min(free, key=lambda ab: min(abs((ab[0] - target).total_seconds()),
                                        abs((ab[1] - target).total_seconds())))
    return best[0] + (best[1] - best[0]) / 2


def build_dataset_npz(clean_path, flares_path, out_path, **overrides):
    p = {**DEFAULTS, **overrides}
    d = pd.read_parquet(clean_path)
    fl = _load_flares(flares_path)

    m = _resample(d, p["cadence"])
    cad_s = float(pd.Series(m.index).diff().dt.total_seconds().median())
    m = _features(m, cad_s, p["smooth_min"], p["floor"])

    y_prog, y_cls = _in_progress(m.index, fl, p["k_efold"], p["efold_min"])
    y_warn, warn_started = _warning(m.index, fl, p["horizon_min"], p["min_warn_class"])

    W = int(p["window_min"] * 60 / cad_s)
    S = max(1, int(p["stride_s"] / cad_s))
    feat = m[FEATURES].astype(np.float32).values
    sw = np.ascontiguousarray(sliding_window_view(feat, W, axis=0).transpose(0, 2, 1))

    X = sw[::S].copy()
    t_end = m.index[W - 1:][::S]
    yp, yc = y_prog[W - 1:][::S], y_cls[W - 1:][::S]
    yw, ws = y_warn[W - 1:][::S], warn_started[W - 1:][::S]

    vmask = sliding_window_view(m["valid"].values, W)[::S].all(axis=1)
    keep = vmask & np.isfinite(X).all(axis=(1, 2))
    X, t_end, yp, yc, yw, ws = X[keep], t_end[keep], yp[keep], yc[keep], yw[keep], ws[keep]

    # event_id (06): later flare wins on overlap
    t_end = pd.to_datetime(t_end)
    event_id = np.full(len(t_end), -1, dtype=np.int32)
    te_ns = t_end.values.astype("datetime64[ns]").astype("int64")
    for i, f in fl.iterrows():
        s = np.datetime64(f["event_starttime"]).astype("datetime64[ns]").astype("int64")
        e = np.datetime64(_label_end(f, p["k_efold"], p["efold_min"])).astype("datetime64[ns]").astype("int64")
        event_id[(te_ns >= s) & (te_ns <= e)] = i

    # chronological split on flare-free midpoints, with a window-wide gap band
    free = _free_intervals(fl, t_end[0], t_end[-1], p["window_min"], p["k_efold"], p["efold_min"])
    span = t_end[-1] - t_end[0]
    b_tr = _pick_boundary(t_end[0] + p["train_frac"] * span, free)
    b_va = _pick_boundary(t_end[0] + p["val_frac"] * span, free)
    gap = pd.Timedelta(minutes=p["window_min"])
    is_tr = (t_end <= b_tr)
    is_va = ((t_end > b_tr + gap) & (t_end <= b_va))
    is_te = (t_end > b_va + gap)

    # scaler on train only
    F = X.shape[2]
    flat = X[is_tr].reshape(-1, F)
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    np.savez_compressed(
        out_path, X=Xn, t_end=t_end.values.astype("datetime64[ns]"),
        y_progress=yp, y_class=yc, y_warn=yw, warn_started=ws, event_id=event_id,
        is_train=is_tr, is_val=is_va, is_test=is_te, scaler_mean=mu, scaler_std=sd,
        features=np.array(FEATURES),
        meta=np.array([p["cadence"], str(p["window_min"]), str(p["stride_s"]),
                       str(p["horizon_min"]), "efold", str(p["k_efold"]),
                       str(b_tr), str(b_va)]))
    return out_path

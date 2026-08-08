#!/usr/bin/env python
"""
Whole pipeline in one go: download GOES -> clean -> features -> windows -> split.
Same thing notebooks 02-07 do, just scripted so I can re-run it for any range.

    python run_pipeline.py
    python run_pipeline.py --start 2024-08-01 --end 2025-01-01
    python run_pipeline.py --force            # ignore caches
    python run_pipeline.py --stages clean     # stop after cleaning
    python run_pipeline.py --dry-run          # just print the plan

Every chunk caches its cleaned output, so if it dies halfway through a long run
you don't lose the earlier months - just run it again.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHUNKS = DATA / "chunks"
DATA.mkdir(exist_ok=True)
CHUNKS.mkdir(exist_ok=True)

CLS = {"A": 0, "B": 0, "C": 1, "M": 2, "X": 3}


_t0 = time.time()


def log(msg: str, level: int = 0) -> None:
    print(f"[{time.time()-_t0:7.1f}s] {'  '*level}{msg}", flush=True)


def nan_runs(mask: np.ndarray):
    # start index + length of each run of True
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts, ends - starts


def label_end(f) -> pd.Timestamp:
    if C.LABEL_SCHEME == "catalogued":
        return f["event_endtime"]
    return f["event_peaktime"] + pd.Timedelta(minutes=C.K_EFOLD * C.EFOLD_MIN)


# ---- download ----

def fetch_flares(t0: str, t1: str, chunk_days: int = 30) -> pd.DataFrame:
    # HEK chokes on wide ranges - one 150-day query can hang for minutes or
    # just time out. monthly slices are faster and you can watch progress.
    from sunpy.net import Fido, attrs as a

    cur, end = pd.Timestamp(t0), pd.Timestamp(t1)
    step = pd.Timedelta(days=chunk_days)
    n = max(1, int(np.ceil((end - cur) / step)))
    frames, i = [], 0

    while cur < end:
        nxt = min(cur + step, end)
        i += 1
        t = time.time()
        res = Fido.search(
            a.Time(str(cur), str(nxt)),
            a.hek.EventType("FL"),
            a.hek.OBS.Observatory == "GOES",
            a.hek.FRM.Name == "SWPC",
        )
        try:
            tbl = res["hek"]["event_starttime", "event_peaktime",
                             "event_endtime", "fl_goescls", "ar_noaanum"]
            part = tbl.to_pandas()
        except (KeyError, IndexError, TypeError):
            part = pd.DataFrame()          # empty month, fine
        frames.append(part)
        log(f"HEK [{i}/{n}] {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: "
            f"{len(part)} rows in {time.time()-t:.0f}s", 1)
        cur = nxt

    fl = pd.concat([f for f in frames if len(f)], ignore_index=True)
    before = len(fl)

    # dupes come from two places: same flare listed twice (one copy has
    # ar_noaanum=0), and the chunk seams (HEK ranges are inclusive both ends).
    # sort AR desc so the copy with a real region number wins.
    fl = (fl.sort_values("ar_noaanum", ascending=False)
            .drop_duplicates(subset=["event_starttime", "event_peaktime",
                                     "event_endtime"], keep="first")
            .sort_values("event_starttime")
            .reset_index(drop=True))

    for c in ["event_starttime", "event_peaktime", "event_endtime"]:
        fl[c] = pd.to_datetime(fl[c])
    fl["class_letter"] = fl["fl_goescls"].astype(str).str[0]
    fl["cls_id"] = fl["class_letter"].map(CLS).fillna(0).astype(int)

    scale = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

    def to_flux(s):
        s = str(s).strip()
        if not s or s[0] not in scale:
            return np.nan
        try:
            return float(s[1:]) * scale[s[0]]
        except ValueError:
            return np.nan

    fl["peak_flux"] = fl["fl_goescls"].apply(to_flux)
    log(f"flares: {before} rows -> {len(fl)} after dedup", 1)
    return fl


def _make_downloader():
    # parfive with an actual timeout. without one a stalled socket hangs
    # forever. the timeout arg moved around between versions, hence the fallback.
    import parfive

    try:
        import aiohttp
        from parfive import SessionConfig
        cfg = SessionConfig(timeouts=aiohttp.ClientTimeout(
            total=C.DL_TIMEOUT_TOTAL, sock_read=C.DL_TIMEOUT_READ))
        return parfive.Downloader(max_conn=C.MAX_CONN, progress=False, config=cfg)
    except Exception:
        pass
    try:
        return parfive.Downloader(
            max_conn=C.MAX_CONN, progress=False,
            timeouts={"total": C.DL_TIMEOUT_TOTAL, "sock_read": C.DL_TIMEOUT_READ})
    except Exception:
        log("WARNING: could not set download timeouts; a stalled server "
            "may hang this chunk", 2)
        return parfive.Downloader(max_conn=C.MAX_CONN, progress=False)


_NC_RE = re.compile(r"[^\s'\"]+\.nc")


def _purge_bad_files(files) -> int:
    # sunpy's error only names the FIRST bad file, so reacting to it clears one
    # per retry - a chunk with 3 corrupt files needs 3 retries. test them all
    # up front instead. size check first since truncated files are tiny; only
    # parse-test the ones that pass it.
    from sunpy.timeseries import TimeSeries

    paths = [Path(f) for f in files if Path(f).exists()]
    if not paths:
        return 0

    removed = 0
    sizes = sorted(p.stat().st_size for p in paths)
    median = sizes[len(sizes) // 2]

    for p in paths:
        size = p.stat().st_size
        if size < 0.5 * median:
            log(f"removed truncated {p.name} ({size/1e6:.2f} MB vs "
                f"{median/1e6:.2f} MB median)", 2)
            p.unlink(missing_ok=True)
            removed += 1
            continue
        try:
            TimeSeries(str(p))
        except Exception:
            log(f"removed unparseable {p.name}", 2)
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def fetch_raw(t0: pd.Timestamp, t1: pd.Timestamp, retries: int = None) -> pd.DataFrame:
    # GOES XRS flux for [t0, t1). NOAA times out under load and leaves a half
    # -written .nc behind, and sunpy won't re-download something already on
    # disk - so the chunk fails the same way every time until the file is gone.
    from sunpy.net import Fido, attrs as a
    from sunpy.timeseries import TimeSeries

    retries = retries if retries is not None else C.FETCH_RETRY
    last = None
    for attempt in range(1, retries + 1):
        search = Fido.search(
            a.Time(str(t0), str(t1)),
            a.Instrument.xrs,
            a.Resolution(C.RESOLUTION),
            a.goes.SatelliteNumber(C.SATELLITE),
        )
        if not search:
            raise RuntimeError(f"no GOES data found for {t0} .. {t1}")

        files = Fido.fetch(search, downloader=_make_downloader())
        errs = list(getattr(files, "errors", []) or [])

        try:
            raw = TimeSeries(files, concatenate=True).to_dataframe()
            return raw[(raw.index >= t0) & (raw.index < t1)]
        except Exception as exc:
            last = exc
            n = _purge_bad_files(files)
            if attempt < retries:
                wait = 5 * attempt
                log(f"purged {n} bad file(s), {len(errs)} download error(s); "
                    f"retry {attempt}/{retries} after {wait}s", 2)
                time.sleep(wait)

    raise RuntimeError(f"failed after {retries} attempts: "
                       f"{str(last).splitlines()[0][:120]}")


# ---- clean ----

def clean_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in ["xrsa", "xrsb"] if c in raw.columns]
    df = raw[fcols].copy()

    for f in fcols:
        invalid = raw[f].isna()
        q = f + "_quality"
        if q in raw.columns:
            invalid = invalid | (raw[q] != 0)   # a present value can still be flagged bad

        s = raw[f].where(~invalid)
        starts, lens = nan_runs(invalid.values)

        # a short gap is fine to interpolate over; mark those spots valid again
        valid = ~invalid.values.copy()
        for st, ln in zip(starts, lens):
            if ln <= C.GAP_LIMIT_S:
                valid[st:st + ln] = True

        df[f + "_clean"] = s.interpolate(limit=C.GAP_LIMIT_S, limit_area="inside")
        df[f + "_valid"] = valid & df[f + "_clean"].notna().values

    # background: 5th percentile over 4h, past-only. compute on a 1min series
    # because a rolling quantile over the full 1s array is painfully slow.
    minute = df.loc[df["xrsb_valid"], "xrsb_clean"].resample("1min").median()
    bg = minute.rolling(C.BG_WINDOW, min_periods=10).quantile(C.BG_Q).dropna()

    # gotcha: GOES 1s stamps have a sub-second offset but resample() labels on
    # whole minutes, so nothing lines up and plain .reindex() gives all-NaN
    # with no error. ffill lines it up AND stays causal.
    df["bg_causal"] = bg.reindex(df.index, method="ffill")

    for f in fcols:
        df["log_" + f] = np.log10(df[f + "_clean"].clip(lower=C.FLOOR))
    return df


def downsample(df: pd.DataFrame) -> pd.DataFrame:
    # max on flux to keep the peaks; last on background since it's a running
    # estimate and max would peek forward.
    m = df.resample(C.CADENCE).agg({
        "xrsb_clean": "max",
        "xrsa_clean": "max",
        "bg_causal":  "last",
    })
    m["valid"] = (df["xrsb_valid"].resample(C.CADENCE).min().astype(bool)
                  & df["xrsa_valid"].resample(C.CADENCE).min().astype(bool))
    m.loc[m["bg_causal"].isna(), "valid"] = False
    return m


# ---- features ----

def add_features(m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # everything here is causal - backward diff, trailing roll. do NOT swap in
    # np.gradient, it reads one step ahead and leaks the future into training.
    cad_s = float(pd.Series(m.index).diff().dt.total_seconds().median())
    n_sm = max(1, int(round(C.SMOOTH_MIN * 60 / cad_s)))

    m["log_xrsb"] = np.log10(m["xrsb_clean"].clip(lower=C.FLOOR))
    m["log_xrsa"] = np.log10(m["xrsa_clean"].clip(lower=C.FLOOR))
    m["log_ratio"] = np.log10((m["xrsb_clean"] / m["bg_causal"]).clip(lower=1e-3))
    m["hardness"] = np.log10((m["xrsa_clean"] / m["xrsb_clean"]).clip(lower=1e-6))
    m["rate_smooth"] = m["log_xrsb"].diff().rolling(n_sm, min_periods=1).mean()
    m["rate_a"] = m["log_xrsa"].diff().rolling(n_sm, min_periods=1).mean()

    feats = ["log_xrsb", "log_xrsa", "log_ratio", "hardness", "rate_smooth", "rate_a"]
    return m, feats


# ---- labels ----

def build_labels(index: pd.DatetimeIndex, fl: pd.DataFrame):
    # returns (in_progress, class_id, warning, warn_started, event_id)
    n = len(index)
    y_prog = np.zeros(n, dtype=np.int8)
    y_cls = np.zeros(n, dtype=np.int8)
    y_warn = np.zeros(n, dtype=np.int8)
    started = np.zeros(n, dtype=np.int8)
    event_id = np.full(n, -1, dtype=np.int32)

    # nasty one: a parquet round-trip drops the index from ns to us precision,
    # so index.astype(int64) is microseconds but np.datetime64(ts) is nanos.
    # nothing ever matches and every label comes out 0 with no error. pin both to ns.
    ns = index.values.astype("datetime64[ns]").astype("int64")
    H = pd.Timedelta(minutes=C.HORIZON_MIN)

    def to_ns(t):
        return np.datetime64(pd.Timestamp(t)).astype("datetime64[ns]").astype("int64")

    for i, f in fl.iterrows():
        s, e = to_ns(f["event_starttime"]), to_ns(label_end(f))
        sel = (ns >= s) & (ns <= e)
        y_prog[sel] = 1
        y_cls[sel] = np.maximum(y_cls[sel], f["cls_id"])
        event_id[sel] = i          # loop is start-ordered so a later flare wins overlaps

        if f["cls_id"] >= C.MIN_WARN_CLASS:
            pk = to_ns(f["event_peaktime"])
            w = (ns > to_ns(f["event_peaktime"] - H)) & (ns <= pk)
            y_warn[w] = 1
            started[w & (ns >= s)] = 1

    # if we had flares but labelled nothing, the times didn't line up (units/tz).
    # fail loud rather than train on a silently empty target.
    if len(fl) and y_prog.sum() == 0:
        raise RuntimeError(
            f"{len(fl)} flares supplied but no window was labelled. "
            f"index {index[0]} .. {index[-1]} ({index.dtype}); "
            f"first flare {fl['event_starttime'].iloc[0]}"
        )
    return y_prog, y_cls, y_warn, started, event_id


# ---- windows ----

def build_windows(m: pd.DataFrame, feats: list[str], labels):
    from numpy.lib.stride_tricks import sliding_window_view

    cad_s = float(pd.Series(m.index).diff().dt.total_seconds().median())
    W = int(C.WINDOW_MIN * 60 / cad_s)
    S = max(1, int(C.STRIDE_S / cad_s))

    arr = m[feats].astype(np.float32).values
    sw = sliding_window_view(arr, W, axis=0)              # (N-W+1, F, W)
    X = np.ascontiguousarray(sw.transpose(0, 2, 1))[::S].copy()

    t_end = m.index[W-1:][::S]
    out = [lab[W-1:][::S] for lab in labels]

    vmask = sliding_window_view(m["valid"].values, W)[::S].all(axis=1)
    finite = np.isfinite(X).all(axis=(1, 2))

    # if a chunk failed there's a hole in the series, but sliding_window_view
    # doesn't know that - it'll happily glue Oct and Nov into one "window".
    # so drop any window whose span isn't the length it should be.
    idx_ns = m.index.values.astype("datetime64[ns]").astype("int64")
    spans = sliding_window_view(idx_ns, W)[::S]
    elapsed = spans[:, -1] - spans[:, 0]
    expected = int((W - 1) * cad_s * 1e9)
    contiguous = np.abs(elapsed - expected) <= int(cad_s * 1e9)

    keep = vmask & finite & contiguous
    log(f"windows {len(X):,} -> kept {keep.sum():,} "
        f"(invalid {(~vmask).sum():,}, non-finite {(vmask & ~finite).sum():,}, "
        f"spanning a gap {(~contiguous).sum():,})", 1)
    return X[keep], t_end[keep], [o[keep] for o in out]


# ---- split ----

def free_intervals(fl, t0, t1, pad_min):
    # stretches with no flare in them (padded), so a split boundary can land
    # in a quiet gap instead of chopping a flare in half.
    pad = pd.Timedelta(minutes=pad_min)
    busy = sorted((f["event_starttime"] - pad, label_end(f) + pad)
                  for _, f in fl.iterrows())
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
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


def pick_boundary(target, free):
    if not free:
        raise RuntimeError("no flare-free interval; widen the date range")
    best = min(free, key=lambda ab: min(abs((ab[0]-target).total_seconds()),
                                        abs((ab[1]-target).total_seconds())))
    return best[0] + (best[1] - best[0]) / 2


def make_split(t_end, event_id, fl):
    free = free_intervals(fl, t_end[0], t_end[-1], pad_min=C.WINDOW_MIN)

    # aim at the quantile of actual window times, not a linear slice of the
    # clock - with a gap in the data a clock-based target can land in the hole.
    n = len(t_end)
    b_tr = pick_boundary(t_end[min(int(C.TRAIN_FRAC * n), n - 1)], free)
    b_va = pick_boundary(t_end[min(int(C.VAL_FRAC * n), n - 1)], free)
    gap = pd.Timedelta(minutes=C.WINDOW_MIN)

    is_tr = t_end <= b_tr
    is_va = (t_end > b_tr + gap) & (t_end <= b_va)
    is_te = t_end > b_va + gap

    for nm, msk in [("train", is_tr), ("val", is_va), ("test", is_te)]:
        if np.sum(msk) == 0:
            raise RuntimeError(
                f"the '{nm}' split is empty. This normally means chunks failed "
                f"and left a hole in the series. Either re-run to fetch them, "
                f"or set TEND in config.py to the last date that succeeded."
            )

    # the two checks this whole stage exists for: real gap between splits, and
    # no flare with windows on both sides of a boundary.
    g1 = (t_end[is_va][0] - t_end[is_tr][-1]).total_seconds() / 60
    g2 = (t_end[is_te][0] - t_end[is_va][-1]).total_seconds() / 60
    assert g1 > C.WINDOW_MIN and g2 > C.WINDOW_MIN, "boundary gap too small"

    e_tr = set(event_id[is_tr][event_id[is_tr] >= 0])
    e_va = set(event_id[is_va][event_id[is_va] >= 0])
    e_te = set(event_id[is_te][event_id[is_te] >= 0])
    assert not (e_tr & e_va) and not (e_tr & e_te) and not (e_va & e_te), \
        "event bleed between splits"

    log(f"split boundaries {b_tr} | {b_va}", 1)
    log(f"gaps {g1:.0f} / {g2:.0f} min, no shared events -- PASS", 1)
    return is_tr, is_va, is_te, (b_tr, b_va), (e_tr, e_va, e_te)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c-h), min(1.0, c+h)


# ---- main ----

def chunk_ranges(t0: pd.Timestamp, t1: pd.Timestamp):
    out, cur = [], t0
    step = pd.Timedelta(days=C.CHUNK_DAYS)
    while cur < t1:
        out.append((cur, min(cur + step, t1)))
        cur += step
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=C.TSTART)
    ap.add_argument("--end", default=C.TEND)
    ap.add_argument("--force", action="store_true", help="ignore caches")
    ap.add_argument("--stages", default="all",
                    choices=["all", "flares", "clean", "windows", "split"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
    chunks = chunk_ranges(t0, t1)

    log(f"range   {t0}  ->  {t1}   ({(t1-t0).days} days)")
    log(f"chunks  {len(chunks)} x {C.CHUNK_DAYS} days, {C.LEADIN_HOURS} h lead-in")
    log(f"config  cadence {C.CADENCE}, window {C.WINDOW_MIN} min, "
        f"horizon {C.HORIZON_MIN} min, label '{C.LABEL_SCHEME}' K={C.K_EFOLD}")

    # cache files are named by chunk boundaries, so bumping CHUNK_DAYS orphans
    # every existing chunk and quietly re-downloads the lot. flag it.
    wanted = {f"m10_{a:%Y%m%d}_{b:%Y%m%d}.parquet" for a, b in chunks}
    orphans = [p for p in CHUNKS.glob("m10_*.parquet") if p.name not in wanted]
    if orphans:
        mb = sum(p.stat().st_size for p in orphans) / 1e6
        log(f"NOTE: {len(orphans)} cached chunk(s) ({mb:.0f} MB) do not match "
            f"CHUNK_DAYS={C.CHUNK_DAYS}", 1)
        log("and will be re-downloaded. Delete data/chunks/ to reclaim the space, "
            "or restore", 1)
        log("the previous CHUNK_DAYS value to reuse them.", 1)
    if args.dry_run:
        for a, b in chunks:
            p = CHUNKS / f"m10_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
            log(f"{a:%Y-%m-%d} -> {b:%Y-%m-%d}   {'cached' if p.exists() else 'FETCH'}", 1)
        return 0

    # flares
    fl_path = DATA / "flares.parquet"
    rng_path = DATA / "flares_range.txt"
    want = f"{t0}|{t1}"
    have = rng_path.read_text().strip() if rng_path.exists() else ""

    # only reuse the cached catalogue if it was built for THIS range - otherwise
    # a one-week catalogue gets reused for a 5-month run and 95% of it is
    # labelled "no flare". been there.
    if args.force or not fl_path.exists() or have != want:
        if fl_path.exists() and have != want:
            log(f"STAGE flares -- cache was built for [{have}], need [{want}]; refetching")
        else:
            log("STAGE flares")
        fl = fetch_flares(str(t0), str(t1))
        fl.to_parquet(fl_path)
        rng_path.write_text(want)
    else:
        fl = pd.read_parquet(fl_path)
        log(f"STAGE flares -- cached ({len(fl)}) for the requested range")

    # do the dtype fixup on both paths - if only the cache branch does it you
    # get a TypeError three stages downstream.
    for c in ["event_starttime", "event_peaktime", "event_endtime"]:
        fl[c] = pd.to_datetime(fl[c])
    if "cls_id" not in fl.columns:
        fl["class_letter"] = fl["fl_goescls"].astype(str).str[0]
        fl["cls_id"] = fl["class_letter"].map(CLS).fillna(0).astype(int)
    fl = fl.sort_values("event_starttime").reset_index(drop=True)
    log(fl["class_letter"].value_counts().to_dict(), 1)

    # sanity: does the catalogue actually cover the range we asked for?
    if len(fl):
        cov = (fl["event_starttime"].iloc[-1] - fl["event_starttime"].iloc[0])
        span = t1 - t0
        log(f"catalogue spans {cov.days} d of the {span.days} d requested", 1)
        if cov < span * 0.5:
            log("WARNING: flare catalogue covers less than half the range.", 1)
            log("         Delete data/flares.parquet and re-run.", 1)
    if args.stages == "flares":
        return 0

    # clean, chunk by chunk
    log("STAGE clean")
    parts, failed = [], []
    lead = pd.Timedelta(hours=C.LEADIN_HOURS)
    for i, (a, b) in enumerate(chunks, 1):
        cache = CHUNKS / f"m10_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if cache.exists() and not args.force:
            parts.append(pd.read_parquet(cache))
            log(f"[{i}/{len(chunks)}] {a:%Y-%m-%d} cached", 1)
            continue
        try:
            ta = time.time()
            raw = fetch_raw(max(a - lead, t0), b)      # pull the lead-in too
            tb = time.time()
            cleaned = clean_chunk(raw)
            tc = time.time()
            m = downsample(cleaned)
            m = m[(m.index >= a) & (m.index < b)]      # then trim it back off
            m.to_parquet(cache)
            parts.append(m)
            td = time.time()
            log(f"[{i}/{len(chunks)}] {a:%Y-%m-%d} -> {len(m):,} rows "
                f"| fetch {tb-ta:.0f}s  clean {tc-tb:.0f}s  resample {td-tc:.0f}s", 1)
            del raw, cleaned
            gc.collect()          # ~1.3M-row frames, don't let them stack up
        except Exception as exc:
            failed.append((a, b, str(exc).splitlines()[0][:110]))
            log(f"[{i}/{len(chunks)}] {a:%Y-%m-%d} FAILED: {str(exc).splitlines()[0][:110]}", 1)
            gc.collect()

    if failed:
        log("")
        log(f"{len(failed)} of {len(chunks)} chunk(s) failed:")
        for a, b, msg in failed:
            log(f"{a:%Y-%m-%d} -> {b:%Y-%m-%d}   {msg}", 1)
        log("")
        log("Re-run to retry only these; the rest stay cached.")
        log("If they keep failing, NOAA is refusing your requests. Either wait,")
        log("or set TEND in config.py to the last date that succeeded and")
        log("proceed with what you have -- more data can be added later.")
        log("")

    if not parts:
        log("no data. Aborting.")
        return 1
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated(keep="first")]           # lead-in overlaps at the seams
    del parts
    log(f"combined {len(m):,} rows at {C.CADENCE}, "
        f"{100*m['valid'].mean():.2f}% valid", 1)
    if args.stages == "clean":
        return 0

    # windows
    log("STAGE windows")
    m, feats = add_features(m)
    labels = build_labels(m.index, fl)
    X, t_end, (yp, yc, yw, ws, ev) = build_windows(m, feats, labels)
    log(f"X {X.shape}  ({X.nbytes/1e6:.0f} MB)", 1)
    log(f"flare-in-progress {100*yp.mean():.1f}%, warning {100*yw.mean():.2f}%", 1)
    if args.stages == "windows":
        np.savez_compressed(DATA / "windows.npz", X=X,
                            t_end=t_end.values.astype("datetime64[ns]"),
                            y_progress=yp, y_class=yc, y_warn=yw,
                            warn_started=ws, event_id=ev,
                            features=np.array(feats))
        return 0

    # split + scale
    log("STAGE split")
    is_tr, is_va, is_te, bounds, (e_tr, e_va, e_te) = make_split(t_end, ev, fl)

    # fit the scaler on train ONLY, then apply to all three
    F = X.shape[2]
    mu = X[is_tr].reshape(-1, F).mean(axis=0)
    sd = X[is_tr].reshape(-1, F).std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    out = DATA / "dataset.npz"
    np.savez_compressed(
        out, X=Xn, t_end=t_end.values.astype("datetime64[ns]"),
        y_progress=yp, y_class=yc, y_warn=yw, warn_started=ws, event_id=ev,
        is_train=np.asarray(is_tr), is_val=np.asarray(is_va), is_test=np.asarray(is_te),
        scaler_mean=mu, scaler_std=sd, features=np.array(feats),
        meta=np.array([C.CADENCE, str(C.WINDOW_MIN), str(C.STRIDE_S),
                       str(C.HORIZON_MIN), C.LABEL_SCHEME, str(C.K_EFOLD),
                       str(bounds[0]), str(bounds[1])]),
    )

    log("")
    log("=" * 62)
    rows = []
    for nm, msk, ev_set in [("train", is_tr, e_tr), ("val", is_va, e_va), ("test", is_te, e_te)]:
        cl = fl.loc[sorted(ev_set), "cls_id"] if ev_set else pd.Series(dtype=int)
        rows.append({"split": nm, "windows": int(np.sum(msk)), "events": len(ev_set),
                     "C": int((cl == 1).sum()), "M": int((cl == 2).sum()),
                     "X": int((cl == 3).sum()),
                     "warn_%": round(100 * yw[msk].mean(), 2)})
    print(pd.DataFrame(rows).set_index("split").to_string())

    # events, not windows, is the real sample size (one flare ~ 30 windows)
    n_m = sum(1 for e in e_te if fl.loc[e, "cls_id"] >= C.MIN_WARN_CLASS)
    log("")
    log(f"Test set: {len(e_te)} events, {n_m} of class M or above.")
    if n_m:
        lo, hi = wilson(int(round(0.8 * n_m)), n_m)
        log(f"At 80% recall the 95% interval is [{lo:.0%}, {hi:.0%}] "
            f"-- quote n={n_m} beside every metric.")
    if n_m < 20:
        log("WARNING: fewer than 20 M+ test events. Results will not be")
        log("         distinguishable from noise. Extend the date range.")
    log(f"Saved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # parfive spews secondary errors on ctrl-c cleanup; ignore them,
        # finished chunks are already saved.
        print()
        log("interrupted. Completed chunks are cached -- re-run to continue "
            "from where this stopped.")
        sys.exit(130)

# phase 4: turn the messy PRADAN FITS tree into a clean two-channel 1s series,
# the aditya analog of goes_clean_1s.parquet. soft = SoLEXS SDD2 (2-22 keV),
# hard = HEL1OS CdTe1 20-30 keV (genuinely non-thermal, above SoLEXS's ceiling -
# the thing GOES can't see). downstream feature code treats soft like xrsb, hard
# like xrsa.
#
# gotchas this file exists to handle:
#  - the two instruments nest files differently and repeat date dirs, so we
#    DISCOVER by rglob + parse the date from the filename, never build paths.
#  - SoLEXS TIME is unix seconds, HEL1OS is MJD. convert, don't assume.
#  - HEL1OS bands have DIFFERENT row counts and start ~11s past midnight, off the
#    integer second. floor + groupby + reindex to a full-day grid, assert
#    coverage - same class of silent-NaN bug we hit with reindex in phase 1.
#  - real gaps stay NaN (downstream windowing is gap-aware). we only bridge
#    sub-second jitter, never fill missing hours with stale values.

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

HARD_BAND = "20.00KEV_TO_30.00KEV"   # HEL1OS CdTe1 hard channel (match by EXTNAME)
SEC_PER_DAY = 86400
FILL_LIMIT = 5                        # bridge only tiny jitter, keep real gaps NaN

_SLX_DATE = re.compile(r"AL1_SOLEXS_(\d{8})_SDD2")
_HLS_DATE = re.compile(r"HLS_(\d{8})_")


def mjd_to_unix(mjd):
    # MJD epoch is 1858-11-17; unix epoch 1970-01-01 = MJD 40587.
    return (np.asarray(mjd, dtype="float64") - 40587.0) * SEC_PER_DAY


def day_start_unix(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def find_solexs(root):
    # date -> soft .lc.gz path. if a day has both v1.0 and v1.1, keep the later.
    out = {}
    for p in Path(root).rglob("*_SDD2_L1.lc.gz"):
        m = _SLX_DATE.search(p.name)
        if not m:
            continue
        d = m.group(1)
        if d not in out or "v1.1" in str(p):     # prefer newer version dir
            out[d] = p
    return out


def find_hel1os(root):
    # date -> list of cdte1 light-curve paths (usually two ~12h dumps per day).
    out = {}
    for p in Path(root).rglob("lightcurve_cdte1.fits"):
        m = _HLS_DATE.search(str(p))
        if not m:
            continue
        out.setdefault(m.group(1), []).append(p)
    return out


def read_solexs_soft(path):
    # RATE extension: TIME (unix s), COUNTS. already 1Hz for the full day.
    with fits.open(path) as h:
        r = h["RATE"].data
        t = np.asarray(r["TIME"], dtype="float64")
        c = np.asarray(r["COUNTS"], dtype="float64")
    return pd.Series(c, index=np.floor(t).astype("int64"))


def read_hel1os_hard(path):
    # pick the 20-30 keV extension BY NAME (hdu order isn't guaranteed). MJD->unix.
    with fits.open(path) as h:
        ext = next((hdu for hdu in h
                    if getattr(hdu, "name", "").endswith(HARD_BAND)), None)
        if ext is None:
            raise ValueError(f"no {HARD_BAND} band in {path}")
        d = ext.data
        t = mjd_to_unix(np.asarray(d["MJD"], dtype="float64"))
        c = np.asarray(d["CTR"], dtype="float64")
    return pd.Series(c, index=np.floor(t).astype("int64"))


def _to_grid(series, t0):
    # collapse duplicate seconds (keep last), reindex onto the full-day 1s grid,
    # bridge only tiny jitter. returns (aligned series, coverage fraction).
    s = series[(series.index >= t0) & (series.index < t0 + SEC_PER_DAY)]
    s = s.groupby(level=0).last()
    grid = np.arange(t0, t0 + SEC_PER_DAY, dtype="int64")
    s = s.reindex(grid)
    cov = float(s.notna().mean())
    s = s.ffill(limit=FILL_LIMIT)
    return s, cov


def build_day(date_str, solexs_path, hel1os_paths):
    # one fused day. concat the day's HEL1OS dumps, grid both channels, join.
    t0 = day_start_unix(date_str)
    soft_raw = read_solexs_soft(solexs_path)
    hard_raw = pd.concat([read_hel1os_hard(p) for p in hel1os_paths])

    soft, soft_cov = _to_grid(soft_raw, t0)
    hard, hard_cov = _to_grid(hard_raw, t0)

    df = pd.DataFrame({"time": soft.index, "soft": soft.values, "hard": hard.values})
    return df, soft_cov, hard_cov


def build_dataset(root, out_path="data/aditya_clean_1s.parquet",
                  min_hard_cov=0.05, min_soft_cov=0.90):
    # fuse every day that has BOTH instruments AND enough of each channel. fusion
    # needs both, so a day strong in one and empty in the other (e.g. Oct 13:
    # 99% hard, 13% soft) is useless and gets dropped - checking only one channel
    # let those through silently.
    root = Path(root)
    slx = find_solexs(root / "solexs")
    hls = find_hel1os(root / "hel1os")
    days = sorted(set(slx) & set(hls))     # overlap only - fusion needs both
    print(f"solexs days {len(slx)}  hel1os days {len(hls)}  overlap {len(days)}")

    frames, dropped = [], []
    for d in days:
        df, sc, hc = build_day(d, slx[d], hls[d])
        why = "low soft cov" if sc < min_soft_cov else \
              "low hard cov" if hc < min_hard_cov else None
        tag = f"  DROPPED ({why})" if why else ""
        print(f"  {d}  soft {sc:5.1%}  hard {hc:5.1%}{tag}")
        if why:
            dropped.append(d)
            continue
        frames.append(df)

    if not frames:
        raise SystemExit("no usable overlap days - check the download/extraction")

    full = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    both = full[["soft", "hard"]].notna().all(axis=1).mean()
    print(f"\nwrote {out_path}  rows {len(full):,}  kept {len(frames)} days "
          f"(dropped {len(dropped)})")
    print(f"fused coverage (both channels present): {both:.1%} of rows")
    return full

#!/usr/bin/env python
"""
smoke test for the phase 4 loader. no real data - fabricates FITS that match the
PRADAN layout (SoLEXS RATE, HEL1OS's five named bands, unix vs MJD, two dumps a
day, a mid-day gap) and checks the loader parses, converts time, aligns, reports
coverage, and drops a day that's weak in either channel. catches the silent-NaN,
time-base, and single-channel-hole bugs on CPU.

    python tests/smoke_aditya.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aditya_load import build_dataset, mjd_to_unix, SEC_PER_DAY, day_start_unix  # noqa: E402

BANDS = ["5.00KEV_TO_20.00KEV", "20.00KEV_TO_30.00KEV", "30.00KEV_TO_40.00KEV",
         "40.00KEV_TO_60.00KEV", "1.80KEV_TO_90.00KEV"]


def write_solexs(base, date, nrows):
    # nrows < 86400 simulates a partial SoLEXS day (Oct-13-style hole)
    d = base / "solexs" / "s" / f"AL1_SLX_L1_{date}_v1.1" / "SDD2"
    d.mkdir(parents=True)
    t0 = day_start_unix(date)
    t = np.arange(t0, t0 + nrows, dtype="float64")
    counts = np.full(nrows, 200.0)
    hdu = fits.BinTableHDU(Table({"TIME": t, "COUNTS": counts}), name="RATE")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        d / f"AL1_SOLEXS_{date}_SDD2_L1.lc.gz", overwrite=True)


def _cdte(path, start_unix, n, gap=None):
    path.mkdir(parents=True)
    unix = start_unix + 11.7 + np.arange(n, dtype="float64")   # off the integer second
    if gap:
        keep = (np.arange(n) < gap[0]) | (np.arange(n) >= gap[1])
        unix = unix[keep]
    mjd = unix / SEC_PER_DAY + 40587.0
    isot = np.array(["2024-08-02T00:00:00.000"] * len(unix))
    hdus = [fits.PrimaryHDU()]
    for b in BANDS:
        ctr = np.full(len(unix), 5.0)
        tbl = Table({"MJD": mjd, "ISOT": isot, "CTR": ctr, "STAT_ERR": np.sqrt(ctr)})
        hdus.append(fits.BinTableHDU(tbl, name=f"CDTE1_LC_BAND_{b}"))
    fits.HDUList(hdus).writeto(path / "lightcurve_cdte1.fits", overwrite=True)


def write_hel1os(base, date):
    t0 = day_start_unix(date)
    root = base / "hel1os" / "h"
    _cdte(root / f"HLS_{date}_000011_x" / "cdte", t0, 21600, gap=(10000, 11000))
    _cdte(root / f"HLS_{date}_120005_x" / "cdte", t0 + 12 * 3600, 21600)


# time-base conversion against the real first-row value from your files
assert abs(mjd_to_unix(60524.00013577062) - 1722556811.73) < 0.1, "MJD->unix off"

GOOD, BAD = "20240802", "20240803"
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    write_solexs(base, GOOD, SEC_PER_DAY)     # full soft day -> kept
    write_hel1os(base, GOOD)
    write_solexs(base, BAD, 8640)             # 10% soft -> must be dropped
    write_hel1os(base, BAD)
    df = build_dataset(base, out_path=str(base / "out.parquet"),
                       min_hard_cov=0.05, min_soft_cov=0.90)

T0 = day_start_unix(GOOD)
assert list(df.columns) == ["time", "soft", "hard"], df.columns.tolist()
# only the good day survives -> exactly one day of rows, all from GOOD
assert len(df) == SEC_PER_DAY, f"expected 1 kept day, got {len(df)} rows"
assert df["time"].min() == T0 and df["time"].max() == T0 + SEC_PER_DAY - 1

soft_cov = df["soft"].notna().mean()
hard_cov = df["hard"].notna().mean()
assert soft_cov > 0.99, f"soft coverage {soft_cov:.3f}"
assert 0.40 < hard_cov < 0.55, f"hard coverage {hard_cov:.3f}"

def hard_at(sec):
    return df.loc[df["time"] == T0 + sec, "hard"].iloc[0]

assert not np.isnan(hard_at(100)), "hard should exist 100s in"
assert np.isnan(hard_at(8 * 3600)), "hard should be NaN between dumps"

print(f"\nsmoke passed: kept 1/2 days (low-soft day dropped), soft {soft_cov:.1%}, "
      f"hard {hard_cov:.1%}, MJD->unix exact, gaps preserved.")

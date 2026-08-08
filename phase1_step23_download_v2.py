"""
Phase 1, Steps 2 & 3  —  v2 (fixes three bugs found in the v1 run)

FIXED SINCE v1:
  1. Satellite duplication — v1 pulled GOES-16 AND GOES-18, doubling every
     timestamp (23,040 rows where 11,520 was correct). Now pinned to one.
  2. Time-window mismatch — the XRS client returns whole daily FILES, so it
     gave 8 full days, while HEK stopped at Aug 8 00:00. The extra day had
     data but no labels. Flux is now trimmed to the label window.
  3. Duplicate flares — the SWPC filter cut most, but not all. Same flare
     appears twice, once with an active region and once with ar_noaanum = 0.

Run:  python phase1_step23_download_v2.py
"""

import pandas as pd
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Explicit times, not bare dates — this is what keeps flux and labels aligned.
TSTART = "2024-08-01 00:00"
TEND = "2024-08-08 00:00"

# Pin ONE satellite. Mixing GOES-16 and GOES-18 injects calibration offsets
# that a model would happily learn as if they were solar physics.
SATELLITE = 16


# ----------------------------------------------------------------------
# STEP 2 — GOES XRS flux (the signal, X)
# ----------------------------------------------------------------------
print("=" * 70)
print("STEP 2: GOES XRS")
print("=" * 70)

xrs_search = Fido.search(
    a.Time(TSTART, TEND),
    a.Instrument.xrs,
    a.Resolution("avg1m"),
    a.goes.SatelliteNumber(SATELLITE),   # <-- FIX 1
)
print(xrs_search)
print()

xrs_files = Fido.fetch(xrs_search)
goes = TimeSeries(xrs_files, concatenate=True)
df = goes.to_dataframe()

# FIX 2: the client hands back whole daily files, so clip to the exact window.
df = df[(df.index >= TSTART) & (df.index < TEND)]

# Belt and braces: prove no timestamp repeats. Should print 0.
dupe_times = df.index.duplicated().sum()
print(f"Duplicate timestamps: {dupe_times}   (must be 0)")

expected = int((pd.Timestamp(TEND) - pd.Timestamp(TSTART)).total_seconds() // 60)
print(f"Rows: {len(df)}   Expected ~{expected} (1 row per minute)")
print(f"Range: {df.index[0]}  ->  {df.index[-1]}")
print()
print(df[["xrsa", "xrsb"]].head())
print()

# Background level check — this decides whether B-class is even detectable.
bg = df["xrsb"].min()
print(f"Background xrsb : {bg:.2e} W/m^2")
print(f"Peak xrsb       : {df['xrsb'].max():.2e} W/m^2")
if bg >= 1e-6:
    print("NOTE: background is at C-class. B-class flares are BELOW it")
    print("      and cannot be detected this week. Expect zero B labels.")
print()
print(f"Missing values  : xrsa={df['xrsa'].isna().sum()}, xrsb={df['xrsb'].isna().sum()}")
print()


# ----------------------------------------------------------------------
# STEP 3 — HEK flare catalogue (the labels, y)
# ----------------------------------------------------------------------
print("=" * 70)
print("STEP 3: HEK flare catalogue")
print("=" * 70)

hek_search = Fido.search(
    a.Time(TSTART, TEND),
    a.hek.EventType("FL"),
    a.hek.OBS.Observatory == "GOES",
    a.hek.FRM.Name == "SWPC",
)

flares = hek_search["hek"][
    "event_starttime", "event_peaktime", "event_endtime",
    "fl_goescls", "ar_noaanum",
]

fl = flares.to_pandas()
print(f"Rows returned by HEK: {len(fl)}")

# FIX 3: identical (start, peak, end) = one physical flare listed twice.
# Sort ar_noaanum descending first so the copy WITH an active region wins,
# since ar_noaanum = 0 means "unassigned" and carries less information.
before = len(fl)
fl = (
    fl.sort_values("ar_noaanum", ascending=False)
      .drop_duplicates(
          subset=["event_starttime", "event_peaktime", "event_endtime"],
          keep="first",
      )
      .sort_values("event_starttime")
      .reset_index(drop=True)
)
print(f"After deduplication : {len(fl)}   (removed {before - len(fl)})")
print()

# Class counts — your imbalance problem, made visible early.
fl["class_letter"] = fl["fl_goescls"].astype(str).str[0]
print("Count by class:")
for letter in ["A", "B", "C", "M", "X"]:
    n = (fl["class_letter"] == letter).sum()
    print(f"  {letter}: {n}")
print()

# Convert "M5.2" -> 5.2e-5 W/m^2. You will reuse this constantly.
SCALE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

def goes_class_to_flux(cls):
    """'M5.2' -> 5.2e-5 W/m^2. Returns None if unparseable."""
    cls = str(cls).strip()
    if not cls or cls[0] not in SCALE:
        return None
    try:
        return float(cls[1:]) * SCALE[cls[0]]
    except ValueError:
        return None

fl["peak_flux"] = fl["fl_goescls"].apply(goes_class_to_flux)
print("Largest flares by peak flux:")
print(fl.nlargest(3, "peak_flux")[["event_starttime", "fl_goescls", "peak_flux"]])
print()


# ----------------------------------------------------------------------
# CROSS-CHECK — the two datasets should agree
# ----------------------------------------------------------------------
print("=" * 70)
print("CROSS-CHECK")
print("=" * 70)
measured = df["xrsb"].max()
catalogued = fl["peak_flux"].max()
print(f"Peak flux measured in flux data : {measured:.3e}")
print(f"Peak flux listed in catalogue   : {catalogued:.3e}")
ratio = measured / catalogued
print(f"Ratio: {ratio:.2f}   (should be close to 1.0)")
if 0.8 < ratio < 1.25:
    print("PASS — signal and labels agree.")
else:
    print("MISMATCH — investigate before building labels.")
print()

first_flare = pd.to_datetime(fl["event_starttime"].iloc[0])
last_flare = pd.to_datetime(fl["event_starttime"].iloc[-1])
print(f"Flux window   : {df.index[0]}  ->  {df.index[-1]}")
print(f"Flare window  : {first_flare}  ->  {last_flare}")
print("Both flare times must sit inside the flux window.")


# ----------------------------------------------------------------------
# SAVE — so Step 4 does not re-download
# ----------------------------------------------------------------------
df.to_parquet("data/goes_xrs.parquet")
fl.to_parquet("data/flares.parquet")
print()
print("Saved -> data/goes_xrs.parquet, data/flares.parquet")

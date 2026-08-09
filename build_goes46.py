#!/usr/bin/env python
"""
phase 12.2: paired same-days GOES. subset the phase-3 GOES dataset.npz down to
the exact calendar dates aditya observed, so walk_forward compares goes vs aditya
on the SAME flares instead of 80 days vs 46. removes the "those 46 days were
easier" confound.

we subset the npz rather than re-clean from scratch on purpose: it reuses the
identical phase-3 features (same 6-feature pipeline, same scaler), so nothing
about the goes side changes except which windows are present. note the stale-file
trap in the state doc - goes_clean_1s.parquet on disk is only the 7-day slice, so
"filter the clean parquet" isn't actually available without a full run_pipeline
re-download. this path sidesteps that.

walk_forward reads t_end + event_id and uses X as-is, so a plain window subset is
a valid dataset. the 80-day scaler stays baked in - the 46 days are drawn from
the same 80, so its mean/std barely move, invisible to the net.

    python build_goes46.py
    python build_goes46.py --out data/goes46_dataset.npz
then on colab:
    python walk_forward.py --arch tcn --warn-weight 10 --data data/goes46_dataset.npz
    python walk_forward.py --arch tcn --warn-weight 10 --data data/aditya_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _dates(t_end):
    # window timestamps -> set of calendar dates (day resolution)
    return pd.to_datetime(t_end).normalize().unique()


def _subset(z, mask):
    # subset every per-window array; leave scaler/features/meta alone. per-window
    # arrays are the ones whose first axis matches the window count.
    n = mask.shape[0]
    out = {}
    for k in z.files:
        a = z[k]
        out[k] = a[mask] if (a.ndim >= 1 and a.shape[0] == n) else a
    return out


def _report(tag, d):
    ev = d["event_id"]
    warn_ev = np.unique(ev[(d["y_warn"] == 1) & (ev >= 0)])
    det_ev = np.unique(ev[ev >= 0])
    print(f"  {tag}: {len(d['t_end']):,} win, {len(det_ev)} events "
          f"({len(warn_ev)} M+ warning)  "
          f"det+ {d['y_progress'].mean():.1%}  warn+ {d['y_warn'].mean():.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", default="data/dataset.npz")
    ap.add_argument("--aditya", default="data/aditya_dataset.npz")
    ap.add_argument("--out", default="data/goes46_dataset.npz")
    ap.add_argument("--aditya-matched-out", default="data/aditya_matched.npz")
    args = ap.parse_args()

    zg = np.load(ROOT / args.goes, allow_pickle=False)
    za = np.load(ROOT / args.aditya, allow_pickle=False)

    g_dates = _dates(zg["t_end"])
    a_dates = _dates(za["t_end"])
    shared = np.intersect1d(g_dates, a_dates)
    missing = np.setdiff1d(a_dates, g_dates)   # aditya days goes never processed

    print(f"goes days {len(g_dates)}   aditya days {len(a_dates)}   shared {len(shared)}")
    if len(missing):
        # if goes is missing some aditya days, the pair isn't clean until aditya
        # is cut to the same set too - handled below, don't silently ignore.
        print(f"  {len(missing)} aditya day(s) absent from goes: "
              f"{[str(x)[:10] for x in missing[:6]]}{' ...' if len(missing) > 6 else ''}")
    assert len(shared) > 0, "no shared dates - wrong files or a date/tz mismatch, not a real result"

    gmask = np.isin(_dates_of(zg["t_end"]), shared)
    goes46 = _subset(zg, gmask)
    np.savez_compressed(ROOT / args.out, **goes46)
    print(f"\nwrote {args.out}")
    _report("goes46", goes46)

    # keep the pair honest: if goes lacked some aditya days, cut aditya to the
    # shared set so both sides cover identical dates. if goes had them all,
    # aditya_dataset.npz already IS the matched counterpart - say so, write nothing.
    if len(missing):
        amask = np.isin(_dates_of(za["t_end"]), shared)
        adm = _subset(za, amask)
        np.savez_compressed(ROOT / args.aditya_matched_out, **adm)
        print(f"wrote {args.aditya_matched_out}  (aditya cut to the shared days)")
        _report("aditya_matched", adm)
        print(f"\ncompare: {args.out}  vs  {args.aditya_matched_out}")
    else:
        print(f"\naditya covers no extra days - compare {args.out} vs {args.aditya} directly")


def _dates_of(t_end):
    # per-window date array (day-truncated), for the isin mask
    return pd.to_datetime(t_end).normalize().values


if __name__ == "__main__":
    main()

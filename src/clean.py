# notebook 04's cleaning, factored so aditya runs the SAME background estimator
# as goes. verified: bg_causal reproduces the stored column exactly (0.0 err).
#
# the background is computed on a 1-MINUTE MEDIAN series, not the 1s data - a
# 4h rolling quantile over 600k rows is painfully slow and the background only
# varies over hours. then mapped back by ffill.
#
# gotcha that cost hours in phase 1: resample("1min") labels bins on clean
# boundaries but the 1s index carries a sub-second offset (...00.982231), so NOT
# ONE label exists in the index. plain .reindex() matches nothing and returns
# 100% NaN silently. method="ffill" is what makes it work - and it's also the
# causally correct choice, since it only ever carries a PAST value forward.

import numpy as np
import pandas as pd

BG_WINDOW = "4h"
BG_Q = 0.05
BG_MIN_PERIODS = 10
GAP_LIMIT_S = 60
FLOOR = 1e-9


def nan_runs(mask):
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts, ends - starts


def clean_channel(s, gap_limit_s=GAP_LIMIT_S):
    # blank invalid, interpolate only short gaps, mark long ones unusable.
    invalid = s.isna()
    starts, lens = nan_runs(invalid.values)
    valid = ~invalid.values.copy()
    for st, ln in zip(starts, lens):
        if ln <= gap_limit_s:
            valid[st:st + ln] = True          # short gap, interpolation trusted
    clean = s.where(~invalid).interpolate(limit=gap_limit_s, limit_area="inside")
    return clean, valid & clean.notna().values


def causal_background(clean, valid, window=BG_WINDOW, q=BG_Q,
                      min_periods=BG_MIN_PERIODS):
    minute = clean[valid].resample("1min").median()
    bg = minute.rolling(window, min_periods=min_periods).quantile(q)
    return bg.dropna().reindex(clean.index, method="ffill")


def clean_frame(soft, hard, gap_limit_s=GAP_LIMIT_S):
    # soft/hard -> the schema featurize expects (xrsb=soft, xrsa=hard).
    df = pd.DataFrame(index=soft.index)
    df["xrsb"], df["xrsa"] = soft.values, hard.values
    df["xrsb_clean"], df["xrsb_valid"] = clean_channel(soft, gap_limit_s)
    df["xrsa_clean"], df["xrsa_valid"] = clean_channel(hard, gap_limit_s)
    df["log_xrsb"] = np.log10(df["xrsb_clean"].clip(lower=FLOOR))
    df["log_xrsa"] = np.log10(df["xrsa_clean"].clip(lower=FLOOR))
    df["bg_causal"] = causal_background(df["xrsb_clean"], df["xrsb_valid"])
    df["excess"] = df["xrsb_clean"] - df["bg_causal"]
    df["ratio"] = df["xrsb_clean"] / df["bg_causal"]
    df["log_ratio"] = np.log10(df["ratio"].clip(lower=1e-3))
    return df


def aditya_to_clean(parquet_in, parquet_out, gap_limit_s=GAP_LIMIT_S):
    # aditya_clean_1s.parquet (time/soft/hard counts) -> goes-shaped clean frame.
    # soft = SoLEXS 2-22 keV, hard = HEL1OS 20-30 keV. units differ from goes
    # (counts vs W/m^2) but every feature is log'd then z-scored on train, so
    # absolute scale washes out - that's what makes the comparison fair.
    a = pd.read_parquet(parquet_in)
    idx = pd.to_datetime(a["time"], unit="s")
    soft = pd.Series(a["soft"].values, index=idx).sort_index()
    hard = pd.Series(a["hard"].values, index=idx).sort_index()
    soft = soft[~soft.index.duplicated(keep="last")]
    hard = hard[~hard.index.duplicated(keep="last")]

    df = clean_frame(soft, hard, gap_limit_s)
    cov = df["bg_causal"].notna().mean()
    assert cov > 0.5, f"background mostly NaN ({cov:.1%}) - index alignment broken"

    df.to_parquet(parquet_out)
    print(f"wrote {parquet_out}  rows {len(df):,}")
    print(f"  xrsb(soft) valid {df['xrsb_valid'].mean():.1%}   "
          f"xrsa(hard) valid {df['xrsa_valid'].mean():.1%}   "
          f"bg coverage {cov:.1%}")
    return df

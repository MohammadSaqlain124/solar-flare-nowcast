#!/usr/bin/env python
"""
proves src/featurize.py reproduces the pipeline that built data/dataset.npz.
aligns on shared t_end (the cached goes_clean parquet may be only the dev slice,
the npz is the full run), then checks labels + event_id exactly and unscaled
features to float32 precision. split/scaler depend on the full range so they're
not compared here - the reproduction is of the per-window feature+label logic.

    python tests/reproduce_dataset.py
"""
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.featurize import build_dataset_npz     # noqa: E402

DATA = ROOT / "data"
fd, out = tempfile.mkstemp(suffix=".npz")
os.close(fd)                      # we only wanted the path
build_dataset_npz(DATA / "goes_clean_1s.parquet", DATA / "flares.parquet", out)

# load fully into memory and CLOSE the file before touching it again - on windows
# a file with an open handle can't be deleted (WinError 32).
with np.load(DATA / "dataset.npz", allow_pickle=False) as z:
    ref = {k: z[k] for k in z.files}
with np.load(out, allow_pickle=False) as z:
    got = {k: z[k] for k in z.files}

rt = ref["t_end"].astype("datetime64[ns]")
gt = got["t_end"].astype("datetime64[ns]")
common, ri, gi = np.intersect1d(rt, gt, return_indices=True)
assert len(common) > 1000, f"too few shared windows ({len(common)}) - wrong parquet?"

for k in ["y_progress", "y_class", "y_warn", "warn_started", "event_id"]:
    assert np.array_equal(ref[k][ri], got[k][gi]), f"{k} diverges"

ru = ref["X"][ri] * ref["scaler_std"] + ref["scaler_mean"]
gu = got["X"][gi] * got["scaler_std"] + got["scaler_mean"]
md = float(np.abs(ru - gu).max())
assert md < 1e-3, f"features diverge, max|diff|={md:.3e}"

print(f"reproduced pipeline on {len(common):,} shared windows: "
      f"labels + event_id exact, features max|diff| {md:.1e}")

try:
    os.unlink(out)               # best-effort cleanup; harmless if it can't
except OSError:
    pass

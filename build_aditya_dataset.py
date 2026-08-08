#!/usr/bin/env python
"""
phase 4: aditya_clean_1s.parquet -> aditya_dataset.npz, in the exact schema
train.py / walk_forward.py already eat. runs the SAME cleaning (src/clean) and
the SAME windowing/labels/split (src/featurize) as goes - only the input channel
differs. that's what makes soft-vs-fused a measurement of the hard x-rays and
not of my code.

labels come from flares.parquet (GOES/HEK) unchanged - same sun, same UTC, so
the catalogue applies to aditya windows as-is.

    python build_aditya_dataset.py
    python build_aditya_dataset.py --out data/aditya_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.clean import aditya_to_clean
from src.featurize import build_dataset_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="data/aditya_clean_1s.parquet")
    ap.add_argument("--clean-out", default="data/aditya_goesshape_1s.parquet")
    ap.add_argument("--flares", default="data/flares.parquet")
    ap.add_argument("--out", default="data/aditya_dataset.npz")
    args = ap.parse_args()

    print("cleaning + background")
    aditya_to_clean(args.fused, args.clean_out)

    print("\nwindows + labels + split")
    build_dataset_npz(args.clean_out, args.flares, args.out)

    with np.load(args.out, allow_pickle=False) as z:
        print(f"\nwrote {args.out}")
        print(f"  X {z['X'].shape}  train {int(z['is_train'].sum()):,} "
              f"val {int(z['is_val'].sum()):,} test {int(z['is_test'].sum()):,}")
        ev = z["event_id"]
        te_ev = np.unique(ev[z["is_test"] & (ev >= 0)])
        warn_ev = np.unique(ev[(z["y_warn"] == 1) & z["is_test"] & (ev >= 0)])
        print(f"  detection positives {z['y_progress'].mean():.1%}   "
              f"warning positives {z['y_warn'].mean():.2%}")
        print(f"  test events {len(te_ev)}  (M+ warning events {len(warn_ev)})")
        print(f"  meta {[str(x) for x in z['meta']]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
soft-only ablation. drop the three hard-derived features (log_xrsa, hardness,
rate_a) from aditya_dataset.npz, keep the soft trio (log_xrsb, log_ratio,
rate_smooth), retrain same days / same folds. if soft-only matches soft+hard on
warning recall AND first-fire lead, the hard channel is PROVEN redundant for the
5-min warning task - the definitive close on the 12.2 / lead-time story.

channel drop is valid without re-scaling: featurize standardises each feature
independently, so the kept channels are still correctly scaled on their own. we
subset X, scaler, and the feature names by name (not a hardcoded index, in case
FEATURES order ever moves) and leave labels / event_id / t_end / masks alone.

needs the one-line trainer patch (n_feat inferred from X) or the 3-channel net
won't build.

    python build_ablation.py
    python build_ablation.py --data data/aditya_dataset.npz --out data/aditya_soft_dataset.npz
then, same as the fused run:
    python walk_forward.py --arch tcn --warn-weight 10 --data data/aditya_soft_dataset.npz
    python lead_time.py    --arch tcn --warn-weight 10 --data data/aditya_soft_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

DROP = ["log_xrsa", "hardness", "rate_a"]     # every feature that touches the hard channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aditya_dataset.npz")
    ap.add_argument("--out", default="data/aditya_soft_dataset.npz")
    ap.add_argument("--drop", nargs="*", default=DROP)
    args = ap.parse_args()

    z = dict(np.load(ROOT / args.data, allow_pickle=False))
    feats = [str(f) for f in z["features"]]
    keep = [i for i, f in enumerate(feats) if f not in args.drop]
    kept_names = [feats[i] for i in keep]
    dropped = [f for f in feats if f in args.drop]

    assert len(keep) < len(feats), f"nothing dropped - names {args.drop} not in {feats}"
    print(f"features {feats}")
    print(f"  drop {dropped}")
    print(f"  keep {kept_names}  ({len(keep)} channels)")

    out = dict(z)
    out["X"] = np.ascontiguousarray(z["X"][:, :, keep])      # (N, T, F) -> keep soft channels
    out["features"] = np.array(kept_names)
    out["scaler_mean"] = z["scaler_mean"][keep]
    out["scaler_std"] = z["scaler_std"][keep]

    np.savez_compressed(ROOT / args.out, **out)
    print(f"\nwrote {args.out}   X {out['X'].shape}")


if __name__ == "__main__":
    main()

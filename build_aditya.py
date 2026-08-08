#!/usr/bin/env python
"""
phase 4 driver: fuse SoLEXS soft + HEL1OS hard into data/aditya_clean_1s.parquet,
the aditya analog of goes_clean_1s.parquet. prints a per-day coverage report so
you can see exactly which of the overlap days survive.

    python build_aditya.py
    python build_aditya.py --root data/aditya --min-hard-cov 0.05
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.aditya_load import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/aditya")   # holds solexs/ and hel1os/
    ap.add_argument("--out", default="data/aditya_clean_1s.parquet")
    ap.add_argument("--min-hard-cov", type=float, default=0.05)
    ap.add_argument("--min-soft-cov", type=float, default=0.90)
    args = ap.parse_args()
    build_dataset(args.root, out_path=args.out,
                  min_hard_cov=args.min_hard_cov, min_soft_cov=args.min_soft_cov)


if __name__ == "__main__":
    main()

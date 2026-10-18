"""
build_xsection.py
=================
Build the cross-sectional feature panel consumed by
``env.FlyTradeEnv(use_xsection=True)``.

Usage
-----
    python build_xsection.py                 # all data/*_daily.csv
    python build_xsection.py --tickers NVDA,JPM,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR
from xsection import build_panel, default_panel_path, save_panel


def main():
    p = argparse.ArgumentParser(description="Build the cross-sectional feature panel")
    p.add_argument("--tickers", default=None,
                   help="comma-separated universe (default: every data/*_daily.csv)")
    p.add_argument("--out", default=None, help="output .npz path")
    args = p.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    panel = build_panel(DATA_DIR, tickers)
    out = Path(args.out) if args.out else default_panel_path(DATA_DIR)
    save_panel(panel, out)

    n_dates, n_ticks = panel["features"].shape[1], panel["features"].shape[2]
    print(f"[xsection] built panel: {n_dates} dates x {n_ticks} tickers "
          f"x {panel['features'].shape[0]} features")
    print(f"[xsection] date range: {panel['dates'].min()} .. {panel['dates'].max()}")
    print(f"[xsection] saved -> {out}")


if __name__ == "__main__":
    main()

"""Download the OHLCV CSVs used by the trainers and the dashboard.

The checked-in ``data_manifest.json`` records the exact coverage of the local
snapshot.  ``--from-manifest`` is the reproducible bootstrap for a clean clone:
it downloads each ticker over the same first/last dates as the manifest so the
committed artifacts and the local data stay aligned.

Examples
--------
    python data_fetch.py --from-manifest data_manifest.json
    python data_fetch.py --tickers AAPL,MSFT
    python data_fetch.py AAPL --start 2022-01-03 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* using yfinance.

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close,
    Volume.  The multi-level header written by current yfinance is preserved so
    the CSVs match the format already consumed by ``env._load_ohlcv``.
    """
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Ensure proper column order and drop NaNs (e.g. non-trading days).
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return data


def save_data(df: pd.DataFrame, ticker: str, out_dir: Path = Path("data")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_daily.csv"
    df.to_csv(out_path)
    return out_path


def _manifest_jobs(path: Path) -> list[tuple[str, str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for entry in payload.get("files", []):
        first = entry.get("first_date")
        last = entry.get("last_date")
        if not first or not last:
            raise SystemExit(f"Manifest entry for {entry.get('ticker')} has no date coverage")
        # yfinance's ``end`` is exclusive; include the manifest's last session.
        end = (dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat()
        jobs.append((str(entry["ticker"]), first, end))
    return jobs


def _fetch(ticker: str, start: str, end: str, out_dir: Path, force: bool) -> bool:
    out_path = out_dir / f"{ticker}_daily.csv"
    if out_path.exists() and not force:
        print(f"[skip] {ticker}: {out_path} already exists (use --force to refresh)")
        return True
    try:
        df = download_stock_data(ticker, start=start, end=end)
        if df.empty:
            print(f"[fail] {ticker}: no rows returned")
            return False
        save_data(df, ticker, out_dir=out_dir)
        print(f"[ok] {ticker}: {len(df)} rows -> {out_path}")
        return True
    except Exception as exc:  # yfinance raises a variety of network/parse errors
        print(f"[fail] {ticker}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--tickers", help="Comma-separated ticker list")
    parser.add_argument("--from-manifest", type=Path,
                        help="Download every ticker in data_manifest.json using its coverage")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true",
                        help="Re-download files that already exist")
    args = parser.parse_args()

    if args.from_manifest:
        jobs = _manifest_jobs(args.from_manifest)
    elif args.tickers:
        jobs = [(t.strip().upper(), args.start, args.end)
                for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        jobs = [(args.ticker.upper(), args.start, args.end)]
    else:
        parser.error("provide a ticker, --tickers, or --from-manifest")

    ok = True
    for ticker, start, end in jobs:
        ok = _fetch(ticker, start, end, args.out_dir, args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

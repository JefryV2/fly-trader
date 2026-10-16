import yfinance as yf
import pandas as pd
from pathlib import Path

def download_stock_data(ticker: str, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* using yfinance.

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
    """
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Ensure proper column order and drop NaNs (e.g., non‑trading days)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data

def save_data(df: pd.DataFrame, ticker: str, out_dir: Path = Path("data")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_daily.csv"
    df.to_csv(out_path)
    return out_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    args = parser.parse_args()
    df = download_stock_data(args.ticker, start=args.start, end=args.end)
    path = save_data(df, args.ticker)
    print(f"Saved {len(df)} rows to {path}")

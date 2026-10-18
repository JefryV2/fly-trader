"""
xsection.py
===========
Cross-sectional (universe-relative) features.

The single-asset observation tells the policy about *one* ticker's own price
history. It cannot know whether that ticker is strong or weak relative to the
rest of the market — which is the signal that actually predicts relative
returns (momentum / relative strength). Every documented edge in the fly's
regime block (trend, volatility) is *absolute*; this module adds the *relative*
axis.

The panel is built once from the whole universe and cached to
``data/xsection_panel.npz``. ``env.FlyTradeEnv(use_xsection=True)`` appends the
relevant slice (for its ticker and dates) to the observation, and
``fly_brain`` folds it into the sensory summary the read-out actually sees.

Feature block (k = config.XSECTION_OBS_DIM = 8), all NaN-safe and clipped:

    0 rel_mom20    cross-sectional percentile of 20-day return, centred
    1 rel_mom60    ... 60-day
    2 rel_mom120   ... 120-day
    3 rs_vs_mkt    own 60-day return minus the universe median
    4 range_pos    close / 252-day high - 1  (own range position, <= 0)
    5 vol_rel      own annualised vol20 / universe median - 1
    6 breadth      fraction of universe above its 200-day SMA, centred
    7 mkt_mom60    universe equal-weight 60-day return (market trend)

No look-ahead: every feature at date t uses only data up to and including t,
and the cross-sectional ranks use only tickers that have data at t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Feature order is part of the on-disk/observation contract. Do not reorder.
FEATURE_NAMES = [
    "rel_mom20", "rel_mom60", "rel_mom120", "rs_vs_mkt",
    "range_pos", "vol_rel", "breadth", "mkt_mom60",
]
N_FEATURES = len(FEATURE_NAMES)

# Clip bounds (keeps early-history/edge cases from producing huge inputs).
_CLIP = np.array([0.5, 0.5, 0.5, 2.0, 1.0, 3.0, 0.5, 2.0], dtype=np.float32)

_PANEL_CACHE: dict | None = None
_PANEL_CACHE_PATH: Path | None = None


def default_panel_path(data_dir: Path) -> Path:
    return Path(data_dir) / "xsection_panel.npz"


# ── build ──────────────────────────────────────────────────────────────────
def build_panel(data_dir: Path, tickers: list[str] | None = None) -> dict:
    """Compute the cross-sectional feature panel over the whole universe.

    Returns a dict with ``dates`` (int64 ns), ``tickers`` (list) and
    ``features`` ((k, n_dates, n_tickers) float32).
    """
    data_dir = Path(data_dir)
    if tickers is None:
        tickers = sorted(
            p.name[: -len("_daily.csv")] for p in data_dir.glob("*_daily.csv")
        )
    from env import _load_ohlcv  # local import: avoids a circular import at module load

    series = {}
    for t in tickers:
        path = data_dir / f"{t}_daily.csv"
        if not path.exists():
            continue
        df = _load_ohlcv(path)
        if len(df) > 0:
            series[t] = df["Close"].astype(np.float64)
    if not series:
        raise SystemExit(f"No usable tickers found in {data_dir}")

    # Outer-join onto the union of trading dates.
    close = pd.DataFrame(series).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    ret1 = close.pct_change(fill_method=None)

    mom20 = close / close.shift(20) - 1.0
    mom60 = close / close.shift(60) - 1.0
    mom120 = close / close.shift(120) - 1.0
    vol20 = ret1.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    high252 = close.rolling(252, min_periods=60).max()
    sma200 = close.rolling(200, min_periods=60).mean()

    # Cross-sectional quantities (per row/date).
    def rank_centred(m):
        return m.rank(axis=1, pct=True) - 0.5

    rel_mom20 = rank_centred(mom20)
    rel_mom60 = rank_centred(mom60)
    rel_mom120 = rank_centred(mom120)
    med_mom60 = mom60.median(axis=1)
    rs_vs_mkt = mom60.sub(med_mom60, axis=0)
    range_pos = close / (high252 + 1e-12) - 1.0
    med_vol = vol20.median(axis=1)
    vol_rel = vol20.div(med_vol + 1e-12, axis=0) - 1.0

    above200 = (close > sma200).astype(float)
    above200[sma200.isna()] = np.nan
    breadth = above200.mean(axis=1, skipna=True) - 0.5
    mkt_mom60 = mom60.mean(axis=1, skipna=True)

    # Broadcast the market-level series across ticker columns for a uniform layout.
    breadth_m = pd.DataFrame(
        np.repeat(breadth.values[:, None], close.shape[1], axis=1),
        index=close.index, columns=close.columns,
    )
    mkt_mom60_m = pd.DataFrame(
        np.repeat(mkt_mom60.values[:, None], close.shape[1], axis=1),
        index=close.index, columns=close.columns,
    )

    stack = [rel_mom20, rel_mom60, rel_mom120, rs_vs_mkt,
             range_pos, vol_rel, breadth_m, mkt_mom60_m]
    feats = np.empty((N_FEATURES, len(close.index), close.shape[1]), dtype=np.float32)
    for i, m in enumerate(stack):
        arr = np.nan_to_num(m.values.astype(np.float32), nan=0.0,
                            posinf=0.0, neginf=0.0)
        feats[i] = np.clip(arr, -_CLIP[i], _CLIP[i])

    return {
        "dates": close.index.values.astype("datetime64[ns]").astype(np.int64),
        "tickers": list(close.columns),
        "features": feats,
        "feature_names": FEATURE_NAMES,
    }


def save_panel(panel: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dates=panel["dates"],
        tickers=np.array(panel["tickers"], dtype=object),
        features=panel["features"],
        feature_names=np.array(panel["feature_names"], dtype=object),
    )
    return path


def load_panel(path: Path) -> dict:
    """Load (and cache) the panel; raise a clear error if it is missing."""
    global _PANEL_CACHE, _PANEL_CACHE_PATH
    path = Path(path)
    if _PANEL_CACHE is not None and _PANEL_CACHE_PATH == path:
        return _PANEL_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"Cross-sectional panel not found at {path}. "
            f"Build it with:  python build_xsection.py"
        )
    z = np.load(path, allow_pickle=True)
    panel = {
        "dates": z["dates"],
        "tickers": list(z["tickers"]),
        "features": z["features"],
        "feature_names": list(z["feature_names"]),
    }
    _PANEL_CACHE, _PANEL_CACHE_PATH = panel, path
    return panel


def features_for_ticker(panel: dict, ticker: str, dates) -> np.ndarray:
    """Return the (len(dates), k) cross-sectional block for ``ticker``.

    Missing (date, ticker) cells and dates absent from the panel are zero.
    """
    dates = pd.DatetimeIndex(dates)
    panel_dates = pd.DatetimeIndex(panel["dates"])
    tickers = list(panel["tickers"])
    n = len(dates)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    if ticker not in tickers:
        return out
    row_idx = panel_dates.get_indexer(dates)
    col = tickers.index(ticker)
    feats = panel["features"]
    valid = row_idx >= 0
    if valid.any():
        out[valid] = feats[:, row_idx[valid], col].T
    return out

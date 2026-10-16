"""Tests for the shared masking / selection / metrics helpers."""
import numpy as np
import pytest

from config import MASK_PRESETS
from strategy import apply_action_mask, apply_preset, performance_metrics, sample_action


def test_mask_zeroes_invalid_sells():
    p = apply_action_mask(np.ones(7), shares=0.0, cash=1e5)
    assert np.allclose(p[4:], 0.0)
    assert p.sum() == pytest.approx(1.0)


def test_mask_zeroes_invalid_buys():
    p = apply_action_mask(np.ones(7), shares=10.0, cash=0.0)
    assert np.allclose(p[1:4], 0.0)
    assert p.sum() == pytest.approx(1.0)


def test_mask_preserves_existing_distribution_without_heuristics():
    raw = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    p = apply_action_mask(raw, shares=5.0, cash=1e5, hold_cap=None, buy_floor=None)
    # all actions valid -> distribution unchanged
    assert np.allclose(p, raw / raw.sum(), atol=1e-9)


def test_all_presets_sum_to_one():
    for name, cfg in MASK_PRESETS.items():
        for shares, cash in [(0.0, 1e5), (5.0, 1e5), (5.0, 0.0), (0.0, 0.0)]:
            p = apply_action_mask(np.ones(7), shares=shares, cash=cash, **cfg)
            assert p.sum() == pytest.approx(1.0), name
            assert np.all(p >= 0), name


def test_sample_action_deterministic():
    p = np.array([0.1, 0.1, 0.7, 0.05, 0.02, 0.02, 0.01])
    assert sample_action(p, deterministic=True) == 2


def test_deterministic_never_picks_invalid():
    # With shares=0, sells are masked; deterministic pick must be a buy/hold.
    p = apply_action_mask(np.ones(7), shares=0.0, cash=1e5)
    assert sample_action(p, deterministic=True) in (0, 1, 2, 3)


def test_performance_metrics_basic():
    hist = [100.0, 110.0, 100.0, 120.0]
    m = performance_metrics(hist, initial_cash=100.0)
    assert m["total_return_pct"] == pytest.approx(20.0)
    assert m["max_drawdown_pct"] > 0
    assert 0 <= m["win_rate"] <= 100


def test_baseline_buy_hold_buys_then_holds():
    from strategy import baseline_probs
    from config import OBS_DIM
    obs = np.ones(OBS_DIM)
    p = baseline_probs("buy_hold", obs, shares=0.0, cash=1e5, prev_action=0)
    assert p[3] == 1.0
    p = baseline_probs("buy_hold", obs, shares=1.0, cash=0.0, prev_action=3)
    assert p[0] == 1.0


def test_compare_on_env_runs_all_baselines():
    from config import DATA_DIR
    from env import FlyTradeEnv
    from strategy import compare_on_env

    def factory():
        return FlyTradeEnv(DATA_DIR / "AAPL_daily.csv", window_size=30,
                           start_date="2024-01-01", end_date="2024-06-30")

    res = compare_on_env(factory, None, include_legacy=False)
    assert set(res.keys()) == {"buy_hold", "momentum", "trend200", "random"}
    for m in res.values():
        assert "total_return_pct" in m
        assert "sharpe" in m
        assert "trades" in m
        assert m["steps"] > 0


def test_compare_on_env_buy_hold_matches_price_ratio():
    """buy&hold's return should closely track the raw close-price ratio."""
    from config import DATA_DIR
    from env import FlyTradeEnv, _load_ohlcv
    from strategy import compare_on_env

    start, end = "2024-01-01", "2024-06-30"
    s = _load_ohlcv(DATA_DIR / "AAPL_daily.csv").loc[start:end]["Close"]
    raw = (s.iloc[-2] / s.iloc[29] - 1) * 100  # env trades on window-end close

    def factory():
        return FlyTradeEnv(DATA_DIR / "AAPL_daily.csv", window_size=30,
                           start_date=start, end_date=end)

    bh = compare_on_env(factory, None, include_legacy=False)["buy_hold"]
    # allow for the 0.1% buy cost and the env's final-bar off-by-one
    assert abs(bh["total_return_pct"] - raw) < 2.0

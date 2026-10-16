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
    obs = np.ones(390)
    p = baseline_probs("buy_hold", obs, shares=0.0, cash=1e5, prev_action=0)
    assert p[3] == 1.0
    p = baseline_probs("buy_hold", obs, shares=1.0, cash=0.0, prev_action=3)
    assert p[0] == 1.0

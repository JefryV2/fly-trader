"""Tests for the multi-seed confidence-interval reporting."""
import numpy as np

from scripts.seed_sweep import _flatten_single_eval, _mean_ci, summarize


def test_mean_ci_uses_t_interval_for_small_samples():
    out = _mean_ci([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert out["mean"] == 2.0
    assert out["sd"] == 1.0
    assert out["interval"] == "t"
    assert out["ci_low"] < out["mean"] < out["ci_high"]
    # The t interval for n=3 is wider than the normal approximation would be.
    half = out["ci_high"] - out["mean"]
    assert half > 1.959963984540054 * out["se"]


def test_mean_ci_single_seed_is_a_point():
    out = _mean_ci([7.5])
    assert out["n"] == 1
    assert out["mean"] == out["ci_low"] == out["ci_high"] == 7.5
    assert out["sd"] == 0.0


def _row(seed, fly, bh, sharpe, maxdd, val_fly=0.0):
    return {
        "seed": seed,
        "validation": {"fly": val_fly, "excess": val_fly - bh},
        "test": {"fly": fly, "buyhold": bh, "excess": fly - bh,
                 "sharpe": sharpe, "maxdd": maxdd},
    }


def test_summarize_reports_distribution_not_best_seed():
    rows = [
        _row(0, 10.0, 5.0, 1.0, 2.0),
        _row(1, 20.0, 5.0, 2.0, 4.0),
        _row(2, 30.0, 5.0, 3.0, 6.0),
    ]
    rows[1]["guard_failed"] = True
    s = summarize(rows)
    assert s["test_fly"]["mean"] == 20.0
    assert s["test_excess"]["mean"] == 15.0
    # The mean is reported, not the best seed (30%).
    assert s["test_excess"]["mean"] < max(r["test"]["excess"] for r in rows)
    assert s["test_excess"]["positive_seeds"] == 3
    assert s["test_excess"]["positive_fraction"] == 1.0
    assert s["test_buyhold"]["sd"] == 0.0
    assert np.isclose(s["test_fly"]["sd"], 10.0)
    assert s["guard_failed_seeds"] == 1


def test_single_eval_shape_matches_portfolio_eval_shape():
    out = _flatten_single_eval({
        "mean_fly": 12.0, "mean_buyhold": 10.0, "mean_excess": 2.0,
        "mean_sharpe": 1.1, "mean_maxdd": 4.0, "mean_bh_maxdd": 5.0,
        "per_ticker": {"X": {}},
    })
    assert out["fly"] == 12.0
    assert out["buyhold"] == 10.0
    assert out["excess"] == 2.0
    assert out["sharpe"] == 1.1
    assert out["maxdd"] == 4.0
    assert out["bh_maxdd"] == 5.0
    assert out["per_ticker"] == {"X": {}}

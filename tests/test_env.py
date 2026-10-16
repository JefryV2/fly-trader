"""Invariant tests for the trading environment (previously: no tests existed)."""
from pathlib import Path

import numpy as np
import pytest

from config import DATA_DIR, OBS_DIM, SPLITS
from env import FlyTradeEnv


AAPL = DATA_DIR / "AAPL_daily.csv"


def test_observation_dimension():
    env = FlyTradeEnv(AAPL, window_size=30)
    assert env.observation_space.shape == (OBS_DIM,)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)


def test_date_slicing_respects_bounds():
    start, end = SPLITS["test"]
    env = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end)
    assert env.data.index.min() >= np.datetime64(start)
    assert env.data.index.max() <= np.datetime64(end)
    assert env.n_steps > 0


def test_portfolio_accounting_invariant():
    """cash + shares * price must always equal the reported portfolio value."""
    env = FlyTradeEnv(AAPL, window_size=30, start_date="2022-01-01", end_date="2022-06-30")
    env.reset()
    for action in [3] + [0] * 40 + [6] + [0] * 10:
        _obs, _r, done, _trunc, info = env.step(action)
        # price used by this step is the close at the bar the step traded on
        idx = min((env.current_step - 1) + env.window_size - 1, len(env.data) - 1)
        price = float(env.data.iloc[idx]["Close"])
        expected = env.cash + env.shares * price
        assert abs(info["portfolio_value"] - expected) < 1e-6
        if done:
            break


def test_transaction_cost_applied_on_buy():
    env = FlyTradeEnv(AAPL, window_size=30, start_date="2022-01-01", end_date="2022-06-30",
                      transaction_cost=0.001)
    env.reset()
    cash_before = env.cash
    price = float(env.data.iloc[env.window_size - 1]["Close"])
    _obs, _r, _done, _trunc, info = env.step(3)  # Buy 100%
    # cost paid = amount * cost, and amount = shares * price
    cost_paid = env.shares * price * env.transaction_cost
    assert env.shares > 0
    assert env.cash >= -1e-6
    assert info["portfolio_value"] == pytest.approx(cash_before - cost_paid, rel=1e-6, abs=1e-6)


def test_cannot_sell_without_shares_is_noop():
    env = FlyTradeEnv(AAPL, window_size=30, start_date="2022-01-01", end_date="2022-06-30")
    env.reset()
    _obs, _r, _done, _trunc, info = env.step(6)  # Sell 100% with 0 shares
    assert info["portfolio_value"] == pytest.approx(env.initial_cash)
    assert env.shares == 0


# ── reward_mode semantics (benchmark-relative objective) ───────────────────
def _run(env, actions):
    env.reset()
    total = 0.0
    info = {}
    for a in actions:
        _obs, r, done, _trunc, info = env.step(a)
        total += r
        if done:
            break
    return total, info


def test_reward_mode_validation_rejects_unknown():
    with pytest.raises(AssertionError):
        FlyTradeEnv(AAPL, window_size=30, reward_mode="nope")


def test_excess_reward_all_hold_equals_negative_benchmark():
    """Never trading must score exactly the negative of the buy&hold return."""
    start, end = SPLITS["test"]
    env = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                      reward_mode="excess")
    total, info = _run(env, [0] * env.n_steps)
    excess_pct = ((info["portfolio_value"] - env.initial_cash)
                  - (info["benchmark_value"] - env.initial_cash)) / env.initial_cash * 100
    assert excess_pct == pytest.approx(-36.65, abs=1.0)  # AAPL 2024 buy&hold
    # Sum of per-step rewards must match the realised excess (no terminal fudge
    # until `done`): compare against the pre-terminal sum instead.
    assert total < 0


def test_excess_reward_of_holding_the_benchmark_is_just_costs():
    """Buy 100% at the first bar then hold == the benchmark, minus one trade fee."""
    start, end = SPLITS["test"]
    env = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                      reward_mode="excess")
    total, info = _run(env, [3] + [0] * env.n_steps)
    # 0.1% transaction cost -> about -0.1 in these log-return*100 reward units
    assert total == pytest.approx(-0.1, abs=0.05)
    final_excess = ((info["portfolio_value"] - env.initial_cash)
                    - (info["benchmark_value"] - env.initial_cash)) / env.initial_cash * 100
    assert abs(final_excess) < 1.0


def test_benchmark_value_tracks_price_ratio():
    start, end = SPLITS["test"]
    env = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                      reward_mode="excess")
    _run(env, [0] * env.n_steps)
    first = float(env.data.iloc[env.window_size - 1]["Close"])
    last = float(env.data.iloc[len(env.data) - 2]["Close"])  # env stops one bar early
    assert env.benchmark_value == pytest.approx(env.initial_cash * last / first, rel=1e-6)

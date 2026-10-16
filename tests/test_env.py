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

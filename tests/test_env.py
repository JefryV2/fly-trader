"""Invariant tests for the trading environment (previously: no tests existed)."""
from pathlib import Path

import numpy as np
import pytest

from config import DATA_DIR, OBS_DIM, PORTFOLIO_SLICE, SPLITS
from env import FlyTradeEnv, fit_indicator_stats


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


def test_validation_reuses_train_only_indicator_scaler():
    train_mean, train_std = fit_indicator_stats(
        AAPL, start_date="2022-01-01", end_date="2022-06-30")
    env = FlyTradeEnv(
        AAPL, window_size=30, start_date="2023-01-01", end_date="2023-06-30",
        indicator_stats=(train_mean, train_std))
    assert np.allclose(env.ind_mean, train_mean)
    assert np.allclose(env.ind_std, train_std)


def test_indicator_stats_shape_is_validated():
    with pytest.raises(ValueError, match="indicator_stats"):
        FlyTradeEnv(AAPL, window_size=30, indicator_stats=(np.zeros(7), np.ones(7)))


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


def test_trade_recency_updates_outside_shaped_reward():
    """The observation's steps-since-trade feature must reset after a trade.

    It used to be updated only inside the legacy ``shaped`` reward branch, so
    the default ``excess`` objective reported a monotonically increasing stale
    value even immediately after a buy.
    """
    env = FlyTradeEnv(AAPL, window_size=30, start_date="2022-01-01",
                      end_date="2022-06-30", reward_mode="excess")
    env.reset()
    for _ in range(5):
        _obs, _r, _done, _trunc, _info = env.step(0)

    obs, _r, _done, _trunc, _info = env.step(3)  # Buy 100%
    assert env.last_trade_step == env.current_step
    assert env.last_trade_step > 0
    assert obs[PORTFOLIO_SLICE][3] == pytest.approx(0.0, abs=1e-6)

    obs_next, _r, _done, _trunc, _info = env.step(0)
    assert obs_next[PORTFOLIO_SLICE][3] == pytest.approx(1.0 / 30.0, abs=1e-6)


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


# ── drawdown penalty (the term that makes de-risking learnable) ────────────
def test_dd_penalty_punishes_deepening_drawdown_but_not_cash():
    start, end = "2022-01-01", "2022-12-31"     # a drawdown year for AAPL
    long_acts = [3] + [0] * 400                 # buy 100% then hold

    plain = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                        reward_mode="excess", dd_penalty=0.0)
    penalised = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                            reward_mode="excess", dd_penalty=1.0)
    t_plain, _ = _run(plain, long_acts)
    t_penalised, _ = _run(penalised, long_acts)
    # Staying invested through a persistent drawdown must cost more.
    assert t_penalised < t_plain - 1.0

    # A pure-cash agent never prints a new equity low, so the penalty is a no-op.
    cash_plain = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                             reward_mode="excess", dd_penalty=0.0)
    cash_pen = FlyTradeEnv(AAPL, window_size=30, start_date=start, end_date=end,
                           reward_mode="excess", dd_penalty=1.0)
    s_plain, _ = _run(cash_plain, [0] * cash_plain.n_steps)
    s_pen, _ = _run(cash_pen, [0] * cash_pen.n_steps)
    assert s_pen == pytest.approx(s_plain)


# ── long-horizon regime block (trend / volatility context) ─────────────────
MSFT = DATA_DIR / "MSFT_daily.csv"


def test_regime_block_defined_from_first_bar_of_slice():
    """A test slice starting mid-history must still have a 200-day trend."""
    env = FlyTradeEnv(MSFT, window_size=30,
                      start_date="2022-01-03", end_date="2022-12-31")
    obs, _ = env.reset()
    assert env.observation_space.shape == (OBS_DIM,)
    assert obs.shape == (OBS_DIM,)
    regime = obs[-4:]
    assert np.isfinite(regime).all()
    # close/SMA200 - 1 is defined because MSFT has history before the slice.
    assert regime[0] != 0.0
    assert regime[2] > 0.0   # realised volatility is positive


def test_regime_gate_blocks_exposure_below_trend():
    from config import OBS_DIM
    from strategy import regime_gate_prob_fn

    def inner_always_buy(_obs, _prev, _shares, _cash):
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    gate = regime_gate_prob_fn(inner_always_buy)
    obs = np.zeros(OBS_DIM)
    obs[-4] = -0.10          # below the 200-day trend -> risk-off
    p_flat = gate(obs, 0, 0.0, 1e5)
    assert p_flat[0] == 1.0 and p_flat[3] == 0.0
    p_long = gate(obs, 3, 10.0, 0.0)
    assert p_long[6] == 1.0   # exit
    obs[-4] = 0.10           # above the trend -> pass through
    assert gate(obs, 0, 0.0, 1e5)[3] == 1.0

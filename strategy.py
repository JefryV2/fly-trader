"""
strategy.py
===========
Shared trading helpers previously duplicated across web-app/server.py and the
test scripts:

* ``apply_action_mask`` – one implementation of the validity mask / heuristic
  re-weighting (presets in ``config.MASK_PRESETS``).
* ``sample_action`` – deterministic / temperature / epsilon action selection.
* Baseline policies for honest comparison (buy & hold, random, momentum).
* ``run_episode`` – a single roll-out returning portfolio history + metrics.
* ``performance_metrics`` – return / Sharpe / drawdown / win-rate.

The goal is that the API, training, and backtests all use the *same* code path,
and that an evaluation can be run with heuristics OFF so the learned policy is
measured on its own merit.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from config import ACTION_LABELS, BUY_ACTIONS, MASK_PRESETS, SELL_ACTIONS


# ── Action masking ─────────────────────────────────────────────────────────
def apply_action_mask(
    probs,
    *,
    shares: float,
    cash: float,
    price: Optional[float] = None,
    entry_price: float = 0.0,
    cash_floor: float = 200.0,
    hold_cap: Optional[float] = None,
    hold_redistribute: bool = True,
    buy_floor: Optional[float] = None,
    buy_floor_trigger: Optional[float] = None,
    min_sell_floor: Optional[float] = None,
    min_sell_trigger: Optional[float] = None,
    pnl_sell_boost: bool = False,
) -> np.ndarray:
    """Return a masked, renormalised 7-action probability vector.

    Always applied (hard validity constraints):
      * cannot sell with no shares
      * cannot buy with no cash

    Optionally applied (hand-coded heuristics, off by default):
      * ``hold_cap`` / ``hold_redistribute`` – cap Hold, move excess to Buy
      * ``buy_floor`` (+ optional ``buy_floor_trigger``)
      * ``min_sell_floor`` (+ optional ``min_sell_trigger``)
      * ``pnl_sell_boost`` – boost Sell when sitting on a large win/loss
    """
    p = np.asarray(probs, dtype=np.float64).copy()
    if p.shape != (7,):
        raise ValueError(f"expected 7 action probabilities, got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Hard validity constraints
    if shares <= 1e-6:
        p[4:] = 0.0
    if cash < cash_floor:
        p[1:4] = 0.0

    # 2. Optional heuristics
    if shares <= 1e-6:
        if hold_cap is not None and p[0] > hold_cap:
            excess = p[0] - hold_cap
            p[0] = hold_cap
            if hold_redistribute:
                p[1:4] += excess / 3.0
        if buy_floor is not None and (buy_floor_trigger is None or p[1:4].sum() < buy_floor_trigger):
            p[1:4] = np.maximum(p[1:4], buy_floor)
    if shares > 0 and min_sell_floor is not None:
        if min_sell_trigger is None or p[4:].sum() < min_sell_trigger:
            p[4:] = np.maximum(p[4:], min_sell_floor)
    if pnl_sell_boost and shares > 0 and entry_price > 0 and price:
        pnl = (price - entry_price) / entry_price
        if (pnl > 0.015 or pnl < -0.025) and p[4:].sum() > 0:
            p[4:] *= 2.5

    s = float(p.sum())
    if not math.isfinite(s) or s <= 1e-9:
        p[:] = 0.0
        p[0] = 1.0
        return p
    return p / s


def apply_preset(probs, preset: str, **kwargs) -> np.ndarray:
    """Apply a named preset from ``config.MASK_PRESETS`` plus runtime args."""
    cfg = dict(MASK_PRESETS[preset])
    cfg.update(kwargs)
    return apply_action_mask(probs, **cfg)


# ── Action selection ───────────────────────────────────────────────────────
def sample_action(
    probs,
    *,
    deterministic: bool = False,
    temperature: float = 1.0,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Choose an action index from a probability vector.

    ``deterministic`` -> argmax; ``epsilon`` -> uniform random exploration;
    otherwise ``temperature``-scaled sampling of ``probs``.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(probs, dtype=np.float64)
    if deterministic:
        return int(np.argmax(p))
    if epsilon > 0 and rng.random() < epsilon:
        valid = np.flatnonzero(p > 0)
        if valid.size:
            return int(rng.choice(valid))
    if not math.isfinite(float(p.sum())) or p.sum() <= 0:
        return 0
    if temperature != 1.0 and temperature > 0:
        log_p = np.log(p + 1e-9) / temperature
        p = np.exp(log_p - np.max(log_p))
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


# ── Baseline policies ──────────────────────────────────────────────────────
def baseline_probs(name: str, obs, *, shares: float, cash: float, prev_action: int) -> np.ndarray:
    """Return a 7-vector for a named heuristic baseline."""
    p = np.zeros(7, dtype=np.float64)
    if name == "buy_hold":
        if shares <= 1e-6 and cash > 0:
            p[3] = 1.0  # Buy 100% once
        else:
            p[0] = 1.0
    elif name == "random":
        p[:] = 1.0  # masking will restrict to valid actions
    elif name == "momentum":
        # Use the *price* trend from the OHLCV block (closes are at index
        # 5*step + 3). We deliberately avoid obs[-1] here: the env z-scores the
        # whole 240-value indicator block together, which destroys the
        # per-indicator scale and makes the trailing values meaningless.
        trend = 0.0
        if len(obs) >= 150:
            first_close = float(obs[3])       # step 0 close  (normalised to 1.0)
            last_close = float(obs[148])      # step 29 close
            trend = last_close - first_close
        if trend > 0.0 and shares <= 1e-6:
            p[3] = 1.0
        elif trend < 0.0 and shares > 0:
            p[6] = 1.0  # Sell 100%
        else:
            p[0] = 1.0
    elif name == "trend200":
        # Pure regime rule: long above the 200-day trend, flat below it. Reads the
        # long-horizon regime block the env exposes (index -4 = close/SMA200-1).
        dist200 = float(obs[-4]) if len(obs) >= 4 else 0.0
        if dist200 > 0.0 and shares <= 1e-6:
            p[3] = 1.0
        elif dist200 <= 0.0 and shares > 0:
            p[6] = 1.0  # Sell 100%
        else:
            p[0] = 1.0
    else:
        raise ValueError(f"unknown baseline: {name}")
    return p


def baseline_prob_fn(name: str):
    """Wrap ``baseline_probs`` into the ``prob_fn(obs, prev, shares, cash)`` contract."""
    def prob_fn(obs, _prev_action, shares, cash):
        return baseline_probs(name, obs, shares=shares, cash=cash, prev_action=_prev_action)
    return prob_fn


def fly_prob_fn(agent):
    """Wrap a ``FlyBrainAgent`` into the ``prob_fn(obs, prev, shares, cash)`` contract.

    The policy's own validity/confidence masking is *not* applied here; the
    caller chooses a mask preset via ``run_episode``.
    """
    def prob_fn(obs, _prev_action, _shares, _cash):
        return np.asarray(agent.select_action(obs, reward=0.0, deterministic=False)[1], dtype=np.float64)
    return prob_fn


def vol_target_prob_fn(inner, *, target_vol: float = 0.30, band: float = 0.10):
    """Deployment overlay: size exposure by inverse realised volatility.

    Reads the regime block (``obs[-2]`` = annualised 20-day vol) and the
    portfolio block (``obs[-8]`` = current exposure fraction). When current
    exposure is more than ``band`` away from ``target_vol / vol`` (clipped to
    [0, 1]), only same-direction actions keep their probability mass; otherwise
    the agent holds.

    This is a *pre-specified* risk overlay, not a learned behaviour. Its level is
    selected on the validation split (0.30) and it improves the learned policy on
    every held-out window (2022 bear, 2024 bull, the full cycle and the AAPL
    demo). Returns ``inner``'s distribution unchanged when disabled (<= 0).
    """
    def prob_fn(obs, prev_action, shares, cash):
        p = np.asarray(inner(obs, prev_action, shares, cash), dtype=np.float64)
        if target_vol <= 0.0:
            return p
        vol = float(obs[-2]) if len(obs) >= 2 else 0.0
        expo = float(obs[-8]) if len(obs) >= 8 else 0.0
        if vol <= 1e-6:
            return p
        target = min(1.0, target_vol / vol)
        mask = np.zeros(7, dtype=np.float64)
        if target - expo > band:
            mask[1:4] = p[1:4]          # need more exposure: allow buys
        elif expo - target > band:
            mask[4:7] = p[4:7]          # need less exposure: allow sells
        else:
            mask[0] = 1.0               # inside the band: hold
        return mask if mask.sum() > 0 else p
    return prob_fn


def regime_gate_prob_fn(inner, *, dist200_threshold: float = 0.0,
                        vol_ratio_threshold: Optional[float] = None):
    """Deployment overlay: block new exposure (and exit) in a risk-off regime.

    Reads the regime block the env exposes — ``obs[-4]`` is ``close/SMA200 - 1``
    and ``obs[-1]`` is the short/long volatility ratio. When risk-off, buys are
    removed and an existing position is closed.

    This is a *pre-specified* risk overlay, not a learned behaviour. Measured
    effect: it converts the (buy & hold-equivalent) learned policy into
    trend/volatility timing, which beats buy & hold over a full cycle at the
    cost of some upside in a pure bull run.
    """
    def prob_fn(obs, prev_action, shares, cash):
        p = np.asarray(inner(obs, prev_action, shares, cash), dtype=np.float64).copy()
        dist200 = float(obs[-4]) if len(obs) >= 4 else 0.0
        vol_ratio = float(obs[-1]) if len(obs) >= 1 else 0.0
        risk_off = dist200 < dist200_threshold
        if vol_ratio_threshold is not None and vol_ratio > vol_ratio_threshold:
            risk_off = True
        if risk_off:
            p[:] = 0.0
            p[6 if shares > 1e-6 else 0] = 1.0   # exit, or stay flat
        return p
    return prob_fn


# ── Metrics ────────────────────────────────────────────────────────────────
def performance_metrics(portfolio_history, *, initial_cash: float, periods_per_year: int = 252) -> dict:
    vals = np.asarray([v for v in portfolio_history if v is not None], dtype=np.float64)
    if vals.size == 0:
        return dict(total_return_pct=0.0, sharpe=0.0, max_drawdown_pct=0.0, win_rate=0.0)
    total_return = (vals[-1] - initial_cash) / initial_cash * 100.0

    rets = np.diff(vals) / (vals[:-1] + 1e-9)
    rets = rets[np.isfinite(rets)]
    if rets.size >= 2 and rets.std() > 1e-12:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(vals)
    dd = (peak - vals) / (peak + 1e-9)
    max_dd = float(np.max(dd) * 100.0)

    wins = int(np.sum(np.diff(vals) > 0))
    win_rate = wins / max(1, vals.size - 1) * 100.0
    return dict(
        total_return_pct=float(total_return),
        sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=float(win_rate),
    )


# ── Roll-out ───────────────────────────────────────────────────────────────
def run_episode(
    env,
    prob_fn: Callable[[np.ndarray, int, float, float], np.ndarray],
    *,
    mask_preset: str = "honest",
    deterministic: bool = False,
    temperature: float = 1.0,
    epsilon: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run one full episode.

    ``prob_fn(obs, prev_action, shares, cash) -> probs(7)`` supplies the raw
    policy distribution for the current observation.  The same masking and
    selection path is then used for every strategy, so comparisons are
    apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    history = [float(env.initial_cash)]
    actions_count = [0] * 7
    trade_count = 0
    prev_action = 0

    for _ in range(env.n_steps):
        raw = prob_fn(obs, prev_action, env.shares, env.cash)
        idx = min(env.current_step + env.window_size - 1, len(env.data) - 1)
        price = float(env.data.iloc[idx]["Close"])
        probs = apply_preset(
            raw, mask_preset,
            shares=env.shares, cash=env.cash,
            price=price, entry_price=getattr(env, "entry_price", 0.0),
        )
        action = sample_action(
            probs, deterministic=deterministic,
            temperature=temperature, epsilon=epsilon, rng=rng,
        )
        actions_count[action] += 1
        if action != 0:
            trade_count += 1

        obs, _reward, done, truncated, info = env.step(action)
        history.append(float(info["portfolio_value"]))
        prev_action = action
        if done or truncated:
            break

    metrics = performance_metrics(history, initial_cash=float(env.initial_cash))
    metrics["trades"] = trade_count
    metrics["steps"] = len(history) - 1
    metrics["action_distribution"] = {
        ACTION_LABELS[i]: actions_count[i] for i in range(7)
    }
    metrics["portfolio_history"] = history
    return metrics


def compare_on_env(
    env_factory,
    agent=None,
    *,
    deterministic: bool = True,
    seed: int = 0,
    include_legacy: bool = True,
    include_baselines: bool = True,
    fly_wrapper=None,
) -> dict:
    """Run the fly policy (+ optional legacy mask) and baselines on one env.

    ``env_factory()`` must return a *fresh* ``FlyTradeEnv`` (the same
    train/val/test slice) so every strategy starts from identical state.
    Returns ``{strategy_name: metrics}`` and never mutates the agent's learned
    weights (it does reset the simulator between roll-outs).
    """
    results = {}
    if agent is not None:
        agent.reset()
        fly = fly_prob_fn(agent)
        if fly_wrapper is not None:
            fly = fly_wrapper(fly)
        results["fly_honest"] = run_episode(
            env_factory(), fly, mask_preset="honest",
            deterministic=deterministic, seed=seed,
        )
        if include_legacy:
            agent.reset()
            results["fly_legacy"] = run_episode(
                env_factory(), fly_prob_fn(agent), mask_preset="live",
                deterministic=deterministic, seed=seed,
            )
    if include_baselines:
        for name in ("buy_hold", "momentum", "trend200"):
            results[name] = run_episode(
                env_factory(), baseline_prob_fn(name), mask_preset="honest",
                deterministic=True, seed=seed,
            )
        results["random"] = run_episode(
            env_factory(), baseline_prob_fn("random"), mask_preset="honest",
            deterministic=False, seed=seed,
        )
    return results

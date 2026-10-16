"""
trainer.py
==========
Reusable training roll-out for the FlyBrain agent, extracted so that the CLI
trainer, the API's ``/train_batch`` endpoint and future experiments all share
one implementation instead of copy-pasting it.

The fly brain is a *fixed* feature extractor; only the small read-out
(feature encoder + actor + critic) is trained, online, from experience
replayed out of a bounded buffer.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from strategy import apply_preset, sample_action


class RewardNormalizer:
    """Welford running mean/variance with safe std and clipping."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def normalize(self, x: float) -> float:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count
        self.var = max(self.var, 1e-6)
        std = math.sqrt(self.var) + 1e-4
        return float(np.clip((x - self.mean) / std, -5.0, 5.0))


def shaped_reward(env_reward: float, portfolio_history, window: int = 20,
                  sharpe_cap: float = 3.0, sharpe_blend: float = 0.4) -> float:
    """Blend the env reward with a rolling Sharpe term.

    ``sharpe_blend=0.0`` uses the environment reward as-is. That is preferred
    for the benchmark-relative (``excess``) objective, where the env already
    encodes "beat buy & hold" and adding a hand-rolled Sharpe term only muddies
    the credit assignment.
    """
    if sharpe_blend <= 0.0:
        return float(env_reward)
    sharpe = 0.0
    if len(portfolio_history) >= window:
        arr = np.asarray(portfolio_history[-window:], dtype=np.float64)
        rets = np.diff(arr) / (arr[:-1] + 1e-9)
        if rets.size and rets.std() > 1e-12:
            sharpe = float(np.clip(rets.mean() / (rets.std() + 1e-9), -sharpe_cap, sharpe_cap))
    return (1.0 - sharpe_blend) * float(env_reward) + sharpe_blend * sharpe


def train_episode(
    agent,
    env,
    *,
    mask_preset: str = "train",
    epsilon: float = 0.30,
    train_every: int = 4,
    batch_size: int = 64,
    min_buffer: int = 16,
    normalize_reward: bool = True,
    sharpe_blend: float = 0.4,
    max_steps: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Run one training episode, updating the agent online. Returns metrics."""
    rng = np.random.default_rng(seed)
    norm = RewardNormalizer()
    obs, _ = env.reset(seed=seed)
    history = [float(env.initial_cash)]
    losses = []
    previous_shaped = 0.0
    trades = 0
    step = 0
    done = False

    def _price():
        idx = min(env.current_step + env.window_size - 1, len(env.data) - 1)
        return float(env.data.iloc[idx]["Close"])

    limit = max_steps if max_steps is not None else env.n_steps
    while not done and step < limit:
        _, raw_probs, _, _ = agent.select_action(obs, reward=previous_shaped)
        probs = apply_preset(
            raw_probs, mask_preset,
            shares=env.shares, cash=env.cash,
            price=_price(), entry_price=getattr(env, "entry_price", 0.0),
        )
        action = sample_action(probs, epsilon=epsilon, rng=rng)
        if action != 0:
            trades += 1

        next_obs, reward, done, truncated, info = env.step(action)
        pv = float(info["portfolio_value"])
        history.append(pv)

        sr = shaped_reward(reward, history, sharpe_blend=sharpe_blend)
        previous_shaped = sr
        r = norm.normalize(sr) if normalize_reward else sr
        agent.record_experience(obs, action, r, next_obs, bool(done))

        if step % train_every == 0 and len(agent.replay_buffer) >= min_buffer:
            try:
                losses.append(agent.train_online_step(batch_size=min(batch_size, len(agent.replay_buffer))))
            except RuntimeError as e:
                # Autograd in-place safety: drop the buffer and keep going.
                print(f"  [trainer] skipped online step ({e}); clearing buffer")
                agent.replay_buffer.clear()
                try:
                    agent.optimizer.state.clear()
                except Exception:
                    pass

        obs = next_obs
        step += 1
        if done or truncated:
            break

    return {
        "steps": step,
        "trades": trades,
        "portfolio_history": history,
        "final_value": history[-1],
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }

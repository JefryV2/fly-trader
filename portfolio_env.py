"""
portfolio_env.py
================
Long-only, multi-asset portfolio environment.

Scope expansion: instead of trading ONE ticker, the fly allocates capital across
a whole cross-section simultaneously.

Allocation rule (the important design choice)
---------------------------------------------
The *same* single-asset policy is applied to every asset each step. Each asset's
7-way action is read as a **conviction level** and the convictions are turned
into weights by a **softmax over a fixed budget that includes cash**::

    d_a    = DESIRES[action_a]                     # conviction in [-1, 1]
    w_a    = exp(beta * d_a) / (exp(0) + sum_b exp(beta * d_b))
    cash   = 1 / (exp(0) + sum_b exp(beta * d_b))

Why a softmax and not independent target weights? With independent targets the
reward ``w_a * ret_a`` is monotone in ``w_a``, so every asset's head is pushed to
"all-in"; after renormalisation that is exactly *equal weight = buy & hold*, i.e.
a degenerate optimum. The softmax puts every asset in competition for one fixed
budget, so the gradient is proportional to ``(ret_a - ret_portfolio)``: the fly
is rewarded for *tilting into the relatively strongest names* and for *raising
cash when the whole cross-section falls* (the cash slot earns 0). That is the
multi-asset generalisation of the trend/momentum edge we measured single-asset.

Breadth gate
------------
The softmax is optionally *gated* to a subset of the assets (see
``alloc_gate``). Gating moves the cross-sectional *selection* out of the shared
asset-local policy -- which cannot express "I am in the top-k" -- and into the
env, so the policy only has to rank. With ``alloc_gate="positive"`` the breadth
k is itself a policy output: exactly the assets the policy flags with a positive
conviction compete for the budget.

Reward (per asset, decomposed so the shared policy gets a clean gradient)::

    r_a = w_a * ret_a - w_a_is_cost - turnover_penalty * |w_a - w_a_prev|

``sum_a r_a`` is the net portfolio return. Per-asset observations carry the
asset's *portfolio* weight, cash weight, unrealised P&L and trade recency, so the
shared policy can condition on the book it already holds.

Evaluation is portfolio-level: total return, max drawdown, annualised Sharpe,
turnover -- compared against equal-weight buy & hold and momentum rotation.
"""
from __future__ import annotations

import numpy as np

from config import DATA_DIR, HOLD
from env import FlyTradeEnv

# Conviction attached to each of the 7 actions:
#   [Hold, Buy25, Buy50, Buy100, Sell25, Sell50, Sell100]
DESIRES = np.array([0.0, 0.25, 0.6, 1.0, -0.25, -0.6, -1.0], dtype=np.float64)


class PortfolioEnv:
    """Multi-asset long-only portfolio. Not a gym.Env: the action is a vector
    (one per asset), so it is driven by a shared per-asset policy instead."""

    def __init__(self, tickers, start_date=None, end_date=None, window_size: int = 30,
                 use_xsection: bool = False, transaction_cost: float = 0.001,
                 turnover_penalty: float = 0.002, indicator_norm: str = "per_feature",
                 beta: float = 2.0, max_weight: float = 1.0,
                 use_portfolio_obs: bool = True, rebalance_every: int = 1,
                 alloc_topk: int = None, alloc_gate: str = None,
                 alloc_min_k: int = 0, alloc_max_k: int = None,
                 action_mask: bool = True):
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.envs = [
            FlyTradeEnv(DATA_DIR / f"{t}_daily.csv", window_size=window_size,
                        start_date=start_date, end_date=end_date,
                        indicator_norm=indicator_norm, reward_mode="logret",
                        use_xsection=use_xsection)
            for t in self.tickers
        ]
        self.window_size = window_size

        # Align to the dates every asset actually trades (non-contiguous
        # calendars would break the windowed features, so require equality).
        common = self.envs[0].data.index
        for e in self.envs[1:]:
            common = common.intersection(e.data.index)
        common = common.sort_values()
        for t, e in zip(self.tickers, self.envs):
            restricted = e.data.index[e.data.index.isin(common)]
            if not restricted.equals(common):
                raise SystemExit(
                    f"{t} calendar does not match the universe; drop it or use a "
                    f"ticker with the same trading days")
        self.dates = common
        self.n_steps = len(common) - window_size
        if self.n_steps < 2:
            raise SystemExit("Not enough common dates for the portfolio window")
        self.obs_dim = int(self.envs[0].observation_space.shape[0])
        self.observation_space = self.envs[0].observation_space

        # Precompute asset-local features once: they depend only on the asset's
        # own history, not on portfolio state. Without this the pandas work per
        # (asset, step) makes multi-asset evaluation and training very slow.
        self._prices = [e.data["Close"].values.astype(np.float64) for e in self.envs]
        self._market_cache = [
            np.stack([e._market_observation(i) for i in range(self.n_steps)]).astype(np.float32)
            for e in self.envs
        ]
        self._regime_cache = [
            np.stack([e._regime_observation(i) for i in range(self.n_steps)]).astype(np.float32)
            for e in self.envs
        ]

        self.transaction_cost = transaction_cost
        self.turnover_penalty = turnover_penalty
        self.beta = float(beta)
        self.max_weight = max_weight
        # The allocation is an *absolute* conviction (weight = softmax(beta*d)),
        # so the policy does not need to observe its own position. Feeding the
        # current weight back in creates a limit cycle ("I hold a lot -> sell ->
        # now I hold little -> buy") that burns transaction costs. Set False to
        # zero the portfolio block -- must match between training and evaluation.
        self.use_portfolio_obs = bool(use_portfolio_obs)
        # Rebalance horizon in bars. Daily rebalancing gives the policy an
        # immediate reward driven by *next-day* returns, whose conditional edge
        # is ~0 -- so the gradient is noise and the greedy policy wanders. The
        # cross-sectional momentum edge only appears over weeks/months, so the
        # reward must be measured over that horizon (e.g. 21 bars = monthly).
        self.rebalance_every = max(1, int(rebalance_every))
        # Optional breadth gate on the softmax allocation. With asset-local
        # features a shared policy can only express "my own momentum is above an
        # absolute threshold", never "I am in the top-k of the cross-section" --
        # so the breadth it holds on an unseen universe is an accident of where
        # that universe's momentum levels fall. Gating the softmax moves the
        # cross-sectional selection *into the env*: the policy only has to rank
        # assets, which a monotone momentum map does.
        #
        #   "none"     : every asset competes (the plain softmax).
        #   "topk"     : only the k most-convicted assets compete. k is a fixed
        #                hyperparameter.
        #   "positive" : only assets the policy *itself* flags with positive
        #                conviction compete (plus cash). Here **k is a policy
        #                output**: breadth is the number of assets the policy
        #                gives one of the three buy-actions, so the policy can
        #                widen and narrow the book step by step instead of
        #                holding a hard-coded k. An all-negative book means all
        #                cash, i.e. a learned full de-risk.
        self.alloc_topk = None if alloc_topk is None else max(1, int(alloc_topk))
        if alloc_gate is None:
            alloc_gate = "topk" if self.alloc_topk is not None else "none"
        alloc_gate = str(alloc_gate).lower()
        if alloc_gate not in ("none", "topk", "positive"):
            raise ValueError(f"alloc_gate must be one of none|topk|positive, "
                             f"got {alloc_gate!r}")
        if alloc_gate == "topk" and self.alloc_topk is None:
            raise ValueError("alloc_gate='topk' requires alloc_topk")
        self.alloc_gate = alloc_gate
        # Optional bounds on the gated breadth. With the "positive" gate the
        # policy decides k, but its conviction scale need not generalise to an
        # unseen cross-section: on a new universe the learned threshold can flag
        # far fewer (or more) assets than the teacher does. A floor keeps the
        # book invested (``max(min_k, #positive)``) and a cap stops it
        # over-concentrating, while the policy still owns everything in between.
        self.alloc_min_k = max(0, int(alloc_min_k))
        self.alloc_max_k = None if alloc_max_k is None else max(1, int(alloc_max_k))
        if self.alloc_max_k is not None and self.alloc_min_k > self.alloc_max_k:
            raise ValueError("alloc_min_k must be <= alloc_max_k")
        self.gated = (self.alloc_gate != "none" or self.alloc_min_k > 0
                      or self.alloc_max_k is not None)
        # Whether the trainer should suppress "buy" actions on fully-invested
        # assets and "sell" actions on empty ones. That mask belongs to
        # directional actions; with conviction-style actions (and especially with
        # any breadth gate) it is wrong, so it is off by default there.
        self.action_mask = bool(action_mask) and not self.gated
        self.reset()

    # ── helpers ────────────────────────────────────────────────────────────
    def _price(self, a: int, bar: int) -> float:
        p = self._prices[a]
        return float(p[min(max(bar, 0), len(p) - 1)])

    def _bar(self, i: int) -> int:
        """Index of the *current* bar (close) for step i."""
        return min(i + self.window_size - 1, len(self.dates) - 1)

    def _obs_for(self, a: int, i: int) -> np.ndarray:
        if 0 <= i < self.n_steps:
            market = self._market_cache[a][i]
            regime = self._regime_cache[a][i]
        else:                                   # safety fallback
            e = self.envs[a]
            market = e._market_observation(i)
            regime = e._regime_observation(i)
        w = float(self.weights[a])
        entry = self.entry[a]
        price = self._price(a, self._bar(i))
        unreal = float(np.clip(price / entry - 1.0, -1.0, 1.0)) if entry else 0.0
        recency = float(np.clip((i - self.last_trade_step[a]) / 30.0, 0.0, 1.0))
        if self.use_portfolio_obs:
            port = np.array([w, float(self.cash), unreal, recency], dtype=np.float32)
        else:
            # Break the position-feedback loop (see __init__).
            port = np.zeros(4, dtype=np.float32)
        return np.concatenate([market, port, np.asarray(regime, dtype=np.float32)])

    def observe(self) -> np.ndarray:
        return np.stack([self._obs_for(a, self.step_i) for a in range(self.n_assets)])

    def weights_from_actions(self, actions) -> np.ndarray:
        """Softmax conviction allocation over (assets + cash).

        The breadth *gate* decides which assets may compete for the budget::

          * ``none``     -- all of them;
          * ``topk``     -- the ``alloc_topk`` most-convicted ones;
          * ``positive`` -- the ones the policy flagged with a positive
                            conviction (so **k is a policy output**).

        The env then supplies the cross-sectional selection, so the policy only
        needs to *rank* assets rather than learn an absolute threshold.
        """
        d = DESIRES[np.asarray(actions, dtype=int)]
        logits = np.concatenate([self.beta * d, [0.0]])          # last = cash
        if self.gated:
            n = self.n_assets
            asset_logits = logits[:n]
            if self.alloc_gate == "positive":
                chosen = np.flatnonzero(d > 0.0)
            elif self.alloc_gate == "topk":
                chosen = np.argsort(-asset_logits)[:min(self.alloc_topk, n)]
            else:
                chosen = np.arange(n)
            # Floor: never hold fewer than alloc_min_k names, whatever the
            # policy's absolute conviction looks like on an unseen universe.
            if self.alloc_min_k > chosen.size:
                order = np.argsort(-asset_logits)
                chosen = np.unique(np.concatenate([chosen, order[:self.alloc_min_k]]))
            # Cap: never hold more than alloc_max_k.
            if self.alloc_max_k is not None and chosen.size > self.alloc_max_k:
                chosen = chosen[np.argsort(-asset_logits[chosen])[:self.alloc_max_k]]
            mask = np.zeros(n + 1, dtype=bool)
            mask[chosen] = True
            mask[-1] = True                                      # cash always stays
            logits = np.where(mask, logits, -np.inf)
        logits -= logits.max()
        e = np.exp(logits)
        w = e / e.sum()
        return np.clip(w[:-1], 0.0, self.max_weight)

    def reset(self) -> np.ndarray:
        self.step_i = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float64)  # all cash
        self.cash = 1.0
        self.entry = [None] * self.n_assets
        self.last_trade_step = [-10] * self.n_assets
        self.value = 1.0
        self.history = [1.0]
        self.breadth = []           # names held (weight > 0), one entry per decision
        self.turnover_total = 0.0
        self.n_trades = 0
        return self.observe()

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, actions):
        """actions: length-N sequence of discrete actions.

        Returns (next_obs, rewards(N,), done, info). ``rewards`` are the
        per-asset *decomposed* contributions so the shared policy gets a clean
        per-asset gradient; they sum to the net portfolio return.
        """
        targets = self.weights_from_actions(actions)
        return self._advance(targets)

    def _advance(self, targets: np.ndarray):
        bar = self._bar(self.step_i)
        next_step = min(self.step_i + self.rebalance_every, self.n_steps)
        next_bar = self._bar(next_step)
        targets = np.clip(np.asarray(targets, dtype=np.float64), 0.0, self.max_weight)

        rets = np.zeros(self.n_assets, dtype=np.float64)
        for a in range(self.n_assets):
            p0, p1 = self._price(a, bar), self._price(a, next_bar)
            rets[a] = (p1 / (p0 + 1e-12) - 1.0) if p0 > 0 else 0.0

        dw = targets - self.weights
        turnover = np.abs(dw)
        cost = turnover * self.transaction_cost
        rewards = targets * rets - cost - self.turnover_penalty * turnover

        self.weights = targets
        self.cash = float(max(0.0, 1.0 - targets.sum()))
        for a in range(self.n_assets):
            if abs(dw[a]) > 1e-6:
                if self.weights[a] > 1e-6 and (self.entry[a] is None
                                               or self.entry[a] <= 0):
                    self.entry[a] = self._price(a, next_bar)
                self.last_trade_step[a] = self.step_i
                self.n_trades += 1
            if self.weights[a] <= 1e-6:
                self.entry[a] = None
        self.turnover_total += float(turnover.sum())
        self.breadth.append(int((self.weights > 1e-6).sum()))
        port_ret = float(targets @ rets - cost.sum())
        self.value *= (1.0 + port_ret)
        self.history.append(self.value)
        self.step_i = next_step
        done = self.step_i >= self.n_steps
        obs = (self.observe() if not done
               else np.zeros((self.n_assets, self.obs_dim), dtype=np.float32))
        info = {
            "value": self.value,
            "weights": self.weights.copy(),
            "cash": self.cash,
            "turnover": float(turnover.sum()),
            "portfolio_return": port_ret,
        }
        return obs, rewards, done, info

    # ── metrics ────────────────────────────────────────────────────────────
    def metrics(self) -> dict:
        h = np.asarray(self.history, dtype=np.float64)
        total = (h[-1] - 1.0) * 100.0
        peak = np.maximum.accumulate(h)
        dd = (peak - h) / (peak + 1e-12)
        rets = np.diff(h) / (h[:-1] + 1e-12)
        sharpe = 0.0
        if rets.size > 2 and rets.std() > 1e-12:
            # Annualise by the *decision* horizon: a monthly env has ~12
            # decisions a year, not 252.
            periods = 252.0 / max(1, self.rebalance_every)
            sharpe = float(rets.mean() / rets.std() * np.sqrt(periods))
        return {
            "total_return_pct": float(total),
            "max_drawdown_pct": float(dd.max() * 100.0),
            "sharpe": sharpe,
            "turnover": float(self.turnover_total),
            "n_trades": int(self.n_trades),
            # Breadth actually held (weight > 0). With the "positive" gate this
            # is the k the *policy* chose, so reporting it is how we show the
            # gate is adaptive rather than hard-coded.
            "avg_positions": float(np.mean(self.breadth)) if self.breadth else 0.0,
            "min_positions": int(min(self.breadth)) if self.breadth else 0,
            "max_positions": int(max(self.breadth)) if self.breadth else 0,
        }


# ── baselines ──────────────────────────────────────────────────────────────
def buy_hold_weights(env: PortfolioEnv, i: int = 0) -> np.ndarray:
    return np.full(env.n_assets, 1.0 / env.n_assets)


def trend200_weights(env: PortfolioEnv, i: int) -> np.ndarray:
    """Equal-weight the assets trading above their 200-day SMA, else cash."""
    above = [a for a, e in enumerate(env.envs)
             if (r := e._regime_observation(i)).size and r[0] > 0]
    w = np.zeros(env.n_assets)
    if above:
        w[above] = 1.0 / len(above)
    return w


def momentum_weights(env: PortfolioEnv, i: int, lookback: int = 120,
                     top_k: int = 3) -> np.ndarray:
    """Equal-weight the top-k assets by trailing ``lookback``-day return."""
    bar = env._bar(i)
    scores = np.array([env._price(a, bar) / (env._price(a, max(0, bar - lookback)) + 1e-12) - 1.0
                       for a in range(env.n_assets)])
    k = min(top_k, env.n_assets)
    w = np.zeros(env.n_assets)
    w[np.argsort(scores)[-k:]] = 1.0 / k
    return w


def run_static_weights(env: PortfolioEnv, weight_fn, rebalance_every: int = 1) -> dict:
    """Run a fixed weight rule (no learning). Used for the baselines.

    ``rebalance_every`` counts *decisions*: 1 recomputes the rule every time the
    env advances (which is one rebalance horizon)."""
    env.reset()
    w = weight_fn(env, 0)
    done, d = False, 0
    while not done:
        if d % max(1, rebalance_every) == 0:
            w = weight_fn(env, env.step_i)
        _, _, done, _ = env._advance(w)
        d += 1
    return env.metrics()

"""
config.py
=========
Single source of truth for constants that were previously hardcoded and
duplicated across env.py, fly_brain.py, web-app/server.py and the scripts.

Nothing here changes model behaviour on its own; it only centralises values so
training, evaluation and the API agree.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "web-app" / "checkpoints"

# ── Action space (7 discrete actions) ──────────────────────────────────────
ACTION_LABELS = [
    "Hold",
    "Buy 25%", "Buy 50%", "Buy 100%",
    "Sell 25%", "Sell 50%", "Sell 100%",
]
N_ACTIONS = len(ACTION_LABELS)
HOLD, BUY_25, BUY_50, BUY_100, SELL_25, SELL_50, SELL_100 = range(N_ACTIONS)
BUY_ACTIONS = (BUY_25, BUY_50, BUY_100)
SELL_ACTIONS = (SELL_25, SELL_50, SELL_100)

# ── Observation layout (must match env.FlyTradeEnv) ────────────────────────
WINDOW_SIZE = 30
N_OHLCV_FEATURES = WINDOW_SIZE * 5          # 150
N_INDICATOR_FEATURES = WINDOW_SIZE * 8      # 240
MARKET_OBS_DIM = N_OHLCV_FEATURES + N_INDICATOR_FEATURES  # 390
# The observation also carries the agent's *own portfolio state* so the policy
# can condition on whether it is already long/flat. Without this it cannot tell
# the difference between "about to open a position" and "already fully invested",
# which is why it could never commit to a full position.
#   [exposure_frac, cash_frac, unrealised_pnl_frac, steps_since_trade/30]
PORTFOLIO_OBS_DIM = 4
OBS_DIM = MARKET_OBS_DIM + PORTFOLIO_OBS_DIM  # 394
# Explicit slice of the *current-step* indicator vector inside the observation
# (row 29 of the 30x8 indicator window). The spiking encoder and the sensory
# summary must use this rather than [-8:], because the portfolio block now
# occupies the tail of the observation.
CUR_INDICATOR_SLICE = slice(MARKET_OBS_DIM - 8, MARKET_OBS_DIM)

# ── Anatomical partitions as fractions of N_NEURONS ────────────────────────
# AL / KC / CX / DAN / MBON  (128 / 640 / 128 / 64 / 64 out of 1024)
REGION_RATIOS = {
    "AL": 128 / 1024,
    "KC": 640 / 1024,
    "CX": 128 / 1024,
    "DAN": 64 / 1024,
    "MBON": 64 / 1024,
}
REGION_ORDER = ["AL", "KC", "CX", "DAN", "MBON"]

# ── Default train / validation / test split (temporal, no leakage) ─────────
# NOTE: the on-disk data files do NOT share a date range:
#   * AAPL_daily.csv         2022-01-03 .. 2024-12-31 (~753 rows)
#   * TSLA/NVDA/MSFT_daily   2015-01-02 .. 2025-12-30 (~2765 rows)
# Training/evaluating on the FULL file therefore reports ~11-year buy-and-hold
# returns for TSLA/NVDA/MSFT (e.g. TSLA ~+3000%, NVDA ~+38000%) while AAPL is
# only a 3-year window — the two are not comparable.
# These splits give every ticker the same held-out TEST window (2024) and the
# same VAL window (2023 H2); TRAIN uses everything before the val window that
# the file actually contains.
SPLIT_PLANS = {
    "default": {
        "train": ("2015-01-01", "2023-06-30"),
        "val":   ("2023-07-01", "2023-12-31"),
        "test":  ("2024-01-01", "2024-12-31"),
    },
    # Honest bear-regime plan: train/validate before the 2022 drawdown and test
    # on it. Answers "can the fly preserve capital when buy & hold cannot?".
    # Only the long-history tickers (TSLA/NVDA/MSFT) have pre-2022 data.
    "bear2022": {
        "train": ("2015-01-01", "2020-12-31"),
        "val":   ("2021-01-01", "2021-12-31"),
        "test":  ("2022-01-01", "2022-12-31"),
    },
    # Full-cycle plan: the test slice spans a choppy half-year, a bear market
    # (2022) *and* a bull market (2023-2024). That is the only honest way to
    # judge a strategy whose whole purpose is to survive drawdowns: evaluated
    # one year at a time, trend/vol timing looks terrible in the bull year, but
    # over a full cycle it wins, because avoiding a -50% drawdown compounds.
    # AAPL lacks pre-2022 data, so this plan uses TSLA/NVDA/MSFT only.
    "cycle": {
        "train": ("2015-01-01", "2019-12-31"),
        "val":   ("2020-01-01", "2021-06-30"),
        "test":  ("2021-07-01", "2024-12-31"),
    },
}
SPLITS = SPLIT_PLANS["default"]
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]

# Defaults for the API (server.py). Training must not see the test window.
SERVER_TRAIN_SPLIT = "train"
SERVER_EVAL_SPLIT = "test"

# ── Action-masking presets ─────────────────────────────────────────────────
# These preserve the exact per-site behaviour that was previously copy-pasted
# into server.py (see _run_step / train_batch / eval_episode).  Pass the preset
# to strategy.apply_action_mask(), or set heuristics=False for a clean,
# policy-only evaluation.
CASH_FLOOR = 200.0

MASK_PRESETS = {
    # server._run_step  (live stepping)
    "live": dict(cash_floor=CASH_FLOOR, hold_cap=0.20, buy_floor=0.22, pnl_sell_boost=True),
    # server.train_batch (training rollouts)
    "train": dict(cash_floor=CASH_FLOOR, hold_cap=0.20, buy_floor=0.22,
                  min_sell_floor=0.07, min_sell_trigger=0.20),
    # server.eval_episode (frozen policy evaluation)
    "eval": dict(cash_floor=CASH_FLOOR, hold_cap=0.45, hold_redistribute=False,
                 buy_floor=0.10, buy_floor_trigger=0.20,
                 min_sell_floor=0.05, min_sell_trigger=0.15),
    # policy-only: only zero out invalid actions, no heuristic re-weighting
    "honest": dict(cash_floor=CASH_FLOOR),
}

# ── Default RL hyperparameters (mirror FlyBrainAgent defaults) ─────────────
RL_DEFAULTS = dict(
    gamma=0.99,
    entropy_coef=0.08,
    value_coef=0.5,
    max_grad_norm=1.0,
    lr=3e-4,
    weight_decay=1e-4,
    max_buffer_size=4000,
)

# ── Transaction / reward defaults ──────────────────────────────────────────
TRANSACTION_COST = 0.001
INITIAL_CASH = 1e5

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
OBS_DIM = N_OHLCV_FEATURES + N_INDICATOR_FEATURES  # 390

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
# Data on disk currently covers 2022-01-03 .. 2024-12-31.
SPLITS = {
    "train": ("2022-01-01", "2023-06-30"),
    "val":   ("2023-07-01", "2023-12-31"),
    "test":  ("2024-01-01", "2024-12-31"),
}
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]

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

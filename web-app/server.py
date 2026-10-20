"""
web-app/server.py  –  FlyTrader FastAPI backend (v0.4)
=======================================================
• Serves the static front-end files at http://127.0.0.1:8000/
• MaleCNS 1024-LIF Spiking Connectome + Dopaminergic Reinforcement Learning
• REST endpoints:
    GET  /ping
    GET  /reset
    POST /step            (with autonomous Fly Brain deliberation or manual action)
    POST /train_batch     (fast multi-episode policy optimization)
    GET  /brain_status    (real-time neurochemical & learning metrics)
    GET  /info
    GET  /learning_history
    GET  /brain_atlas
"""

import sys, os, math, datetime as _dt
from pathlib import Path
import struct
import json
import threading
import time
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from config import SPLITS, SERVER_EVAL_SPLIT, SERVER_TRAIN_SPLIT
from env import FlyTradeEnv
from fly_brain import FlyBrainAgent
from strategy import apply_preset, baseline_prob_fn, baseline_probs, compare_on_env, run_episode

# ── Brain Atlas Data Loader ────────────────────────────────────────────────
def _load_brain_atlas():
    """Load FlyWire MaleCNS v1.0 brain atlas data from binary files."""
    atlas_dir = Path(__file__).parent.parent / "fly_connectome" / "public" / "data" / "brain-atlas"
    positions_path = atlas_dir / "positions.bin"
    ids_path = atlas_dir / "ids.bin"
    groups_path = atlas_dir / "groups.bin"
    
    try:
        with open(positions_path, 'rb') as f:
            positions_data = f.read()
        with open(ids_path, 'rb') as f:
            ids_data = f.read()
        with open(groups_path, 'rb') as f:
            groups_data = f.read()
        
        # Positions: float32 x, y, z for each soma (140024 neurons)
        n_positions = len(positions_data) // 4  # 4 bytes per float32
        positions = struct.unpack(f'{n_positions}f', positions_data)
        positions = [(positions[i], positions[i+1], positions[i+2]) for i in range(0, len(positions), 3)]
        
        # IDs: int32 for each neuron
        n_ids = len(ids_data) // 4
        neuron_ids = struct.unpack(f'{n_ids}i', ids_data)
        
        # Groups: int32 for each neuron's group classification
        n_groups = len(groups_data) // 4
        neuron_groups = struct.unpack(f'{n_groups}i', groups_data)
        
        # Build group lookup (map group ID to color/name)
        # From manifest.json groups: optic=0, central=1, descending=2, vnc=3, other=4
        group_info = {
            0: {"name": "optic", "color": "#81b5c8", "count": 90807},
            1: {"name": "central", "color": "#cfcacf", "count": 32170},
            2: {"name": "descending", "color": "#dfb672", "count": 1312},
            3: {"name": "vnc", "color": "#af9bc3", "count": 15362},
            4: {"name": "other", "color": "#6c7777", "count": 373},
        }
        
        # Count neurons per group
        group_counts = {gid: 0 for gid in group_info}
        for g in neuron_groups:
            if g in group_counts:
                group_counts[g] += 1
        
        return {
            "positions": positions,
            "neuron_ids": neuron_ids,
            "neuron_groups": neuron_groups,
            "group_info": group_info,
            "group_counts": group_counts,
            "total_neurons": len(neuron_ids),
        }
    except Exception as e:
        print(f"Warning: Could not load brain atlas: {e}")
        return None

# Load atlas data at startup
_brain_atlas_cache = None

def get_brain_atlas():
    global _brain_atlas_cache
    if _brain_atlas_cache is None:
        _brain_atlas_cache = _load_brain_atlas()
    return _brain_atlas_cache

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="FlyTrader API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "AAPL_daily.csv"
DATA_DIR = DATA_PATH.parent
if not DATA_PATH.is_file():
    raise RuntimeError(f"Missing {DATA_PATH}. Run: python data_fetch.py AAPL")

# Every ticker the dashboard can trade. This is the raw data/ inventory; a
# multi-stock session additionally needs one *shared* trading calendar, which is
# validated when the session is built.
AVAILABLE_TICKERS = sorted(
    p.name[: -len("_daily.csv")] for p in DATA_DIR.glob("*_daily.csv")
)

# Live session: what is being traded, with what starting budget, and whether the
# fly chooses a continuous position size instead of the 25/50/100 buckets.
SESSION = {
    "tickers": ["AAPL"],
    "budget": 100000.0,
    "dynamic_sizing": os.environ.get(
        "FLYBRAIN_DYNAMIC_SIZING", "true").lower() == "true",
}

# Every live episode covers the same three years, whatever the ticker or basket,
# so switching universe never silently changes the test period (the raw CSVs
# range from 2022-2024 for AAPL to 2015-2025 for most others).
LIVE_WINDOW_START = "2022-01-03"
LIVE_WINDOW_END = "2024-12-31"

# Objective used by the server's environment(s). "excess" (agent return minus a
# passive buy & hold benchmark) matches how the shipped checkpoint is trained.
SERVER_REWARD_MODE = os.environ.get("FLYBRAIN_REWARD_MODE", "excess")
# Indicator normalisation must match whatever the served checkpoint was trained
# with, or the observations mean something different to the network.
SERVER_INDICATOR_NORM = os.environ.get("FLYBRAIN_INDICATOR_NORM", "per_feature")
# Pre-specified risk overlay. When on, the served policy is not allowed to hold
# new exposure while price is below its 200-day trend (and an open position is
# closed). This converts an otherwise buy & hold-equivalent policy into
# regime/tail timing, which beats buy & hold over a full cycle at the cost of
# upside in a pure bull run (see README, "Does the fly beat buy & hold?").
# Off by default so the raw learned policy can be evaluated honestly.
SERVER_REGIME_GATE = os.environ.get("FLYBRAIN_REGIME_GATE", "false").lower() == "true"
# Deployed strategy. Default "fly": serve the live learned policy so it trades
# and keeps learning online. "trend200" is the pre-specified overlay that
# maximises raw full-cycle return (see README cross-regime matrix) but it is a
# static rule that sits flat below the 200-day trend — set it explicitly when
# you want max return rather than a learning trader.
SERVER_STRATEGY = os.environ.get("FLYBRAIN_STRATEGY", "fly").lower()
RULE_STRATEGIES = {"trend200", "buy_hold", "momentum"}
# Pre-specified volatility-targeting overlay for the learned policy (see
# strategy.vol_target_prob_fn). The level was selected on the *validation* split
# (0.30) and improves the fly on every held-out window, so it is on by default;
# set FLYBRAIN_VOL_TARGET=0 to serve the raw learned policy.
SERVER_VOL_TARGET = float(os.environ.get("FLYBRAIN_VOL_TARGET", "0.30"))
# Regime block layout appended to each observation by env._compute_regime_block:
# [close/SMA200-1, close/SMA50-1, ann_vol20, vol20/vol60].
REGIME_DIST200_IDX = -4
REGIME_VOL20_IDX = -2
# Portfolio block: [exposure_frac, cash_frac, unrealised_pnl_frac, steps/30].
PORTFOLIO_EXPOSURE_IDX = -8

def _single_env(ticker: str, budget: float = 100000.0):
    """Build a single-asset live env.

    Every ticker goes through the same feature pipeline, so the served brain and
    its observation width transfer unchanged when the universe is switched.
    """
    path = DATA_DIR / f"{ticker.upper()}_daily.csv"
    if not path.is_file():
        raise ValueError(f"Unknown ticker {ticker!r}")
    return FlyTradeEnv(path, window_size=30, enable_sl_tp=False,
                       initial_cash=float(budget),
                       start_date=LIVE_WINDOW_START, end_date=LIVE_WINDOW_END,
                       reward_mode=SERVER_REWARD_MODE,
                       indicator_norm=SERVER_INDICATOR_NORM)


env = _single_env(SESSION["tickers"][0], SESSION["budget"])

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "fly_brain.pt"
BEST_CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "fly_brain_best.pt"
BEST_SCORE_PATH = Path(__file__).parent / "checkpoints" / "fly_brain_best_score.json"
SIMULATION_HISTORY_PATH = Path(__file__).parent / "checkpoints" / "simulation_history.json"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_NEURONS = int(os.environ.get("FLYBRAIN_N_NEURONS", 1024))
USE_CONNECTOME = os.environ.get("FLYBRAIN_USE_CONNECTOME", "true").lower() == "true"
brain = FlyBrainAgent(
    obs_dim=env.observation_space.shape[0],
    use_connectome=USE_CONNECTOME,
    N_NEURONS=N_NEURONS
)

# Online-learning exploration target. The brain default is 1.30 nats out of a
# maximum of ln 7 ~= 1.95, i.e. almost uniform — with that target the entropy
# tuner inside train_online_step drives entropy_coef to its 0.2 cap and the
# policy degenerates into ~2000 random trades per episode (churn, and a much
# worse return). The shipped checkpoint was trained with train.py
# --target-entropy 0.5, so we match it here: online learning then *refines* the
# deployed policy instead of wrecking it. Override with FLYBRAIN_TARGET_ENTROPY.
brain.target_entropy = float(os.environ.get("FLYBRAIN_TARGET_ENTROPY", "0.5"))
# Optional probability floor (see select_action). 0.0 = the honest, decisive
# policy. Raise it (e.g. 0.03) if you want a visibly higher-turnover demo at the
# cost of return — measured on AAPL: floor 0.03 turns ~36 trades into ~365 and
# $140k into $102k. Override with FLYBRAIN_ACTION_FLOOR.
brain.action_floor = float(os.environ.get("FLYBRAIN_ACTION_FLOOR", "0.0"))

chosen_cp = None
for candidate in [BEST_CHECKPOINT_PATH, CHECKPOINT_PATH]:
    if candidate.exists():
        try:
            data = torch.load(candidate, map_location="cpu", weights_only=False)
            if 'simulator_state' in data:
                chosen_cp = (candidate, data)
                break
            elif chosen_cp is None:
                chosen_cp = (candidate, data)
        except Exception as e:
            print(f"Warning loading {candidate}: {e}")

if chosen_cp:
    cp_path_used, cp_data = chosen_cp
    brain.set_policy_state(cp_data)
    print(f"Loaded checkpoint from {cp_path_used.name} (has simulator_state: {'simulator_state' in cp_data})")

# Policy health check: if actor head is saturated (Buy == 0.0), restore smart prior weights
dummy_obs = env._get_observation(0)
_, test_probs, _, _ = brain.select_action(dummy_obs, deterministic=False)
if sum(test_probs[1:4]) < 0.01:
    print("Warning: Actor head was saturated. Applying smart prior weights to restore proactive trading.")
    brain._init_smart_weights()
    try:
        torch.save(brain.get_policy_state(), CHECKPOINT_PATH)
        torch.save(brain.get_policy_state(), BEST_CHECKPOINT_PATH)
    except Exception as e:
        print(f"Warning saving repaired checkpoint: {e}")

obs, _ = env.reset()
_history = [env.initial_cash]
_trades  = []
_total_episodes_trained = 0
_online_learning_enabled = True
_learning_history = []
_simulation_runs = []   # Persistent log of every /train_batch call
_collapse_events = []   # Episodes where the policy's entropy collapsed
if SIMULATION_HISTORY_PATH.exists():
    try:
        with open(SIMULATION_HISTORY_PATH, "r") as f:
            _simulation_runs = json.load(f)
            if _simulation_runs:
                _total_episodes_trained = _simulation_runs[-1].get("total_episodes_after", 0)
    except Exception as e:
        print(f"Warning loading simulation history: {e}")

def _save_simulation_history():
    try:
        with open(SIMULATION_HISTORY_PATH, "w") as f:
            json.dump(_simulation_runs, f, indent=2)
    except Exception as e:
        print(f"Warning saving simulation history: {e}")
_last_shaped_reward = 0.0
_best_score = -999.0  # Track best Sharpe ratio for checkpoint saving
_patience_counter = 0  # Early stopping patience

# Seed the best-Sharpe tracker from disk. Without this it starts at -999, so the
# very first /train_batch episode always counts as a new record and overwrites
# the shipped fly_brain_best.pt with an in-sample (and often worse) policy — i.e.
# clicking "Train" once destroyed the deployed brain. Now training only replaces
# the best checkpoint when it genuinely beats the recorded score.
if BEST_SCORE_PATH.exists():
    try:
        with open(BEST_SCORE_PATH, "r") as _f:
            _best_score = float(json.load(_f).get("best_score", _best_score))
        print(f"Seeded best-Sharpe tracker from {BEST_SCORE_PATH.name}: {_best_score:.4f}")
    except Exception as _e:
        print(f"Warning loading best score: {_e}")
class RewardNormalizer:
    """Welford running mean and variance normalizer with safe standard deviation and clipping."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def normalize(self, x: float) -> float:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        self.var = max(self.var, 1e-6)
        std = math.sqrt(self.var) + 1e-4
        norm = (x - self.mean) / std
        return float(np.clip(norm, -5.0, 5.0))

_reward_norm = RewardNormalizer()

# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _apply_regime_gate(probs, obs, shares):
    """Risk overlay: force flat/exit while price is below its 200-day trend.

    Must be applied *after* the mask preset, whose buy-floor would otherwise
    re-introduce exposure. Mirrors ``strategy.regime_gate_prob_fn`` so live
    serving matches the offline evaluation of the overlay.
    """
    if not SERVER_REGIME_GATE:
        return probs
    try:
        dist200 = float(obs[REGIME_DIST200_IDX])
    except Exception:
        return probs
    if dist200 >= 0.0:
        return probs
    p = np.zeros(7, dtype=np.float64)
    p[6 if shares > 1e-6 else 0] = 1.0   # exit, or stay flat
    return p.astype(probs.dtype)


def _apply_vol_target(probs, obs, shares):
    """Vol-target overlay: size exposure by inverse realised volatility.

    Mirrors ``strategy.vol_target_prob_fn`` so live serving matches the offline
    evaluation that selected the level (validation-selected 0.30). Applied
    *after* the mask preset, whose buy-floor would otherwise re-introduce
    exposure.
    """
    if SERVER_VOL_TARGET <= 0.0:
        return probs
    try:
        vol = float(obs[REGIME_VOL20_IDX])
        expo = float(obs[PORTFOLIO_EXPOSURE_IDX])
    except Exception:
        return probs
    if vol <= 1e-6:
        return probs
    target = min(1.0, SERVER_VOL_TARGET / vol)
    base = np.asarray(probs, dtype=np.float64)
    p = np.zeros(7, dtype=np.float64)
    if target - expo > 0.10:
        p[1:4] = base[1:4]           # need more exposure: allow buys
    elif expo - target > 0.10:
        p[4:7] = base[4:7]           # need less exposure: allow sells
    else:
        p[0] = 1.0                   # inside the band: hold
    if p.sum() <= 0:
        return probs
    return p.astype(probs.dtype)


def _save_checkpoint():
    try:
        torch.save(brain.get_policy_state(), CHECKPOINT_PATH)
    except Exception as e:
        print(f"Warning: failed to save checkpoint: {e}")


def _save_best_checkpoint(score: float):
    """Save best checkpoint based on Sharpe ratio."""
    global _best_score, _patience_counter
    if score > _best_score:
        _best_score = score
        _patience_counter = 0
        try:
            torch.save(brain.get_policy_state(), BEST_CHECKPOINT_PATH)
            with open(BEST_SCORE_PATH, 'w') as f:
                json.dump({'best_score': score, 'total_episodes': _total_episodes_trained}, f)
            print(f"  New best model saved: Sharpe={score:.4f}")
        except Exception as e:
            print(f"Warning: failed to save best checkpoint: {e}")
    else:
        _patience_counter += 1


def _load_best_checkpoint():
    """Load best checkpoint if available."""
    global _best_score
    if BEST_CHECKPOINT_PATH.exists() and BEST_SCORE_PATH.exists():
        try:
            brain.set_policy_state(torch.load(BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=False))
            with open(BEST_SCORE_PATH, 'r') as f:
                score_data = json.load(f)
                _best_score = score_data.get('best_score', -999.0)
            print(f"Loaded best checkpoint (Sharpe={_best_score:.4f})")
        except Exception as e:
            print(f"Warning: failed to load best checkpoint: {e}")

def _compute_sharpe(values):
    if len(values) < 2:
        return 0.0
    arr = np.array([_safe_float(v) for v in values], dtype=np.float64)
    rets = np.diff(arr) / (arr[:-1] + 1e-9)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    mu, std = float(np.mean(rets)), float(np.std(rets))
    if not math.isfinite(mu) or not math.isfinite(std) or std < 1e-9:
        return 0.0
    val = mu / (std + 1e-9) * math.sqrt(252)
    return round(float(val), 4) if math.isfinite(val) else 0.0

# ── Train / validation / test splitting helpers ────────────────────────────
DATA_DIR = DATA_PATH.parent


def _data_path_for(ticker: str) -> Path:
    p = DATA_DIR / f"{ticker}_daily.csv"
    return p if p.is_file() else DATA_PATH


def _split_bounds(split: str):
    if split not in SPLITS:
        raise HTTPException(status_code=400,
                            detail=f"unknown split '{split}'; expected one of {list(SPLITS)}")
    return SPLITS[split]


def _make_env(ticker: str, split: str, *, enable_sl_tp: bool, window_size: int = 30,
              reward_mode: str = None, indicator_norm: str = None) -> FlyTradeEnv:
    start, end = _split_bounds(split)
    return FlyTradeEnv(
        _data_path_for(ticker), window_size=window_size,
        enable_sl_tp=enable_sl_tp, start_date=start, end_date=end,
        reward_mode=reward_mode or SERVER_REWARD_MODE,
        indicator_norm=indicator_norm or SERVER_INDICATOR_NORM,
    )


def _simulator_state_snapshot():
    return (brain.simulator.V.copy(), brain.simulator.last_spikes.copy(),
            getattr(brain.simulator, "_inh", 0.0), brain.simulator.dopamine_level,
            brain.training)


def _restore_simulator_state(snap):
    v, spikes, inh, dopamine, was_training = snap
    brain.simulator.V = v
    brain.simulator.last_spikes = spikes
    if hasattr(brain.simulator, "_inh"):
        brain.simulator._inh = inh
    brain.simulator.dopamine_level = dopamine
    brain.train(was_training)


def _heldout_report(tickers, split: str, *, seed: int = 0) -> dict:
    """Out-of-sample evaluation of the frozen policy vs baselines on ``split``.

    Uses the same masking/selection code path as everything else
    (``strategy.compare_on_env``) and does NOT touch the policy weights or the
    live simulator state.
    """
    snap = _simulator_state_snapshot()
    report = {"split": split, "date_range": list(SPLITS[split]), "per_ticker": {}}
    try:
        brain.eval()
        for t in tickers:
            def env_factory(t=t):
                return _make_env(t, split, enable_sl_tp=False)
            try:
                res = compare_on_env(env_factory, brain, deterministic=True, seed=seed,
                                     include_legacy=False, include_baselines=True)
            except Exception as e:  # pragma: no cover - defensive
                report["per_ticker"][t] = {"error": str(e)}
                continue
            report["per_ticker"][t] = {
                name: {
                    "return_pct": round(m["total_return_pct"], 2),
                    "sharpe": round(m["sharpe"], 4),
                    "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
                    "trades": m["trades"],
                }
                for name, m in res.items()
            }
    finally:
        _restore_simulator_state(snap)

    fly = [v["fly_honest"]["return_pct"] for v in report["per_ticker"].values() if "fly_honest" in v]
    bh = [v["buy_hold"]["return_pct"] for v in report["per_ticker"].values() if "buy_hold" in v]
    report["mean_fly_return_pct"] = round(float(np.mean(fly)), 2) if fly else 0.0
    report["mean_buyhold_return_pct"] = round(float(np.mean(bh)), 2) if bh else 0.0
    report["fly_excess_vs_buyhold_pct"] = round(
        report["mean_fly_return_pct"] - report["mean_buyhold_return_pct"], 2)
    return report


def _get_region_breakdown(spikes_1024, N_NEURONS=1024):
    """Summarizes spiking activity across anatomical Drosophila lobes."""
    al_end = int(round(N_NEURONS * 128 / 1024))
    kc_end = int(round(N_NEURONS * (128 + 640) / 1024))
    cx_end = int(round(N_NEURONS * (128 + 640 + 128) / 1024))
    dan_end = int(round(N_NEURONS * (128 + 640 + 128 + 64) / 1024))
    
    al_rate   = float(np.mean(spikes_1024[0:al_end]))
    kc_rate   = float(np.mean(spikes_1024[al_end:kc_end]))
    cx_rate   = float(np.mean(spikes_1024[kc_end:cx_end]))
    # DAN:  last 64 neurons in the dan block
    # MBON: between cx_end and the start of DAN (dan_end - 64)
    mbon_end  = dan_end - 64
    dan_rate  = float(np.mean(spikes_1024[mbon_end:dan_end])) if mbon_end < dan_end else 0.0
    mbon_rate = float(np.mean(spikes_1024[cx_end:mbon_end]))  if cx_end < mbon_end  else 0.0
    total_spikes = int(np.count_nonzero(spikes_1024))
    return {
        "antennal_lobe": round(al_rate * 100, 1),
        "kenyon_cells": round(kc_rate * 100, 1),
        "central_complex": round(cx_rate * 100, 1),
        "dopamine_dan": round(dan_rate * 100, 1),
        "mbon_output": round(mbon_rate * 100, 1),
        "total_active": total_spikes,
        "spike_rate_pct": round(total_spikes / N_NEURONS * 100, 1),
    }

def _continuous_size(probs, action: int) -> float:
    """Turn the 7-way action distribution into a continuous commitment.

    ``probs`` is the (masked) categorical distribution over
    {hold, buy 25/50/100, sell 25/50/100}. Its *expected* buy magnitude
    ``p1*0.25 + p2*0.50 + p3*1.00`` is a number in [0, 1] -- so a policy that is
    90% sure of a small nibble can commit 3.6% of cash instead of being snapped
    to 25%. Sells use the same expectation over the position. Returns 0.0 for
    hold or an out-of-range action.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.size != 7:
        return 0.0
    if 1 <= action <= 3:
        return float(np.clip(p[1] * 0.25 + p[2] * 0.50 + p[3] * 1.00, 0.0, 1.0))
    if 4 <= action <= 6:
        return float(np.clip(p[4] * 0.25 + p[5] * 0.50 + p[6] * 1.00, 0.0, 1.0))
    return 0.0


# ── Live portfolio session ─────────────────────────────────────────────────
# Two assets or more switches the dashboard to the multi-asset brain: a separate
# checkpoint trained on a cross-section (README, "Adaptive breadth"). It
# rebalances monthly over the same 2022-2024 window as the single-asset demo so
# both modes have a comparable number of decisions.
LIVE_PORTFOLIO_START = LIVE_WINDOW_START
LIVE_PORTFOLIO_END = LIVE_WINDOW_END
ACTION_LABELS = ["Hold", "Buy 25%", "Buy 50%", "Buy 100%",
                 "Sell 25%", "Sell 50%", "Sell 100%"]

_pf_live = {"env": None, "agent": None, "key": None, "history": [], "trades": []}


def _portfolio_live_env(tickers):
    from portfolio_env import PortfolioEnv
    kw = dict(start_date=LIVE_PORTFOLIO_START, end_date=LIVE_PORTFOLIO_END,
              rebalance_every=PORTFOLIO_REBALANCE,
              alloc_min_k=PORTFOLIO_MIN_K, alloc_max_k=PORTFOLIO_MAX_K)
    if PORTFOLIO_GATE == "topk":
        kw["alloc_gate"] = "topk"
        kw["alloc_topk"] = PORTFOLIO_MIN_K
    elif PORTFOLIO_GATE != "none":
        kw["alloc_gate"] = PORTFOLIO_GATE
    return PortfolioEnv(list(tickers), **kw)


def _live_portfolio():
    """The live (env, agent) for the current universe, rebuilt when it changes."""
    global _pf_live
    tickers = [t.upper() for t in SESSION["tickers"]]
    key = tuple(tickers)
    if _pf_live["env"] is None or _pf_live["key"] != key:
        p_env = _portfolio_live_env(tickers)
        p_agent = _get_portfolio_agent(p_env)
        p_env.reset()
        _pf_live = {"env": p_env, "agent": p_agent, "key": key,
                    "history": [1.0], "trades": []}
    return _pf_live["env"], _pf_live["agent"]


def _reset_live_portfolio():
    global _pf_live
    if _pf_live["env"] is not None:
        _pf_live["env"].reset()
        _pf_live["history"] = [1.0]
        _pf_live["trades"] = []


def _run_portfolio_step():
    """One monthly rebalance of the multi-asset fly."""
    global _pf_live
    p_env, p_agent = _live_portfolio()
    if p_env.step_i >= p_env.n_steps:
        p_env.reset()
        _pf_live["history"] = [1.0]
        _pf_live["trades"] = []

    obs = p_env.observe()
    actions, probs, _values = p_agent.select_actions_batch(obs, deterministic=True)
    actions = np.asarray(actions, dtype=int)
    _next, rewards, done, info = p_env.step(actions)

    value = float(info["value"])
    _pf_live["history"].append(value)

    budget = float(SESSION["budget"])
    pv = budget * value
    hist_pv = [budget * h for h in _pf_live["history"]]
    dec_step = len(hist_pv) - 1
    total_decisions = int(np.ceil(p_env.n_steps / max(1, p_env.rebalance_every)))

    weights = [float(w) for w in info["weights"]]
    cash_w = float(info["cash"])
    holdings = []
    for a, t in enumerate(p_env.tickers):
        act = int(actions[a])
        holdings.append({
            "ticker": t,
            "weight": round(weights[a], 4),
            "action": ACTION_LABELS[act],
            "action_idx": act,
            "probs": [round(float(p), 4) for p in probs[a]],
        })
    holdings.sort(key=lambda h: h["weight"], reverse=True)
    held = [h for h in holdings if h["weight"] > 1e-6]

    if held:
        _pf_live["trades"].append({
            "step":      dec_step,
            "action":    " · ".join(f"{h['ticker']} {h['weight']*100:.0f}%" for h in held),
            "price":     None,
            "portfolio": round(pv, 2),
        })

    h = np.asarray(hist_pv, dtype=np.float64)
    total_return = (h[-1] / budget - 1.0) * 100.0 if budget > 0 else 0.0
    peak = np.maximum.accumulate(h)
    drawdown = float(((peak - h) / (peak + 1e-9)).max() * 100.0) if h.size else 0.0
    wins = sum(1 for i in range(1, len(h)) if h[i] > h[i - 1])
    win_rate = wins / max(1, len(h) - 1) * 100.0
    sharpe = _compute_sharpe(hist_pv)

    spikes_1024 = np.asarray(p_agent.simulator.last_spikes, dtype=np.float32).flatten()
    if spikes_1024.size < p_agent.N_NEURONS:
        spikes_1024 = np.pad(spikes_1024, (0, p_agent.N_NEURONS - spikes_1024.size))
    spikes_1024 = spikes_1024[: p_agent.N_NEURONS]
    spikes_list = [int(v > 0) for v in spikes_1024]

    summary = (f"{len(held)} held" if held else "All cash")

    return {
        "step":           dec_step,
        "n_steps":        total_decisions,
        "bar":            int(p_env.step_i),
        "action":         summary,
        "action_idx":     0,
        "action_label":   summary,
        "action_probs":   [0.0] * 7,
        "size_pct":       None,
        "dynamic":        True,
        "mode":           "portfolio",
        "ticker":         None,
        "tickers":        list(p_env.tickers),
        "budget":         budget,
        "holdings":       holdings,
        "cash_weight":    round(cash_w, 4),
        "breadth":        len(held),
        "strategy":       "portfolio-fly",
        "reward":         round(float(info["portfolio_return"]), 6),
        "dopamine":       round(_safe_float(p_agent.simulator.dopamine_level), 4),
        "loss":           0.0,
        "done":           bool(done),
        "portfolio":      round(pv, 2),
        "total_return":   _safe_float(total_return),
        "sharpe":         _safe_float(sharpe),
        "drawdown":       _safe_float(drawdown),
        "win_rate":       _safe_float(win_rate),
        "regions":        _get_region_breakdown(spikes_1024, p_agent.N_NEURONS),
        "voltages":       p_agent.get_region_voltages(),
        "voltages_peak":  p_agent.get_region_peak_voltages(),
        "spikes_sample":  spikes_list[:256],
        "spikes_full":    spikes_list,
        "history":        [round(_safe_float(v), 2) for v in hist_pv[-200:]],
        "trades":         _pf_live["trades"][-20:],
    }


def _dispatch_step(action: int, auto_brain: bool = False,
                   enable_learning: bool = True):
    if len(SESSION["tickers"]) > 1:
        return _run_portfolio_step()
    return _run_step(action, auto_brain=auto_brain, enable_learning=enable_learning)


def _run_step(action: int, auto_brain: bool = False, enable_learning: bool = True):
    global obs, _history, _trades, _last_shaped_reward, _best_score, _patience_counter, _reward_mean, _reward_std, _reward_count

    # Auto-reset episode if completed
    if env.current_step >= env.n_steps:
        _save_checkpoint()
        # Save best model based on Sharpe ratio
        current_sharpe = _compute_sharpe(_history)
        _save_best_checkpoint(current_sharpe)
        
        # Early stopping if patience exceeded
        if _patience_counter > 50:
            print(f"Early stopping: no improvement for {_patience_counter} episodes")
        
        obs, _ = env.reset()
        brain.reset()
        _history = [env.initial_cash]
        _trades.clear()
        _last_shaped_reward = 0.0
        # NB: the reward normaliser is deliberately *not* reset per episode — a
        # consistent reward scale is what keeps the critic stable across
        # episodes (see trainer.py).

    idx = min(env.current_step + env.window_size - 1, len(env.data) - 1)
    price = _safe_float(env.data.iloc[idx]["Close"])

    # If action is -1 or auto_brain, let the 1024-LIF Fly Brain decide autonomously!
    # 7 actions: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 25%, 5=Sell 50%, 6=Sell 100%
    if auto_brain or action < 0 or action > 6:
        # The fly brain always steps so the live visualisation/telemetry stays
        # populated, even when a pre-specified rule is the deployed strategy.
        _, raw_fly_probs, state_value, spikes_1024 = brain.select_action(obs, reward=_last_shaped_reward)

        if SERVER_STRATEGY in RULE_STRATEGIES:
            # Max-return pre-specified rule. Use the same minimal ("honest")
            # mask as the offline evaluation — not the live heuristics — so live
            # serving matches the backtest that produced the headline numbers.
            raw_action_probs = baseline_probs(
                SERVER_STRATEGY, obs,
                shares=env.shares, cash=env.cash, prev_action=0,
            )
            probs_arr = apply_preset(
                raw_action_probs, "honest",
                shares=env.shares, cash=env.cash,
                price=price, entry_price=getattr(env, "entry_price", 0.0),
            ).astype(np.float32)
        else:
            # Serve the learned policy with the SAME minimal ("honest") mask used
            # in the offline backtest, so live behaviour matches the headline
            # numbers. (The "live" preset re-weights the policy with hand-coded
            # heuristics and is not what was evaluated.)
            raw_action_probs = np.asarray(raw_fly_probs, dtype=np.float64)
            probs_arr = apply_preset(
                raw_action_probs, "honest",
                shares=env.shares, cash=env.cash,
                price=price, entry_price=getattr(env, "entry_price", 0.0),
            ).astype(np.float32)
            # Pre-specified vol-targeting overlay (validation-selected level).
            probs_arr = _apply_vol_target(probs_arr, obs, env.shares)
            probs_arr = _apply_regime_gate(probs_arr, obs, env.shares)
        action_probs = probs_arr.tolist()

        # A pre-specified rule is deterministic (its vector is a one-hot), so
        # sample straight from it; the learned policy uses temperature sampling
        # to avoid a static argmax lock.
        if SERVER_STRATEGY in RULE_STRATEGIES or _total_episodes_trained <= 5:
            action = int(np.random.choice(7, p=probs_arr))
        else:
            temp = 0.65
            log_p = np.log(probs_arr + 1e-9) / temp
            exp_p = np.exp(log_p - np.max(log_p))
            sample_p = exp_p / np.sum(exp_p)
            action = int(np.random.choice(7, p=sample_p))

        entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-9))
        max_entropy = np.log(7)
        confidence = float(np.clip(1.0 - (entropy / max_entropy), 0.35, 1.0))
    else:
        # User manual action: still step the connectome to observe neural firing!
        action_probs = [0.0] * 7
        action_probs[action] = 1.0
        spikes_1024 = brain.step_simulation(obs[np.newaxis, :])[0]
        confidence = 1.0  # Manual actions use full position size

    # Ensure 1D array of 1024 elements
    spikes_1024 = np.asarray(spikes_1024).flatten()

    # ── Continuous position sizing ─────────────────────────────────────────
    # The policy is categorical over seven buckets, but its *expected*
    # commitment is continuous: sum(p_buy * size) over the three buy buckets is
    # any value in [0, 1]. Trading that instead of the argmax bucket is what
    # lets the fly buy 3.6% (or sell 11.2%) rather than being snapped to 25%,
    # 50% or 100%. It is the fly's own distribution, not a hand-written rule.
    size_override = None
    size_pct = None
    if SESSION["dynamic_sizing"] and auto_brain and 1 <= action <= 6:
        size_override = _continuous_size(action_probs, action)
        size_pct = round(size_override * 100.0, 2)
        if size_override <= 5e-4:
            # Expected commitment rounds to nothing -- log a hold rather than a
            # zero-share trade.
            size_override = None
            size_pct = None
            action = 0

    # Step the environment with adaptive confidence
    next_obs, reward, done, _, info = env.step(
        action, confidence=confidence, size_override=size_override)

    pv = _safe_float(info.get("portfolio_value", env.initial_cash), env.initial_cash)
    _history.append(pv)

    profit_reward = 0.0
    if len(_history) >= 2:
        prev = _history[-2]
        if prev > 0:
            profit_reward = (pv - prev) / prev

    sharpe_reward = 0.0
    if len(_history) >= 20:
        rets = np.diff(np.array(_history[-20:])) / (np.array(_history[-20:])[:-1] + 1e-9)
        sharpe_reward = float(np.mean(rets) / (np.std(rets) + 1e-9))

    shaped_reward = 0.6 * float(reward) + 0.4 * sharpe_reward
    _last_shaped_reward = shaped_reward  # Store for next step's dopamine signal

    # Online learning step (dopamine-modulated plasticity)
    loss = 0.0
    if enable_learning and _online_learning_enabled:
        normalized_reward = _reward_norm.normalize(shaped_reward)
        brain.record_experience(obs, action, normalized_reward, next_obs, bool(done))
        if len(brain.replay_buffer) >= 16:
            loss = brain.train_online_step(batch_size=min(64, len(brain.replay_buffer)))
        elif len(brain.replay_buffer) >= 8:
            loss = brain.train_online_step(batch_size=8)

    obs = next_obs

    labels = ["Hold", "Buy 25%", "Buy 50%", "Buy 100%", "Sell 25%", "Sell 50%", "Sell 100%"]
    label = labels[action]
    if size_pct is not None and action != 0:
        label = f"{'Buy' if action <= 3 else 'Sell'} {size_pct:.1f}%"
    if action != 0:
        _trades.append({
            "step":      env.current_step,
            "action":    label,
            "price":     round(price, 2),
            "portfolio": round(pv, 2),
        })

    initial      = float(env.initial_cash)
    total_return = round((pv - initial) / initial * 100, 2) if initial > 0 else 0.0
    sharpe       = _compute_sharpe(_history)
    max_val      = max(_history) if _history else pv
    drawdown     = round((max_val - pv) / (max_val + 1e-9) * 100, 2) if max_val > 0 else 0.0
    wins         = sum(1 for i in range(1, len(_history)) if _history[i] > _history[i-1])
    win_rate     = round(wins / max(1, len(_history)-1) * 100, 1)
    clean_reward = round(_safe_float(reward, 0.0), 6)

    # 1024-neuron activity downsampled to 256 for compact wire transmission
    # and full region metrics
    spikes_list = [int(v > 0) for v in spikes_1024]
    regions = _get_region_breakdown(spikes_1024, brain.N_NEURONS)
    voltages = brain.get_region_voltages()

    return {
        "step":           env.current_step,
        "action":         label,
        "action_idx":     action,
        "action_label":   label,
        "size_pct":       size_pct,
        "dynamic":        bool(SESSION["dynamic_sizing"]),
        "mode":           "single",
        "ticker":         SESSION["tickers"][0],
        "tickers":        list(SESSION["tickers"]),
        "budget":         float(env.initial_cash),
        "n_steps":        int(env.n_steps),
        "strategy":       SERVER_STRATEGY,
        "action_probs":   [round(_safe_float(p), 4) for p in action_probs],
        "reward":         clean_reward,
        "dopamine":       round(_safe_float(brain.simulator.dopamine_level), 4),
        "loss":           round(_safe_float(loss), 5),
        "done":           bool(done),
        "portfolio":      round(pv, 2),
        "total_return":   _safe_float(total_return),
        "sharpe":         _safe_float(sharpe),
        "drawdown":       _safe_float(drawdown),
        "win_rate":       _safe_float(win_rate),
        "regions":        regions,
        "voltages":       voltages,
        "voltages_peak":  brain.get_region_peak_voltages(),
        "spikes_sample":  spikes_list[:256],      # Top 256 for 3D visualizer
        "spikes_full":    spikes_list,            # Full 1024 binary spikes
        "history":        [round(_safe_float(v), 2) for v in _history[-200:]],
        "trades":         _trades[-20:],
    }

# ── API Endpoints ──────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "FlyTrader Connectome API is live"}

@app.get("/reset")
def reset():
    global obs, _history, _trades, _last_shaped_reward, _patience_counter, _reward_mean, _reward_std, _reward_count
    if len(SESSION["tickers"]) > 1:
        _reset_live_portfolio()
        p_env, _ = _live_portfolio()
        return {
            "message":     "Portfolio episode reset",
            "portfolio":   SESSION["budget"],
            "total_steps": int(np.ceil(p_env.n_steps / max(1, p_env.rebalance_every))),
        }
    obs, _ = env.reset()
    brain.reset()
    _history = [env.initial_cash]
    _trades.clear()
    _last_shaped_reward = 0.0
    _patience_counter = 0
    _reward_mean = 0.0
    _reward_std = 1.0
    _reward_count = 0
    return {
        "message":     "Episode reset",
        "portfolio":   env.initial_cash,
        "total_steps": env.n_steps,
    }

class ActionPayload(BaseModel):
    action: int = -1
    policy: str = "brain"   # "brain", "momentum", "mean_revert", "manual"
    learning: bool = False  # Disable online learning by default to prevent policy collapse

@app.post("/step")
def take_step(payload: ActionPayload):
    auto_brain = (payload.action == -1)
    return _dispatch_step(payload.action, auto_brain=auto_brain,
                          enable_learning=payload.learning)


# ── Session config: universe, budget, sizing mode ──────────────────────────
class SessionPayload(BaseModel):
    tickers: list[str] | None = None
    budget: float | None = None
    dynamic_sizing: bool | None = None


def _session_summary() -> dict:
    mode = "portfolio" if len(SESSION["tickers"]) > 1 else "single"
    out = {
        "session": {
            "tickers": list(SESSION["tickers"]),
            "budget": float(SESSION["budget"]),
            "dynamic_sizing": bool(SESSION["dynamic_sizing"]),
            "mode": mode,
        },
        "available_tickers": AVAILABLE_TICKERS,
    }
    if mode == "portfolio":
        p_env, _ = _live_portfolio()
        out["n_steps"] = int(np.ceil(p_env.n_steps / max(1, p_env.rebalance_every)))
        out["step"] = int(p_env.step_i // max(1, p_env.rebalance_every))
        out["n_bars"] = int(p_env.n_steps)
        out["rebalance_every"] = int(p_env.rebalance_every)
        out["gate"] = {"mode": PORTFOLIO_GATE, "min_k": PORTFOLIO_MIN_K,
                       "max_k": PORTFOLIO_MAX_K}
        out["window"] = [LIVE_PORTFOLIO_START, LIVE_PORTFOLIO_END]
    else:
        out["n_steps"] = int(env.n_steps)
        out["step"] = int(env.current_step)
    return out


@app.get("/config")
def get_config():
    return _session_summary()


@app.post("/config")
def set_config(payload: SessionPayload):
    """Switch universe / budget / sizing mode, then reset the affected episode."""
    global env, obs, _history, _trades, _last_shaped_reward

    tickers = list(SESSION["tickers"])
    budget = float(SESSION["budget"])
    dynamic = bool(SESSION["dynamic_sizing"])

    if payload.tickers is not None:
        picked = []
        for t in payload.tickers:
            t = str(t).strip().upper()
            if not t:
                continue
            if t not in AVAILABLE_TICKERS:
                raise HTTPException(400, f"Unknown ticker {t!r}")
            if t not in picked:
                picked.append(t)
        if not picked:
            raise HTTPException(400, "Pick at least one ticker")
        tickers = picked

    if payload.budget is not None:
        b = float(payload.budget)
        if not (b > 0) or not math.isfinite(b):
            raise HTTPException(400, "Budget must be a positive number")
        budget = b

    if payload.dynamic_sizing is not None:
        dynamic = bool(payload.dynamic_sizing)

    if len(tickers) > 1 and not PORTFOLIO_CKPT.is_file():
        raise HTTPException(
            400, f"Portfolio checkpoint missing ({PORTFOLIO_CKPT.name}); "
                 "train it with the recipe in README, or pick one ticker.")

    prev = dict(SESSION)
    new_env = None
    try:
        SESSION["tickers"], SESSION["budget"], SESSION["dynamic_sizing"] = \
            tickers, budget, dynamic
        if len(tickers) > 1:
            _reset_live_portfolio()
            _live_portfolio()
        else:
            new_env = _single_env(tickers[0], budget)
    except (SystemExit, Exception) as exc:
        SESSION.clear()
        SESSION.update(prev)
        raise HTTPException(400, f"Could not build that session: {exc}")

    if new_env is not None:
        env = new_env
        obs, _ = env.reset()
        brain.reset()
        _history = [env.initial_cash]
        _trades.clear()
        _last_shaped_reward = 0.0

    return _session_summary()

class TrainBatchPayload(BaseModel):
    episodes: int = 25
    split: str = SERVER_TRAIN_SPLIT

@app.post("/train_batch")
def train_batch(payload: TrainBatchPayload):
    """Fast-forward multi-episode training with detailed telemetry of what the brain learned."""
    global _total_episodes_trained, _last_shaped_reward, _reward_count, _reward_mean, _reward_std, _learning_history, _simulation_runs, _collapse_events
    import datetime as _dt
    n_episodes = min(max(1, payload.episodes), 100)

    # Capture initial synaptic state and baseline probabilities
    weights_before = brain.get_synaptic_weights_sample()
    dummy_obs = env._get_observation(0)
    _, probs_before, _, _ = brain.select_action(dummy_obs)

    episodes_data = []
    returns = []
    losses = []
    split = payload.split
    _split_bounds(split)  # validate early (raises 400 on bad split)

    for ep in range(n_episodes):
        # NB: the reward normaliser persists across episodes on purpose; a
        # moving reward scale is one of the things that used to destabilise the
        # critic (see trainer.py).

        # Curriculum learning: cycle through the assets
        ticker = MULTI_ASSET_TICKERS[ep % len(MULTI_ASSET_TICKERS)]
        # Train on the requested time slice only (default: everything before the
        # held-out val/test windows). Never trains on the test split.
        train_env = _make_env(ticker, split, enable_sl_tp=True)
        
        cur_obs, _ = train_env.reset()
        ep_history = [train_env.initial_cash]
        done = False
        step_count = 0
        ep_trades = 0
        ep_wins = 0
        ep_entropies = []

        while not done and step_count < train_env.n_steps:
            last_shaped = _last_shaped_reward if step_count == 0 else shaped_reward
            _, raw_action_probs, _, _ = brain.select_action(cur_obs, reward=last_shaped)
            # Shared masking path (config.MASK_PRESETS['train']); see strategy.py.
            p_arr = apply_preset(
                raw_action_probs, "train",
                shares=train_env.shares, cash=train_env.cash,
                entry_price=getattr(train_env, "entry_price", 0.0),
            ).astype(np.float32)

            # Exploration decay - slower decay to ensure sufficient exploration
            epsilon = max(0.05, 0.3 * (0.99 ** ep))  # Start at 30%, decay to 5%
            if np.random.random() < epsilon:
                valid_actions = [i for i, p in enumerate(p_arr) if p > 0]
                action = int(np.random.choice(valid_actions)) if valid_actions else 0
            else:
                action = int(np.random.choice(7, p=p_arr))

            if action != 0:
                ep_trades += 1

            next_obs, reward, done, _, info = train_env.step(action)
            pv_now = float(info["portfolio_value"])
            if len(ep_history) > 1 and pv_now > ep_history[-1]:
                ep_wins += 1

            profit_reward = 0.0
            if len(ep_history) >= 2 and ep_history[-1] > 0:
                profit_reward = (pv_now - ep_history[-1]) / ep_history[-1]
            sharpe_reward = 0.0
            if len(ep_history) >= 20:
                rets = np.diff(np.array(ep_history[-20:])) / (np.array(ep_history[-20:])[:-1] + 1e-9)
                raw_sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9))
                sharpe_reward = float(np.clip(raw_sharpe, -3.0, 3.0))  # Cap Sharpe to prevent domination
            shaped_reward = 0.6 * float(reward) + 0.4 * sharpe_reward
            _last_shaped_reward = shaped_reward  # Keep dopamine signal fresh during training

            normalized_reward = _reward_norm.normalize(shaped_reward)
            brain.record_experience(cur_obs, action, normalized_reward, next_obs, bool(done))
            if step_count % 4 == 0 and len(brain.replay_buffer) >= 16:
                try:
                    loss = brain.train_online_step(batch_size=min(64, len(brain.replay_buffer)))
                    losses.append(loss)
                except RuntimeError as e:
                    if "inplace operation" in str(e):
                        print(f"Training step skipped (in-place op): {e}")
                        brain.replay_buffer.clear()
                        brain.optimizer.state.clear()
                    else:
                        raise

            # Track entropy for policy collapse detection
            with torch.no_grad():
                obs_t = torch.tensor(cur_obs, dtype=torch.float32).unsqueeze(0)
                logits, _, _ = brain(obs_t)
                probs_t = F.softmax(logits, dim=-1)
                entropy_val = -(probs_t * torch.log(probs_t + 1e-9)).sum(dim=-1).mean().item()
                ep_entropies.append(entropy_val)

            cur_obs = next_obs
            ep_history.append(pv_now)
            step_count += 1

        # Episode-end reinforcement updates
        for _ in range(3):
            if len(brain.replay_buffer) >= 32:
                try:
                    loss = brain.train_online_step(batch_size=32)
                    losses.append(loss)
                except RuntimeError as e:
                    print(f"Episode-end step skipped: {e}")

        # Anti-collapse *telemetry only*. The entropy coefficient is already
        # auto-tuned toward brain.target_entropy inside train_online_step, so we
        # do not need to re-initialise weights or clear the buffer/optimizer
        # here. The old handler did exactly that, which is why the online brain
        # never improved across episodes: every time it started to commit to a
        # strategy its weights were wiped and it began again from the prior.
        if ep_entropies:
            avg_entropy = float(np.mean(ep_entropies))
            if avg_entropy < 0.35 * float(np.log(7)):  # < ~0.68 nats
                _collapse_events.append({"episode": _total_episodes_trained + ep + 1,
                                         "avg_entropy": round(avg_entropy, 4)})

        final_pv = ep_history[-1]
        ep_ret = (final_pv - train_env.initial_cash) / train_env.initial_cash * 100
        ep_winrate = round(ep_wins / max(1, step_count) * 100, 1)
        ep_sharpe = _compute_sharpe(ep_history)

        # Automatically preserve best checkpoint when performance improves
        if ep_sharpe > _best_score and ep_ret > 0:
            _save_best_checkpoint(ep_sharpe)

        returns.append(ep_ret)
        ep_start = str(train_env.data.index.min().date())
        ep_end = str(train_env.data.index.max().date())
        ep_data = {
            "episode": _total_episodes_trained + ep + 1,
            "ticker": ticker,
            "split": split,
            "start_date": ep_start,
            "end_date": ep_end,
            "steps": step_count,
            "return_pct": round(ep_ret, 2),
            "win_rate": ep_winrate,
            "sharpe": ep_sharpe,
            "trades": ep_trades,
            "final_portfolio": round(final_pv, 2),
            "loss": round(float(np.mean(losses[-5:])) if losses else 0.0, 5)
        }
        episodes_data.append(ep_data)
        # Update learning history for performance graphs
        _learning_history.append({
            "episode": _total_episodes_trained + ep + 1,
            "ticker": ticker,
            "return_pct": round(ep_ret, 2),
            "sharpe": ep_sharpe,
            "loss": round(float(np.mean(losses[-5:])) if losses else 0.0, 5),
            "win_rate": ep_winrate,
        })

    _total_episodes_trained += n_episodes

    weights_after = brain.get_synaptic_weights_sample()
    _, probs_after, _, _ = brain.select_action(dummy_obs)

    # Compute synaptic weight plasticity deltas (LTP vs LTD)
    deltas = [round(a - b, 5) for a, b in zip(weights_after, weights_before)]
    ltp_count = sum(1 for d in deltas if d > 0.0001)  # Long-Term Potentiation
    ltd_count = sum(1 for d in deltas if d < -0.0001) # Long-Term Depression

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_return = float(np.mean(returns)) if returns else 0.0
    best_return = float(np.max(returns)) if returns else 0.0
    worst_return = float(np.min(returns)) if returns else 0.0
    median_return = float(np.median(returns)) if returns else 0.0

    # Out-of-sample evaluation on the held-out split (never trained on).
    # This is the honest number: it is NOT the max over stochastic in-sample
    # episodes, and it is measured over the same window for every ticker.
    heldout = _heldout_report(MULTI_ASSET_TICKERS, SERVER_EVAL_SPLIT, seed=0)

    # Save latest checkpoint
    _save_checkpoint()
    # If a best checkpoint exists, activate it
    _load_best_checkpoint()

    # --- Record this simulation run in the persistent history log ---
    run_record = {
        "run": len(_simulation_runs) + 1,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": n_episodes,
        "total_episodes_after": _total_episodes_trained,
        "train_split": split,
        "train_date_range": list(SPLITS[split]),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(median_return, 2),
        "best_return_pct": round(best_return, 2),
        "worst_return_pct": round(worst_return, 2),
        "avg_win_rate": round(float(np.mean([ep["win_rate"] for ep in episodes_data])) if episodes_data else 0.0, 1),
        "avg_trades": round(float(np.mean([ep["trades"] for ep in episodes_data])) if episodes_data else 0.0, 1),
        "avg_loss": round(avg_loss, 5),
        # NOTE: best_return_pct is the MAX over stochastic in-sample episodes
        # (each episode samples a different ticker and action noise). Median and
        # the held-out block below are the trustworthy figures.
        "heldout": heldout,
        "plasticity": {
            "ltp_count": ltp_count,
            "ltd_count": ltd_count,
            "weights_before": [round(w, 4) for w in weights_before],
            "weights_after": [round(w, 4) for w in weights_after],
            "deltas": deltas,
        },
        "valence_evolution": {
            "before": [round(p, 3) for p in probs_before],
            "after":  [round(p, 3) for p in probs_after]
        },
        "episodes_data": episodes_data,
    }
    _simulation_runs.append(run_record)
    _save_simulation_history()

    return {
        "status": "success",
        "run": run_record["run"],
        "episodes_trained": n_episodes,
        "total_episodes": _total_episodes_trained,
        "train_split": split,
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(median_return, 2),
        "best_return_pct": round(best_return, 2),
        "worst_return_pct": round(worst_return, 2),
        "heldout": heldout,
        "avg_loss": round(avg_loss, 5),
        "episodes_data": episodes_data,
        "plasticity": run_record["plasticity"],
        "valence_evolution": run_record["valence_evolution"],
    }

class EvalEpisodePayload(BaseModel):
    ticker: str = "AAPL"
    deterministic: bool = True
    enable_sl_tp: bool = True
    split: str = SERVER_EVAL_SPLIT

@app.post("/eval_episode")
def eval_episode(payload: EvalEpisodePayload):
    """
    Run an offline-style full evaluation episode using the living brain in eval mode.
    Does not update policy weights or pollute the replay buffer.
    """
    eval_env = _make_env(payload.ticker, payload.split, enable_sl_tp=payload.enable_sl_tp)
    cur_obs, _ = eval_env.reset()

    # Save simulator state so live server session state is cleanly preserved
    state_v = brain.simulator.V.copy()
    state_spikes = brain.simulator.last_spikes.copy()
    state_inh = getattr(brain.simulator, '_inh', 0.0)
    state_dopamine = brain.simulator.dopamine_level

    brain.eval()
    portfolio_history = [eval_env.initial_cash]
    trades = []
    actions_taken = [0] * 7
    labels = ["Hold", "Buy 25%", "Buy 50%", "Buy 100%", "Sell 25%", "Sell 50%", "Sell 100%"]
    total_reward = 0.0
    step_count = 0
    done = False
    shaped_reward = 0.0

    while not done and step_count < eval_env.n_steps:
        _, raw_probs, val, spikes = brain.select_action(cur_obs, reward=shaped_reward, deterministic=False)
        # Shared masking path (config.MASK_PRESETS['eval']); see strategy.py.
        p_arr = apply_preset(
            raw_probs, "eval",
            shares=eval_env.shares, cash=eval_env.cash,
            entry_price=getattr(eval_env, "entry_price", 0.0),
        ).astype(np.float32)
        p_arr = _apply_regime_gate(p_arr, cur_obs, eval_env.shares)

        if payload.deterministic:
            action = int(np.argmax(p_arr))
        else:
            action = int(np.random.choice(7, p=p_arr))

        actions_taken[action] += 1
        price = float(eval_env.data.iloc[min(eval_env.current_step + eval_env.window_size - 1, len(eval_env.data) - 1)]["Close"])

        next_obs, rew, done, truncated, info = eval_env.step(action)
        pv = float(info["portfolio_value"])
        portfolio_history.append(pv)
        total_reward += rew

        if action != 0:
            trades.append({
                "step": step_count,
                "action": labels[action],
                "price": round(price, 2),
                "portfolio": round(pv, 2),
            })

        sharpe_rew = 0.0
        if len(portfolio_history) >= 20:
            rets = np.diff(np.array(portfolio_history[-20:])) / (np.array(portfolio_history[-20:])[:-1] + 1e-9)
            raw_s = float(np.mean(rets) / (np.std(rets) + 1e-9))
            sharpe_rew = float(np.clip(raw_s, -3.0, 3.0))
        shaped_reward = 0.6 * float(rew) + 0.4 * sharpe_rew

        cur_obs = next_obs
        step_count += 1

    # Restore simulator state
    brain.simulator.V = state_v
    brain.simulator.last_spikes = state_spikes
    if hasattr(brain.simulator, '_inh'):
        brain.simulator._inh = state_inh
    brain.simulator.dopamine_level = state_dopamine

    initial = float(eval_env.initial_cash)
    final = float(portfolio_history[-1])
    ret_pct = round((final - initial) / initial * 100, 2)
    sharpe = _compute_sharpe(portfolio_history)
    max_pv = max(portfolio_history)
    max_dd = round((max_pv - min(portfolio_history)) / max_pv * 100, 2) if max_pv > 0 else 0.0
    wins = sum(1 for i in range(1, len(portfolio_history)) if portfolio_history[i] > portfolio_history[i-1])
    win_rate = round(wins / max(1, len(portfolio_history) - 1) * 100, 1)

    # Reference benchmark: true buy & hold over the SAME window (no SL/TP, so it
    # is a real buy-and-hold, not a stop-loss churn strategy).
    bh_env = _make_env(payload.ticker, payload.split, enable_sl_tp=False)
    bh = run_episode(bh_env, baseline_prob_fn("buy_hold"), mask_preset="honest",
                     deterministic=True, seed=0)
    buy_hold_pct = round(bh["total_return_pct"], 2)

    return {
        "status": "success",
        "ticker": payload.ticker,
        "split": payload.split,
        "start_date": str(eval_env.data.index.min().date()),
        "end_date": str(eval_env.data.index.max().date()),
        "initial_portfolio": initial,
        "final_portfolio": round(final, 2),
        "return_pct": ret_pct,
        "buy_hold_return_pct": buy_hold_pct,
        "excess_vs_buyhold_pct": round(ret_pct - buy_hold_pct, 2),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_steps": step_count,
        "total_trades": len(trades),
        "action_distribution": {labels[i]: actions_taken[i] for i in range(7)},
        "trades_sample": trades[:30],
        "portfolio_history_sampled": [round(v, 2) for v in portfolio_history[::max(1, len(portfolio_history)//100)]],
    }

@app.get("/brain_status")
def brain_status():
    return {
        "total_episodes_trained": _total_episodes_trained,
        "dopamine_level": round(brain.simulator.dopamine_level, 4),
        "experience_buffer_size": len(brain.replay_buffer),
        "online_learning_enabled": _online_learning_enabled,
        "learning_history": _learning_history[-50:],  # Last 50 episodes
        "N_NEURONS": brain.N_NEURONS,
    }

@app.get("/info")
def get_info():
    return {
        "ticker":       "AAPL",
        "total_steps":  env.n_steps,
        "window_size":  env.window_size,
        "initial_cash": env.initial_cash,
        "current_step": env.current_step,
    }

@app.get("/learning_history")
def learning_history():
    """Return learning progress history for performance graphs."""
    return {
        "history": _learning_history,
        "total_episodes": _total_episodes_trained,
    }

@app.get("/simulation_history")
def simulation_history():
    """Return log of every /train_batch (Evolve / Run Simulations) call."""
    return {
        "runs": _simulation_runs,
        "total_runs": len(_simulation_runs),
        "total_episodes": _total_episodes_trained,
    }

@app.post("/simulation_history/clear")
def clear_simulation_history():
    """Clear all simulation run history records."""
    global _simulation_runs
    _simulation_runs = []
    _save_simulation_history()
    return {"status": "cleared", "total_runs": 0}

_EDGE_CACHE: dict[str, np.ndarray] = {}


@app.get("/connectome_edges.bin")
def connectome_edges(top: int = 12000):
    """Real FlyWire-derived synapse list for the 1,024 model neurons.

    Returns little-endian float32 triples (source_index, target_index, |w|),
    strongest first, capped at ``top``. The browser draws these as lines, so
    the 3-D view shows the actual connectivity instead of an empty overlay.
    """
    key = "all"
    if key not in _EDGE_CACHE:
        # Iterate the sparse matrix: the dense copy is unnecessary and scales
        # as N^2 (268 MB at 8,192 neurons).
        W = brain.simulator.W.tocoo()
        keep = W.row != W.col                        # drop self-loops
        src = W.row[keep].astype(np.int32)
        tgt = W.col[keep].astype(np.int32)
        w = np.abs(W.data[keep]).astype(np.float32)
        order = np.argsort(-w, kind="stable")
        _EDGE_CACHE[key] = np.stack(
            [src[order], tgt[order], w[order]], axis=1).astype("<f4")
    edges = _EDGE_CACHE[key]
    n = max(1, min(int(top), len(edges)))
    body = edges[:n].tobytes()
    return Response(content=body, media_type="application/octet-stream", headers={
        "X-Total-Edges": str(len(edges)),
        "X-Returned-Edges": str(n),
        "Cache-Control": "public, max-age=3600",
    })


@app.get("/brain_atlas")
def brain_atlas():
    """Return real MaleCNS brain atlas data for 3D visualization."""
    atlas = get_brain_atlas()
    if atlas is None:
        return {"error": "Brain atlas not available"}
    
    # Return subset of positions and groups for frontend rendering
    # Sample every 5th neuron for performance
    positions = atlas["positions"]
    neuron_ids = atlas["neuron_ids"]
    neuron_groups = atlas["neuron_groups"]
    
    sample_rate = 5
    sampled_positions = positions[::sample_rate]
    sampled_ids = neuron_ids[::sample_rate]
    sampled_groups = neuron_groups[::sample_rate]
    
    return {
        "positions": sampled_positions,
        "neuron_ids": sampled_ids,
        "neuron_groups": sampled_groups,
        "group_info": atlas["group_info"],
        "group_counts": atlas["group_counts"],
        "total_sampled": len(sampled_positions),
        "total_original": atlas["total_neurons"],
    }

# ── Portfolio (cross-section) engine ───────────────────────────────────────
# The single-asset engine above is what the 3-D brain and the live telemetry
# drive. This is the *scope expansion*: one shared policy allocates the whole
# book across N assets plus a cash slot, rebalanced on a monthly horizon, and
# the numbers below come from the same honest, split-aware, cost-inclusive,
# **ticker-holdout** evaluation the README reports (unseen tickers, never-trained
# window, 10 bps per side). See portfolio_env.PortfolioEnv for the allocation.
PORTFOLIO_CKPT = Path(os.environ.get(
    "FLYBRAIN_PORTFOLIO_CKPT",
    str(Path(__file__).parent / "checkpoints" / "portfolio_fly.pt")))
PORTFOLIO_PLAN = os.environ.get("FLYBRAIN_PORTFOLIO_PLAN", "cycle")
PORTFOLIO_SPLIT = os.environ.get("FLYBRAIN_PORTFOLIO_SPLIT", "test")
PORTFOLIO_UNIVERSE = os.environ.get(
    "FLYBRAIN_PORTFOLIO_TICKERS", "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN")
PORTFOLIO_REBALANCE = int(os.environ.get("FLYBRAIN_PORTFOLIO_REBALANCE", "21"))
# Breadth gate. "positive" makes the held breadth k a *policy output*; the floor
# keeps the book invested where the policy's learned conviction threshold
# under-fires on an unseen cross-section (README, "Adaptive breadth").
PORTFOLIO_GATE = os.environ.get("FLYBRAIN_PORTFOLIO_GATE", "positive")
PORTFOLIO_MIN_K = int(os.environ.get("FLYBRAIN_PORTFOLIO_MIN_K", "2"))
PORTFOLIO_MAX_K = int(os.environ.get("FLYBRAIN_PORTFOLIO_MAX_K", "4"))
PORTFOLIO_FULL_READOUT = os.environ.get(
    "FLYBRAIN_PORTFOLIO_FULL_READOUT", "true").lower() == "true"
# Optional: run the no-RL distilled policy. The checkpoint carries its own
# weights, so this only matters for the (documented) RL fine-tuning variant.
_portfolio_cache: dict | None = None
_portfolio_agent = None


def _portfolio_tickers():
    return [t.strip().upper() for t in PORTFOLIO_UNIVERSE.split(",") if t.strip()]


def _build_portfolio_env():
    from portfolio_env import PortfolioEnv
    from config import SPLIT_PLANS
    start, end = SPLIT_PLANS[PORTFOLIO_PLAN][PORTFOLIO_SPLIT]
    kw = dict(start_date=start, end_date=end, rebalance_every=PORTFOLIO_REBALANCE,
              alloc_min_k=PORTFOLIO_MIN_K, alloc_max_k=PORTFOLIO_MAX_K)
    if PORTFOLIO_GATE == "topk":
        kw["alloc_gate"] = "topk"
        kw["alloc_topk"] = PORTFOLIO_MIN_K
    elif PORTFOLIO_GATE != "none":
        kw["alloc_gate"] = PORTFOLIO_GATE
    return PortfolioEnv(_portfolio_tickers(), **kw)


def _load_portfolio_agent(obs_dim: int):
    """A private portfolio-shaped brain instance, loaded from the checkpoint.

    Used by the live session (cached) and by background combo searches (one
    fresh instance per job, so a search can never race the live telemetry).
    """
    agent = FlyBrainAgent(obs_dim=obs_dim, use_connectome=USE_CONNECTOME,
                          N_NEURONS=N_NEURONS,
                          use_full_readout=PORTFOLIO_FULL_READOUT)
    agent.set_policy_state(torch.load(PORTFOLIO_CKPT, map_location="cpu",
                                      weights_only=False))
    agent.eval()
    return agent


def _get_portfolio_agent(env):
    """A second, portfolio-shaped brain instance (the live one is 398-dim
    single-asset). Built lazily so server startup stays fast."""
    global _portfolio_agent
    if _portfolio_agent is not None and _portfolio_agent.obs_dim == env.obs_dim:
        return _portfolio_agent
    _portfolio_agent = _load_portfolio_agent(env.obs_dim)
    return _portfolio_agent


# ── Combo Lab ──────────────────────────────────────────────────────────────
# "Which k names should the fly be allowed to trade?" Enumerate every k-stock
# subset of the current universe, run the *deployed* portfolio checkpoint on the
# validation split, and rank by the fly's own return (excess vs that basket's
# buy & hold is reported on every row). The winner is then evaluated once on
# test and reported separately. Validation is the only split allowed to drive
# selection, so the test number is the honest generalisation estimate.
#
# The search runs as a background job: the dashboard polls its progress instead
# of holding a request open for a minute. One job at a time keeps the CPU (and
# the live brain) responsive.
COMBO_TRAIN_TICKERS = {
    t.strip().upper() for t in os.environ.get(
        "FLYBRAIN_PORTFOLIO_TRAIN_TICKERS",
        "MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG").split(",") if t.strip()
}
_COMBO_JOBS: dict[str, dict] = {}
_COMBO_LOCK = threading.Lock()
_COMBO_ROWS: dict[tuple, dict] = {}


def _combo_key(combo, split):
    try:
        stamp = PORTFOLIO_CKPT.stat().st_mtime
    except OSError:
        stamp = 0.0
    return (tuple(combo), str(split), PORTFOLIO_CKPT.name, stamp,
            PORTFOLIO_PLAN, PORTFOLIO_GATE, PORTFOLIO_MIN_K, PORTFOLIO_MAX_K,
            PORTFOLIO_REBALANCE)


def _combo_env(tickers, split):
    from portfolio_env import PortfolioEnv
    from config import SPLIT_PLANS
    start, end = SPLIT_PLANS[PORTFOLIO_PLAN][split]
    kw = dict(start_date=start, end_date=end,
              rebalance_every=PORTFOLIO_REBALANCE,
              alloc_min_k=PORTFOLIO_MIN_K, alloc_max_k=PORTFOLIO_MAX_K)
    if PORTFOLIO_GATE == "topk":
        kw["alloc_gate"] = "topk"
        kw["alloc_topk"] = PORTFOLIO_MIN_K
    elif PORTFOLIO_GATE != "none":
        kw["alloc_gate"] = PORTFOLIO_GATE
    return PortfolioEnv(list(tickers), **kw)


def _combo_rollout(combo, split, agent):
    """One deterministic fly rollout plus the same basket's buy & hold."""
    from portfolio_env import buy_hold_weights, run_static_weights
    env = _combo_env(combo, split)
    env.reset()
    done = False
    while not done:
        actions, _probs, _values = agent.select_actions_batch(
            env.observe(), deterministic=True)
        _, _, done, _ = env.step(np.asarray(actions, dtype=int))
    fly = env.metrics()
    bh = run_static_weights(env, buy_hold_weights, 1)
    decisions = max(1, len(env.history) - 1)
    return {
        "combo": list(combo),
        "split": split,
        "start": str(env.dates[0].date()),
        "end": str(env.dates[-1].date()),
        "return_pct": round(float(fly["total_return_pct"]), 2),
        "bh_return_pct": round(float(bh["total_return_pct"]), 2),
        "excess_pct": round(float(fly["total_return_pct"]
                                  - bh["total_return_pct"]), 2),
        "sharpe": round(float(fly["sharpe"]), 2),
        "max_drawdown_pct": round(float(fly["max_drawdown_pct"]), 2),
        "avg_positions": round(float(fly["avg_positions"]), 2),
        "turnover_per_decision": round(float(fly["turnover"]) / decisions, 3),
        "decisions": int(decisions),
    }


def _combo_worker(job_id, combos, refresh):
    from combo_lab import rank, summarise
    job = _COMBO_JOBS[job_id]
    t0 = time.time()
    results = []
    with _COMBO_LOCK:
        job["status"] = "running"
    try:
        probe = _combo_env(combos[0], "val")
        agent = _load_portfolio_agent(probe.obs_dim)
        del probe
        for i, combo in enumerate(combos):
            key = _combo_key(combo, "val")
            row = None if refresh else _COMBO_ROWS.get(key)
            if row is None:
                row = _combo_rollout(combo, "val", agent)
                _COMBO_ROWS[key] = row
            results.append(row)
            with _COMBO_LOCK:
                job["done"] = i + 1
                job["results"] = rank(results)[:12]
                job["summary"] = summarise(results, len(combos))
        ranked = rank(results)
        best = None
        if ranked:
            top = ranked[0]
            key = _combo_key(top["combo"], "test")
            test = None if refresh else _COMBO_ROWS.get(key)
            if test is None:
                with _COMBO_LOCK:
                    job["phase"] = "test"
                test = _combo_rollout(top["combo"], "test", agent)
                _COMBO_ROWS[key] = test
            best = dict(top)
            best["test"] = test
            best["test_is_ticker_holdout"] = not any(
                t in COMBO_TRAIN_TICKERS for t in best["combo"])
        with _COMBO_LOCK:
            job.update(status="done", phase=None, best=best,
                       results=ranked[:12],
                       summary=summarise(results, len(combos)),
                       elapsed_s=round(time.time() - t0, 1))
    except Exception as exc:                      # keep the job introspectable
        with _COMBO_LOCK:
            job.update(status="error", error=str(exc), phase=None,
                       elapsed_s=round(time.time() - t0, 1))


class ComboPayload(BaseModel):
    tickers: list[str] | None = None
    size: int = 3
    refresh: bool = False


@app.post("/combo_search")
def start_combo_search(payload: ComboPayload):
    """Start a background search over every ``size``-stock combo of a universe."""
    from combo_lab import MAX_COMBOS, combos_for, normalise_pool
    pool = normalise_pool(payload.tickers if payload.tickers is not None
                          else SESSION["tickers"])
    size = int(payload.size)
    if not 2 <= size <= 4:
        raise HTTPException(400, "Combo size must be 2, 3 or 4")
    if len(pool) < size:
        raise HTTPException(
            400, f"Need at least {size} stocks in the universe, got {len(pool)}")
    unknown = [t for t in pool if t not in AVAILABLE_TICKERS]
    if unknown:
        raise HTTPException(400, f"Unknown ticker(s): {', '.join(unknown)}")
    if not PORTFOLIO_CKPT.is_file():
        raise HTTPException(
            400, f"Portfolio checkpoint missing ({PORTFOLIO_CKPT.name}); "
                 "train it with the recipe in README first.")
    combos = combos_for(pool, size)
    if len(combos) > MAX_COMBOS:
        raise HTTPException(
            400, f"C({len(pool)},{size}) = {len(combos)} combos, above the "
                 f"{MAX_COMBOS}-combo limit. Pick fewer stocks or a smaller size.")
    with _COMBO_LOCK:
        if any(j["status"] in ("queued", "running")
               for j in _COMBO_JOBS.values()):
            raise HTTPException(409, "A combo search is already running")
        job_id = uuid.uuid4().hex[:12]
        _COMBO_JOBS[job_id] = {
            "job_id": job_id, "status": "queued", "phase": "val",
            "pool": pool, "size": size, "done": 0, "total": len(combos),
            "results": [], "summary": {"evaluated": 0, "total": len(combos)},
            "best": None, "error": None, "elapsed_s": 0.0,
        }
    threading.Thread(target=_combo_worker,
                     args=(job_id, combos, bool(payload.refresh)),
                     daemon=True).start()
    return {"job_id": job_id, "pool": pool, "size": size,
            "total": len(combos)}


@app.get("/combo_search/{job_id}")
def combo_search_status(job_id: str):
    with _COMBO_LOCK:
        job = _COMBO_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown combo search")
        return dict(job)


def _portfolio_sample(points, stride):
    return points[::stride]


@app.get("/portfolio_eval")
def portfolio_eval(refresh: bool = False):
    """Held-out cross-section backtest: fly vs buy & hold / trend / momentum.

    Cached after the first call (building the env reads and aligns ~10 CSVs and
    the brain load is another second); ``?refresh=true`` recomputes it.
    """
    global _portfolio_cache
    if _portfolio_cache is not None and not refresh:
        return _portfolio_cache

    if not PORTFOLIO_CKPT.is_file():
        return {
            "available": False,
            "checkpoint": PORTFOLIO_CKPT.name,
            "recipe": ("python train_portfolio.py --tickers MSFT,TSLA,AMD,GOOGL,"
                       "AMZN,KO,COST,HD,V,MA,ORCL,PG --plan cycle --episodes 0 "
                       "--rebalance-every 21 --full-readout --alloc-gate positive "
                       "--alloc-min-k 2 --alloc-max-k 4 --imitate-episodes 10 "
                       f"--out {PORTFOLIO_CKPT}"),
        }

    from portfolio_env import (buy_hold_weights, momentum_weights,
                               trend200_weights)

    env = _build_portfolio_env()
    agent = _get_portfolio_agent(env)

    def rollout(step_fn):
        env.reset()
        equity = [1.0]
        done = False
        while not done:
            _, _, done, info = step_fn(env)
            equity.append(float(info["value"]))
        m = env.metrics()
        m["equity"] = equity
        return m

    def rule(fn):
        return lambda e: e._advance(fn(e, e.step_i))

    def fly(e):
        obs = e.observe()
        actions = [agent.select_action(obs[a], deterministic=True)[0]
                   for a in range(e.n_assets)]
        return e.step(actions)

    strategies = {
        "fly": rollout(fly),
        "buy&hold": rollout(rule(lambda e, i: buy_hold_weights(e))),
        "trend200": rollout(rule(trend200_weights)),
        "mom120-top3-m": rollout(rule(lambda e, i: momentum_weights(e, i, 120, 3))),
    }

    n_pts = min(len(v["equity"]) for v in strategies.values())
    stride = max(1, n_pts // 240)
    # Each equity point is one *decision*, i.e. `rebalance_every` bars apart.
    bars = [env.dates[env._bar(min(i, env.n_steps) * PORTFOLIO_REBALANCE)]
            for i in range(n_pts)]

    out_strats = {}
    for name, m in strategies.items():
        out_strats[name] = {
            "return_pct": round(m["total_return_pct"], 2),
            "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
            "sharpe": round(m["sharpe"], 3),
            "turnover": round(m["turnover"], 2),
            "n_trades": m["n_trades"],
            "avg_positions": round(m["avg_positions"], 2),
            "curve": _portfolio_sample(m["equity"][:n_pts], stride),
        }

    bh = out_strats["buy&hold"]["return_pct"]
    _portfolio_cache = {
        "available": True,
        "checkpoint": PORTFOLIO_CKPT.name,
        "plan": PORTFOLIO_PLAN,
        "split": PORTFOLIO_SPLIT,
        "date_range": [str(env.dates.min().date()), str(env.dates.max().date())],
        "tickers": env.tickers,
        "n_assets": env.n_assets,
        "n_steps": env.n_steps,
        "rebalance_every": PORTFOLIO_REBALANCE,
        "gate": {"mode": PORTFOLIO_GATE, "min_k": PORTFOLIO_MIN_K,
                 "max_k": PORTFOLIO_MAX_K},
        "full_readout": PORTFOLIO_FULL_READOUT,
        "cost_bps": 10,
        "curve_dates": [d.strftime("%Y-%m-%d") for d in bars[::stride]],
        "strategies": out_strats,
        "fly_excess_vs_buyhold_pp": round(out_strats["fly"]["return_pct"] - bh, 2),
    }
    print(f"[portfolio] {env.n_assets} assets {PORTFOLIO_PLAN}/{PORTFOLIO_SPLIT}: "
          f"fly {out_strats['fly']['return_pct']:.1f}% vs bh {bh:.1f}%")
    return _portfolio_cache


# ── Serve brain atlas, flybody & static front-end ──────────────────────────
_STATIC = Path(__file__).parent
_ROOT = _STATIC.parent

_ATLAS_DIR = _ROOT / "fly_connectome" / "public" / "data" / "brain-atlas"
if _ATLAS_DIR.is_dir():
    app.mount("/data/brain-atlas", StaticFiles(directory=str(_ATLAS_DIR)), name="brain-atlas")

_FLYBODY_DIR = _ROOT / "fly_connectome" / "public" / "data" / "flybody"
if _FLYBODY_DIR.is_dir():
    app.mount("/data/flybody", StaticFiles(directory=str(_FLYBODY_DIR)), name="flybody")

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(str(_STATIC / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
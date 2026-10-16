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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
MULTI_ASSET_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]
if not DATA_PATH.is_file():
    raise RuntimeError(f"Missing {DATA_PATH}. Run: python data_fetch.py AAPL")

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
# Deployed strategy. On the honest full-cycle test the learned policy only
# narrowly beats buy & hold (+6pp), while the *pre-specified* 200-day trend
# overlay beats it outright (+17pp) with a lower drawdown and a higher Sharpe
# (see README, "cross-regime matrix"). So the max-return default is "trend200";
# set FLYBRAIN_STRATEGY=fly to serve the raw learned policy instead, or
# "buy_hold" / "momentum" for the other pre-specified rules.
SERVER_STRATEGY = os.environ.get("FLYBRAIN_STRATEGY", "trend200").lower()
RULE_STRATEGIES = {"trend200", "buy_hold", "momentum"}
# Regime block layout appended to each observation by env._compute_regime_block:
# [close/SMA200-1, close/SMA50-1, ann_vol20, vol20/vol60].
REGIME_DIST200_IDX = -4

env   = FlyTradeEnv(DATA_PATH, window_size=30, enable_sl_tp=False,
                    reward_mode=SERVER_REWARD_MODE,
                    indicator_norm=SERVER_INDICATOR_NORM)

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
            # Shared masking path (config.MASK_PRESETS['live']); see strategy.py.
            # Heuristics are opt-in per preset so the policy can also be
            # evaluated honestly (see scripts/backtest.py).
            raw_action_probs = np.asarray(raw_fly_probs, dtype=np.float64)
            probs_arr = apply_preset(
                raw_action_probs, "live",
                shares=env.shares, cash=env.cash,
                price=price, entry_price=getattr(env, "entry_price", 0.0),
            ).astype(np.float32)
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

    # Step the environment with adaptive confidence
    next_obs, reward, done, _, info = env.step(action, confidence=confidence)

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
    if action != 0:
        _trades.append({
            "step":      env.current_step,
            "action":    labels[action],
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
        "action":         labels[action],
        "action_idx":     action,
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
    return _run_step(payload.action, auto_brain=auto_brain, enable_learning=payload.learning)

class TrainBatchPayload(BaseModel):
    episodes: int = 25
    split: str = SERVER_TRAIN_SPLIT

@app.post("/train_batch")
def train_batch(payload: TrainBatchPayload):
    """Fast-forward multi-episode training with detailed telemetry of what the brain learned."""
    global _total_episodes_trained, _last_shaped_reward, _reward_count, _reward_mean, _reward_std, _learning_history, _simulation_runs
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

        # Policy collapse detection: reset replay buffer and reinitialize weights if entropy collapsed
        if ep_entropies:
            avg_entropy = float(np.mean(ep_entropies))
            max_entropy = np.log(7)
            if avg_entropy < 0.4 * max_entropy:  # Less than 40% of max entropy
                # Reinitialize to the buy & hold prior (see FlyBrainAgent._init_smart_weights)
                with torch.no_grad():
                    brain.actor[-1].bias.copy_(torch.tensor([0.0, -1.5, -0.5, 1.0, -2.0, -2.0, -2.0]))
                brain.replay_buffer.clear()
                brain.optimizer.state.clear()

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
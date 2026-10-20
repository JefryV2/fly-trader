"""Get the benefits of a bigger brain from the same 1,024 neurons.

Two levers that extra neurons would normally buy, scored on the same honest
pooled probe (10 tickers, ~27k observations, time-ordered 70/30 split, 5-day
forward return, ridge + permutation null):

  A. a wider read-out - project all 1,024 spikes to 256 dims (what 4,096
     neurons offer as 256 MBONs), instead of reading only the 64 MBONs;
  B. temporal memory - a causal decaying spike trace (rho=0.9), so the
     read-out sees history across days (the encoding is stateless otherwise).

Feature probe, not a trading policy.  Usage: python scripts/probe_readout.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import FlyTradeEnv                      # noqa: E402
from fly_brain import FlyBrainAgent              # noqa: E402

TICKERS = ["NVDA", "JPM", "JNJ", "XOM", "WMT", "CAT", "META", "DIS", "T", "LIN"]
TRAIN_FRAC = 0.70
HORIZON = 5
RHO = 0.90
WIDE = 256


def collect(ticker):
    env = FlyTradeEnv(os.path.join(ROOT, "data", f"{ticker}_daily.csv"), window_size=30)
    obs, _ = env.reset()
    rows, closes = [], []
    for _ in range(env.n_steps):
        a = np.asarray(obs, dtype=np.float32)
        rows.append(a)
        closes.append(float(a[148]))
        obs, *_ = env.step(0)
    closes = np.asarray(closes)
    fwd = np.full(len(closes), np.nan)
    fwd[:-HORIZON] = closes[HORIZON:] / closes[:-HORIZON] - 1.0
    ok = np.isfinite(fwd)
    return np.stack(rows)[ok], fwd[ok]


def blocks(agent, obs, rng):
    sim = agent.simulator
    spikes = sim.encode_batch(obs).astype(np.float32)
    mbon = spikes[:, sim.IDX_MBON.start:sim.IDX_MBON.stop]
    sens = np.stack([agent.sensory_from_obs(o) for o in obs]).astype(np.float32)

    proj = spikes @ (rng.standard_normal((spikes.shape[1], WIDE)).astype(np.float32)
                     / np.sqrt(spikes.shape[1]))

    trace = np.zeros_like(mbon)
    state = np.zeros(mbon.shape[1], dtype=np.float32)
    for i in range(len(mbon)):
        state = RHO * state + mbon[i]
        trace[i] = state
    return {"mbon": mbon, "sens": sens, "proj": proj, "trace": trace}


def ridge_fit_predict(Xtr, ytr, Xte):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_val = max(20, int(0.15 * len(Xtr)))
    Xa, ya = Xtr[:-n_val], ytr[:-n_val]
    Xv, yv = Xtr[-n_val:], ytr[-n_val:]
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xa, Xv = (Xa - mu) / sd, (Xv - mu) / sd
    best, best_a = -9.9, alphas[0]
    for a in alphas:
        w = np.linalg.solve(Xa.T @ Xa + a * np.eye(Xa.shape[1]), Xa.T @ ya)
        r = np.corrcoef(Xv @ w, yv)[0, 1]
        if r > best:
            best, best_a = r, a
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + best_a * np.eye(Xs.shape[1]), Xs.T @ ytr)
    return ((Xte - mu) / sd) @ w


def main():
    agent = FlyBrainAgent(use_connectome=True, N_NEURONS=1024)
    rng = np.random.default_rng(42)
    parts = {}
    for t in TICKERS:
        obs, y = collect(t)
        b = blocks(agent, obs, rng)
        cut = int(TRAIN_FRAC * len(y))
        for k, v in b.items():
            parts.setdefault(k, []).append((v[:cut], v[cut:]))
        parts.setdefault("y", []).append((y[:cut], y[cut:]))

    def stack(k):
        return np.vstack([p[0] for p in parts[k]]), np.vstack([p[1] for p in parts[k]])

    ytr = np.concatenate([p[0] for p in parts["y"]])
    yte = np.concatenate([p[1] for p in parts["y"]])
    print(f"pooled: {len(ytr)} train / {len(yte)} test")

    configs = {
        "baseline  MBON+sens": ["mbon", "sens"],
        "wide      proj256+sens": ["proj", "sens"],
        "memory    MBON+sens+trace": ["mbon", "sens", "trace"],
        "both      proj256+sens+trace": ["proj", "sens", "trace"],
    }
    perm = np.random.default_rng(0).permutation(len(ytr))
    for name, keys in configs.items():
        Xtr = np.hstack([stack(k)[0] for k in keys])
        Xte = np.hstack([stack(k)[1] for k in keys])
        r = float(np.corrcoef(ridge_fit_predict(Xtr, ytr, Xte), yte)[0, 1])
        null = abs(float(np.corrcoef(
            ridge_fit_predict(Xtr, ytr[perm], Xte), yte)[0, 1]))
        print(f"{name:32s} d={Xtr.shape[1]:3d}  r={r:+.3f}  null={null:.3f}")


if __name__ == "__main__":
    main()

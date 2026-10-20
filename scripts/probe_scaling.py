"""Does a bigger brain help when there is more data and a nonlinear read-out?

Feature probe, not a trading policy.  It asks one question: given the fly's
deterministic encoding, is there a *learnable* mapping from features to the
next 5-day return, and does 4,096 neurons beat 1,024 when the sample is large?

Setup
-----
* 10 tickers pooled (~27k observations) instead of 723 AAPL rows.
* Time-ordered split per ticker: first 70 % train, last 30 % test.
* Read-outs: ridge (linear) and a 1-hidden-layer MLP (nonlinear).
* Feature blocks reported separately: MBON spikes only, the 24-dim sensory
  summary only, and both.
* Null: shuffle training targets, refit, score the same test set (10 reps);
  report the 95th percentile of |r|.  A model only counts if |r| clears it.

Caveat: all ten tickers share the same test window (2022-2025), so per-ticker
correlations are not independent draws — this is one market regime, not ten.

Usage: python scripts/probe_scaling.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import FlyTradeEnv                      # noqa: E402
from fly_brain import FlyBrainAgent              # noqa: E402

TICKERS = ["NVDA", "JPM", "JNJ", "XOM", "WMT", "CAT", "META", "DIS", "T", "LIN"]
TRAIN_FRAC = 0.70
HORIZON = 5
SEED = 0


def collect(ticker):
    """Observations and their forward 5-day returns, in time order."""
    env = FlyTradeEnv(os.path.join(ROOT, "data", f"{ticker}_daily.csv"), window_size=30)
    obs, _ = env.reset()
    rows, closes = [], []
    for _ in range(env.n_steps):
        a = np.asarray(obs, dtype=np.float32)
        rows.append(a)
        closes.append(float(a[148]))     # latest close in the 30x5 OHLCV window
        obs, *_ = env.step(0)
    closes = np.asarray(closes)
    fwd = np.full(len(closes), np.nan)
    fwd[:-HORIZON] = closes[HORIZON:] / closes[:-HORIZON] - 1.0
    ok = np.isfinite(fwd)
    return np.stack(rows)[ok], fwd[ok]


def features(agent, obs):
    spikes = agent.simulator.encode_batch(obs)
    mbon = spikes[:, agent.simulator.IDX_MBON.start:agent.simulator.IDX_MBON.stop]
    sens = np.stack([agent.sensory_from_obs(o) for o in obs]).astype(np.float32)
    return np.concatenate([mbon, sens], axis=1).astype(np.float32)


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


class MLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_fit_predict(Xtr, ytr, Xte, seed=SEED):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = MLP(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(300):
        opt.zero_grad()
        lossf(model(Xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()


def score(pred, y):
    return float(np.corrcoef(pred, y)[0, 1])


def main():
    data = {t: collect(t) for t in TICKERS}
    print(f"pooled observations: {sum(len(v[1]) for v in data.values())} "
          f"over {len(TICKERS)} tickers")

    for n_neurons in (1024, 4096):
        agent = FlyBrainAgent(use_connectome=True, N_NEURONS=n_neurons)
        n_mbon = agent.simulator.IDX_MBON.stop - agent.simulator.IDX_MBON.start

        Xtr, ytr, Xte, yte, te_slices = [], [], [], [], []
        for t, (obs, y) in data.items():
            f = features(agent, obs)
            cut = int(TRAIN_FRAC * len(y))
            Xtr.append(f[:cut]); ytr.append(y[:cut])
            Xte.append(f[cut:]); yte.append(y[cut:])
            te_slices.append((t, f[cut:], y[cut:]))
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        Xte, yte = np.vstack(Xte), np.concatenate(yte)

        rng = np.random.default_rng(SEED)
        perms = [rng.permutation(len(ytr)) for _ in range(10)]

        blocks = (("MBON", slice(0, n_mbon)),
                  ("sensory", slice(n_mbon, None)),
                  ("both", slice(None)))
        for name, sl in blocks:
            pred = ridge_fit_predict(Xtr[:, sl], ytr, Xte[:, sl])
            print(f"N={n_neurons:4d}  ridge  {name:7s} d={Xtr[:, sl].shape[1]:3d}  "
                  f"r={score(pred, yte):+.3f}")

        pred = ridge_fit_predict(Xtr, ytr, Xte)
        nulls = [abs(score(ridge_fit_predict(Xtr, ytr[p], Xte), yte)) for p in perms]
        q95 = float(np.quantile(nulls, 0.95))
        per = [(t, score(ridge_fit_predict(Xtr, ytr, fx), fy)) for t, fx, fy in te_slices]
        print(f"N={n_neurons:4d}  ridge  both    r={score(pred, yte):+.3f}  "
              f"null95={q95:.3f}  per-ticker mean={np.mean([v for _, v in per]):+.3f}  "
              f"range={min(v for _, v in per):+.2f}..{max(v for _, v in per):+.2f}")

        pred = mlp_fit_predict(Xtr, ytr, Xte)
        nulls = [abs(score(mlp_fit_predict(Xtr, ytr[p], Xte), yte)) for p in perms]
        q95 = float(np.quantile(nulls, 0.95))
        print(f"N={n_neurons:4d}  mlp    both    r={score(pred, yte):+.3f}  null95={q95:.3f}")


if __name__ == "__main__":
    main()

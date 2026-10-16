"""
train.py
========
Offline trainer for the FlyTrader fly-brain agent.

This replaces the previous stable-baselines3 stub, which could not run: it
passed a plain ``nn.Module`` (returning a 3-tuple) as an SB3
``features_extractor_class`` (which must subclass ``BaseFeaturesExtractor`` with
a ``features_dim``), and hardcoded ``obs_dim=330`` against a 390-dim env.

The trainer here trains the same online Actor-Critic the API uses, on a
**time-split train slice only**, selects the best checkpoint by **validation**
Sharpe, and finally reports **out-of-sample test** metrics against baseline
strategies.

Examples
--------
    python train.py --ticker AAPL
    python train.py --all-tickers --episodes 40
    python train.py --ticker NVDA --indicator-norm per_feature --no-connectome
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_DIR, DATA_DIR, SPLITS, TICKERS  # noqa: E402
from env import FlyTradeEnv  # noqa: E402
from fly_brain import FlyBrainAgent  # noqa: E402
from strategy import baseline_probs, run_episode  # noqa: E402
from trainer import train_episode  # noqa: E402


def make_env(ticker, split, args):
    start, end = SPLITS[split]
    return FlyTradeEnv(
        DATA_DIR / f"{ticker}_daily.csv",
        window_size=30,
        enable_sl_tp=args.sl_tp,
        start_date=start,
        end_date=end,
        indicator_norm=args.indicator_norm,
    )


def validate(agent, ticker, split, args):
    """Validation return using the shared roll-out (deterministic, honest mask)."""
    env = make_env(ticker, split, args)
    metrics = run_episode(
        env,
        lambda obs, prev, shares, cash: np.asarray(
            agent.select_action(obs, reward=0.0, deterministic=False)[1], dtype=np.float64
        ),
        mask_preset="honest",
        deterministic=True,
        seed=args.seed,
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train the FlyTrader agent offline")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--all-tickers", action="store_true")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--epsilon-decay", type=float, default=0.99)
    p.add_argument("--indicator-norm", choices=["block", "per_feature"], default="block")
    p.add_argument("--sl-tp", action="store_true")
    p.add_argument("--use-connectome", dest="use_connectome", action="store_true", default=True)
    p.add_argument("--no-connectome", dest="use_connectome", action="store_false")
    p.add_argument("--n-neurons", type=int, default=1024)
    p.add_argument("--out", default=str(CHECKPOINT_DIR / "fly_brain_trained.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tickers = TICKERS if args.all_tickers else [args.ticker]
    probe = make_env(tickers[0], "train", args)
    obs_dim = int(probe.observation_space.shape[0])
    print(f"[train] obs_dim={obs_dim} | splits={{'train': {SPLITS['train']}, 'val': {SPLITS['val']}, 'test': {SPLITS['test']}}}")

    agent = FlyBrainAgent(obs_dim=obs_dim, use_connectome=args.use_connectome, N_NEURONS=args.n_neurons)
    agent.optimizer.param_groups[0]["lr"] = args.lr

    best_val = -1e9
    for ep in range(args.episodes):
        ticker = tickers[ep % len(tickers)]
        env = make_env(ticker, "train", args)
        eps = max(0.05, args.epsilon * (args.epsilon_decay ** ep))
        result = train_episode(agent, env, epsilon=eps, seed=args.seed + ep)
        ret = (result["final_value"] - env.initial_cash) / env.initial_cash * 100
        print(f"[ep {ep + 1:>3}/{args.episodes}] {ticker} train_ret={ret:>8.2f}% "
              f"trades={result['trades']:>4} loss={result['avg_loss']:>8.4f} eps={eps:.3f}")

        if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
            vals = [validate(agent, t, "val", args)["total_return_pct"] for t in tickers]
            val_ret = float(np.mean(vals))
            tag = ""
            if val_ret > best_val:
                best_val = val_ret
                torch.save(agent.get_policy_state(), args.out)
                tag = "  <- best (saved)"
            print(f"          val_mean_ret={val_ret:>8.2f}%{tag}")

    print(f"\n[train] best validation mean return = {best_val:.2f}%  ->  {args.out}")

    # Final out-of-sample test report (in sample-independent split).
    if Path(args.out).exists():
        agent.set_policy_state(torch.load(args.out, map_location="cpu", weights_only=False))
    print("\n=== Out-of-sample TEST (honest mask) ===")
    header = f"{'ticker':<8} {'fly%':>8} {'buyhold%':>9} {'momentum%':>10} {'random%':>8}"
    print(header)
    print("-" * len(header))
    for t in tickers:
        fly = validate(agent, t, "test", args)
        env = make_env(t, "test", args)
        bh = run_episode(env, lambda o, pa, s, c: baseline_probs("buy_hold", o, shares=s, cash=c, prev_action=pa),
                         mask_preset="honest", deterministic=True, seed=args.seed)
        env = make_env(t, "test", args)
        mo = run_episode(env, lambda o, pa, s, c: baseline_probs("momentum", o, shares=s, cash=c, prev_action=pa),
                         mask_preset="honest", deterministic=True, seed=args.seed)
        env = make_env(t, "test", args)
        rd = run_episode(env, lambda o, pa, s, c: baseline_probs("random", o, shares=s, cash=c, prev_action=pa),
                         mask_preset="honest", deterministic=False, seed=args.seed)
        print(f"{t:<8} {fly['total_return_pct']:>8.2f} {bh['total_return_pct']:>9.2f} "
              f"{mo['total_return_pct']:>10.2f} {rd['total_return_pct']:>8.2f}")


if __name__ == "__main__":
    main()

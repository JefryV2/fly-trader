"""
scripts/backtest.py
===================
Honest, out-of-sample evaluation for FlyTrader.

Runs the *same* masking/selection code path for every strategy and reports
metrics on a held-out time split only.  The point is to answer the question the
project previously could not: does the fly policy beat trivial baselines on
data it was not trained on?

Usage
-----
    python scripts/backtest.py --ticker AAPL --split test
    python scripts/backtest.py --ticker NVDA --all-splits
    python scripts/backtest.py --all-tickers --split test --checkpoint best

Strategies compared
-------------------
    fly_honest   fly policy, validity masking only  (no hand-coded heuristics)
    fly_legacy   fly policy, with the server's legacy buy-forcing heuristics
    buy_hold     buy 100% on the first bar, then hold
    momentum     buy when EMA-trend > 0, sell when < 0
    random       uniformly random over currently valid actions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_DIR, DATA_DIR, SPLITS, TICKERS  # noqa: E402
from env import FlyTradeEnv  # noqa: E402
from fly_brain import FlyBrainAgent  # noqa: E402
from strategy import compare_on_env  # noqa: E402


def load_agent(args, obs_dim, device="cpu"):
    agent = FlyBrainAgent(
        obs_dim=obs_dim,
        use_connectome=args.use_connectome,
        N_NEURONS=args.n_neurons,
    )
    ckpt = None
    if args.checkpoint == "best":
        ckpt = CHECKPOINT_DIR / "fly_brain_best.pt"
    elif args.checkpoint == "latest":
        ckpt = CHECKPOINT_DIR / "fly_brain.pt"
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)

    if ckpt and ckpt.exists():
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        agent.set_policy_state(data)
        has_sim = "simulator_state" in data
        print(f"[checkpoint] loaded {ckpt.name} (simulator_state={has_sim})")
    else:
        print(f"[checkpoint] none found ({ckpt}); using fresh initialisation")

    # Mirror the server's policy-health repair so evaluation reflects the
    # model the API would actually serve.
    env0 = FlyTradeEnv(DATA_DIR / f"{args.ticker}_daily.csv", window_size=30)
    dummy = env0._get_observation(0)
    _, probs, _, _ = agent.select_action(dummy, deterministic=False)
    if sum(probs[1:4]) < 0.01:
        if args.repair:
            print("[checkpoint] actor collapsed (Buy probs ~0) -> applying smart priors")
            agent._init_smart_weights()
        else:
            print("[checkpoint] WARNING: actor collapsed (Buy probs ~0); pass --repair to restore priors")
    return agent


def evaluate_ticker(agent, ticker, split, args):
    start, end = SPLITS[split]
    data_path = DATA_DIR / f"{ticker}_daily.csv"
    if not data_path.is_file():
        print(f"  !! {data_path.name} missing, skipping")
        return None

    def build_env():
        return FlyTradeEnv(
            data_path, window_size=30,
            enable_sl_tp=args.sl_tp,
            start_date=start, end_date=end,
        )

    env_probe = build_env()
    if env_probe.n_steps < 5:
        print(f"  !! {ticker}/{split}: only {env_probe.n_steps} steps, skipping")
        return None

    return compare_on_env(
        build_env,
        agent,
        deterministic=args.deterministic,
        seed=args.seed,
        include_legacy=True,
        include_baselines=True,
    )


def print_table(ticker, split, rows):
    start, end = SPLITS[split]
    print(f"\n=== {ticker} [{split}] {start} .. {end} ===")
    header = f"{'strategy':<12} {'return%':>10} {'sharpe':>8} {'maxDD%':>8} {'win%':>7} {'trades':>7}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        print(
            f"{name:<12} {m['total_return_pct']:>10.2f} {m['sharpe']:>8.3f} "
            f"{m['max_drawdown_pct']:>8.2f} {m['win_rate']:>7.1f} {m['trades']:>7d}"
        )


def main():
    p = argparse.ArgumentParser(description="FlyTrader out-of-sample backtest")
    p.add_argument("--ticker", default="AAPL", choices=TICKERS)
    p.add_argument("--all-tickers", action="store_true")
    p.add_argument("--split", default="test", choices=list(SPLITS))
    p.add_argument("--all-splits", action="store_true")
    p.add_argument("--checkpoint", default="best",
                   help="'best', 'latest', a path, or '' for fresh init")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--sl-tp", action="store_true", help="enable stop-loss / take-profit in env")
    p.add_argument("--repair", action="store_true", help="re-apply smart priors if actor collapsed")
    p.add_argument("--use-connectome", dest="use_connectome", action="store_true", default=True)
    p.add_argument("--no-connectome", dest="use_connectome", action="store_false")
    p.add_argument("--n-neurons", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Build once to learn obs_dim (single source of truth).
    probe_env = FlyTradeEnv(DATA_DIR / f"{args.ticker}_daily.csv", window_size=30)
    obs_dim = int(probe_env.observation_space.shape[0])
    print(f"[env] observation dim = {obs_dim}")

    tickers = TICKERS if args.all_tickers else [args.ticker]
    splits = list(SPLITS) if args.all_splits else [args.split]

    agent = load_agent(args, obs_dim)
    for ticker in tickers:
        for split in splits:
            rows = evaluate_ticker(agent, ticker, split, args)
            if rows:
                print_table(ticker, split, rows)


if __name__ == "__main__":
    main()

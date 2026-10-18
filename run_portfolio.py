"""
run_portfolio.py
================
Evaluate multi-asset portfolio strategies on a held-out window:

  * equal-weight buy & hold      (the bar to beat)
  * trend200                     (equal-weight assets above their 200d SMA)
  * momentum top-k rotation      (equal-weight the k strongest by trailing return)
  * the fly (optional --ckpt)    (shared single-asset policy applied per asset)

Usage
-----
    python run_portfolio.py --plan cycle_b --split test
    python run_portfolio.py --ckpt <path> --use-xsection --turnover-penalty 0.002
    python run_portfolio.py --ckpt <path> --alloc-gate positive --alloc-min-k 2 --alloc-max-k 4

The allocation has an optional *breadth gate* (``--alloc-gate``): ``topk`` keeps
only the k most-convicted assets, while ``positive`` keeps exactly the assets the
policy flags with a positive conviction, so the held breadth k is a **policy
output**. ``--alloc-min-k``/``--alloc-max-k`` bound it. The gate used to train a
checkpoint must be passed again here or the allocation will not match.
"""
from __future__ import annotations

import argparse

import numpy as np

from config import SPLIT_PLANS
from portfolio_env import (PortfolioEnv, buy_hold_weights, momentum_weights,
                           run_static_weights, trend200_weights)

DEFAULT_TICKERS = "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN"


def make_env(args) -> PortfolioEnv:
    start, end = SPLIT_PLANS[args.plan][args.split]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    return PortfolioEnv(
        tickers, start_date=start, end_date=end,
        use_xsection=args.use_xsection,
        transaction_cost=args.cost,
        turnover_penalty=args.turnover_penalty,
        indicator_norm=args.indicator_norm,
        beta=args.beta,
        use_portfolio_obs=not args.no_portfolio_obs,
        rebalance_every=args.rebalance_every,
        alloc_topk=args.alloc_topk,
        alloc_gate=getattr(args, "alloc_gate", None),
        alloc_min_k=getattr(args, "alloc_min_k", 0),
        alloc_max_k=getattr(args, "alloc_max_k", None),
    )


def run_fly(env: PortfolioEnv, args) -> dict:
    from fly_brain import FlyBrainAgent
    agent = FlyBrainAgent(obs_dim=env.obs_dim, use_connectome=True,
                          use_xsection=args.use_xsection,
                          use_full_readout=args.use_full_readout)
    import torch
    agent.set_policy_state(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    env.reset()
    done = False
    while not done:
        obs = env.observe()
        actions = []
        for a in range(env.n_assets):
            act, _, _, _ = agent.select_action(obs[a], deterministic=not args.sample)
            actions.append(act)
        _, _, done, _ = env.step(actions)
    return env.metrics()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default=DEFAULT_TICKERS)
    p.add_argument("--plan", default="cycle_b", choices=list(SPLIT_PLANS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--use-xsection", dest="use_xsection", action="store_true")
    p.add_argument("--full-readout", dest="use_full_readout", action="store_true",
                   help="checkpoint was trained with the whole-brain read-out")
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--turnover-penalty", type=float, default=0.0)
    p.add_argument("--indicator-norm", default="per_feature")
    p.add_argument("--beta", type=float, default=2.0,
                   help="softmax conviction temperature for the allocation")
    p.add_argument("--no-portfolio-obs", dest="no_portfolio_obs", action="store_true",
                   help="zero the position block in the observation (breaks the "
                        "position-feedback limit cycle); must match training")
    p.add_argument("--rebalance-every", type=int, default=1,
                   help="rebalance horizon in bars (21 = monthly); the reward is the "
                        "return over that horizon")
    p.add_argument("--ckpt", default=None, help="fly checkpoint to evaluate")
    p.add_argument("--alloc-topk", type=int, default=None,
                   help="restrict the softmax allocation to the k most-convicted "
                        "assets (plus cash), matching the env used in training")
    p.add_argument("--alloc-gate", default=None, choices=["none", "topk", "positive"],
                   help="breadth gate, must match training: 'positive' lets the "
                        "policy decide k (only assets it flags with positive "
                        "conviction compete for the budget)")
    p.add_argument("--alloc-min-k", type=int, default=0,
                   help="floor on the gated breadth (must match training)")
    p.add_argument("--alloc-max-k", type=int, default=None,
                   help="cap on the gated breadth (must match training)")
    p.add_argument("--sample", action="store_true", help="sample (default greedy)")
    p.add_argument("--momentum-lookback", type=int, default=120)
    p.add_argument("--momentum-topk", type=int, default=3)
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"[portfolio] n={len(tickers)} plan={args.plan}/{args.split} "
          f"xsection={args.use_xsection} cost={args.cost} "
          f"turnover_penalty={args.turnover_penalty}")
    print(f"[portfolio] tickers={tickers}")

    results = {}
    env = make_env(args)          # build once: asset-local features are cached
    # When the env itself rebalances on a horizon, every baseline recomputes its
    # rule at each decision (i.e. at each horizon), not every N decisions.
    cad = (lambda k: k) if args.rebalance_every == 1 else (lambda k: 1)
    results["buy&hold"] = run_static_weights(env, lambda e, i: buy_hold_weights(e), cad(1))
    results["trend200-daily"] = run_static_weights(env, trend200_weights, cad(1))
    results["trend200-weekly"] = run_static_weights(env, trend200_weights, cad(5))
    results["mom%d-top%d-m" % (args.momentum_lookback, args.momentum_topk)] = run_static_weights(
        env,
        lambda e, i: momentum_weights(e, i, args.momentum_lookback, args.momentum_topk),
        cad(21))
    if args.ckpt:
        results["fly"] = run_fly(env, args)

    print()
    header = (f"{'strategy':<22} {'ret%':>9} {'maxDD%':>8} {'sharpe':>8} "
              f"{'turnover':>10} {'trades':>8} {'pos':>6}")
    print(header)
    print("-" * len(header))
    for k, v in results.items():
        print(f"{k:<22} {v['total_return_pct']:>9.2f} {v['max_drawdown_pct']:>8.2f} "
              f"{v['sharpe']:>8.2f} {v['turnover']:>10.1f} {v.get('n_trades', 0):>8d} "
              f"{v.get('avg_positions', 0.0):>6.2f}")


if __name__ == "__main__":
    main()

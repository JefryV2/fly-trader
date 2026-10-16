"""
train.py
========
Offline trainer for the FlyTrader fly-brain agent.

Goal: train the fly to *beat buy & hold* out-of-sample.

To make that a well-posed objective, training defaults to the
``reward_mode="excess"`` environment reward: each step the agent is rewarded for
its return *relative to a passive buy & hold benchmark*. Simply holding then
earns ~0 reward, so the only way to score is to time entries/exits better than
the benchmark. The hand-coded "always buy" pressures (regime bonuses, buy-margin
penalty, buy-floor masking) are switched off by default so the fly is free to
learn that cash is sometimes the best position.

The trainer:
  1. trains on the **train** split only,
  2. selects the best checkpoint by **validation** excess return (fly - buy&hold),
  3. reports the **out-of-sample test** split vs buy&hold / momentum / random.

Examples
--------
    python train.py --all-tickers --episodes 40
    python train.py --ticker NVDA --episodes 60 --reward-mode excess
    python train.py --all-tickers --indicator-norm per_feature --max-steps 600
    python train.py --all-tickers --reward-mode shaped      # legacy objective
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_DIR, DATA_DIR, SPLIT_PLANS, TICKERS  # noqa: E402
from env import FlyTradeEnv  # noqa: E402
from fly_brain import FlyBrainAgent  # noqa: E402
from strategy import compare_on_env  # noqa: E402
from trainer import train_episode  # noqa: E402


def plan(args):
    return SPLIT_PLANS[args.plan]


def make_env(ticker, split, args):
    start, end = plan(args)[split]
    return FlyTradeEnv(
        DATA_DIR / f"{ticker}_daily.csv",
        window_size=30,
        enable_sl_tp=args.sl_tp,
        start_date=start,
        end_date=end,
        indicator_norm=args.indicator_norm,
        reward_mode=args.reward_mode,
    )


def usable_tickers(tickers, args):
    """Drop tickers that have too little data for this plan's train slice."""
    ok = []
    for t in tickers:
        try:
            if make_env(t, "train", args).n_steps >= 30:
                ok.append(t)
        except AssertionError:
            continue
    return ok


def evaluate_split(agent, tickers, split, args, *, deterministic=True, n_seeds=1):
    """Fly vs buy&hold on a split, via the shared roll-out path.

    ``deterministic=False`` samples from the learned action distribution and
    averages over ``n_seeds`` seeds. That is the right estimator for *model
    selection*: the greedy (argmax) policy collapses to buy & hold early in
    training, which makes the deterministic excess identically zero and gives no
    signal to choose between checkpoints.
    """
    per_ticker = {}
    for t in tickers:
        flys, bhs, shs, trs, dds, bh_dds = [], [], [], [], [], []
        for k in range(n_seeds):
            def factory(t=t):
                return make_env(t, split, args)
            res = compare_on_env(
                factory, agent, deterministic=deterministic,
                seed=args.seed + 1009 * k,
                include_legacy=False, include_baselines=True,
            )
            flys.append(float(res["fly_honest"]["total_return_pct"]))
            shs.append(float(res["fly_honest"]["sharpe"]))
            dds.append(float(res["fly_honest"]["max_drawdown_pct"]))
            bhs.append(float(res["buy_hold"]["total_return_pct"]))
            bh_dds.append(float(res["buy_hold"]["max_drawdown_pct"]))
            trs.append(int(res["fly_honest"]["trades"]))
        f, b = float(np.mean(flys)), float(np.mean(bhs))
        per_ticker[t] = {
            "fly": f, "buyhold": b, "excess": f - b,
            "sharpe": float(np.mean(shs)), "trades": int(np.mean(trs)),
            "maxdd": float(np.mean(dds)), "bh_maxdd": float(np.mean(bh_dds)),
        }
    fly = [v["fly"] for v in per_ticker.values()]
    bh = [v["buyhold"] for v in per_ticker.values()]
    sh = [v["sharpe"] for v in per_ticker.values()]
    dd = [v["maxdd"] for v in per_ticker.values()]
    bh_dd = [v["bh_maxdd"] for v in per_ticker.values()]
    return {
        "split": split,
        "mean_fly": float(np.mean(fly)) if fly else 0.0,
        "mean_buyhold": float(np.mean(bh)) if bh else 0.0,
        "mean_excess": (float(np.mean(fly)) - float(np.mean(bh))) if fly else 0.0,
        "mean_sharpe": float(np.mean(sh)) if sh else 0.0,
        "mean_maxdd": float(np.mean(dd)) if dd else 0.0,
        "mean_bh_maxdd": float(np.mean(bh_dd)) if bh_dd else 0.0,
        "per_ticker": per_ticker,
    }


def selection_score(val, select_by):
    if select_by == "val_return":
        return val["mean_fly"]
    if select_by == "val_sharpe":
        return val["mean_sharpe"]
    return val["mean_excess"]


def main():
    p = argparse.ArgumentParser(description="Train the FlyTrader agent offline")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--all-tickers", action="store_true")
    p.add_argument("--plan", choices=list(SPLIT_PLANS), default="default",
                   help="'default' (test 2024 bull) or 'bear2022' (test the 2022 drawdown)")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--epsilon-decay", type=float, default=0.99)
    p.add_argument("--reward-mode", choices=["shaped", "excess", "logret"], default="excess")
    p.add_argument("--mask-preset", choices=["honest", "train", "live", "eval"], default="honest",
                   help="masking preset used while *training* (default honest = no forced buying)")
    p.add_argument("--sharpe-blend", type=float, default=0.0,
                   help="blend a rolling-Sharpe term into the env reward (0.0 for clean excess)")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="policy entropy target (lower = sharper/committed policy). "
                        "Default keeps the agent's built-in 1.30.")
    p.add_argument("--select-by", choices=["val_excess", "val_return", "val_sharpe"], default="val_excess")
    p.add_argument("--val-seeds", type=int, default=3,
                   help="seeds averaged for stochastic validation model selection")
    p.add_argument("--no-buy-margin", dest="buy_margin", action="store_false", default=False,
                   help="disable the anti-stall buy-margin penalty (default: disabled)")
    p.add_argument("--buy-margin", dest="buy_margin", action="store_true",
                   help="re-enable the legacy anti-stall buy-margin penalty")
    p.add_argument("--indicator-norm", choices=["block", "per_feature"], default="block")
    p.add_argument("--sl-tp", action="store_true")
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap steps per training episode (0 = full split); speeds up experiments")
    p.add_argument("--use-connectome", dest="use_connectome", action="store_true", default=True)
    p.add_argument("--no-connectome", dest="use_connectome", action="store_false")
    p.add_argument("--n-neurons", type=int, default=1024)
    p.add_argument("--out", default=str(CHECKPOINT_DIR / "fly_brain_trained.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tickers = TICKERS if args.all_tickers else [args.ticker]
    tickers = usable_tickers(tickers, args)
    if not tickers:
        raise SystemExit(f"No tickers have enough data for plan '{args.plan}'")
    probe = make_env(tickers[0], "train", args)
    obs_dim = int(probe.observation_space.shape[0])
    print(f"[train] plan={args.plan} obs_dim={obs_dim} reward_mode={args.reward_mode} "
          f"mask={args.mask_preset} select_by={args.select_by} "
          f"buy_margin={0.35 if args.buy_margin else 0.0}")
    print(f"[train] tickers={tickers}")
    print(f"[train] splits={plan(args)}")

    agent = FlyBrainAgent(obs_dim=obs_dim, use_connectome=args.use_connectome, N_NEURONS=args.n_neurons)
    agent.optimizer.param_groups[0]["lr"] = args.lr
    agent.buy_margin_coef = 0.35 if args.buy_margin else 0.0
    if args.target_entropy is not None:
        agent.target_entropy = float(args.target_entropy)

    max_steps = args.max_steps if args.max_steps > 0 else None
    best_score = -1e9
    for ep in range(args.episodes):
        ticker = tickers[ep % len(tickers)]
        env = make_env(ticker, "train", args)
        eps = max(0.05, args.epsilon * (args.epsilon_decay ** ep))
        result = train_episode(
            agent, env, mask_preset=args.mask_preset, epsilon=eps,
            sharpe_blend=args.sharpe_blend, max_steps=max_steps, seed=args.seed + ep,
        )
        ret = (result["final_value"] - env.initial_cash) / env.initial_cash * 100
        # Benchmark return over the same (possibly truncated) episode prefix.
        bh_ret = 0.0
        try:
            prices = env.data["Close"].values
            i0 = env.window_size - 1
            i1 = min(i0 + result["steps"], len(prices) - 1)
            if prices[i0] > 0:
                bh_ret = (prices[i1] / prices[i0] - 1) * 100
        except Exception:
            pass
        print(f"[ep {ep + 1:>3}/{args.episodes}] {ticker} ret={ret:>8.2f}% "
              f"bh={bh_ret:>8.2f}% exc={ret - bh_ret:>7.2f}% "
              f"trades={result['trades']:>4} loss={result['avg_loss']:>8.4f} eps={eps:.3f}")

        if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
            val = evaluate_split(agent, tickers, "val", args,
                                 deterministic=False, n_seeds=args.val_seeds)
            score = selection_score(val, args.select_by)
            tag = ""
            if score > best_score:
                best_score = score
                torch.save(agent.get_policy_state(), args.out)
                tag = "  <- best (saved)"
            print(f"          VAL fly={val['mean_fly']:>8.2f}% bh={val['mean_buyhold']:>8.2f}% "
                  f"excess={val['mean_excess']:>7.2f}% sharpe={val['mean_sharpe']:.3f}{tag}")

    print(f"\n[train] best validation score ({args.select_by}) = {best_score:.3f}  ->  {args.out}")

    # Final out-of-sample test report (deterministic = greedy deployment policy).
    if Path(args.out).exists():
        agent.set_policy_state(torch.load(args.out, map_location="cpu", weights_only=False))
    test = evaluate_split(agent, tickers, "test", args, deterministic=True, n_seeds=1)
    print("\n=== Out-of-sample TEST (deterministic, honest mask) ===")
    header = (f"{'ticker':<8} {'fly%':>9} {'buyhold%':>10} {'excess%':>9} "
              f"{'sharpe':>8} {'flyMaxDD%':>10} {'bhMaxDD%':>9} {'trades':>7}")
    print(header)
    print("-" * len(header))
    for t, v in test["per_ticker"].items():
        print(f"{t:<8} {v['fly']:>9.2f} {v['buyhold']:>10.2f} {v['excess']:>9.2f} "
              f"{v['sharpe']:>8.3f} {v['maxdd']:>10.2f} {v['bh_maxdd']:>9.2f} {v['trades']:>7d}")
    print("-" * len(header))
    print(f"{'MEAN':<8} {test['mean_fly']:>9.2f} {test['mean_buyhold']:>10.2f} "
          f"{test['mean_excess']:>9.2f} {test['mean_sharpe']:>8.3f} "
          f"{test['mean_maxdd']:>10.2f} {test['mean_bh_maxdd']:>9.2f}")
    verdict = "BEATS" if test["mean_excess"] > 0 else "trails"
    print(f"\n[train] out-of-sample (greedy): fly {verdict} buy & hold by "
          f"{test['mean_excess']:+.2f} pp (mean return); "
          f"max drawdown {test['mean_maxdd']:.2f}% vs buy&hold {test['mean_bh_maxdd']:.2f}%")

    # Stochastic test: samples the learned action distribution (what the policy
    # actually believes) rather than its single greedy mode.
    test_s = evaluate_split(agent, tickers, "test", args,
                            deterministic=False, n_seeds=args.val_seeds)
    print(f"[train] out-of-sample (sampled, {args.val_seeds} seeds): "
          f"fly={test_s['mean_fly']:>8.2f}% bh={test_s['mean_buyhold']:>8.2f}% "
          f"excess={test_s['mean_excess']:>7.2f}% sharpe={test_s['mean_sharpe']:.3f}")


if __name__ == "__main__":
    main()

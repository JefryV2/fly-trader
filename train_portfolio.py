"""
train_portfolio.py
==================
Train the fly to trade a whole **cross-section** (multi-asset portfolio).

Scope expansion over ``train.py``: instead of one ticker per episode, an episode
is one pass over the portfolio timeline, and the shared single-asset policy
allocates capital across every asset at every step.

Why this is a better-posed learning problem than single-asset training
---------------------------------------------------------------------
The portfolio env's reward is *decomposed per asset* and *immediate*::

    r_a = w_a * ret_a - cost_a - turnover_penalty * |dw_a|

so the fly gets a clean, local credit signal ("I raised this asset's weight and
it then rose / fell") instead of the single-asset setting where holding earned
exactly 0 and buy & hold was the global optimum. The softmax allocation puts
every asset in competition for one budget, so the gradient rewards *tilting into
the relatively strongest names* and *raising cash when the whole cross-section
falls* -- i.e. the trend/momentum edge, learned end-to-end.

Selection / honesty
-------------------
* trains on the ``train`` split only (ticker-holdout by default),
* selects the checkpoint by **validation** portfolio return,
* ships the B&H prior unless a checkpoint beats equal-weight buy & hold on
  validation (same guard as ``train.py``),
* reports the **test** split out-of-sample.

Examples
--------
    python train_portfolio.py --plan cycle_b --episodes 200
    python train_portfolio.py --plan cycle_b --episodes 200 --xsection --full-readout
    python train_portfolio.py --plan cycle_b --episodes 200 --init <ckpt> --resume
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))  # noqa: E402

from config import (CHECKPOINT_DIR, SPLIT_PLANS, TICKERS,  # noqa: E402
                    WALKFORWARD_FOLDS, WALKFORWARD_TEST)
from fly_brain import FlyBrainAgent  # noqa: E402
from portfolio_env import PortfolioEnv, buy_hold_weights, run_static_weights  # noqa: E402
from trainer import RewardNormalizer  # noqa: E402

DEFAULT_HOLDOUT = "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN"
BUY_IDX, SELL_IDX = slice(1, 4), slice(4, 7)


def split_dates(plan: str, split: str):
    return SPLIT_PLANS[plan][split]


def fold_dates(args, split: str):
    """Resolve the window for ``split`` taking ``--fold`` into account.

    With ``--fold N`` the train/val windows come from ``WALKFORWARD_FOLDS[N]``
    (validation is the period right after that fold's training window, so the
    first fold validates on the 2018 bear market) and the test window is the
    common full cycle. Without ``--fold`` this is the plain split plan.
    """
    fold = getattr(args, "fold", None)
    if fold is not None:
        tr_s, tr_e, va_s, va_e = WALKFORWARD_FOLDS[fold]
        if split == "train":
            return tr_s, tr_e
        if split == "val":
            return va_s, va_e
        if split == "test":
            return WALKFORWARD_TEST
    return split_dates(args.plan, split)


def make_env(tickers, plan, split, args) -> PortfolioEnv:
    start, end = fold_dates(args, split)
    return PortfolioEnv(
        tickers, start_date=start, end_date=end, window_size=30,
        use_xsection=args.use_xsection, transaction_cost=args.cost,
        turnover_penalty=args.turnover_penalty, indicator_norm=args.indicator_norm,
        beta=args.beta, use_portfolio_obs=not args.no_portfolio_obs,
        rebalance_every=args.rebalance_every,
    )


def masked_probs(probs: np.ndarray, w: float) -> np.ndarray:
    """Honest action mask for a portfolio weight: no buying what you max out,
    no selling what you do not hold."""
    p = np.array(probs, dtype=np.float64)
    if w >= 0.995:
        p[BUY_IDX] = 0.0
    if w <= 0.005:
        p[SELL_IDX] = 0.0
    s = p.sum()
    return p / s if s > 1e-9 else np.array(probs, dtype=np.float64)


def choose_actions(agent, obs, weights, epsilon, rng, deterministic=False):
    actions, probs, values = agent.select_actions_batch(obs, deterministic=False)
    out = []
    for a in range(obs.shape[0]):
        p = masked_probs(probs[a], float(weights[a]))
        if deterministic:
            out.append(int(np.argmax(p)))
            continue
        q = (1.0 - epsilon) * p + epsilon / p.size
        q = masked_probs(q, float(weights[a]))
        out.append(int(rng.choice(q.size, p=q / q.sum())))
    return np.array(out, dtype=int)


def train_episode(agent, env, *, epsilon, train_every, batch_size, min_buffer,
                  rng, max_steps=None) -> dict:
    norm = getattr(agent, "reward_norm", None)
    if norm is None:
        norm = RewardNormalizer()
        agent.reward_norm = norm
    obs = env.reset()
    n = env.n_assets
    limit = max_steps if max_steps else env.n_steps
    losses, step, done = [], 0, False
    feats = agent.encode_obs_batch(obs)
    while not done and step < limit:
        actions = choose_actions(agent, obs, env.weights, epsilon, rng)
        next_obs, rewards, done, _ = env.step(actions)
        next_feats = agent.encode_obs_batch(next_obs) if not done else feats
        for a in range(n):
            r = norm.normalize(float(rewards[a]))
            agent.replay_buffer.append(
                (feats[a], int(actions[a]), r, next_feats[a], bool(done), agent.gamma))
            if len(agent.replay_buffer) > agent.max_buffer_size:
                agent.replay_buffer.pop(0)
        if step % train_every == 0 and len(agent.replay_buffer) >= min_buffer:
            try:
                losses.append(agent.train_online_step(
                    batch_size=min(batch_size, len(agent.replay_buffer))))
            except RuntimeError as e:
                print(f"  [portfolio] skipped online step ({e}); clearing buffer")
                agent.replay_buffer.clear()
                agent.optimizer.state.clear()
        obs, feats = next_obs, next_feats
        step += 1
    return {"steps": step, "final_value": env.value,
            "avg_loss": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def evaluate(agent, tickers, plan, split, args, *, n_seeds=1, max_steps=None) -> dict:
    env = make_env(tickers, plan, split, args)
    bh = run_static_weights(env, buy_hold_weights, 1)
    flys, shs, dds, tos = [], [], [], []
    for k in range(n_seeds):
        rng = np.random.default_rng(args.seed + 1009 * k)
        env.reset()
        limit = max_steps if max_steps else env.n_steps
        done, step = False, 0
        while not done and step < limit:
            actions = choose_actions(agent, env.observe(), env.weights, 0.0, rng,
                                     deterministic=not args.val_sample)
            _, _, done, _ = env.step(actions)
            step += 1
        m = env.metrics()
        flys.append(m["total_return_pct"])
        shs.append(m["sharpe"])
        dds.append(m["max_drawdown_pct"])
        tos.append(m["turnover"] / max(1, env.n_steps))
    return {"fly": float(np.mean(flys)), "buyhold": bh["total_return_pct"],
            "excess": float(np.mean(flys)) - bh["total_return_pct"],
            "sharpe": float(np.mean(shs)), "maxdd": float(np.mean(dds)),
            "bh_maxdd": bh["max_drawdown_pct"],
            "turnover_per_step": float(np.mean(tos))}


def main():
    p = argparse.ArgumentParser(description="Train the fly on a multi-asset portfolio")
    p.add_argument("--tickers", default=None,
                   help="comma-separated training universe (default: config.TICKERS "
                        "minus --holdout)")
    p.add_argument("--holdout", default=DEFAULT_HOLDOUT,
                   help="tickers excluded from training and used for the final OOS report")
    p.add_argument("--plan", default="cycle_b", choices=list(SPLIT_PLANS))
    p.add_argument("--fold", type=int, default=None, choices=range(len(WALKFORWARD_FOLDS)),
                   help="walk-forward fold index. Overrides the plan's train/val "
                        "windows with WALKFORWARD_FOLDS[fold] and tests on the "
                        "common full cycle. Fold 0 validates on the 2018 bear "
                        "market, which is what makes selection able to rank "
                        "concentration (a single bull val window cannot).")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay", type=float, default=0.995)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--target-entropy", type=float, default=0.5,
                   help="policy entropy target (lower = sharper/committed policy). "
                        "The agent default is 1.30 (max is ln 7 = 1.95), which keeps "
                        "the policy near-uniform and unable to commit to a tilt; the "
                        "shipped single-asset runs use 0.5.")
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--turnover-penalty", type=float, default=0.002,
                   help="explicit per-unit-turnover penalty added to the reward; "
                        "this is the direct fix for the churn measured in earlier runs")
    p.add_argument("--beta", type=float, default=2.0,
                   help="temperature of the softmax conviction allocation")
    p.add_argument("--indicator-norm", default="per_feature")
    p.add_argument("--xsection", dest="use_xsection", action="store_true")
    p.add_argument("--full-readout", dest="use_full_readout", action="store_true",
                   help="let the read-out draw on ALL brain regions (AL/KC/CX/DAN/MBON)")
    p.add_argument("--no-portfolio-obs", dest="no_portfolio_obs", action="store_true",
                   help="zero the position block in the observation. The allocation is "
                        "absolute (weight = softmax(conviction)), so the policy does not "
                        "need its own position -- and feeding it back causes a "
                        "buy/sell limit cycle that burns transaction costs.")
    p.add_argument("--rebalance-every", type=int, default=1,
                   help="rebalance horizon in bars. Daily (1) gives an immediate reward "
                        "driven by next-day returns (conditional edge ~0 -> the policy "
                        "chases noise). 21 = monthly, where cross-sectional momentum "
                        "actually pays.")
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-buffer", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps per episode")
    p.add_argument("--val-seeds", type=int, default=2)
    p.add_argument("--val-sample", action="store_true",
                   help="evaluate validation by sampling (default greedy)")
    p.add_argument("--no-guard", dest="guard", action="store_false", default=True,
                   help="ship even if validation does not beat buy & hold (experiments)")
    p.add_argument("--init", default=None, help="warm-start checkpoint")
    p.add_argument("--out", default=str(CHECKPOINT_DIR / "fly_portfolio.pt"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--latest-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    holdout = [t.strip().upper() for t in args.holdout.split(",") if t.strip()]
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [t for t in TICKERS if t not in holdout]

    probe = make_env(tickers, args.plan, "train", args)
    print(f"[portfolio-train] n_assets={probe.n_assets} plan={args.plan} "
          f"steps={probe.n_steps} obs_dim={probe.obs_dim} "
          f"beta={args.beta} turnover_penalty={args.turnover_penalty} "
          f"xsection={args.use_xsection} full_readout={args.use_full_readout}")
    print(f"[portfolio-train] tickers={tickers}")
    print(f"[portfolio-train] holdout ={holdout}")

    agent = FlyBrainAgent(obs_dim=probe.obs_dim, use_connectome=True,
                          use_xsection=args.use_xsection,
                          use_full_readout=args.use_full_readout)
    agent.optimizer.param_groups[0]["lr"] = args.lr
    agent.buy_margin_coef = 0.0
    if args.target_entropy is not None:
        agent.target_entropy = float(args.target_entropy)
    prior_state = copy.deepcopy(agent.get_policy_state())

    latest_path = args.out + ".latest"
    start_ep, saved, best_score, resumed = 0, False, None, False
    if args.init and Path(args.init).exists():
        agent.set_policy_state(torch.load(args.init, map_location="cpu", weights_only=False))
        print(f"[portfolio-train] warm-start from {args.init}")
        prior_state = copy.deepcopy(agent.get_policy_state())
    if args.resume and Path(latest_path).exists():
        ck = torch.load(latest_path, map_location="cpu", weights_only=False)
        agent.set_policy_state(ck.get("state", ck))
        meta = ck.get("meta", {})
        start_ep = int(meta.get("next_ep", 0))
        saved = bool(meta.get("saved", False))
        best_score = meta.get("best_score", None)
        resumed = True
        print(f"[portfolio-train] RESUME ep={start_ep}/{args.episodes} best={best_score}")

    if resumed and best_score is not None:
        guard = float(best_score)
    else:
        base = evaluate(agent, tickers, args.plan, "val", args,
                        n_seeds=1, max_steps=args.max_steps)
        guard = base["buyhold"] if args.guard else -1e9
        print(f"[portfolio-train] val buy&hold={base['buyhold']:.2f}%  "
              f"sharpe={base['sharpe']:.2f}  -> ship threshold={guard:.2f}%")
        best_score = guard
    best_score = float(best_score)

    train_env = make_env(tickers, args.plan, "train", args)
    bh_train = run_static_weights(train_env, buy_hold_weights, 1)
    print(f"[portfolio-train] train-split buy&hold = {bh_train['total_return_pct']:.2f}% "
          f"({train_env.n_steps} steps, {train_env.n_assets} assets)")

    for ep in range(start_ep, args.episodes):
        eps = max(0.03, args.epsilon * (args.epsilon_decay ** ep))
        agent.set_lr(args.lr * (args.lr_decay ** ep))
        rng = np.random.default_rng(args.seed + ep)
        t0 = time.time()
        res = train_episode(agent, train_env, epsilon=eps,
                            train_every=args.train_every, batch_size=args.batch_size,
                            min_buffer=args.min_buffer, rng=rng,
                            max_steps=args.max_steps or None)
        dt = time.time() - t0
        ret = (res["final_value"] - 1.0) * 100.0
        d = getattr(agent, "last_diag", {})
        print(f"[ep {ep + 1:>3}/{args.episodes}] ret={ret:>8.2f}% "
              f"bh={bh_train['total_return_pct']:>8.2f}% "
              f"exc={ret - bh_train['total_return_pct']:>7.2f}% "
              f"loss={res['avg_loss']:>8.4f} eps={eps:.3f} "
              f"H={d.get('entropy', 0.0):.2f} {dt:.1f}s", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
            val = evaluate(agent, tickers, args.plan, "val", args,
                           n_seeds=args.val_seeds, max_steps=args.max_steps)
            tag = ""
            if val["fly"] > best_score:
                best_score = val["fly"]
                torch.save(agent.get_policy_state(), args.out)
                saved = True
                tag = "  <- best (saved)"
            print(f"          VAL fly={val['fly']:>8.2f}% bh={val['buyhold']:>8.2f}% "
                  f"exc={val['excess']:>7.2f}% sharpe={val['sharpe']:.2f} "
                  f"to/step={val['turnover_per_step']:.3f}{tag}")

        if (ep + 1) % args.latest_every == 0 or ep == args.episodes - 1:
            torch.save({"state": agent.get_policy_state(),
                        "meta": {"next_ep": ep + 1, "best_score": best_score,
                                 "saved": saved}}, latest_path)

    if not saved:
        agent.set_policy_state(prior_state)
        torch.save(agent.get_policy_state(), args.out)
        print("[portfolio-train] nothing beat buy & hold on validation -> shipping the prior")

    if Path(latest_path).exists():
        try:
            Path(latest_path).unlink()
        except OSError:
            pass

    if Path(args.out).exists():
        agent.set_policy_state(torch.load(args.out, map_location="cpu", weights_only=False))
    print(f"\n[portfolio-train] TICKER-HOLDOUT test on {len(holdout)} unseen tickers")
    test = evaluate(agent, holdout, args.plan, "test", args, n_seeds=3)
    print(f"  fly={test['fly']:.2f}%  bh={test['buyhold']:.2f}%  "
          f"excess={test['excess']:+.2f}pp  sharpe={test['sharpe']:.2f}  "
          f"maxDD={test['maxdd']:.2f}% (bh {test['bh_maxdd']:.2f}%)  "
          f"turnover/step={test['turnover_per_step']:.2f}")


if __name__ == "__main__":
    main()

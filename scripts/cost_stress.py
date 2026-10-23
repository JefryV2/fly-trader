"""
scripts/cost_stress.py
======================
Stress the deployed portfolio checkpoint at higher transaction costs.

The headline uses a flat 10 bps per side. That is optimistic for a monthly
strategy with 15x annual turnover: it ignores spread widening, slippage and
market impact. This script reruns the *same* checkpoint and the same baselines at
0, 10, 25, 50 and 100 bps, so the reader can see where the edge survives.

It is deliberately an evaluation-only study. The policy is not retrained at each
cost level; a cost-aware retrain is a separate experiment.

Usage
-----
    python scripts/cost_stress.py
    python scripts/cost_stress.py --costs 0,0.001,0.0025,0.005,0.01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SPLIT_PLANS  # noqa: E402
from portfolio_env import (momentum_weights, run_buy_and_hold,
                           run_static_weights)  # noqa: E402

S1_HOLDOUT = "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN"
PLANS = ("cycle", "bear2022", "default")


def _ns(plan: str, cost: float) -> SimpleNamespace:
    return SimpleNamespace(
        plan=plan, fold=None, seed=0, cost=cost, turnover_penalty=0.0,
        indicator_norm="per_feature", beta=2.0, use_xsection=False,
        no_portfolio_obs=False, rebalance_every=21, alloc_topk=None,
        alloc_gate="positive", alloc_min_k=2, alloc_max_k=4,
        no_action_mask=False, val_sample=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Transaction-cost stress test")
    p.add_argument("--ckpt", default=str(ROOT / "web-app" / "checkpoints" / "portfolio_fly.pt"))
    p.add_argument("--tickers", default=S1_HOLDOUT)
    p.add_argument("--costs", default="0,0.001,0.0025,0.005,0.01")
    p.add_argument("--json-out", default=str(ROOT / "reports" / "cost_stress.json"))
    args = p.parse_args()

    import torch
    import train_portfolio as tp
    from fly_brain import FlyBrainAgent

    costs = [float(x) for x in args.costs.split(",") if x.strip()]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    use_connectome = bool(meta.get("use_connectome", True))
    use_full = bool(meta.get("use_full_readout", False))
    proj = int(meta.get("full_readout_proj", 8) or 8)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt),
        "tickers": tickers,
        "costs": costs,
        "cost_units": "one-way cost per unit of turnover",
        "plans": {},
    }
    print(f"[cost] checkpoint={ckpt}")
    print(f"[cost] holdout={tickers} costs={costs}")

    for plan in PLANS:
        ns = _ns(plan, 0.001)
        scalers = tp.build_scalers(tickers, ns)
        probe = tp.make_env(tickers, plan, "train", ns, indicator_stats=scalers)
        agent = FlyBrainAgent(obs_dim=probe.obs_dim,
                              use_connectome=use_connectome,
                              use_full_readout=use_full,
                              full_readout_proj=proj)
        agent.set_policy_state(state)

        rows = []
        for cost in costs:
            ns = _ns(plan, cost)
            fly = tp.evaluate(agent, tickers, plan, "test", ns, n_seeds=1,
                              indicator_stats=scalers)
            env = tp.make_env(tickers, plan, "test", ns, indicator_stats=scalers)
            bh = run_buy_and_hold(env)
            env = tp.make_env(tickers, plan, "test", ns, indicator_stats=scalers)
            mom = run_static_weights(
                env, lambda e, i: momentum_weights(e, i, 120, 3), 1)
            row = {
                "cost": cost,
                "fly": fly,
                "buyhold": bh,
                "momentum": mom,
                "fly_excess_pp": fly["excess"],
                "fly_minus_momentum_pp": fly["fly"] - mom["total_return_pct"],
            }
            rows.append(row)
            print(f"  {plan:<9} cost={cost * 1e4:>4.0f}bps  "
                  f"fly={fly['fly']:>8.2f}%  bh={bh['total_return_pct']:>7.2f}%  "
                  f"excess={fly['excess']:>+7.2f}pp  "
                  f"mom={mom['total_return_pct']:>7.2f}%  "
                  f"dd={fly['maxdd']:>5.2f}%", flush=True)
        report["plans"][plan] = rows

    print("\n=== cost stress: fly excess over buy & hold (pp) ===")
    print(f"{'plan':<10} " + " ".join(f"{c * 1e4:>7.0f}bp" for c in costs))
    for plan in PLANS:
        vals = " ".join(f"{r['fly_excess_pp']:>+8.2f}" for r in report["plans"][plan])
        print(f"{plan:<10} {vals}")
    print("\nThis is evaluation-only: the policy was not retrained at each cost. "
          "It also excludes slippage/impact beyond the flat bps rate.")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[cost] wrote {out}")


if __name__ == "__main__":
    main()

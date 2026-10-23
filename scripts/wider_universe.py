"""
scripts/wider_universe.py
=========================
Compare the narrow 12-ticker portfolio recipe with a wider training universe.

The historical wider arm used 32 tickers and reported that it halved the 2022
bear-market protection (+13.6 pp vs +38.1 pp on a 20-name holdout). That result
was not reproducible from a single committed list, so this script pins both
universes explicitly and evaluates them on the *same* 10 unseen tickers:

  narrow: the 12 tickers used by the shipped ``portfolio_fly.pt`` recipe
  wide:   every long-history ticker except AAPL (short history) and the 10-name
          S1 holdout -> 43 tickers

Both arms use the same distillation, monthly horizon, whole-brain read-out,
positive gate and [2, 4] breadth bounds. The only change is the training
universe, so the comparison is a clean A/B.

It evaluates three windows per checkpoint:
  * cycle test (2021H2–2024) -- the headline holdout
  * bear2022 test (2022)     -- crisis protection
  * default test (2024)      -- bull-year behaviour

Usage
-----
    python scripts/wider_universe.py --n-seeds 3
    python scripts/wider_universe.py --seeds 0,1,2,3,4 --json-out reports/wide.json

The interval is over training seeds on one shared history/universe; it is not a
forecast interval. A wider universe that improves the mean but loses 2022
protection is reported as a negative, not hidden behind the mean.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.seed_sweep import _mean_ci, _split_csv  # noqa: E402

S1_HOLDOUT = "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN"
NARROW = "MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG"
WIDE = ("ABBV,ADBE,AMD,AMGN,AMZN,AXP,BA,BAC,BMY,C,COP,COST,CRM,CSCO,CVX,GE,"
        "GOOGL,GS,HD,HON,IBM,INTC,KO,LLY,MA,MCD,MRK,MS,MSFT,NFLX,NKE,ORCL,PEP,"
        "PFE,PG,QCOM,TMO,TSLA,TXN,UNH,V,VZ,WFC")
EVAL_PLANS = ("cycle", "bear2022", "default")


def _train_cmd(tickers: str, holdout: str, seed: int, out: Path) -> list[str]:
    return [
        sys.executable, str(ROOT / "train_portfolio.py"),
        "--tickers", tickers,
        "--holdout", holdout,
        "--plan", "cycle",
        "--episodes", "0",
        "--rebalance-every", "21",
        "--full-readout",
        "--alloc-gate", "positive",
        "--alloc-min-k", "2",
        "--alloc-max-k", "4",
        "--imitate-episodes", "10",
        "--target-entropy", "0.5",
        "--cost", "0.001",
        "--turnover-penalty", "0.002",
        "--indicator-norm", "per_feature",
        "--beta", "2.0",
        "--val-seeds", "2",
        "--seed", str(seed),
        "--out", str(out),
    ]


def _args_for_plan(plan: str, seed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        plan=plan, fold=None, seed=seed,
        cost=0.001, turnover_penalty=0.0, indicator_norm="per_feature",
        beta=2.0, use_xsection=False, no_portfolio_obs=False,
        rebalance_every=21, alloc_topk=None, alloc_gate="positive",
        alloc_min_k=2, alloc_max_k=4, no_action_mask=False, val_sample=False,
    )


def _evaluate_plan(agent, tickers, plan, scalers) -> dict:
    import train_portfolio as tp
    ns = _args_for_plan(plan)
    return tp.evaluate(agent, tickers, plan, "test", ns, n_seeds=1,
                       indicator_stats=scalers)


def _train_and_eval(arm: str, tickers: str, seeds, temp_dir: Path) -> list[dict]:
    import torch
    from fly_brain import FlyBrainAgent
    import train_portfolio as tp

    holdout = _split_csv(S1_HOLDOUT)
    train = _split_csv(tickers)
    # Fit plan-specific scalers once per arm. Each plan's train window is the
    # walk-forward boundary; using the test window to fit anything would leak.
    scalers_by_plan = {}
    probes = {}
    for plan in EVAL_PLANS:
        ns = _args_for_plan(plan)
        scalers = tp.build_scalers(train + holdout, ns)
        scalers_by_plan[plan] = scalers
        probes[plan] = tp.make_env(train, plan, "train", ns,
                                   indicator_stats=scalers)

    rows = []
    for seed in seeds:
        ckpt = temp_dir / f"{arm}_seed{seed}.pt"
        cmd = _train_cmd(tickers, S1_HOLDOUT, seed, ckpt)
        print(f"\n[wider] {arm} seed {seed}: training {len(train)} assets ...",
              flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0 or not ckpt.exists():
            raise SystemExit(f"{arm} seed {seed}: training failed")
        train_s = time.time() - t0

        agent = FlyBrainAgent(obs_dim=probes["cycle"].obs_dim,
                              use_connectome=True, use_full_readout=True,
                              full_readout_proj=8)
        agent.set_policy_state(torch.load(ckpt, map_location="cpu",
                                          weights_only=False))
        result = {"arm": arm, "seed": seed, "n_train": len(train),
                  "train_seconds": round(train_s, 1)}
        for plan in EVAL_PLANS:
            result[plan] = _evaluate_plan(agent, holdout, plan,
                                           scalers_by_plan[plan])
            m = result[plan]
            print(f"  {arm} seed {seed} {plan:<9} fly={m['fly']:>8.2f}% "
                  f"bh={m['buyhold']:>7.2f}% exc={m['excess']:>+7.2f}pp "
                  f"maxDD={m['maxdd']:>5.2f}% (bh {m['bh_maxdd']:.2f}%)",
                  flush=True)
        rows.append(result)
    return rows


def _summary(rows: list[dict]) -> dict:
    out = {}
    for plan in EVAL_PLANS:
        out[plan] = {
            "fly": _mean_ci([r[plan]["fly"] for r in rows]),
            "buyhold": _mean_ci([r[plan]["buyhold"] for r in rows]),
            "excess": _mean_ci([r[plan]["excess"] for r in rows]),
            "maxdd": _mean_ci([r[plan]["maxdd"] for r in rows]),
            "bh_maxdd": _mean_ci([r[plan]["bh_maxdd"] for r in rows]),
            "positive_seeds": int(sum(1 for r in rows if r[plan]["excess"] > 0)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Narrow vs wide portfolio universe A/B")
    p.add_argument("--seeds", default=None)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--json-out", default=str(ROOT / "reports" / "wider_universe.json"))
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(range(args.n_seeds)))
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "seeds": seeds, "holdout": _split_csv(S1_HOLDOUT),
              "narrow": _split_csv(NARROW), "wide": _split_csv(WIDE),
              "eval_plans": list(EVAL_PLANS), "arms": {}}

    temp_dir = Path(tempfile.mkdtemp(prefix="flytrader_wide_"))
    try:
        for arm, tickers in (("narrow", NARROW), ("wide", WIDE)):
            rows = _train_and_eval(arm, tickers, seeds, temp_dir)
            report["arms"][arm] = {"n_train": len(_split_csv(tickers)),
                                   "rows": rows,
                                   "summary": _summary(rows)}
    finally:
        if not args.keep_checkpoints:
            for f in temp_dir.glob("*.pt"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    print("\n=== narrow vs wide (same 10 unseen tickers) ===")
    header = (f"{'arm':<7} {'plan':<9} {'fly mean':>9} {'excess':>9} "
              f"{'95% CI':>20} {'maxDD':>7} {'bhDD':>7} {'pos':>4}")
    print(header)
    print("-" * len(header))
    for arm in ("narrow", "wide"):
        for plan in EVAL_PLANS:
            s = report["arms"][arm]["summary"][plan]
            fly, ex, dd = s["fly"], s["excess"], s["maxdd"]
            ci = f"[{ex['ci_low']:+.1f}, {ex['ci_high']:+.1f}]"
            print(f"{arm:<7} {plan:<9} {fly['mean']:>9.2f} {ex['mean']:>+9.2f} "
                  f"{ci:>20} {dd['mean']:>7.2f} {s['bh_maxdd']['mean']:>7.2f} "
                  f"{s['positive_seeds']:>2}/{len(seeds)}")
    print("\nCIs are seed intervals, not forecast intervals. A wide-universe gain "
          "that loses 2022 protection is not a recovery.")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[wider] wrote {out}")


if __name__ == "__main__":
    main()

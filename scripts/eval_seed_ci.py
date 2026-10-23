"""
scripts/eval_seed_ci.py
=======================
Evaluate one existing checkpoint over several *evaluation* seeds.

This is the cheap complement to ``scripts/seed_sweep.py``:

* ``seed_sweep.py`` retrains the recipe for each seed and therefore measures
  training-seed sensitivity. It is the right tool, but a 100-episode
  single-asset run takes hours per seed.
* This script loads one already-trained checkpoint and samples the policy over
  several evaluation seeds. It measures path/policy-sampling noise for that
  checkpoint. It does **not** measure training-seed sensitivity.

The distinction is printed and stored in the JSON. Do not present this interval
as a training-seed confidence interval.

Usage
-----
    python scripts/eval_seed_ci.py --n-seeds 10
    python scripts/eval_seed_ci.py --ckpt web-app/checkpoints/fly_brain.pt \
        --tickers TSLA,NVDA,MSFT --holdout AAPL --plan cycle_b --n-seeds 10
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

from scripts.seed_sweep import _mean_ci, _split_csv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation-seed CI for one checkpoint")
    p.add_argument("--ckpt", default=str(ROOT / "web-app" / "checkpoints" / "fly_brain.pt"))
    p.add_argument("--tickers", default="TSLA,NVDA,MSFT",
                   help="training universe used for the train-only scalers")
    p.add_argument("--holdout", default="TSLA,NVDA,MSFT",
                   help="report tickers; the default is the single-asset headline "
                        "universe. Use AAPL only for the demo window.")
    p.add_argument("--plan", default="cycle_b")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indicator-norm", default="per_feature")
    p.add_argument("--reward-mode", default="excess")
    p.add_argument("--greedy", action="store_true",
                   help="use the argmax policy (no distribution/CI variation)")
    p.add_argument("--json-out", default=str(ROOT / "reports" / "single_eval_ci.json"))
    args = p.parse_args()

    import torch
    import train as tr
    from fly_brain import FlyBrainAgent

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    train_tickers = _split_csv(args.tickers)
    report_tickers = _split_csv(args.holdout)
    ns = SimpleNamespace(
        plan=args.plan, seed=args.seed, sl_tp=False,
        indicator_norm=args.indicator_norm, reward_mode=args.reward_mode,
        dd_penalty=0.0, use_xsection=False, val_seeds=1,
        select_by="val_excess",
    )

    scalers = tr.train_scalers(train_tickers + report_tickers, ns)
    report_tickers = [t for t in report_tickers if t in scalers or t == "AAPL"]
    if not report_tickers:
        raise SystemExit("no report tickers have usable history")

    # Fit the architecture to the checkpoint metadata, not to today's defaults.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    use_xsection = bool(meta.get("use_xsection", False))
    use_full = bool(meta.get("use_full_readout", False))
    proj = int(meta.get("full_readout_proj", 8) or 8)
    memory_rho = float(meta.get("memory_rho", 0.0) or 0.0)
    use_connectome = bool(meta.get("use_connectome", True))

    probe = tr.make_env(train_tickers[0], "train", ns,
                        indicator_stats=scalers[train_tickers[0]])
    agent = FlyBrainAgent(
        obs_dim=int(probe.observation_space.shape[0]),
        use_connectome=use_connectome,
        use_xsection=use_xsection,
        use_full_readout=use_full,
        full_readout_proj=proj,
        memory_rho=memory_rho,
    )
    agent.set_policy_state(state)

    print(f"[eval-ci] checkpoint={ckpt}")
    print(f"[eval-ci] plan={args.plan} train={train_tickers} report={report_tickers} "
          f"seeds={args.n_seeds} sampled={not args.greedy}")
    result = tr.evaluate_split(
        agent, report_tickers, "test", ns,
        deterministic=args.greedy, n_seeds=args.n_seeds,
        indicator_stats=scalers, return_per_seed=True,
    )
    per_seed = result.get("per_seed", [])
    summary = {}
    for key in ("fly", "buyhold", "excess", "sharpe", "maxdd", "bh_maxdd"):
        summary[key] = _mean_ci([r[key] for r in per_seed])

    print(f"\n{'seed':>4} {'fly%':>10} {'bh%':>9} {'excess pp':>10} "
          f"{'sharpe':>8} {'maxDD%':>8}")
    print("-" * 55)
    for i, row in enumerate(per_seed):
        print(f"{i:>4} {row['fly']:>10.2f} {row['buyhold']:>9.2f} "
              f"{row['excess']:>10.2f} {row['sharpe']:>8.2f} "
              f"{row['maxdd']:>8.2f}")
    print("-" * 55)
    for key, label in (("fly", "fly"), ("buyhold", "buy & hold"),
                       ("excess", "excess"), ("sharpe", "Sharpe"),
                       ("maxdd", "maxDD"), ("bh_maxdd", "bh maxDD")):
        s = summary[key]
        print(f"{label:<11} mean={s['mean']:>9.2f}  sd={s['sd']:>8.2f}  "
              f"95% CI=[{s['ci_low']:>8.2f}, {s['ci_high']:>8.2f}]")
    print("\nThis interval is over evaluation seeds for ONE checkpoint. "
          "It is not a training-seed interval.")

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt),
        "checkpoint_meta": meta,
        "plan": args.plan,
        "tickers": train_tickers,
        "report_tickers": report_tickers,
        "n_seeds": args.n_seeds,
        "sampled": not args.greedy,
        "interval_scope": "evaluation-seed variability for one checkpoint",
        "summary": summary,
        "per_seed": per_seed,
        "mean_result": {k: v for k, v in result.items() if k != "per_seed"},
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval-ci] wrote {out}")


if __name__ == "__main__":
    main()

"""
scripts/connectome_ablation.py
==============================
Matched A/B for the claim "the connectome block contributes".

The model's read-out consumes spiking features from a 1,024-neuron circuit. Two
circuits are available:

  * ``connectome`` -- the pinned FlyWire-derived sparse matrix;
  * ``random``     -- the deterministic random-weight fallback, same 1024
                      neurons, same region sizes, same seed.

Everything else is held fixed: same portfolio recipe, same 12 training tickers,
same 10 unseen tickers, same distillation budget and seed list. If the connectome
is doing real work, it should beat the random control on the unseen holdout; if
it only adds a fixed random projection, the A/B will show that and the README
should stop claiming otherwise.

The checkpoints record ``use_connectome`` in their metadata, and both evaluation
paths pass the matching flag, so this cannot silently compare two different
architectures.

Usage
-----
    python scripts/connectome_ablation.py --seeds 0,1,2
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.seed_sweep import _mean_ci, _split_csv  # noqa: E402
from scripts.wider_universe import (EVAL_PLANS, NARROW, S1_HOLDOUT,  # noqa: E402
                                    _args_for_plan, _summary)


def _train_cmd(seed: int, out: Path, use_connectome: bool) -> list[str]:
    cmd = [
        sys.executable, str(ROOT / "train_portfolio.py"),
        "--tickers", NARROW,
        "--holdout", S1_HOLDOUT,
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
    if not use_connectome:
        cmd.append("--no-connectome")
    return cmd


def _evaluate(agent, holdout, plan, scalers) -> dict:
    import train_portfolio as tp
    ns = _args_for_plan(plan)
    return tp.evaluate(agent, holdout, plan, "test", ns, n_seeds=1,
                       indicator_stats=scalers)


def main() -> None:
    p = argparse.ArgumentParser(description="Connectome vs random-weight A/B")
    p.add_argument("--seeds", default=None)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--json-out", default=str(ROOT / "reports" / "connectome_ablation.json"))
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(range(args.n_seeds)))
    import torch
    from fly_brain import FlyBrainAgent
    import train_portfolio as tp

    holdout = _split_csv(S1_HOLDOUT)
    train = _split_csv(NARROW)
    scalers_by_plan = {}
    probes = {}
    for plan in EVAL_PLANS:
        ns = _args_for_plan(plan)
        scalers = tp.build_scalers(train + holdout, ns)
        scalers_by_plan[plan] = scalers
        probes[plan] = tp.make_env(train, plan, "train", ns,
                                   indicator_stats=scalers)

    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "seeds": seeds, "holdout": holdout, "tickers": train,
              "eval_plans": list(EVAL_PLANS), "arms": {}}
    temp_dir = Path(tempfile.mkdtemp(prefix="flytrader_connectome_"))
    try:
        for arm, use_connectome in (("connectome", True), ("random", False)):
            rows = []
            for seed in seeds:
                ckpt = temp_dir / f"{arm}_seed{seed}.pt"
                print(f"\n[connectome-ab] {arm} seed {seed}", flush=True)
                t0 = time.time()
                proc = subprocess.run(_train_cmd(seed, ckpt, use_connectome),
                                      cwd=str(ROOT))
                if proc.returncode != 0 or not ckpt.exists():
                    raise SystemExit(f"{arm} seed {seed}: training failed")
                agent = FlyBrainAgent(
                    obs_dim=probes["cycle"].obs_dim,
                    use_connectome=use_connectome,
                    use_full_readout=True, full_readout_proj=8,
                )
                agent.set_policy_state(torch.load(ckpt, map_location="cpu",
                                                  weights_only=False))
                row = {"arm": arm, "seed": seed,
                       "use_connectome": use_connectome,
                       "train_seconds": round(time.time() - t0, 1)}
                for plan in EVAL_PLANS:
                    row[plan] = _evaluate(agent, holdout, plan,
                                          scalers_by_plan[plan])
                    m = row[plan]
                    print(f"  {arm:<10} seed {seed} {plan:<9} "
                          f"fly={m['fly']:>8.2f}% bh={m['buyhold']:>7.2f}% "
                          f"exc={m['excess']:>+7.2f}pp "
                          f"maxDD={m['maxdd']:>5.2f}%", flush=True)
                rows.append(row)
            report["arms"][arm] = {
                "use_connectome": use_connectome,
                "rows": rows,
                "summary": _summary(rows),
            }
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

    print("\n=== connectome vs random-weight control (same 10 unseen tickers) ===")
    header = (f"{'arm':<11} {'plan':<9} {'fly mean':>9} {'excess':>9} "
              f"{'95% CI':>20} {'maxDD':>7} {'pos':>4}")
    print(header)
    print("-" * len(header))
    for arm in ("connectome", "random"):
        for plan in EVAL_PLANS:
            s = report["arms"][arm]["summary"][plan]
            ex = s["excess"]
            ci = f"[{ex['ci_low']:+.1f}, {ex['ci_high']:+.1f}]"
            print(f"{arm:<11} {plan:<9} {s['fly']['mean']:>9.2f} "
                  f"{ex['mean']:>+9.2f} {ci:>20} "
                  f"{s['maxdd']['mean']:>7.2f} "
                  f"{s['positive_seeds']:>2}/{len(seeds)}")

    # Direct paired difference on the headline window: the same seed uses the
    # same read-out initialisation, so this is the cleanest single number.
    paired = []
    for plan in EVAL_PLANS:
        diffs = []
        for a, b in zip(report["arms"]["connectome"]["rows"],
                        report["arms"]["random"]["rows"]):
            assert a["seed"] == b["seed"]
            diffs.append(a[plan]["excess"] - b[plan]["excess"])
        paired.append((plan, _mean_ci(diffs)))
    report["paired_excess_difference"] = {
        plan: ci for plan, ci in paired
    }
    print("\nPaired excess difference (connectome - random), same seed:")
    for plan, ci in paired:
        print(f"  {plan:<9} mean={ci['mean']:+.2f} pp  "
              f"95% CI=[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  n={ci['n']}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[connectome-ab] wrote {out}")


if __name__ == "__main__":
    main()

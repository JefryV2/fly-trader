"""
scripts/seed_sweep.py
=====================
Train a recipe across independent seeds and report honest confidence intervals.

The headline results in the README are point estimates from one checkpoint. A
single seed cannot tell you whether a +345 pp holdout excess is a property of the
recipe or of that particular weight initialisation. This script runs the *same*
recipe under several seeds, evaluates every resulting checkpoint on the same
validation and test windows, and summarises the distribution.

It is deliberately a thin orchestrator:
  * training runs in a subprocess, so torch/NumPy global state cannot leak
    between seeds and a crash in one seed cannot corrupt the next;
  * evaluation uses the same code paths as the trainers (`train_portfolio.
    evaluate` / `train.evaluate_split`), not a reimplementation;
  * the JSON artifact records the exact command, seed, and per-seed metrics.

Usage
-----
    python scripts/seed_sweep.py --mode portfolio --n-seeds 10
    python scripts/seed_sweep.py --mode portfolio --seeds 0,1,2 --sample
    python scripts/seed_sweep.py --mode single --n-seeds 3 \
        --tickers TSLA,NVDA,MSFT --holdout AAPL --plan cycle_b --episodes 100

The 95 % interval is a t interval over training seeds. It describes seed
sensitivity only: all seeds share the same history, universe, and costs, so it is
not a confidence interval for future market performance. The output says so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_DIR, DATA_DIR, TICKERS  # noqa: E402

DEFAULT_TRAIN = "MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG"
DEFAULT_HOLDOUT = "NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN"


def _split_csv(value: str) -> list[str]:
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def _mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean, sample sd, standard error and a t-based interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out = {
        "n": n,
        "mean": float(arr.mean()) if n else 0.0,
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "se": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "ci_low": float(arr.mean()) if n else 0.0,
        "ci_high": float(arr.mean()) if n else 0.0,
    }
    if n > 1:
        try:
            from scipy import stats
            crit = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
        except Exception:  # pragma: no cover - scipy is a project dependency
            # Normal approximation is conservative enough for a fallback; the
            # report always states which one was used.
            crit = 1.959963984540054
            out["interval"] = "normal-approx"
        else:
            out["interval"] = "t"
        out["ci_low"] = out["mean"] - crit * out["se"]
        out["ci_high"] = out["mean"] + crit * out["se"]
    return out


def _checkpoint_path(out_dir: Path, mode: str, seed: int) -> Path:
    return out_dir / f"{mode}_seed{seed}.pt"


def _train_command(args, seed: int, out_path: Path) -> list[str]:
    py = sys.executable
    if args.mode == "portfolio":
        cmd = [
            py, str(ROOT / "train_portfolio.py"),
            "--tickers", args.tickers,
            "--holdout", args.holdout,
            "--plan", args.plan,
            "--episodes", str(args.episodes),
            "--eval-every", str(args.eval_every),
            "--rebalance-every", str(args.rebalance_every),
            "--imitate-episodes", str(args.imitate_episodes),
            "--teacher-lookback", str(args.teacher_lookback),
            "--teacher-topk", str(args.teacher_topk),
            "--target-entropy", str(args.target_entropy),
            "--cost", str(args.cost),
            "--turnover-penalty", str(args.turnover_penalty),
            "--indicator-norm", args.indicator_norm,
            "--beta", str(args.beta),
            "--val-seeds", str(args.val_seeds),
            "--seed", str(seed),
            "--out", str(out_path),
        ]
        if args.full_readout:
            cmd += ["--full-readout",
                    "--full-readout-proj", str(args.full_readout_proj)]
        if args.memory_rho:
            cmd += ["--memory-rho", str(args.memory_rho)]
        if args.alloc_gate:
            cmd += ["--alloc-gate", args.alloc_gate]
        if args.alloc_topk:
            cmd += ["--alloc-topk", str(args.alloc_topk)]
        if args.alloc_min_k:
            cmd += ["--alloc-min-k", str(args.alloc_min_k)]
        if args.alloc_max_k is not None:
            cmd += ["--alloc-max-k", str(args.alloc_max_k)]
        if args.no_portfolio_obs:
            cmd.append("--no-portfolio-obs")
        if args.xsection:
            cmd.append("--xsection")
        if args.no_connectome:
            cmd.append("--no-connectome")
        if args.no_guard:
            cmd.append("--no-guard")
        if args.init:
            cmd += ["--init", args.init]
        return cmd

    cmd = [
        py, str(ROOT / "train.py"),
        "--tickers", args.tickers,
        "--plan", args.plan,
        "--episodes", str(args.episodes),
        "--eval-every", str(args.eval_every),
        "--reward-mode", args.reward_mode,
        "--mask-preset", args.mask_preset,
        "--select-by", args.select_by,
        "--indicator-norm", args.indicator_norm,
        "--val-seeds", str(args.val_seeds),
        "--lr-decay", str(args.lr_decay),
        "--n-step", str(args.n_step),
        "--seed", str(seed),
        "--out", str(out_path),
    ]
    if args.target_entropy is not None:
        cmd += ["--target-entropy", str(args.target_entropy)]
    if args.holdout:
        cmd += ["--holdout", args.holdout]
    if args.full_readout:
        cmd += ["--full-readout",
                "--full-readout-proj", str(args.full_readout_proj)]
    if args.memory_rho:
        cmd += ["--memory-rho", str(args.memory_rho)]
    if args.xsection:
        cmd.append("--xsection")
    if args.no_connectome:
        cmd.append("--no-connectome")
    if args.no_guard:
        cmd.append("--no-guard")
    return cmd


def _eval_portfolio(args, tickers: list[str], holdout: list[str],
                    ckpt: Path) -> dict:
    import torch
    from fly_brain import FlyBrainAgent
    import train_portfolio as tp

    ns = SimpleNamespace(
        plan=args.plan, fold=None, seed=args.seed,
        cost=args.cost, turnover_penalty=args.turnover_penalty,
        indicator_norm=args.indicator_norm, beta=args.beta,
        use_xsection=args.xsection, no_portfolio_obs=args.no_portfolio_obs,
        rebalance_every=args.rebalance_every, alloc_topk=args.alloc_topk,
        alloc_gate=args.alloc_gate, alloc_min_k=args.alloc_min_k,
        alloc_max_k=args.alloc_max_k, no_action_mask=False,
        val_sample=args.sample,
    )
    scalers = tp.build_scalers(tickers + holdout, ns)
    probe = tp.make_env(tickers, args.plan, "train", ns,
                        indicator_stats=scalers)
    agent = FlyBrainAgent(obs_dim=probe.obs_dim,
                          use_connectome=not args.no_connectome,
                          use_xsection=args.xsection,
                          use_full_readout=args.full_readout,
                          full_readout_proj=args.full_readout_proj)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent.set_policy_state(state)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    val = tp.evaluate(agent, tickers, args.plan, "val", ns,
                      n_seeds=args.eval_seeds, indicator_stats=scalers)
    test = tp.evaluate(agent, holdout, args.plan, "test", ns,
                       n_seeds=args.eval_seeds, indicator_stats=scalers)
    return {"validation": val, "test": test,
            "guard_failed": bool(meta.get("guard_failed", False))}


def _flatten_single_eval(result: dict) -> dict:
    """``train.evaluate_split`` returns mean_* keys; the sweep uses one shape."""
    return {
        "fly": result["mean_fly"],
        "buyhold": result["mean_buyhold"],
        "excess": result["mean_excess"],
        "sharpe": result["mean_sharpe"],
        "maxdd": result["mean_maxdd"],
        "bh_maxdd": result["mean_bh_maxdd"],
        "per_ticker": result.get("per_ticker", {}),
    }


def _eval_single(args, tickers: list[str], holdout: list[str],
                 ckpt: Path) -> dict:
    import torch
    from fly_brain import FlyBrainAgent
    import train as tr

    ns = SimpleNamespace(
        plan=args.plan, seed=args.seed, sl_tp=False,
        indicator_norm=args.indicator_norm, reward_mode=args.reward_mode,
        dd_penalty=0.0, use_xsection=args.xsection, val_seeds=args.val_seeds,
        select_by=args.select_by,
    )
    report = holdout or tickers
    scalers = tr.train_scalers(tickers + report, ns)
    report = [t for t in report if t in scalers]
    if not report:
        raise RuntimeError("no report tickers have a train-only scaler")
    probe = tr.make_env(tickers[0], "train", ns,
                        indicator_stats=scalers[tickers[0]])
    agent = FlyBrainAgent(obs_dim=int(probe.observation_space.shape[0]),
                          use_connectome=not args.no_connectome,
                          use_xsection=args.xsection,
                          use_full_readout=args.full_readout,
                          full_readout_proj=args.full_readout_proj,
                          memory_rho=args.memory_rho)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent.set_policy_state(state)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}
    test = tr.evaluate_split(agent, report, "test", ns, deterministic=True,
                             n_seeds=args.eval_seeds,
                             indicator_stats=scalers)
    val = tr.evaluate_split(agent, tickers, "val", ns, deterministic=False,
                            n_seeds=args.val_seeds, indicator_stats=scalers)
    return {"validation": _flatten_single_eval(val),
            "test": _flatten_single_eval(test),
            "guard_failed": bool(meta.get("guard_failed", False))}


def summarize(rows: list[dict], confidence: float = 0.95) -> dict:
    """Aggregate per-seed rows into mean/sd/CI blocks for each metric."""
    metrics = {
        "test_fly": [r["test"]["fly"] for r in rows],
        "test_buyhold": [r["test"]["buyhold"] for r in rows],
        "test_excess": [r["test"]["excess"] for r in rows],
        "test_sharpe": [r["test"]["sharpe"] for r in rows],
        "test_maxdd": [r["test"]["maxdd"] for r in rows],
        "val_fly": [r["validation"]["fly"] for r in rows],
        "val_excess": [r["validation"]["excess"] for r in rows],
    }
    out = {k: _mean_ci(v, confidence) for k, v in metrics.items()}
    positive = int(sum(1 for x in metrics["test_excess"] if x > 0))
    out["test_excess"]["positive_seeds"] = positive
    out["test_excess"]["positive_fraction"] = positive / max(1, len(rows))
    out["guard_failed_seeds"] = int(sum(1 for r in rows if r.get("guard_failed")))
    return out


def _print_report(mode: str, summary: dict, rows: list[dict],
                  confidence: float) -> None:
    print(f"\n=== {mode} seed sweep: {len(rows)} seeds ===")
    print(f"{'seed':>5} {'val fly%':>10} {'test fly%':>11} {'bh%':>9} "
          f"{'excess pp':>10} {'sharpe':>8} {'maxDD%':>8} {'guard':>6}")
    print("-" * 76)
    for r in rows:
        v, t = r["validation"], r["test"]
        print(f"{r['seed']:>5} {v['fly']:>10.2f} {t['fly']:>11.2f} "
              f"{t['buyhold']:>9.2f} {t['excess']:>10.2f} "
              f"{t['sharpe']:>8.2f} {t['maxdd']:>8.2f} "
              f"{'FAIL' if r.get('guard_failed') else 'ok':>6}")
    print("-" * 76)
    for key, label in (("test_fly", "test fly"),
                       ("test_buyhold", "buy & hold"),
                       ("test_excess", "test excess"),
                       ("test_sharpe", "test Sharpe"),
                       ("test_maxdd", "test maxDD"),
                       ("val_excess", "val excess")):
        s = summary[key]
        print(f"{label:<12} mean={s['mean']:>9.2f}  sd={s['sd']:>8.2f}  "
              f"{int(confidence * 100)}% CI=[{s['ci_low']:>8.2f}, {s['ci_high']:>8.2f}]")
    s = summary["test_excess"]
    print(f"positive test seeds: {s['positive_seeds']}/{s['n']} "
          f"({s['positive_fraction']:.0%}); "
          f"guard-rejected seeds: {summary.get('guard_failed_seeds', 0)}/{s['n']}")
    print("The interval is over training seeds on one shared history/universe; "
          "it is not a forecast interval.")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed recipe evaluation")
    p.add_argument("--mode", choices=["portfolio", "single"], default="portfolio")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed list (default: 0..n-1)")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "seed_sweep"))
    p.add_argument("--json-out", default=None)
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep the per-seed .pt files (default: remove them)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the training command for the first seed and exit")

    p.add_argument("--tickers", default=None)
    p.add_argument("--holdout", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0,
                   help="seed used for deterministic evaluation")
    p.add_argument("--eval-seeds", type=int, default=1,
                   help="seeds averaged inside each checkpoint evaluation")
    p.add_argument("--sample", action="store_true",
                   help="sample the policy during evaluation (default: greedy)")

    # portfolio recipe knobs
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--imitate-episodes", type=int, default=10)
    p.add_argument("--teacher-lookback", type=int, default=120)
    p.add_argument("--teacher-topk", type=int, default=3)
    p.add_argument("--target-entropy", type=float, default=0.5)
    p.add_argument("--cost", type=float, default=0.001)
    p.add_argument("--turnover-penalty", type=float, default=0.002)
    p.add_argument("--indicator-norm", default="per_feature")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--full-readout", action="store_true")
    p.add_argument("--no-full-readout", action="store_true",
                   help="turn off the whole-brain read-out (portfolio defaults to on)")
    p.add_argument("--full-readout-proj", type=int, default=8)
    p.add_argument("--memory-rho", type=float, default=0.0)
    p.add_argument("--alloc-topk", type=int, default=None)
    p.add_argument("--alloc-gate", default=None, choices=[None, "none", "topk", "positive"])
    p.add_argument("--alloc-min-k", type=int, default=0)
    p.add_argument("--alloc-max-k", type=int, default=None)
    p.add_argument("--no-portfolio-obs", action="store_true")
    p.add_argument("--xsection", action="store_true")
    p.add_argument("--no-connectome", action="store_true")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--init", default=None)
    p.add_argument("--val-seeds", type=int, default=2)

    # single-asset recipe knobs
    p.add_argument("--reward-mode", default="excess")
    p.add_argument("--mask-preset", default="honest")
    p.add_argument("--select-by", default="val_excess")
    p.add_argument("--lr-decay", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=5)

    args = p.parse_args()

    if args.mode == "portfolio":
        args.tickers = args.tickers or DEFAULT_TRAIN
        args.holdout = args.holdout or DEFAULT_HOLDOUT
        args.plan = args.plan or "cycle"
        args.episodes = 0 if args.episodes is None else args.episodes
        # The shipped portfolio recipe uses the whole-brain read-out; a caller
        # can still select the baseline read-out with --no-full-readout.
        args.full_readout = not args.no_full_readout
        if args.alloc_gate is None and args.alloc_topk is None:
            args.alloc_gate = "positive"
            args.alloc_min_k = 2
            args.alloc_max_k = 4
    else:
        args.tickers = args.tickers or ",".join(TICKERS)
        if args.plan is None:
            args.plan = "cycle_b"
        args.episodes = 40 if args.episodes is None else args.episodes
        if args.no_full_readout:
            args.full_readout = False

    train_tickers = _split_csv(args.tickers)
    holdout = _split_csv(args.holdout or "")
    if args.mode == "portfolio" and not holdout:
        raise SystemExit("portfolio mode needs a non-empty --holdout")
    if args.mode == "single" and not holdout:
        print("[seed-sweep] note: no ticker holdout; the report is the plan's "
              "held-out test split on the training universe (same setup as the "
              "single-asset headline).")

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(range(args.n_seeds)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = Path(args.json_out) if args.json_out else out_dir / f"{args.mode}_summary.json"

    rows = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        ckpt = _checkpoint_path(out_dir, args.mode, seed)
        cmd = _train_command(args, seed, ckpt)
        if args.dry_run:
            print(" ".join(cmd))
            return
        print(f"\n[seed-sweep] seed {seed} ({i + 1}/{len(seeds)}) -> {ckpt}")
        train_t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"training failed for seed {seed} (exit {proc.returncode})")
        if not ckpt.exists():
            raise SystemExit(f"trainer did not write {ckpt}")
        eval_t0 = time.time()
        result = (_eval_portfolio(args, train_tickers, holdout, ckpt)
                  if args.mode == "portfolio"
                  else _eval_single(args, train_tickers, holdout, ckpt))
        result["seed"] = seed
        result["train_seconds"] = round(time.time() - train_t0, 1)
        result["eval_seconds"] = round(time.time() - eval_t0, 1)
        result["command"] = cmd
        rows.append(result)
        print(f"[seed-sweep] seed {seed}: val={result['validation']['fly']:.2f}% "
              f"test={result['test']['fly']:.2f}% "
              f"excess={result['test']['excess']:+.2f}pp "
              f"({result['train_seconds']}s train + {result['eval_seconds']}s eval)")

    summary = summarize(rows, args.confidence)
    payload = {
        "mode": args.mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "reproduce": " ".join(sys.argv),
        "recipe": {
            "tickers": train_tickers,
            "holdout": holdout,
            "plan": args.plan,
            "episodes": args.episodes,
            "imitate_episodes": args.imitate_episodes,
            "rebalance_every": args.rebalance_every,
            "gate": args.alloc_gate,
            "alloc_min_k": args.alloc_min_k,
            "alloc_max_k": args.alloc_max_k,
            "full_readout": bool(args.full_readout),
            "cost": args.cost,
            "turnover_penalty": args.turnover_penalty,
            "sample": bool(args.sample),
            "target_entropy": args.target_entropy,
            "lr_decay": args.lr_decay,
            "n_step": args.n_step,
            "reward_mode": args.reward_mode,
            "mask_preset": args.mask_preset,
            "select_by": args.select_by,
            "val_seeds": args.val_seeds,
            "eval_seeds": args.eval_seeds,
        },
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _print_report(args.mode, summary, rows, args.confidence)
    print(f"\n[seed-sweep] wrote {json_out}")

    if not args.keep_checkpoints:
        for seed in seeds:
            ckpt = _checkpoint_path(out_dir, args.mode, seed)
            try:
                ckpt.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

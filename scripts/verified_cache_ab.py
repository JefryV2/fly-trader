"""
scripts/verified_cache_ab.py
============================
Run the connectome-vs-random A/B with the *auditable pinned rebuild*.

The deployed cache has the pinned 58,421-edge support plus 1,024 diagonal
self-connections, but its values are a legacy transform (see
``reports/connectome_audit.json``). This wrapper temporarily swaps in the
byte-reproducible rebuild, runs ``scripts/connectome_ablation.py`` for the same
seeds, restores the deployed cache in a ``finally`` block, and reports the paired
verified-minus-legacy difference.

Usage
-----
    python scripts/verified_cache_ab.py --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.seed_sweep import _mean_ci  # noqa: E402
from scripts.wider_universe import EVAL_PLANS  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows_by_seed(report: dict, arm: str) -> dict[int, dict]:
    return {int(r["seed"]): r for r in report["arms"][arm]["rows"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Verified vs legacy connectome cache A/B")
    p.add_argument("--verified-cache",
                   default=str(ROOT / "flywire_weights" / "connectome_weights_verified.pkl"))
    p.add_argument("--deployed-cache",
                   default=str(ROOT / "flywire_weights" / "connectome_weights.pkl"))
    p.add_argument("--legacy-report",
                   default=str(ROOT / "reports" / "connectome_ablation.json"))
    p.add_argument("--verified-report",
                   default=str(ROOT / "reports" / "connectome_ablation_verified.json"))
    p.add_argument("--json-out", default=str(ROOT / "reports" / "connectome_cache_ab.json"))
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    verified = Path(args.verified_cache)
    deployed = Path(args.deployed_cache)
    if not verified.exists():
        raise SystemExit(f"verified cache not found: {verified}")
    if not deployed.exists():
        raise SystemExit(f"deployed cache not found: {deployed}")

    before = _sha(deployed)
    backup = Path(tempfile.mkdtemp(prefix="flytrader_cache_backup_")) / "deployed.pkl"
    shutil.copy2(deployed, backup)
    verified_sha = _sha(verified)
    print(f"[cache-ab] deployed sha256={before}")
    print(f"[cache-ab] verified sha256={verified_sha}")
    print(f"[cache-ab] backup={backup}")

    try:
        shutil.copy2(verified, deployed)
        print("[cache-ab] swapped in verified cache; training 2 arms x "
              f"{args.seeds} seeds", flush=True)
        cmd = [
            sys.executable, str(ROOT / "scripts" / "connectome_ablation.py"),
            "--seeds", args.seeds,
            "--json-out", str(args.verified_report),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    finally:
        shutil.copy2(backup, deployed)
        after = _sha(deployed)
        print(f"[cache-ab] restored deployed cache sha256={after}")
        if after != before:
            raise SystemExit("FATAL: deployed cache was not restored byte-for-byte")

    with open(args.verified_report, encoding="utf-8") as f:
        verified_report = json.load(f)
    legacy_path = Path(args.legacy_report)
    if not legacy_path.exists():
        print(f"[cache-ab] WARNING: legacy report missing: {legacy_path}")
        legacy_report = None
    else:
        with open(legacy_path, encoding="utf-8") as f:
            legacy_report = json.load(f)

    combined = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": verified_report["seeds"],
        "deployed_cache_sha256": before,
        "verified_cache_sha256": verified_sha,
        "verified_report": str(args.verified_report),
        "legacy_report": str(legacy_path) if legacy_report else None,
        "verified": verified_report,
        "paired_verified_minus_legacy": {},
        "paired_verified_minus_random": verified_report.get(
            "paired_excess_difference", {}),
    }

    if legacy_report is not None:
        legacy_conn = _rows_by_seed(legacy_report, "connectome")
        verified_conn = _rows_by_seed(verified_report, "connectome")
        legacy_rand = _rows_by_seed(legacy_report, "random")
        verified_rand = _rows_by_seed(verified_report, "random")
        shared = sorted(set(legacy_conn) & set(verified_conn))
        for plan in EVAL_PLANS:
            diff_legacy = [verified_conn[s][plan]["excess"]
                           - legacy_conn[s][plan]["excess"] for s in shared]
            diff_random = [verified_rand[s][plan]["excess"]
                           - legacy_rand[s][plan]["excess"] for s in shared]
            combined["paired_verified_minus_legacy"][plan] = {
                "connectome": _mean_ci(diff_legacy),
                "random_control": _mean_ci(diff_random),
                "seeds": shared,
            }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\n=== verified cache minus legacy cache (paired by seed, excess pp) ===")
    for plan in EVAL_PLANS:
        if plan not in combined["paired_verified_minus_legacy"]:
            continue
        block = combined["paired_verified_minus_legacy"][plan]
        c = block["connectome"]
        r = block["random_control"]
        print(f"{plan:<9} connectome={c['mean']:>+8.2f} pp "
              f"95% CI=[{c['ci_low']:>+8.2f}, {c['ci_high']:>+8.2f}]  "
              f"random-control={r['mean']:>+8.2f} pp")
    print(f"\n[cache-ab] wrote {out}")


if __name__ == "__main__":
    main()

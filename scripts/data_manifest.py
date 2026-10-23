"""Write a reproducible manifest for the local daily-bar cache.

The CSV files are intentionally gitignored (they are large and vendor
specific), so a clean clone cannot know which snapshot produced a backtest.
This script records the SHA-256, byte size and date coverage of every
``data/*_daily.csv`` file, plus row counts for each configured split.  Commit
the resulting JSON next to the code when publishing results.

Usage
-----
    python scripts/data_manifest.py
    python scripts/data_manifest.py --out reports/data_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, SPLIT_PLANS  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("Date", "date", "timestamp", "Datetime", "Price"):
        if column in frame.columns:
            try:
                parsed = pd.to_datetime(frame[column], format="mixed", errors="coerce")
            except TypeError:  # pandas < 2.0
                parsed = pd.to_datetime(frame[column], errors="coerce")
            return parsed.dt.strftime("%Y-%m-%d")
    raise ValueError("no date column found")


def describe_file(path: Path) -> dict:
    frame = pd.read_csv(path)
    dates = _date_series(frame)
    valid = dates.dropna()
    splits = {}
    for plan, windows in SPLIT_PLANS.items():
        splits[plan] = {}
        for split, (start, end) in windows.items():
            mask = (dates >= start) & (dates <= end)
            selected = dates[mask].dropna()
            splits[plan][split] = {
                "rows": int(mask.sum()),
                "first": str(selected.iloc[0]) if not selected.empty else None,
                "last": str(selected.iloc[-1]) if not selected.empty else None,
            }
    return {
        "ticker": path.stem.replace("_daily", ""),
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "first_date": str(valid.iloc[0]) if not valid.empty else None,
        "last_date": str(valid.iloc[-1]) if not valid.empty else None,
        "splits": splits,
    }


def build_manifest(data_dir: Path = DATA_DIR) -> dict:
    files = sorted(data_dir.glob("*_daily.csv"))
    try:
        display_dir = str(data_dir.relative_to(ROOT))
    except ValueError:
        display_dir = str(data_dir)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": display_dir,
        "files": [describe_file(path) for path in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "data_manifest.json")
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {args.out} ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()

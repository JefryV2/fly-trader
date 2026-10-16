"""
scripts/rebuild_connectome.py
=============================
Regenerate ``flywire_weights/connectome_weights.pkl`` from the raw FlyWire v783
files with recorded provenance (source hashes, parameters, nnz).

The old cached pickle was produced by a code path that no longer exists and
references undefined variables, so its contents could not be reproduced or
audited.  This script makes the build reproducible.

Usage
-----
    python scripts/rebuild_connectome.py                       # 1024 neurons, degree-normalised
    python scripts/rebuild_connectome.py --neurons 4096 --normalize degree
    python scripts/rebuild_connectome.py --normalize max --clip 0.5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_brain import build_sparse_weights_from_connectome, save_connectome_weights  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Rebuild FlyWire connectome weights with provenance")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--clip", type=float, default=0.45)
    p.add_argument("--normalize", choices=["degree", "max", "none"], default="degree")
    p.add_argument("--reciprocal", type=float, default=0.0,
                   help="weak reciprocal edge scale (0 disables)")
    p.add_argument("--data-dir", default=str(ROOT / "flywire_data"))
    p.add_argument("--out-dir", default=str(ROOT / "flywire_weights"))
    args = p.parse_args()

    comp = Path(args.data_dir) / "2025_Completeness_783.csv"
    conn = Path(args.data_dir) / "2025_Connectivity_783.parquet"
    if not comp.is_file() or not conn.is_file():
        raise SystemExit(f"Missing FlyWire data in {args.data_dir} (need Completeness csv + Connectivity parquet)")

    W, mapping = build_sparse_weights_from_connectome(
        str(comp), str(conn),
        target_neurons=args.neurons,
        clip=args.clip,
        normalize=args.normalize,
        reciprocal_scale=args.reciprocal,
    )
    if W is None:
        raise SystemExit("Build produced no synapses; nothing saved.")

    prov = dict(mapping["provenance"])
    prov["built_at"] = dt.datetime.now().isoformat(timespec="seconds")
    mapping["provenance"] = prov

    out_dir = Path(args.out_dir)
    save_connectome_weights(
        {"W": W, "mapping": mapping, "provenance": prov, "synapse_count": W.nnz},
        str(out_dir),
    )
    print("\nProvenance:")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()

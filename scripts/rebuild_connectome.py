"""
scripts/rebuild_connectome.py
=============================
Regenerate the FlyWire connectome cache from the raw v783 files, with recorded
provenance (source hashes, selection, parameters, nnz).

Why a *pinned selection* matters
--------------------------------
The builder's default selection is file order (rows 0..N-1).  That is arbitrary
and, for this dataset, nearly disconnected: at N=1024 it yields ~1,130 synapses,
while the deployed 1,024-neuron circuit has 58,421.  A rebuild therefore used to
produce a different brain from the one the checkpoints were trained on.

The deployed selection is now pinned in
``flywire_weights/connectome_selection_1024.json`` (recovered from the shipped
pickle; it reproduces the 58,421 raw synapses exactly).  A 4,096-neuron
degree-ranked core is pinned alongside it.  When a matching selection file
exists it is used automatically.

Usage
-----
    python scripts/rebuild_connectome.py                       # 1024, pinned selection
    python scripts/rebuild_connectome.py --neurons 4096        # pinned dense core
    python scripts/rebuild_connectome.py --selection none      # legacy file-order rows
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

from fly_brain import (build_sparse_weights_from_connectome,  # noqa: E402
                       load_connectome_selection, save_connectome_weights)


def _read_selection(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"] if isinstance(payload, dict) else payload
    return [int(r) for r in rows]


def main():
    p = argparse.ArgumentParser(description="Rebuild FlyWire connectome weights with provenance")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--clip", type=float, default=0.45)
    p.add_argument("--normalize", choices=["degree", "max", "none"], default="degree")
    p.add_argument("--reciprocal", type=float, default=0.0,
                   help="weak reciprocal edge scale (0 disables)")
    p.add_argument("--selection", default=None,
                   help="JSON row list, or 'none' for legacy file-order rows "
                        "(default: auto-detect connectome_selection_<N>.json)")
    p.add_argument("--data-dir", default=str(ROOT / "flywire_data"))
    p.add_argument("--out-dir", default=str(ROOT / "flywire_weights"))
    p.add_argument("--out-name", default=None,
                   help="cache filename (default: connectome_weights_<N>.pkl)")
    args = p.parse_args()

    comp = Path(args.data_dir) / "2025_Completeness_783.csv"
    conn = Path(args.data_dir) / "2025_Connectivity_783.parquet"
    if not comp.is_file() or not conn.is_file():
        raise SystemExit(f"Missing FlyWire data in {args.data_dir} (need Completeness csv + Connectivity parquet)")

    if args.selection == "none":
        selection = None
        print("Selection: none (file-order rows — not the deployed circuit)")
    elif args.selection:
        selection = _read_selection(Path(args.selection))
        print(f"Selection: pinned from {args.selection}")
    else:
        selection = load_connectome_selection(args.out_dir, args.neurons)
        if selection is None:
            print("Selection: no pinned file found; falling back to file order")
        else:
            print(f"Selection: pinned from connectome_selection_{args.neurons}.json")

    W, mapping = build_sparse_weights_from_connectome(
        str(comp), str(conn),
        target_neurons=args.neurons,
        clip=args.clip,
        normalize=args.normalize,
        reciprocal_scale=args.reciprocal,
        selection=selection,
    )
    if W is None:
        raise SystemExit("Build produced no synapses; nothing saved.")

    prov = dict(mapping["provenance"])
    prov["built_at"] = dt.datetime.now().isoformat(timespec="seconds")
    mapping["provenance"] = prov

    out_dir = Path(args.out_dir)
    out_name = args.out_name or f"connectome_weights_{args.neurons}.pkl"
    save_connectome_weights(
        {"W": W, "mapping": mapping, "provenance": prov,
         "synapse_count": W.nnz, "neurons": list(mapping["row_to_model"].keys())},
        str(out_dir), out_name,
    )
    print("\nProvenance:")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()

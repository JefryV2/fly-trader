"""Connectome build invariants.

The deployed 1,024-neuron circuit is defined by a pinned row selection, not by
file order.  File order (rows 0..N-1) is nearly disconnected at this size:
1,130 synapses versus 58,421 for the shipped selection.  These tests skip when
the raw FlyWire files are absent (they are gitignored, ~100 MB).
"""
import json
from pathlib import Path

import pytest

from fly_brain import build_sparse_weights_from_connectome, load_connectome_selection

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "flywire_data"
WEIGHTS = ROOT / "flywire_weights"
COMP = DATA / "2025_Completeness_783.csv"
CONN = DATA / "2025_Connectivity_783.parquet"

pytestmark = pytest.mark.skipif(
    not (COMP.is_file() and CONN.is_file()),
    reason="raw FlyWire files not present (gitignored)",
)


def test_pinned_1024_selection_is_present_and_dense():
    sel = load_connectome_selection(str(WEIGHTS), 1024)
    assert sel is not None, "connectome_selection_1024.json must be committed"
    assert len(sel) == 1024
    assert len(set(sel)) == 1024

    W, mapping = build_sparse_weights_from_connectome(
        str(COMP), str(CONN), target_neurons=1024, selection=sel)
    assert W.shape == (1024, 1024)
    # 58,421 raw synapses within the selection; csr drops nothing here
    assert W.nnz == 58_421
    assert mapping["provenance"]["selection_mode"] == "pinned"
    assert mapping["provenance"]["cell_type_verified"] is False


def test_selection_file_metadata_matches_rows():
    payload = json.loads((WEIGHTS / "connectome_selection_1024.json").read_text())
    assert payload["n"] == 1024
    assert payload["mode"] == "shipped_dense_circuit"
    assert len(payload["rows"]) == 1024

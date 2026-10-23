"""Integration checks for the FastAPI server's safety-critical edges.

Importing the server constructs the live environment and loads the deployment
checkpoint, so these tests are module-scoped and intentionally small. The
startup path is read-only with respect to checkpoint files.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def server():
    if not (ROOT / "data" / "AAPL_daily.csv").is_file():
        pytest.skip("data/AAPL_daily.csv is required for server integration tests")
    path = ROOT / "web-app" / "server.py"
    spec = importlib.util.spec_from_file_location("flytrader_server_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_health_is_cheap_and_reports_session(server):
    payload = server.health()
    assert payload["status"] == "ok"
    assert payload["mode"] in {"single", "portfolio"}
    assert payload["tickers"] == list(server.SESSION["tickers"])


def test_checkpoint_info_is_read_only_metadata(server):
    info = server.checkpoint_info()
    assert "single" in info and "portfolio" in info
    assert "path" in info["single"]
    assert "legacy" in info["single"]
    assert "available" in info["portfolio"]
    assert info["live"]["path"].endswith("fly_brain_live.pt")


def test_serving_scaler_follows_checkpoint_metadata(server):
    meta = server.LOADED_CHECKPOINT_INFO.get("meta", {})
    if meta.get("train_window"):
        assert tuple(meta["train_window"]) == tuple(server.SINGLE_TRAIN_WINDOW)
    # AAPL predates the cycle_b window; the fallback must still produce a
    # usable scaler instead of breaking the default dashboard session.
    assert server._train_indicator_stats("AAPL") is not None


def test_dashboard_checkpoint_is_a_sidecar(server, monkeypatch):
    seen = {}

    def fake_save(_obj, path):
        seen["path"] = str(path)

    monkeypatch.setattr(server.torch, "save", fake_save)
    server._save_checkpoint()
    assert seen["path"].endswith("fly_brain_live.pt")
    assert seen["path"] != str(server.CHECKPOINT_PATH)


def test_promote_requires_a_live_sidecar(server, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "LIVE_CHECKPOINT_PATH", tmp_path / "missing.pt")
    with pytest.raises(HTTPException) as exc:
        server.promote_live_checkpoint()
    assert exc.value.status_code == 400


def test_promote_copies_only_a_validation_winner(server, monkeypatch, tmp_path):
    live = tmp_path / "live.pt"
    deployment = tmp_path / "deployment.pt"
    live.write_bytes(b"candidate")
    deployment.write_bytes(b"deployed")
    monkeypatch.setattr(server, "LIVE_CHECKPOINT_PATH", live)
    monkeypatch.setattr(server, "CHECKPOINT_PATH", deployment)
    monkeypatch.setattr(server.torch, "load", lambda *a, **k: {"meta": {}})
    monkeypatch.setattr(server, "_agent_from_state", lambda state: object())
    monkeypatch.setattr(server.brain, "set_policy_state", lambda state: None)
    monkeypatch.setattr(server, "copy2",
                        lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()))
    scores = iter([
        {"mean_fly_return_pct": 20.0, "mean_buyhold_return_pct": 10.0},
        {"mean_fly_return_pct": 15.0, "mean_buyhold_return_pct": 10.0},
    ])
    monkeypatch.setattr(server, "_validation_stats", lambda agent: next(scores))

    result = server.promote_live_checkpoint()
    assert result["promoted"] is True
    assert deployment.read_bytes() == b"candidate"
    assert (tmp_path / "deployment.pre_promote.pt").read_bytes() == b"deployed"

    # A candidate that merely beats the old deployment but not buy & hold
    # must not be promoted.
    live.write_bytes(b"loser")
    deployment.write_bytes(b"deployed")
    scores = iter([
        {"mean_fly_return_pct": 5.0, "mean_buyhold_return_pct": 10.0},
        {"mean_fly_return_pct": 4.0, "mean_buyhold_return_pct": 10.0},
    ])
    monkeypatch.setattr(server, "_validation_stats", lambda agent: next(scores))
    result = server.promote_live_checkpoint()
    assert result["promoted"] is False
    assert deployment.read_bytes() == b"deployed"


def test_ticker_path_rejects_unknown_symbols(server):
    assert server._data_path_for("aapl").name == "AAPL_daily.csv"
    with pytest.raises(HTTPException) as exc:
        server._data_path_for("NOT_A_TICKER")
    assert exc.value.status_code == 404


def test_config_rejects_unknown_ticker_without_mutating_session(server):
    before = dict(server.SESSION)
    payload = server.SessionPayload(tickers=["NOT_A_TICKER"])
    with pytest.raises(HTTPException) as exc:
        server.set_config(payload)
    assert exc.value.status_code == 400
    assert dict(server.SESSION) == before


def test_train_batch_rejects_non_train_split(server):
    payload = server.TrainBatchPayload(episodes=1, split="test")
    with pytest.raises(HTTPException) as exc:
        server.train_batch(payload)
    assert exc.value.status_code == 400
    assert "validation/test" in str(exc.value.detail)


def test_policy_chips_resolve_server_side(server):
    # Portfolio mode owns allocation; baseline chips are disabled there.
    if len(server.SESSION["tickers"]) > 1:
        assert server._policy_action("momentum") == -1
        return
    for name in ("momentum", "mean-revert", "random"):
        assert 0 <= server._policy_action(name) <= 6
    assert server._policy_action("brain") == -1
    assert server._policy_action("not-a-policy") == -1

"""Wide read-out and causal memory: shapes, determinism, episode reset."""
import numpy as np
import pytest

from fly_brain import FlyBrainAgent


def _agent(**kw):
    return FlyBrainAgent(use_connectome=False, N_NEURONS=1024, **kw)


def _obs(seed=0):
    return np.random.default_rng(seed).standard_normal(398).astype(np.float32)


def test_feature_dim_matches_encoder():
    a = _agent(use_full_readout=True, full_readout_proj=64, memory_rho=0.9)
    feat, _ = a.encode_obs(_obs())
    # 64 MBON + (5 region means + 64 projection) + 64 trace + 24 sensory
    assert feat.shape[0] == 64 + 5 + 64 + 64 + 24
    assert feat.shape[0] == a.feature_encoder[0].in_features


def test_memory_is_causal_and_resets():
    a = _agent(memory_rho=0.9)
    obs = _obs()
    f1 = a.encode_feature(obs)
    f2 = a.encode_feature(obs)
    assert not np.array_equal(f1, f2), "trace did not advance"

    a.reset_memory()
    f3 = a.encode_feature(obs)
    assert np.array_equal(f1, f3), "reset_memory did not restore the start state"


def test_no_memory_is_stateless():
    a = _agent()
    obs = _obs()
    assert np.array_equal(a.encode_feature(obs), a.encode_feature(obs))


def test_select_action_with_feat_does_not_advance_memory():
    a = _agent(memory_rho=0.9)
    obs = _obs()
    feat = a.encode_feature(obs)
    trace_before = a._trace.copy()
    a.select_action(obs, feat=feat)
    assert np.array_equal(a._trace, trace_before)


def test_memory_rejects_cross_section():
    a = _agent(memory_rho=0.9)
    with pytest.raises(ValueError, match="single-asset"):
        a.encode_obs_batch(np.zeros((3, 398), dtype=np.float32))


def test_wide_projection_sees_non_mbon_neurons():
    a = _agent(use_full_readout=True, full_readout_proj=64)
    obs = _obs()
    feat, spikes = a.encode_obs(obs)
    proj = feat[64 + 5:64 + 5 + 64]          # region means, then projection
    expected = spikes @ a.readout_proj.detach().cpu().numpy()
    assert np.allclose(proj, expected, atol=1e-5)

"""Batched encoding must be exact, not approximate.

The spiking encoding is stateless and deterministic (noise_std == 0), so
settling B independent observations in parallel is mathematically the same
operation as B sequential passes.  These tests pin that equivalence; they use
the random-weight fallback so they do not need the gitignored FlyWire files.
"""
import numpy as np

from fly_brain import MaleCNS1024Simulator


def _sim():
    return MaleCNS1024Simulator(use_connectome=False, N_NEURONS=1024)


def test_encode_batch_matches_encode_exactly():
    sim = _sim()
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((8, 398)).astype(np.float32)

    batched = sim.encode_batch(obs)
    looped = sim.encode(obs)

    assert batched.shape == (8, 1024)
    assert batched.dtype == np.float32
    assert np.array_equal(batched, looped), "parallel encoding changed the spikes"


def test_encode_batch_single_observation():
    sim = _sim()
    rng = np.random.default_rng(1)
    obs = rng.standard_normal(398).astype(np.float32)

    batched = sim.encode_batch(obs)
    looped = sim.encode(obs)

    assert batched.shape == (1, 1024)
    assert np.array_equal(batched, looped)

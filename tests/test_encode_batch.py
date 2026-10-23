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


def test_encode_batch_publishes_aggregate_telemetry():
    """Portfolio encoding must update the state the dashboard reads.

    ``encode_batch`` is stateless for feature extraction, but the live
    portfolio path reads ``simulator.V`` / ``last_spikes`` after every
    rebalance. They used to remain at the checkpoint values forever.
    """
    sim = _sim()
    rng = np.random.default_rng(2)
    obs = rng.standard_normal((8, 398)).astype(np.float32)

    spikes = sim.encode_batch(obs)

    assert sim.last_spikes.shape == (1024,)
    assert sim.V.shape == (1024,)
    assert np.isfinite(sim.V).all()
    assert np.allclose(sim.last_spikes, spikes.mean(axis=0), atol=1e-7)
    assert np.isfinite(sim._inh)

"""Tests for the learning core: stability, n-step returns and drift tolerance.

These guard the fixes that make *more training keep helping*: a bounded
target-critic TD target, n-step transitions in the replay buffer, a persistent
reward normaliser, and checkpoints that degrade gracefully across architecture
changes instead of crashing.
"""
import numpy as np
import pytest
import torch

from config import DATA_DIR, OBS_DIM
from env import FlyTradeEnv
from fly_brain import FlyBrainAgent
from trainer import train_episode, RewardNormalizer


AAPL = DATA_DIR / "AAPL_daily.csv"


@pytest.fixture(scope="module")
def agent():
    return FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True)


def _env():
    return FlyTradeEnv(AAPL, window_size=30,
                       start_date="2022-01-03", end_date="2022-06-30",
                       indicator_norm="per_feature", reward_mode="excess")


def test_target_critic_is_frozen(agent):
    assert hasattr(agent, "critic_target")
    assert all(not p.requires_grad for p in agent.critic_target.parameters())
    # It starts as a copy of the critic and diverges only via Polyak updates.
    for tp, op in zip(agent.critic_target.parameters(), agent.critic.parameters()):
        assert torch.allclose(tp, op)


def test_training_keeps_targets_and_values_bounded(agent):
    """The old one-step online bootstrap diverged (loss ran to -52). The target
    critic + clipped targets must keep values/TD targets bounded on adversarial
    inputs."""
    agent.train()
    n_feat = agent.feature_encoder[0].in_features
    rng = np.random.default_rng(0)
    agent.replay_buffer = [
        (rng.standard_normal(n_feat).astype(np.float32),
         int(rng.integers(0, 7)),
         float(rng.normal(0, 5)),          # deliberately large rewards
         rng.standard_normal(n_feat).astype(np.float32),
         0.0,
         float(agent.gamma ** 5))
        for _ in range(400)
    ]
    for _ in range(300):
        agent.train_online_step(batch_size=32)
    diag = agent.last_diag
    assert diag, "diagnostics should be populated after an update"
    assert diag["target_abs_mean"] <= agent.target_clip + 1e-3
    assert diag["value_abs_mean"] < 50.0
    assert np.isfinite(diag["critic_loss"]) and diag["critic_loss"] < 100.0


def test_train_episode_emits_nstep_transitions(agent):
    agent.replay_buffer = []
    if hasattr(agent, "reward_norm"):
        del agent.reward_norm
    env = _env()
    train_episode(agent, env, mask_preset="honest", n_step=5,
                  train_every=10_000, max_steps=60, seed=0)
    assert agent.replay_buffer, "episode should have produced transitions"
    for entry in agent.replay_buffer:
        assert len(entry) == 6, "n-step entries must carry gamma**n"
        assert np.isfinite(entry[2])
    full = [e for e in agent.replay_buffer if abs(e[5] - agent.gamma ** 5) < 1e-6]
    assert full, "expected at least one full-horizon (n-step) transition"
    # Partial (flushed) transitions are shorter horizons -> larger gamma**n.
    assert any(e[5] > agent.gamma ** 5 for e in agent.replay_buffer)


def test_reward_normalizer_persists_across_episodes(agent):
    if hasattr(agent, "reward_norm"):
        del agent.reward_norm
    train_episode(agent, _env(), mask_preset="honest", n_step=3,
                  train_every=10_000, max_steps=40, seed=0)
    norm1 = agent.reward_norm
    count_after_one = norm1.count
    train_episode(agent, _env(), mask_preset="honest", n_step=3,
                  train_every=10_000, max_steps=40, seed=1)
    assert agent.reward_norm is norm1, "normaliser must persist across episodes"
    assert norm1.count > count_after_one


def test_set_policy_state_tolerates_shape_drift():
    src = FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True)
    state = src.get_policy_state()
    # Corrupt one tensor's shape, as an older architecture would.
    state["model_state"]["feature_encoder.0.weight"] = torch.zeros(3, 3)
    fresh = FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True)
    fresh.set_policy_state(state)  # must not raise
    # The healthy tensors were still loaded from the source checkpoint.
    assert torch.allclose(fresh.critic[0].weight, src.critic[0].weight, atol=1e-6)


def test_reward_normalizer_clips_extremes():
    n = RewardNormalizer()
    out = [n.normalize(x) for x in [1.0] * 50 + [1e6]]
    assert max(abs(v) for v in out) <= 5.0

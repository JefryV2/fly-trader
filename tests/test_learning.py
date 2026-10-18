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


def _obs_batch(n=24):
    env = FlyTradeEnv(AAPL, window_size=30,
                      start_date="2022-01-03", end_date="2023-06-30",
                      indicator_norm="per_feature")
    obs, _ = env.reset(seed=0)
    out = []
    for _ in range(n):
        out.append(obs)
        obs, _, done, _, _ = env.step(0)
        if done:
            break
    return np.stack(out)


def test_full_readout_connects_all_brain_regions():
    """The whole-brain read-out must add one mean per region plus a projection of
    the full spike vector, and every region must actually vary with the input."""
    base = FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True)
    full = FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True, use_full_readout=True)
    assert full.full_readout_extra == len(full.REGION_NAMES) + full.full_readout_proj
    assert (full.feature_encoder[0].in_features
            == base.feature_encoder[0].in_features + full.full_readout_extra)
    # Default agent is unchanged (old checkpoints still load bit-for-bit).
    assert base.feature_encoder[0].in_features == 88

    feats = np.stack([full.encode_feature(o) for o in _obs_batch(24)])
    assert feats.shape[1] == full.feature_encoder[0].in_features
    mbon = slice(0, 64)
    means = slice(64, 64 + len(full.REGION_NAMES))
    proj = slice(means.stop, means.stop + full.full_readout_proj)
    assert feats[:, mbon].std(axis=0).mean() > 0.0     # MBON still live
    assert feats[:, means].std(axis=0).min() > 0.0     # every region varies
    assert feats[:, proj].std(axis=0).mean() > 0.0     # projection varies


def test_batched_features_match_single_observation():
    """The portfolio env scores a whole cross-section per step; the batched path
    must produce exactly the same features as the one-at-a-time path."""
    agent = FlyBrainAgent(obs_dim=OBS_DIM, use_connectome=True, use_full_readout=True)
    obs = _obs_batch(8)
    batched = agent.encode_obs_batch(obs)
    single = np.stack([agent.encode_feature(o) for o in obs])
    assert np.allclose(batched, single, atol=1e-5)
    actions, probs, values = agent.select_actions_batch(obs, deterministic=True)
    assert actions.shape == (obs.shape[0],)
    assert probs.shape == (obs.shape[0], 7)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

"""Invariant tests for the multi-asset portfolio environment."""
import numpy as np
import pytest

from portfolio_env import (DESIRES, PortfolioEnv, buy_hold_weights,
                           momentum_weights, run_static_weights, trend200_weights)

TICKERS = ["AAPL", "TSLA", "MSFT"]
WINDOW = {"start_date": "2022-01-01", "end_date": "2022-12-31"}


def make(**kw):
    kw.setdefault("window_size", 30)
    return PortfolioEnv(TICKERS, **WINDOW, **kw)


def test_shapes_and_dimensions():
    env = make()
    assert env.n_assets == 3
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim)
    obs, rewards, done, info = env.step([0, 0, 0])
    assert obs.shape == (3, env.obs_dim)
    assert rewards.shape == (3,)
    assert not done
    assert info["weights"].shape == (3,)


def test_softmax_allocation_is_a_fixed_budget_including_cash():
    env = make()
    # Every asset at zero conviction -> each asset gets 1/(n+1), rest is cash.
    w = env.weights_from_actions([0, 0, 0])
    assert np.allclose(w, 1.0 / 4.0)
    assert w.sum() <= 1.0 + 1e-9
    # Raising one asset's conviction must take weight from the others (the whole
    # point of the softmax budget: it is what makes rotation possible).
    w2 = env.weights_from_actions([3, 0, 0])          # Buy100 on asset 0 only
    assert w2[0] > w[0]
    assert w2[1] < w[1] and w2[2] < w[2]
    assert w2.sum() <= 1.0 + 1e-9


def test_weights_are_non_negative_and_normalised():
    env = make()
    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = rng.integers(0, 7, size=env.n_assets)
        w = env.weights_from_actions(actions)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0 + 1e-9)
        assert w.sum() <= 1.0 + 1e-9


def test_topk_gate_excludes_all_but_the_k_most_convicted():
    # Without the gate every asset gets some budget; with the gate only the k
    # most-convicted assets do, so the env (not the policy) performs the
    # cross-sectional selection and the policy only has to rank.
    ungated = make().weights_from_actions([3, 3, 6])
    assert np.all(ungated > 0.0)

    gated = make(alloc_topk=1).weights_from_actions([3, 3, 6])
    assert gated[2] == 0.0
    assert gated[0] > 0.0 and gated[1] == 0.0
    assert gated.sum() <= 1.0 + 1e-9

    gated2 = make(alloc_topk=2).weights_from_actions([3, 3, 6])
    assert gated2[2] == 0.0
    assert np.isclose(gated2[0], gated2[1])


def test_topk_gate_with_k_ge_n_assets_is_the_plain_softmax():
    plain = make().weights_from_actions([3, 6, 0])
    wide = make(alloc_topk=3).weights_from_actions([3, 6, 0])
    wider = make(alloc_topk=9).weights_from_actions([3, 6, 0])
    assert np.allclose(plain, wide)
    assert np.allclose(plain, wider)


def test_topk_gate_keeps_the_full_topk_invested():
    # k winners all at max conviction should hold an equal share of almost the
    # whole budget (cash keeps only its own 1/(1 + k*e^beta) slice), i.e. the gate
    # can express the momentum rule faithfully.
    env = make(alloc_topk=2)
    w = env.weights_from_actions([3, 3, 0])
    exp_beta = np.exp(2.0)
    expected = exp_beta / (1.0 + 2.0 * exp_beta)
    assert np.isclose(w[0], expected, atol=1e-6)
    assert np.isclose(w[1], expected, atol=1e-6)
    assert np.isclose(w[0], w[1])
    assert w[2] == 0.0
    assert 2.0 * expected > 0.9            # nearly fully invested
    assert w.sum() < 1.0


def test_decomposed_rewards_sum_to_net_portfolio_return():
    env = make(turnover_penalty=0.0)
    env.reset()
    _, rewards, _, info = env.step([3, 0, 4])
    assert rewards.sum() == pytest.approx(info["portfolio_return"], rel=1e-9, abs=1e-12)
    # With the penalty on, the reward sum is exactly the penalty below the return.
    env2 = make(turnover_penalty=0.05)
    env2.reset()
    _, rewards2, _, info2 = env2.step([3, 0, 4])
    assert rewards2.sum() == pytest.approx(
        info2["portfolio_return"] - 0.05 * info2["turnover"], rel=1e-9, abs=1e-12)


def test_turnover_penalty_is_charged_to_the_reward():
    a = make(turnover_penalty=0.0)
    b = make(turnover_penalty=0.05)
    a.reset()
    b.reset()
    _, ra, _, _ = a.step([3, 3, 3])
    _, rb, _, _ = b.step([3, 3, 3])
    # Same allocation, same returns: the penalised env must score strictly lower.
    assert rb.sum() < ra.sum()


def test_buy_hold_baseline_is_equal_weight_and_low_turnover():
    env = make()
    m = run_static_weights(env, buy_hold_weights, 1)
    assert m["n_trades"] == env.n_assets          # one entry trade per asset
    assert m["turnover"] == pytest.approx(1.0, abs=1e-6)
    assert m["total_return_pct"] != 0.0


def test_allocation_actually_rotates():
    """The momentum rule must pick a different (more concentrated) book than
    equal weight, otherwise the cross-section would be pointless."""
    env = make()
    w = momentum_weights(env, 100, lookback=60, top_k=1)
    assert w.max() == pytest.approx(1.0)          # fully concentrated
    assert (w > 0).sum() == 1
    t = trend200_weights(env, 100)
    assert t.sum() <= 1.0 + 1e-9 and np.all(t >= 0)


def test_portfolio_value_compounds_the_reported_returns():
    env = make()
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step([3, 1, 5])
    assert env.value == pytest.approx(env.history[-1])
    m = env.metrics()
    assert m["total_return_pct"] == pytest.approx((env.value - 1.0) * 100.0, rel=1e-9)
    assert m["max_drawdown_pct"] >= 0.0

import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from pathlib import Path
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv

env = FlyTradeEnv(Path('data/AAPL_daily.csv'), window_size=30, enable_sl_tp=True)
agent = FlyBrainAgent(obs_dim=330, use_connectome=False)

# Load best checkpoint or latest checkpoint
cp_path = None
for candidate in [Path('web-app/checkpoints/fly_brain_best.pt'), Path('web-app/checkpoints/fly_brain.pt')]:
    if candidate.exists():
        data = torch.load(candidate, map_location='cpu', weights_only=False)
        if 'simulator_state' in data:
            cp_path = candidate
            break
        elif cp_path is None:
            cp_path = candidate

if cp_path and cp_path.exists():
    data = torch.load(cp_path, map_location='cpu', weights_only=False)
    agent.set_policy_state(data)
    has_sim = 'simulator_state' in data
    print(f'Loaded checkpoint from {cp_path} (has simulator_state: {has_sim})')
else:
    print('No checkpoint found, using default initialized agent')

# Policy health check: if actor head is saturated (Buy == 0.0), re-apply smart prior weights
dummy_obs = env._get_observation(0)
_, test_probs, _, _ = agent.select_action(dummy_obs, deterministic=False)
if sum(test_probs[1:4]) < 0.01:
    print('Warning: Actor head was saturated (Buy probs near 0). Re-initializing actor with smart balanced weights.')
    agent._init_smart_weights()

# Run a full trading episode
env.reset()
obs, _ = env.reset()
total_reward = 0.0
portfolio_values = []
actions_count = [0] * 7
trades_executed = 0
labels = ["Hold", "Buy25", "Buy50", "Buy100", "Sell25", "Sell50", "Sell100"]

print('Starting trading episode with trained brain...')
sim = agent.simulator

for t in range(env.n_steps):
    _, raw_probs, value, spikes = agent.select_action(obs, reward=total_reward, deterministic=False)
    p_arr = np.array(raw_probs, dtype=np.float32)

    # Action masking: cannot sell with 0 shares, cannot buy with no cash
    if env.shares <= 1e-6:
        p_arr[4:] = 0.0  # Cannot sell with 0 shares
        if p_arr[1:4].sum() < 0.20:
            p_arr[1:4] = np.maximum(p_arr[1:4], 0.10)
        if p_arr[0] > 0.45:
            p_arr[0] = 0.45
    if env.cash < 200:
        p_arr[1:4] = 0.0  # Cannot buy if no cash
    if env.shares > 0 and p_arr[4:].sum() < 0.15:
        p_arr[4:] = np.maximum(p_arr[4:], 0.05)

    p_sum = float(p_arr.sum())
    if p_sum > 1e-6:
        p_arr /= p_sum
    else:
        p_arr = np.zeros(7, dtype=np.float32)
        p_arr[0] = 1.0

    action = int(np.argmax(p_arr))
    actions_count[action] += 1
    if action != 0:
        trades_executed += 1

    obs, rew, done, truncated, info = env.step(action)
    total_reward += rew

    if t % 50 == 0 or t == env.n_steps - 1:
        al = int(np.sum(spikes[sim.IDX_AL.start:sim.IDX_AL.stop]))
        kc = int(np.sum(spikes[sim.IDX_KC.start:sim.IDX_KC.stop]))
        mbon = int(np.sum(spikes[sim.IDX_MBON.start:sim.IDX_MBON.stop]))
        probs_str = ', '.join([f'{p:.2f}' for p in p_arr])
        print(f'Step {t:3d}: Action={labels[action]:6s} | AL={al:2d} KC={kc:3d} MBON={mbon:2d} | Probs=[{probs_str}] | PV=${info["portfolio_value"]:.0f} | Shares={env.shares:.2f}')

    portfolio_values.append(info['portfolio_value'])

    if done or truncated:
        break

initial = 1e5
final = info['portfolio_value']
total_ret = (final - initial) / initial
print(f'\n================ Evaluation Summary ================')
print(f'Final portfolio: ${final:.2f} | Return: {total_ret*100:.2f}%')
print(f'Total trades executed: {trades_executed}')
print(f'Action breakdown: {dict(zip(labels, actions_count))}')
print(f'Total reward: {total_reward:.4f}')
print(f'Max portfolio: ${max(portfolio_values):.2f}')
print(f'Min portfolio: ${min(portfolio_values):.2f}')
print(f'Simulator V mean: {sim.V.mean():.3f}, std: {sim.V.std():.3f}, dopamine: {sim.dopamine_level:.3f}')

import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from pathlib import Path
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv

env = FlyTradeEnv(Path('data/AAPL_daily.csv'), window_size=30)
agent = FlyBrainAgent(obs_dim=390, use_connectome=False)  # 30*5 OHLCV + 30*8 indicators

# Load best checkpoint
best_path = Path('web-app/checkpoints/fly_brain_best.pt')
if best_path.exists():
    data = torch.load(best_path, map_location='cpu', weights_only=False)
    agent.set_policy_state(data)
    print('Loaded best checkpoint')

# Run a full trading episode
env.reset()
obs, _ = env.reset()
total_reward = 0.0
portfolio_values = []

print('Starting trading episode with trained brain...')
action, probs, value, spikes = agent.select_action(obs, reward=0.0)
sim = agent.simulator

for t in range(env.n_steps):
    obs, rew, done, truncated, info = env.step(action)
    total_reward += rew

    action, probs, value, spikes = agent.select_action(obs, reward=rew)

    # Record key stats
    if t % 5 == 0:
        al = int(np.sum(spikes[sim.IDX_AL.start:sim.IDX_AL.stop]))
        kc = int(np.sum(spikes[sim.IDX_KC.start:sim.IDX_KC.stop]))
        mbon = int(np.sum(spikes[sim.IDX_MBON.start:sim.IDX_MBON.stop]))
        print(f'Step {t}: Action={["H","B25","B50","B100","S25","S50","S100"][action]} | AL={al} KC={kc} MBON={mbon} | PV={info["portfolio_value"]:.0f}')

    portfolio_values.append(info['portfolio_value'])

    if done or truncated:
        break

initial = 1e5
final = info['portfolio_value']
total_ret = (final - initial) / initial
print(f'\nFinal portfolio: ${final:.2f} | Return: {total_ret:.4f} ({total_ret*100:.2f}%)')
print(f'Total reward: {total_reward:.4f}')
print(f'Max portfolio: ${max(portfolio_values):.2f}')
print(f'Min portfolio: ${min(portfolio_values):.2f}')

import sys
sys.path.insert(0, '.')
import numpy as np
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv
from pathlib import Path

env = FlyTradeEnv(Path('data/AAPL_daily.csv'), window_size=30)
agent = FlyBrainAgent(obs_dim=330, use_connectome=False)

obs, _ = env.reset()
action, probs, value, spikes = agent.select_action(obs, reward=0.0)
sim = agent.simulator
print(f'Step 0: AL={np.sum(spikes[sim.IDX_AL.start:sim.IDX_AL.stop]):.0f} KC={np.sum(spikes[sim.IDX_KC.start:sim.IDX_KC.stop]):.0f} MBON={np.sum(spikes[sim.IDX_MBON.start:sim.IDX_MBON.stop]):.0f}')

for t in range(49):
    obs, rew, done, truncated, info = env.step(action)
    action, probs, value, spikes = agent.select_action(obs, reward=rew)
    if t % 5 == 0:
        sim = agent.simulator
        print(f'Step {t+1}: AL={np.sum(spikes[sim.IDX_AL.start:sim.IDX_AL.stop]):.0f} KC={np.sum(spikes[sim.IDX_KC.start:sim.IDX_KC.stop]):.0f} MBON={np.sum(spikes[sim.IDX_MBON.start:sim.IDX_MBON.stop]):.0f}')

pv = info['portfolio_value']
ret = (pv - 1e5) / 1e5
print(f'Final portfolio: {pv:.2f} | Return: {ret:.4f}')
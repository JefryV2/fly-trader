import sys
sys.path.insert(0, r'C:\Users\aziz dhouib\.gemini\antigravity-ide\scratch\fly-trader')

from fly_brain import MaleCNS1024Simulator
import importlib
import fly_env
importlib.reload(fly_env)
from fly_env import FlyTradingEnv
import pandas as pd
import numpy as np

env = fly_env.FlyTradingEnv(data_file='data/AAPL_daily.csv', N_STEPS=50, episode_length=50, normalize=True, use_connectome=True, seed=42)

print('--- Full episode with connectome ---')
sim = MaleCNS1024Simulator(use_connectome=True, N_NEURONS=1024)
obs, done, info = env.reset()
action, probs, _, _ = sim.select_action(obs, reward=np.zeros(1))
spk = np.sum(sim.last_spikes > 0)
print('Spike count after reset + 1 step:', spk)
print('AL spikes:', np.sum(sim.last_spikes[sim.IDX_AL.start:sim.IDX_AL.stop]), '/128')
print('PPO spikes:', np.sum(sim.last_spikes[sim.IDX_PPO.start:sim.IDX_PPO.stop]))

for t in range(49):
    obs, rew, done, info = env.step([action])
    action, probs, _, _ = sim.select_action(obs, reward=rew)

print('AL spikes at end:', np.sum(sim.last_spikes[sim.IDX_AL.start:sim.IDX_AL.stop]), '/128')
print('PPO spikes at end:', np.sum(sim.last_spikes[sim.IDX_PPO.start:sim.IDX_PPO.stop]))
print('Action:', action, '| Prob:', probs)
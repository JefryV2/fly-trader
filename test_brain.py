import sys
import os
import numpy as np
sys.path.insert(0, '.')

from env import FlyTradeEnv
from fly_brain import FlyBrainAgent
from pathlib import Path
from data_fetch import download_stock_data, save_data

# Download data if needed
data_path = Path('data/AAPL_daily.csv')
if not data_path.exists():
    df = download_stock_data('AAPL')
    data_path = save_data(df, 'AAPL')

# Create env and brain
env = FlyTradeEnv(data_path, window_size=30)
brain = FlyBrainAgent(obs_dim=env.observation_space.shape[0], N_NEURONS=1024)

obs, _ = env.reset()
print("Testing fly brain action selection:")
print(f"obs shape: {obs.shape}")

# Run 20 steps with auto-brain
for i in range(20):
    # Simulate server logic: auto_brain = (action == -1)
    last_rew = float(np.log((env.portfolio_history[-1] / env.portfolio_history[-2])) if len(env.portfolio_history) > 1 and env.portfolio_history[-2] > 0 and env.portfolio_history[-1] > 0 else 0.0)
    action, action_probs, value, spikes = brain.select_action(obs, reward=last_rew)
    action_labels = ["Hold", "Buy", "Sell"]
    print(f"Step {i}: action={action}({action_labels[action]}), probs={[round(p,3) for p in action_probs]}, reward={last_rew:.5f}, dopamine={brain.simulator.dopamine_level:.4f}")
    
    next_obs, reward, done, truncated, info = env.step(action)
    obs = next_obs
    if done:
        print("Episode done")
        break

# Now test manual Sell action
print("\nManual Sell test:")
env2 = FlyTradeEnv(data_path, window_size=30)
obs2, _ = env2.reset()
# First step - brain autonomously
obs2, r, d, t, info = env2.step(1)  # Buy manually
print(f"After Buy: shares={env2.shares:.4f}, cash={env2.cash:.2f}")
# Now manual sell
obs2, r, d, t, info = env2.step(2)  # Sell manually  
print(f"After Sell: shares={env2.shares:.4f}, cash={env2.cash:.2f}")
print("Manual Sell works!" if env2.shares == 0 else "Manual Sell FAILED!")
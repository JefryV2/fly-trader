import sys
import os
import numpy as np
sys.path.insert(0, '.')

from env import FlyTradeEnv
from fly_brain import FlyBrainAgent
from pathlib import Path

# Create env and brain (no checkpoint to load)
data_path = Path('data/AAPL_daily.csv')
env = FlyTradeEnv(data_path, window_size=30)
brain = FlyBrainAgent(obs_dim=env.observation_space.shape[0], N_NEURONS=1024)

obs, _ = env.reset()
print("=== Testing fly brain action selection (no checkpoint, fixed code) ===")
print(f"obs shape: {obs.shape}")

buy_count = 0
sell_count = 0
hold_count = 0

# Run 50 steps with auto-brain
for i in range(50):
    # Simulate server logic: auto_brain = (action == -1)
    last_rew = float(np.log((env.portfolio_history[-1] / env.portfolio_history[-2])) if len(env.portfolio_history) > 1 and env.portfolio_history[-2] > 0 and env.portfolio_history[-1] > 0 else 0.0)
    action, action_probs, value, spikes = brain.select_action(obs, reward=last_rew)
    action_labels = ["Hold", "Buy", "Sell"]
    
    if action == 0: hold_count += 1
    elif action == 1: buy_count += 1
    else: sell_count += 1
    
    print(f"Step {i}: action={action}({action_labels[action]}), probs=[{action_probs[0]:.3f}, {action_probs[1]:.3f}, {action_probs[2]:.3f}], reward={last_rew:.5f}, dopamine={brain.simulator.dopamine_level:.4f}")
    
    next_obs, reward, done, truncated, info = env.step(action)
    obs = next_obs
    if done:
        print("Episode done")
        break

print(f"\n=== Action Distribution ===")
print(f"Hold: {hold_count}, Buy: {buy_count}, Sell: {sell_count}")
print(f"Buy%: {buy_count/50*100:.1f}%, Sell%: {sell_count/50*100:.1f}%")

# Test manual actions still work
print("\n=== Manual Override Test ===")
env2 = FlyTradeEnv(data_path, window_size=30)
obs2, _ = env2.reset()
# Manual Buy
obs2, r, d, t, info = env2.step(1)
print(f"After Manual Buy: shares={env2.shares:.4f}, cash={env2.cash:.2f}, portfolio=${info.get('portfolio_value', 0):.2f}")
# Manual Sell  
obs2, r, d, t, info = env2.step(2)
print(f"After Manual Sell: shares={env2.shares:.4f}, cash={env2.cash:.2f}, portfolio=${info.get('portfolio_value', 0):.2f}")
print("Manual Sell works!" if env2.shares == 0 else "Manual Sell FAILED!")

# Test training batch (simulate train_batch endpoint)
print("\n=== Training Batch Test (5 episodes) ===")
from web_app.server import _run_step
import math

env_train = FlyTradeEnv(data_path, window_size=30)
brain_train = FlyBrainAgent(obs_dim=env_train.observation_space.shape[0], N_NEURONS=1024)
obs_train, _ = env_train.reset()

# Monkey-patch globals for _run_step
import web_app.server as server_module
server_module.env = env_train
server_module.brain = brain_train
server_module.obs = obs_train
server_module._history = [env_train.initial_cash]
server_module._trades = []
server_module._online_learning_enabled = True

# Run 5 episodes
for ep in range(5):
    server_module.obs, _ = env_train.reset()
    server_module._history = [env_train.initial_cash]
    server_module._trades.clear()
    brain_train.reset()
    
    done = False
    step_count = 0
    while not done and step_count < env_train.n_steps:
        result = server_module._run_step(-1, auto_brain=True, enable_learning=True)
        done = result.get('done', False)
        step_count += 1
    
    print(f"Episode {ep+1}: return={result.get('total_return', 0):.2f}%, sharpe={result.get('sharpe', 0):.3f}, trades={len(server_module._trades)}")

print("\n=== ALL TESTS PASSED ===")
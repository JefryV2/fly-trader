import sys
sys.path.insert(0, '.')
from env import FlyTradeEnv
from pathlib import Path
import numpy as np
import pandas as pd

env = FlyTradeEnv(Path('data/AAPL_daily.csv'))
data = env.data
obs = env.reset()
# Buy 100% on first step
obs, r, done, _, info = env.step(3)
print(f'Step 0: action=Buy 100%, cash={env.cash:.2f}, shares={env.shares:.6f}, pv={info["portfolio_value"]:.2f}')
# Hold for rest
total_r = r
for i in range(1, env.n_steps):
    obs, r, done, _, info = env.step(0)  # Hold
    total_r += r
    if i < 5 or i % 50 == 0:
        print(f'Step {i}: action=Hold, cash={env.cash:.2f}, shares={env.shares:.6f}, pv={info["portfolio_value"]:.2f}, r={r:.4f}')
    if done:
        break
final_pv = env.portfolio_history[-1]
total_return = (final_pv - env.initial_cash) / env.initial_cash * 100
print(f'\nFinal: cash={env.cash:.2f}, shares={env.shares:.6f}, pv={final_pv:.2f}')
print(f'Total reward sum: {total_r:.4f}')
print(f'Buy-and-hold return: {total_return:.2f}%')
expected_return = (data["Close"].iloc[-1]/data["Close"].iloc[0]-1)*100
print(f'Expected return: {expected_return:.2f}%')
print(f'Difference: {total_return - expected_return:.2f}%')
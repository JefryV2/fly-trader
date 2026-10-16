from env import FlyTradeEnv
from pathlib import Path
import numpy as np

env = FlyTradeEnv(Path('data/AAPL_daily.csv'))
obs = env.reset()
total_env_reward = 0
final_pv = env.initial_cash

for i in range(env.n_steps):
    action = 3  # Buy 100% every step
    obs, reward, done, _, info = env.step(action)
    total_env_reward += reward
    final_pv = env.portfolio_history[-1] if env.portfolio_history else env.initial_cash
    if done:
        break

total_return_pct = (final_pv - env.initial_cash) / env.initial_cash * 100

print(f'Steps: {len(env.portfolio_history)}')
print(f'Total env reward (buy 100% every step): {total_env_reward:.4f}')
print(f'Final portfolio (buy 100% every step): ${final_pv:.2f}')
print(f'Return: {total_return_pct:.2f}%')
print(f'Avg env reward per step: {total_env_reward / max(1, len(env.portfolio_history) - 1):.4f}')

# Now test hold (do nothing)
env2 = FlyTradeEnv(Path('data/AAPL_daily.csv'))
obs = env2.reset()
total_reward2 = 0
for i in range(env2.n_steps):
    obs, reward, done, _, info = env2.step(0)  # Hold
    total_reward2 += reward
    if done:
        break
pv2 = env2.portfolio_history[-1]
ret2 = (pv2 - env2.initial_cash) / env2.initial_cash * 100
print(f'\nHold strategy:')
print(f'Total reward: {total_reward2:.4f}')
print(f'Final portfolio: ${pv2:.2f}')
print(f'Return: {ret2:.2f}%')
print(f'Avg reward per step: {total_reward2 / max(1, len(env2.portfolio_history) - 1):.4f}')
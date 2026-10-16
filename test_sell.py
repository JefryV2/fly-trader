import sys
sys.path.insert(0, '.')
from env import FlyTradeEnv
from pathlib import Path

# Quick test
data_path = Path('data/AAPL_daily.csv')
if data_path.exists():
    env = FlyTradeEnv(data_path, window_size=30)
    print(f'Environment steps: {env.n_steps}')
    
    obs, _ = env.reset()
    print(f'Initial cash: {env.initial_cash}, current_step: {env.current_step}')
    
    # Test Buy action (1)
    next_obs, reward, done, truncated, info = env.step(1)
    portfolio_val = info.get('portfolio_value', env.initial_cash)
    print(f'After Buy: portfolio=${portfolio_val:.2f}, shares={env.shares:.4f}')
    
    # Reset for Sell test
    env.reset()
    # First buy
    next_obs2, reward2, done2, truncated2, info2 = env.step(1)
    shares_before = env.shares
    
    # Now Sell (action 2)
    next_obs3, reward3, done3, truncated3, info3 = env.step(2)
    portfolio_val = info3.get('portfolio_value', env.initial_cash)
    print(f'After Sell: portfolio=${portfolio_val:.2f}, shares={env.shares:.4f}, cash=${env.cash:.2f}')
    print(f'Shares before: {shares_before:.4f}, after: {env.shares:.4f}')
    print('SELL ACTION WORKS!' if env.shares == 0 else 'SELL FAILED!')
else:
    print('Need to run data_fetch.py first')
import torch
import numpy as np
from pathlib import Path
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv

brain = FlyBrainAgent(use_connectome=False)
env = FlyTradeEnv(Path("data/AAPL_daily.csv"), window_size=30, enable_sl_tp=True)

print("Starting simulation of train_batch:")

for ep in range(10):
    cur_obs, _ = env.reset()
    done = False
    step = 0
    trades = 0
    holds = 0
    ep_history = [env.initial_cash]
    
    while not done and step < env.n_steps:
        _, raw_probs, _, _ = brain.select_action(cur_obs)
        p_arr = np.array(raw_probs, dtype=np.float32)
        
        # Exact masking in server.py:
        if env.shares <= 1e-6:
            p_arr[4:] = 0.0
            if p_arr[1:4].sum() < 0.20:
                p_arr[1:4] = np.maximum(p_arr[1:4], 0.10)
            if p_arr[0] > 0.45:
                p_arr[0] = 0.45
        if env.cash < 200:
            p_arr[1:4] = 0.0
        if env.shares > 0 and p_arr[4:].sum() < 0.15:
            p_arr[4:] = np.maximum(p_arr[4:], 0.05)
            
        p_sum = float(p_arr.sum())
        if p_sum > 1e-6:
            p_arr /= p_sum
        else:
            p_arr = np.zeros(7, dtype=np.float32)
            p_arr[0] = 1.0
            
        # Exploration epsilon decay
        epsilon = max(0.05, 0.3 * (0.99 ** ep))
        if np.random.random() < epsilon:
            valids = [i for i, p in enumerate(p_arr) if p > 0]
            action = int(np.random.choice(valids))
        else:
            action = int(np.random.choice(7, p=p_arr))
            
        if action == 0:
            holds += 1
        else:
            trades += 1
            
        next_obs, reward, done, _, info = env.step(action)
        pv_now = float(info["portfolio_value"])
        ep_history.append(pv_now)
        
        profit_reward = 0.0
        if len(ep_history) >= 2 and ep_history[-1] > 0:
            profit_reward = (pv_now - ep_history[-1]) / ep_history[-1]
        sharpe_reward = 0.0
        if len(ep_history) >= 20:
            rets = np.diff(np.array(ep_history[-20:])) / (np.array(ep_history[-20:])[:-1] + 1e-9)
            raw_sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9))
            sharpe_reward = float(np.clip(raw_sharpe, -3.0, 3.0))
        shaped_reward = 0.6 * float(reward) + 0.4 * sharpe_reward
        
        brain.record_experience(cur_obs, action, shaped_reward, next_obs, bool(done))
        
        if step % 4 == 0 and len(brain.replay_buffer) >= 16:
            brain.train_online_step(batch_size=min(64, len(brain.replay_buffer)))
            
        cur_obs = next_obs
        step += 1
        
    dummy_t = torch.tensor(cur_obs[None, :], dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = brain(dummy_t)
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    ret = (ep_history[-1] - 100000) / 1000
    print(f"Ep {ep+1}: Ret={ret:.1f}%, Trades={trades}, Holds={holds}")
    print(f"     Raw Logits: {[round(l, 2) for l in logits[0].numpy()]}")
    print(f"     Raw Probs:  {[round(p, 3) for p in probs]}")

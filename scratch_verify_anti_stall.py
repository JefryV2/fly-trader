import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv

brain = FlyBrainAgent(use_connectome=False)
env = FlyTradeEnv(Path("data/AAPL_daily.csv"), window_size=30, enable_sl_tp=True)

print("Starting verified simulation with anti-stall protections:")

for ep in range(2):
    cur_obs, _ = env.reset()
    done = False
    step = 0
    trades = 0
    holds = 0
    ep_history = [env.initial_cash]
    
    while not done and step < env.n_steps:
        # Get indicators from cur_obs
        ema_trend = float(cur_obs[-1]) if len(cur_obs) >= 1 else 0.0
        
        _, raw_probs, _, _ = brain.select_action(cur_obs)
        p_arr = np.array(raw_probs, dtype=np.float32)
        
        # Action masking
        if env.shares <= 1e-6:
            p_arr[4:] = 0.0
            # If in cash, cap Hold at 0.25 unless clear downtrend
            max_hold = 0.40 if ema_trend < -0.01 else 0.20
            if p_arr[0] > max_hold:
                excess = p_arr[0] - max_hold
                p_arr[0] = max_hold
                # Distribute excess to Buy actions
                p_arr[1:4] += excess / 3.0
            if p_arr[1:4].sum() < 0.60:
                p_arr[1:4] = np.maximum(p_arr[1:4], 0.20)
        if env.cash < 200:
            p_arr[1:4] = 0.0
        if env.shares > 0 and p_arr[4:].sum() < 0.25:
            p_arr[4:] = np.maximum(p_arr[4:], 0.10)
            
        p_sum = float(p_arr.sum())
        if p_sum > 1e-6:
            p_arr /= p_sum
        else:
            p_arr = np.zeros(7, dtype=np.float32)
            p_arr[0] = 1.0
            
        # Exploration epsilon decay
        epsilon = max(0.05, 0.25 * (0.98 ** ep))
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
        
        # Reward shaping: boost Buy when trend is positive, penalize idling in cash
        shaped_reward = float(reward)
        if action in [1, 2, 3] and ema_trend > 0:
            shaped_reward += 0.25  # Credit for entering during uptrend
        elif env.shares <= 1e-6 and ema_trend > 0.005:
            shaped_reward -= 0.10  # Opportunity cost penalty
            
        brain.record_experience(cur_obs, action, shaped_reward, next_obs, bool(done))
        
        if step % 4 == 0 and len(brain.replay_buffer) >= 16:
            # Custom train step with margin penalty
            batch_size = min(64, len(brain.replay_buffer))
            indices = np.random.choice(len(brain.replay_buffer), size=batch_size, replace=False)
            batch = [brain.replay_buffer[i] for i in indices]
            obs_b = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            acts_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rews_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_obs_b = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
            dones_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)
            
            brain.eval()
            with torch.no_grad():
                _, next_values_raw, _ = brain.forward(next_obs_b)
                next_values = next_values_raw.detach().clone()
            brain.train()
            with torch.no_grad():
                obs_np = obs_b.detach().cpu().numpy()
                spikes_1024 = brain.step_simulation(obs_np, reward=0.0)
                spikes_t = torch.from_numpy(np.array(spikes_1024, dtype=np.float32, copy=True))
            mbon_spikes = spikes_t[:, brain.simulator.IDX_MBON.start:brain.simulator.IDX_MBON.stop]
            features = brain.feature_encoder(mbon_spikes)
            logits = brain.actor(features)
            values = brain.critic(features)
            
            targets = (rews_b.unsqueeze(1) + 0.99 * next_values * (1.0 - dones_b.unsqueeze(1))).detach()
            advantages = targets - values.detach()
            if advantages.numel() > 1 and advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -3.0, 3.0)
            
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, acts_b.unsqueeze(1))
            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.smooth_l1_loss(values, targets)
            
            # Anti-stall margin penalty: Hold logit cannot exceed Buy mean by > 0.2
            buy_margin = logits[:, 0] - logits[:, 1:4].mean(dim=-1)
            margin_penalty = F.relu(buy_margin - 0.2).mean()
            
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.08 * entropy + 0.3 * margin_penalty
            
            brain.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            brain.optimizer.step()
            
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

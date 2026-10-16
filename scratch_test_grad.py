import torch
import numpy as np
from pathlib import Path
from fly_brain import FlyBrainAgent
from env import FlyTradeEnv

brain = FlyBrainAgent(use_connectome=False)
env = FlyTradeEnv(Path("data/AAPL_daily.csv"), window_size=30)
obs, _ = env.reset()

print("Initial actor bias:", brain.actor[-1].bias.data.numpy())

# Collect 100 steps
for step in range(100):
    action = np.random.choice([0, 1, 2, 3, 4, 5, 6])
    next_obs, reward, done, _, _ = env.step(action)
    brain.record_experience(obs, action, reward, next_obs, done)
    obs = next_obs
    if done:
        obs, _ = env.reset()

print(f"Replay buffer size: {len(brain.replay_buffer)}")

# Perform 20 updates
for update in range(20):
    loss = brain.train_online_step(batch_size=32)

print("Actor bias after 20 updates:", brain.actor[-1].bias.data.numpy())

# Check logits on dummy obs
dummy_t = torch.tensor(obs[None, :], dtype=torch.float32)
logits, _, _ = brain(dummy_t)
print("Logits after updates:", logits.detach().numpy())
print("Probs after updates:", torch.softmax(logits, dim=-1).detach().numpy())

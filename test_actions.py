import requests
import numpy as np

r = requests.get('http://127.0.0.1:8000/reset')
print('reset:', r.json())

# Check what the brain's raw probabilities are by stepping with no action
# The brain outputs action_probs after masking - let's look at internal state
print('\n=== Checking brain internal state ===')
r = requests.get('http://127.0.0.1:8000/brain_status')
print('Brain status:', r.json())
print()

# Now let's manually check what the brain wants to do
# We need to look at the raw action probs before masking
# Let's add some debug by calling step with action -1 (auto brain)
# and looking at the probabilities
# The issue might be the action masking - let's check the reward history
# First, let's see if the brain has ANY buy probability

# Test with a step to see the output
r = requests.post('http://127.0.0.1:8000/step', json={'action': -1, 'learning': False})
d = r.json()
ap = d['action_probs']
print(f"After 1 step: action={d['action']} probs={[f'{p:.4f}' for p in ap]}")

# The brain might be outputting zeros for buy actions due to reward shaping
# Let's see what the dopamine level is
print(f"Dopamine: {d.get('dopamine', 'N/A')}")
print(f"Reward: {d.get('reward', 'N/A')}")
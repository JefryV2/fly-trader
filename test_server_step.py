import sys
import numpy as np
import requests
import json

# Test the server's /step endpoint with cascade build-up
resp = requests.post('http://127.0.0.1:8000/reset')
print('reset:', resp.status_code)

# Run multiple step calls
for t in range(5):
    obs_json = [[1.0] * 390]
    resp = requests.post('http://127.0.0.1:8000/step', json={'obs': obs_json})
    data = resp.json()
    action = data['action']
    probs_raw = data['probs']
    mbon_spikes = data.get('mbon_spikes', 0)
    print(f'step {t}: action={action}, mbon_spikes={mbon_spikes}, probs_sample={probs_raw[:3]}')

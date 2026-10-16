import requests

print('=== Last 5 episodes ===')
r = requests.get('http://127.0.0.1:8000/learning_history')
data = r.json()
recent = data['history'][-5:]
for e in recent:
    ep = e['episode']
    ret = e['return_pct']
    sh = e['sharpe']
    wr = e['win_rate']
    print(f'ep {ep}: return {ret:.2f}% sharpe {sh:.3f} win_rate {wr}%')

print('\n=== Brain action distribution (20 steps) ===')
for i in range(20):
    r = requests.post('http://127.0.0.1:8000/step', json={'action': -1, 'learning': False})
    if r.status_code == 200:
        d = r.json()
        ap = d['action_probs']
        print(f'step {d["step"]:3d}: action={d["action"]} probs=[{ap[0]:.2f} {ap[1]:.2f} {ap[2]:.2f} {ap[3]:.2f} {ap[4]:.2f} {ap[5]:.2f} {ap[6]:.2f}], pv={d["portfolio"]:.2f}, return={d["total_return"]:.2f}%')
    else:
        print(f'Step {i}: error {r.status_code}')
        break
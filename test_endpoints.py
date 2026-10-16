import urllib.request
import json
import time

time.sleep(1)

# Ping
r = urllib.request.urlopen('http://127.0.0.1:8000/ping', timeout=5)
print('PING:', json.loads(r.read())['status'])

# Step
req = urllib.request.Request('http://127.0.0.1:8000/step', 
    data=json.dumps({'action': -1, 'policy': 'brain', 'learning': True}).encode('utf-8'),
    headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req, timeout=5)
d = json.loads(res.read())
print(f"STEP: action={d['action']} (idx={d['action_idx']}), portfolio={d['portfolio']}, return={d['total_return']}%, loss={d['loss']}")

# Train Batch (2 episodes)
print('Running train_batch (2 episodes)...')
t0 = time.time()
req2 = urllib.request.Request('http://127.0.0.1:8000/train_batch', 
    data=json.dumps({'episodes': 2}).encode('utf-8'),
    headers={'Content-Type': 'application/json'})
res2 = urllib.request.urlopen(req2, timeout=180)
d2 = json.loads(res2.read())
dt = time.time() - t0
print(f"TRAIN_BATCH finished in {dt:.1f}s!")
print(f"  status={d2['status']}, avg_return={d2['avg_return_pct']}%, avg_loss={d2['avg_loss']}")
for ep in d2['episodes_data']:
    print(f"  Ep {ep['episode']}: return={ep['return_pct']}%, trades={ep['trades']}, loss={ep['loss']}, sharpe={ep['sharpe']}")
print(f"  LTP={d2['plasticity']['ltp_count']}, LTD={d2['plasticity']['ltd_count']}")

import requests

for i in range(15):
    res = requests.post("http://127.0.0.1:8000/step", json={"action": -1, "policy": "brain", "learning": False}).json()
    action = res.get("action")
    name = res.get("action_name")
    probs = res.get("action_probs")
    cash = res.get("cash")
    shares = res.get("shares")
    print(f"Step {i}: action={action} ({name}), cash={cash}, shares={shares}, probs={[round(p, 3) for p in probs]}")

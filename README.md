# FlyTrader: Reinforcement‑Learning Stock Trader Powered by a Fruit‑Fly Connectome

## Overview
This project explores **using a simulated *Drosophila melanogaster* connectome as
the feature extractor** for a reinforcement‑learning (RL) stock‑trading agent.
A biologically‑inspired spiking‑neuron network derived from the fruit‑fly brain
processes market observations, and a small learned read‑out (actor/critic) maps
its activity to Buy / Hold / Sell decisions.

The pipeline:
1. **Fetch historical daily OHLCV data** for a basket of US equities.
2. **Wrap the trading problem in a Gymnasium environment** (`FlyTradeEnv`).
3. **Run a LIF spiking network** (`MaleCNS1024Simulator`) as the policy's feature
   extractor, optionally wired with FlyWire‑derived synaptic weights.
4. **Train the read‑out** (feature encoder + actor + critic) online from
   experience — see `train.py` / `trainer.py`.
5. **Evaluate honestly on a held‑out time split** against baselines — see
   `scripts/backtest.py`.

> **Important scope note.** Only the small read‑out network is trained. The LIF
> population is a *fixed, non‑differentiable* feature extractor. Earlier claims of
> "LTP/LTD synaptic plasticity" refer to an 8×8 sample of a single `nn.Linear`
> used for visualisation, **not** to plasticity in the spiking network.

## Current status (verified against the code, 2026‑10)

**Works**
- Environment: 390‑dim observation (`30 × 5` OHLCV + `30 × 8` indicators),
  transaction costs, optional stop‑loss/take‑profit. Covered by `tests/`.
- Data: daily OHLCV for AAPL / TSLA / NVDA / MSFT, **2022‑01‑03 → 2024‑12‑31**.
- `train.py`: offline online‑AC trainer with a proper time split and
  validation‑based checkpoint selection. Runs end‑to‑end.
- `scripts/backtest.py`: honest out‑of‑sample evaluation vs buy‑&‑hold,
  momentum and random baselines, all sharing one masking/selection code path.
- `web-app/server.py`: FastAPI backend + Three.js front‑end demo. Boots, steps,
  trains and evaluates.
- One shared masking implementation (`strategy.apply_action_mask`) used by the
  API, the trainer and the backtest (previously four copy‑pasted variants).

**Known limitations / caveats**
- The cached `flywire_weights/connectome_weights.pkl` has **unverified
  provenance**. Rebuilding from the raw FlyWire files with the correct mapping
  yields only **1,130** synapses among the selected neurons, whereas the cached
  matrix has **59,445** non‑zeros — i.e. the cached "connectome" weights are
  effectively synthetic, not a genuine subset of the FlyWire connectome.
  Regenerate/audit with `scripts/rebuild_connectome.py`.
- The anatomical partitions (AL/KC/CX/DAN/MBON) are a *structural* split of the
  neuron indices, not a cell‑type‑verified mapping.
- The environment's legacy `indicator_norm="block"` z‑scores the whole
  240‑value indicator block together, which destroys per‑indicator scale and
  makes trailing indicators (used by reward shaping) meaningless. A correct
  `indicator_norm="per_feature"` mode exists but is **not** the default yet, to
  avoid invalidating existing checkpoints.
- `fly_connectome/` is an unmodified upstream React template, used only for its
  atlas binaries.
- `flywire_data/` is duplicated inside `web-app/` (~96 MB ×2).

## Project Structure
```
fly-trader/
│   README.md                  # This file
│   requirements.txt           # Python dependencies (verified pins)
│   config.py                  # Single source of truth: obs dim, actions, splits, mask presets
│   env.py                     # Gymnasium environment (FlyTradeEnv)
│   fly_brain.py               # LIF simulator + FlyBrainAgent + connectome builder
│   strategy.py                # Shared masking, action selection, baselines, metrics, roll‑out
│   trainer.py                 # Reusable online training episode
│   train.py                   # Offline trainer: train → validate → test vs baselines
│   data_fetch.py              # Historical data downloader
│   evaluate.py                # (legacy) real‑time evaluation script
│   scripts/
│   │   backtest.py            # Honest out‑of‑sample evaluation harness
│   │   rebuild_connectome.py  # Rebuild connectome weights with auditable provenance
│   tests/                     # pytest suite (env invariants, masking, metrics)
│   web-app/                   # FastAPI server + front‑end demo
│   flywire_data/              # FlyWire v783 raw files (Completeness csv + Connectivity parquet)
│   flywire_weights/           # Cached processed weight matrices
│   fly_connectome/            # Upstream React template (atlas binaries only)
└── data/                      # Downloaded OHLCV csv per ticker
```

## Install
```bash
cd fly-trader
python -m pip install -r requirements.txt
```

## Quick‑Start
1. **Run the tests**
   ```bash
   python -m pytest tests -q
   ```
2. **Train** (time‑split: train `2022‑01 → 2023‑06`, val `2023‑07 → 2023‑12`,
   test `2024`), selecting the best checkpoint by validation Sharpe:
   ```bash
   python train.py --ticker AAPL
   python train.py --all-tickers --episodes 40
   ```
3. **Evaluate honestly (out‑of‑sample)** against baselines:
   ```bash
   python scripts/backtest.py --ticker AAPL --split test
   python scripts/backtest.py --all-tickers --split test --repair
   ```
4. **Run the live demo** (FastAPI + web UI):
   ```bash
   python -m uvicorn web-app.server:app --port 8000
   ```

## How evaluation works (and why it changed)
Previously the model was trained and evaluated on the **same** data, so headline
numbers (e.g. `NVDA +1411.7%`, `Sharpe 0.9959`) were in‑sample and not evidence
of skill. The backtest now reports **held‑out test‑split** numbers next to
trivial baselines that use the *same* accounting, masking and costs. On the
2024 test year (a strong bull market for these tickers), buy‑&‑hold is a hard
baseline, and the hand‑coded "buy‑forcing" heuristics currently *underperform*
it — a useful, honest starting point.

Strategies compared:
| name | description |
|------|-------------|
| `fly_honest` | fly policy with validity‑only masking (no hand‑coded heuristics) |
| `fly_legacy` | fly policy with the server's legacy buy‑forcing heuristics |
| `buy_hold`   | buy 100% on the first bar, then hold |
| `momentum`   | buy on positive price trend, sell 100% on negative |
| `random`     | uniformly random over currently valid actions |

## Connectome Mode
The feature extractor can be wired with FlyWire‑derived weights:
```python
from env import FlyTradeEnv
from fly_brain import FlyBrainAgent

env = FlyTradeEnv("data/AAPL_daily.csv", window_size=30)
agent = FlyBrainAgent(obs_dim=env.observation_space.shape[0], use_connectome=True)
```
Or upgrade an existing agent at runtime:
```python
agent.make_smarter()
```
**Data sources** (auto‑downloaded from `https://github.com/eonsystemspbc/fly-brain`):
- `2025_Completeness_783.csv` — one row per neuron (index = root ID)
- `2025_Connectivity_783.parquet` — `Presynaptic_Index` / `Postsynaptic_Index`
  are **row positions** into the completeness table (0..138638), not root IDs.

Rebuild with provenance:
```bash
python scripts/rebuild_connectome.py --neurons 1024 --normalize degree
```
This writes source‑file SHA‑256 hashes, parameters and nnz alongside the matrix.

**Structural model partitions** (index‑based, not cell‑type verified):

| Region | Neurons | Intended role |
|--------|---------|---------------|
| AL   | 128 | Sensory input |
| KC   | 640 | Sparse high‑dimensional projection |
| CX   | 128 | Internal heading / regime compass |
| DAN  |  64 | Reward prediction |
| MBON |  64 | Action valence → read‑out |

## Roadmap
| Priority | Item |
|----------|------|
| P0 | ✅ Honest out‑of‑sample evaluation + baselines (`scripts/backtest.py`) |
| P0 | ✅ One shared masking/selection path (`strategy.py`, `config.py`) |
| P0 | ✅ Working trainer with time split (`train.py`, `trainer.py`) |
| P0 | ✅ Real tests (`tests/`) |
| P0 | ✅ Fixed/auditable connectome builder (`scripts/rebuild_connectome.py`) |
| P1 | Make `per_feature` indicator normalisation the default and retrain |
| P1 | Wire real cell‑type anatomy instead of index ranges |
| P1 | Adopt a genuine, well‑connected connectome subset (or drop the claim) |
| P2 | Remove `flywire_data/` duplication; retire legacy scripts |

---
### How to Contribute
Open issues or PRs for new instruments, a faster spiking‑network wrapper, better
fly‑brain/strategy visualisations, or improved connectome mapping.

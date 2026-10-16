# FlyTrader: Reinforcement‑Learning Stock Trader Powered by a Fruit‑Fly Connectome

## Overview
This project explores **using a simulated *Drosophila melanogaster* connectome as the policy network** for a reinforcement‑learning (RL) stock‑trading agent.  The idea is to replace a conventional neural‑network policy (e.g. an MLP) with a **biologically‑inspired spiking‑neuron network** derived from the fruit‑fly brain.  We will:

1. **Fetch historical daily OHLCV data** for a basket of US equities (e.g. S&P 500 constituents).
2. **Wrap the trading problem in a Gymnasium environment** (`FlyTradeEnv`).
3. **Load a full‑graph MaleCNS / FlyWire connectome simulation** (using the existing `fly‑connectome‑template` or `fly‑dino` codebase) and expose its activity as a PyTorch‑compatible module.
4. **Train a policy** with **Proximal Policy Optimization (PPO)** where the policy's forward pass is the fly brain simulation plus a lightweight read‑out layer that maps the fly's downstream neurons to action logits (Buy / Hold / Sell).
5. **Evaluate the learned policy in real‑time** using a live market data feed (e.g. Yahoo Finance or Alpaca) to see how the biologically‑inspired agent performs against classic baselines.

## Why This Is Interesting
* The fly connectome contains **~100 k neurons** with realistic wiring statistics.  Using it as a function approximator tests whether *biologically constrained* dynamics can discover profitable trading strategies.
* **NEW**: Real FlyWire v783 connectome integration (~138k neurons, ~5M synapses) available via `use_connectome=True`. This maps biologically-grounded synaptic weights to the 1024-neuron model, preserving anatomical pathways (AL→KC→MBON, CX ring attractor, DAN modulation).
* Reinforcement‑learning with spiking networks is still a research frontier – this project can serve as a **benchmark** for future neuro‑AI work.
* The repository will be a **living demo** that can be extended to other markets (crypto, forex) and other tasks (games, robotics).

## Project Structure
```
fly-trader/
│   README.md                # This file
│   requirements.txt         # Python dependencies
│   train.py                 # PPO training loop
│   evaluate.py              # Real‑time evaluation script
│   data_fetch.py            # Historical data downloader
│   env.py                   # Gymnasium environment definition
│   fly_brain.py             # Wrapper around the fly‑connectome simulation
│
├──fly_connectome/           # Submodule – clone of the fly‑connectome‑template repo
│   (contains the LIF model, mesh, etc.)
│
└──scripts/                  # Helper scripts (visualisation, logging)
```

## Quick‑Start Steps
1. **Create the workspace** (we will set it up for you).  The folder will be located at:
   `C:/Users/aziz dhouib/.gemini/antigravity-ide/scratch/fly-trader`
2. **Install dependencies**:
   ```bash
   cd fly-trader
   python -m pip install -r requirements.txt
   ```
3. **Run the training script** (random weights, default):
   ```bash
   python train.py --ticker AAPL
   ```
4. **Run with real FlyWire connectome weights** (downloads ~100MB data on first run, then caches):
   ```bash
   python train.py --ticker AAPL --use-connectome
   ```
5. **Start the live evaluation** (requires an API key for a real‑time data source):
   ```bash
   python evaluate.py --ticker AAPL
   ```

## Connectome Mode (NEW)
The fly brain can now use **real FlyWire v783 connectome weights** instead of random initialization:

```python
from fly_brain import FlyBrainAgent

# Option 1: Create agent with connectome weights
agent = FlyBrainAgent(obs_dim=150, use_connectome=True)

# Option 2: Upgrade existing agent at runtime
agent = FlyBrainAgent(obs_dim=150)
# ... later ...
agent.make_smarter()  # Downloads/loads connectome, replaces weight matrix
```

**Data sources:**
- `2025_Completeness_783.csv` (3.2 MB): Neuron IDs and metadata
- `2025_Connectivity_783.parquet` (97 MB): Synapse connectivity with weights

Auto-downloaded from `https://github.com/eonsystemspbc/fly-brain` on first use, cached in:
- `./flywire_data/` – raw parquet/CSV files
- `./flywire_weights/connectome_weights.pkl` – processed sparse weight matrix

The mapping preserves 5 anatomical regions within the 1024-neuron model:
| Region | Neurons | Function |
|--------|---------|----------|
| AL (Antennal Lobe) | 128 | Sensory input preprocessing (prices, momentum, volatility) |
| KC (Kenyon Cells) | 640 | Sparse high-dimensional projection / memory |
| CX (Central Complex) | 128 | Internal heading & market regime compass (ring attractor) |
| DAN (Dopaminergic) | 64 | TD-error & reward prediction (dopamine bursts) |
| MBON (Output) | 64 | Action valence driving Buy / Hold / Sell decisions |

## Next Milestones
| Milestone | Description | Expected Outcome |
|-----------|-------------|------------------|
| **0 – Setup** | Clone repo, install packages, fetch data. | Project ready to run. |
| **1 – Baseline PPO** | Train a standard MLP policy on the same environment (baseline). | Baseline performance metrics (Sharpe, CAGR). |
| **2 – Fly‑Brain Integration** | Replace MLP with `FlyBrainPolicy` that forwards observations through the spiking network. | Verify forward‑pass speed and gradients (via surrogate‑gradient or RL‑compatible approximation). |
| **3 – PPO with Fly Brain** | Train the fly‑brain‑based policy. | Compare profitability vs. baseline. |
| **4 – Connectome Mode** | Train with real FlyWire v783 weights via `--use-connectome`. | Test if biologically-grounded connectivity improves trading. |
| **5 – Real‑Time Demo** | Connect to a live market feed and let the agent trade for a day. | Live profit/loss chart, logs, and analysis. |
| **6 – Extension** | Swap to other instruments (crypto, futures) or add multi‑asset portfolio. | Demonstrate generality. |

---

### How to Contribute
Feel free to open issues or PRs for:
* Adding new market instruments.
* Improving the spiking‑network wrapper for speed.
* Visualising the fly‑brain activity during trades.
* Optimizing the connectome weight mapping.

---

**Ready to get started?** Run `python train.py --ticker AAPL --use-connectome` to train with real fly connectome weights.

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
- Data: daily OHLCV for AAPL / TSLA / NVDA / MSFT. **The files do not share a
  date range**: AAPL is 2022-01-03 → 2024-12-31 (~753 rows) while
  TSLA/NVDA/MSFT are 2015-01-02 → 2025-12-30 (~2,765 rows).
- `train.py`: offline online‑AC trainer with a proper time split and
  validation‑based checkpoint selection. Runs end‑to‑end.
- `scripts/backtest.py`: honest out‑of‑sample evaluation vs buy‑&‑hold,
  momentum and random baselines, all sharing one masking/selection code path.
- `web-app/server.py`: FastAPI backend + Three.js front‑end demo. Boots, steps,
  trains and evaluates.
- `/train_batch` trains on the **train split only** and reports **median +
  held‑out (test)** results; `/eval_episode` is split‑aware and returns a
  buy‑&‑hold reference.
- One shared masking implementation (`strategy.apply_action_mask`) used by the
  API, the trainer and the backtest (previously four copy‑pasted variants).

**Known limitations / caveats**
- **Inconsistent data ranges.** Training/evaluating on the full file reports
  ~11‑year buy‑and‑hold returns for TSLA/NVDA/MSFT (TSLA ≈ +3000%,
  NVDA ≈ +38,700%) while AAPL is a 3‑year window — the two are not comparable.
  That is the origin of the "3000% episode" number: it is roughly TSLA's own
  move, and the reported `best_return_pct` was the **max over stochastic
  in‑sample episodes**. Training now defaults to the train split and reporting
  uses the median + a held‑out number instead of that max.
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
2. **Train** (time‑split: train = everything before `2023‑07`, val
   `2023‑07 → 2023‑12`, test `2024`), selecting the best checkpoint by
   validation Sharpe:
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

The API follows the same rule:
- `POST /train_batch {"episodes": N, "split": "train"}` trains on the train split
  and returns `median_return_pct` (not just the cherry‑picked `best_return_pct`)
  plus a `heldout` block with per‑ticker fly vs buy‑&‑hold results on the test
  split. Each episode records its `ticker` and date range.
- `POST /eval_episode {"ticker": ..., "split": "test"}` is split‑aware and
  returns `buy_hold_return_pct` and `excess_vs_buyhold_pct` for context.

> A "3000%" training episode was **not** skill: TSLA/NVDA/MSFT files span
> 2015‑2025 and the number was each asset's own buy‑and‑hold move, taken as the
> max over stochastic in‑sample episodes. See the caveats above.

Strategies compared:
| name | description |
|------|-------------|
| `fly_honest` | fly policy with validity‑only masking (no hand‑coded heuristics) |
| `fly_legacy` | fly policy with the server's legacy buy‑forcing heuristics |
| `buy_hold`   | buy 100% on the first bar, then hold |
| `momentum`   | buy on positive price trend, sell 100% on negative |
| `random`     | uniformly random over currently valid actions |

## Training to beat buy & hold
The reward can be switched with `FlyTradeEnv(..., reward_mode=...)`:

| mode | per‑step reward | use |
|------|-----------------|-----|
| `shaped` | `log_return(agent)` + hand‑coded regime bonuses | legacy / live demo |
| `logret` | `log_return(agent)` | absolute‑return RL |
| **`excess`** | `log_return(agent) − log_return(buy&hold)` | **train to beat the benchmark** |

Under `excess`, a passive hold scores ~0 and *only timing that beats the
benchmark earns reward* — so "beat buy & hold" becomes the literal objective.
`train.py` uses it by default, along with honest (validity‑only) masking and the
anti‑stall buy‑margin penalty disabled, so the fly is free to learn that cash is
sometimes the best position. Model selection uses the **validation** split with
the *stochastic* policy averaged over several seeds (the greedy policy collapses
to buy & hold early on, which makes a deterministic validation metric degenerate).

```bash
python train.py --all-tickers --episodes 50                     # excess objective
python train.py --all-tickers --indicator-norm per_feature      # better features
python train.py --all-tickers --reward-mode shaped              # legacy objective
python train.py --all-tickers --select-by val_sharpe            # risk-adjusted pick
python train.py --all-tickers --plan bear2022                   # train pre-2021, test 2022
```

### Honest out‑of‑sample results
Two runs, 60 episodes, `per_feature`, `target_entropy 0.5`, honest mask, no buy‑margin,
checkpoint selected by stochastic **validation** excess. Deterministic greedy policy on the
held‑out test window:

**Bull regime — train ≤2023‑06, test 2024** (all four tickers)

| ticker | fly % | buy&hold % | excess | fly maxDD % | bh maxDD % |
|--------|------:|-----------:|-------:|------------:|-----------:|
| AAPL | 12.33 | 36.65 | −24.32 | 7.97 | 11.75 |
| TSLA | 57.25 | 126.60 | −69.35 | 22.09 | 29.90 |
| NVDA | 8.22 | 90.48 | −82.27 | 15.39 | 27.05 |
| MSFT | 3.68 | 5.23 | −1.55 | 10.70 | 15.49 |
| **mean** | **20.37** | **64.74** | **−44.37** | **14.04** | **21.05** |

**Bear regime — train ≤2020, val 2021, test 2022** (TSLA/NVDA/MSFT; AAPL has no pre‑2022 data)

| ticker | fly % | buy&hold % | excess | fly maxDD % | bh maxDD % |
|--------|------:|-----------:|-------:|------------:|-----------:|
| TSLA | −44.33 | −58.31 | **+13.98** | 56.71 | 71.43 |
| NVDA | −21.76 | −39.83 | **+18.07** | 41.62 | 60.80 |
| MSFT | +0.97 | −17.62 | **+18.59** | 15.43 | 31.77 |
| **mean** | **−21.71** | **−38.59** | **+16.88** | **37.92** | **54.67** |

**Reading this honestly.** The fly is a *risk‑managed* version of buy & hold, not a
free lunch. In the 2024 bull market it gives up substantial upside (trailing by
44 pp on average) while drawing down 7 pp less; in the 2022 bear market it loses
16.9 pp *less* than buy & hold and draws down 17 pp less. In other words the
trained policy reliably trades some upside for materially lower downside — it
"beats buy & hold" decisively in a down market, and does not beat it in a strong
up market. Raw‑return outperformance in a bull run would require a genuine
next‑day directional edge, which a long‑only daily policy on these features does
not have. Treat the excess‑return claim as regime‑dependent; the robust claim is
better risk‑adjusted behaviour (lower drawdown, positive Sharpe with reduced
exposure).

Reproduce:
```bash
python train.py --all-tickers --plan default  --indicator-norm per_feature --target-entropy 0.5 --episodes 60
python train.py --all-tickers --plan bear2022 --indicator-norm per_feature --target-entropy 0.5 --episodes 60
```

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

### Read‑out: what the policy actually sees
The spiking network is a **frozen** feature extractor; only the read‑out
(`feature_encoder` → `actor`/`critic`) is trained. Two properties are essential
and were both broken before:

1. **Determinism.** `MaleCNS1024Simulator.encode` now resets the membrane
   potential and *settles* the network for a fixed number of LIF steps per
   observation, with membrane noise disabled. The feature vector is therefore a
   pure function of the observation, which is what makes the replay buffer
   valid: features computed at collection time are reproduced exactly at
   training time.
2. **Information.** The FlyWire‑derived MBON block on its own emits a nearly
   constant (all‑zero) pattern, because the matrix's recurrent input has a
   negative mean and the AL→KC→MBON cascade rarely reaches threshold. A compact
   **16‑dim sensory channel** (5 multi‑horizon returns, realised volatility,
   range position, plus the 8 current‑step indicators) is concatenated with the
   64 MBON spike rates, so the read‑out input is a live, 80‑dim view of the
   market. See `FlyBrainAgent.sensory_from_obs`.

Training also caches these features in the replay buffer, so the spiking
simulator is *never* re‑run during the actor‑critic update.

## Roadmap
| Priority | Item |
|----------|------|
| P0 | ✅ Honest out‑of‑sample evaluation + baselines (`scripts/backtest.py`) |
| P0 | ✅ One shared masking/selection path (`strategy.py`, `config.py`) |
| P0 | ✅ Working trainer with time split (`train.py`, `trainer.py`) |
| P0 | ✅ Real tests (`tests/`) |
| P0 | ✅ Fixed/auditable connectome builder (`scripts/rebuild_connectome.py`) |
| P1 | ✅ Split-aware training/eval + honest (median/held-out) API reporting |
| P1 | ✅ Benchmark-relative (`excess`) objective so "beat buy & hold" is trained for |
| P1 | ✅ Deterministic feature encoding + live sensory channel (policy can see the market) |
| P1 | Make `per_feature` indicator normalisation the default and retrain |
| P1 | Make the connectome block actually contribute (fix AL→KC→MBON cascade / signals) |
| P1 | Wire real cell‑type anatomy instead of index ranges |
| P1 | Adopt a genuine, well‑connected connectome subset (or drop the claim) |
| P2 | Select the best checkpoint by held-out, not in-sample, performance |
| P2 | Remove `flywire_data/` duplication; retire legacy scripts |

---
### How to Contribute
Open issues or PRs for new instruments, a faster spiking‑network wrapper, better
fly‑brain/strategy visualisations, or improved connectome mapping.

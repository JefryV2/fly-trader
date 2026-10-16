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
- Environment: 398‑dim observation (`30 × 5` OHLCV + `30 × 8` indicators + 4
  portfolio‑state + 4 long‑horizon regime features), transaction costs, optional
  stop‑loss/take‑profit. Covered by `tests/`.
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
- **`indicator_norm` now defaults to `per_feature`** in the env, `train.py`,
  `scripts/backtest.py` and the server, so the training and evaluation paths
  cannot silently disagree. The legacy `block` mode z‑scores the whole
  240‑value indicator block together, which destroys per‑indicator scale and
  makes trailing indicators meaningless; it is kept only so old checkpoints can
  still be loaded. (A `block`‑trained checkpoint that is evaluated with
  `per_feature` produces garbage — this is exactly the mismatch that is now
  impossible by default.)
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
Two runs, 80 episodes, `per_feature`, `target_entropy 0.3`, honest mask, no buy‑margin,
`action_floor=0`, position‑aware observation (394‑dim), checkpoint selected by stochastic
**validation** excess with a buy & hold floor. Deterministic greedy policy on the held‑out
test window:

**Bull regime — train ≤2023‑06, val 2023H2, test 2024** (all four tickers)

| ticker | fly % | buy&hold % | excess | fly Sharpe | fly maxDD % | bh maxDD % | exposure |
|--------|------:|-----------:|-------:|-----------:|------------:|-----------:|---------:|
| AAPL | 36.65 | 36.65 | 0.00 | 1.670 | 11.75 | 11.75 | 1.00 |
| TSLA | 136.80 | 126.60 | **+10.20** | 1.835 | 29.90 | 29.90 | 1.00 |
| NVDA | 74.94 | 90.48 | −15.55 | 1.504 | 27.05 | 27.05 | 0.99 |
| MSFT | 5.86 | 5.23 | +0.63 | 0.425 | 15.49 | 15.49 | 1.00 |
| **mean** | **63.56** | **64.74** | **−1.18** | **1.359** | **21.05** | **21.05** | **0.99** |

**Bear regime — train ≤2020, val 2021, test 2022** (TSLA/NVDA/MSFT; AAPL has no pre‑2022 data)

| ticker | fly % | buy&hold % | excess | fly maxDD % | bh maxDD % |
|--------|------:|-----------:|-------:|------------:|-----------:|
| TSLA | −58.31 | −58.31 | 0.00 | 71.43 | 71.43 |
| NVDA | −39.83 | −39.83 | 0.00 | 60.80 | 60.80 |
| MSFT | −17.62 | −17.62 | 0.00 | 31.77 | 31.77 |
| **mean** | **−38.59** | **−38.59** | **0.00** | **54.67** | **54.67** |

With the buy & hold floor, a trained checkpoint is shipped only if it beats buy & hold on
validation; otherwise the untrained buy & hold prior is written. So the fly can no longer
*lose* to buy & hold (−44.37 pp → −1.18 pp in the bull test), but on these tickers and this
era it also no longer *beats* it. (These numbers are from the bull‑only `default` and
`bear2022` plans. Training on a full cycle that contains a bear does produce a small,
reproducible out‑of‑sample win — see **three‑way verdict** below.)

### Why the fly converges to buy & hold — and what would beat it
This is a data/objective fact, not a bug. Nine standard daily timing rules were run against
buy & hold (transaction costs included):

```
2022 (bear)   buy&hold -44.19 | sma200  -0.88 | trend+vol  -3.47 | sma20  -9.58
2024 (bull)   buy&hold  72.70 | sma200  20.48 | mom60      34.45 | sma20  24.58
2022-2024     buy&hold 125.35 | sma200 254.64 | mom60     179.13 | sma20 165.07
```

1. In a **pure bull year nothing beats buy & hold** — not one rule. Trailing in 2024 was
   never a bug to fix by cleverness.
2. Over a **full cycle** trend/vol rules win by ~130 pp, because avoiding a −50 % drawdown
   compounds. This advantage is invisible if you score one bull year at a time.
3. The edge is **not** per‑day predictability. Pooled next‑day returns conditioned on
   price vs SMA20:

   ```
   2015-2019 pooled   mu|above 0.105%   mu|below 0.246%   edge -0.142 pp
   2015-2025 pooled   mu|above 0.193%   mu|below 0.186%   edge +0.006 pp
   ```

   Below‑trend days had *higher* next‑day returns in the training window. So a per‑step
   `excess` reward sees ~zero conditional edge and correctly optimises to always‑long —
   i.e. to buy & hold.
4. On the **training window** de‑risking has *negative* excess (2015‑2019: buy & hold 492 vs
   sma200 391 vs trend+vol 42), so there is no signal there to learn de‑risking from. Any
   risk penalty that makes flat/under‑exposed states attractive has a trivial optimum —
   a penalty on the agent's own drawdown collapses to all‑cash (this is why `--dd-penalty`
   ships disabled and is documented as a negative result).

The honest conclusion: on these tickers the reachable edge is *regime/tail* timing, which
pays over a cycle but requires training data containing a sustained bear, and which is a
second‑order (variance/compounding) effect that a per‑step reward can barely see. Buy & hold
is the correct optimum for a bull‑dominated training window; the fly now matches it rather
than losing to it.

Reproduce:
```bash
python train.py --all-tickers --plan default  --indicator-norm per_feature --target-entropy 0.3 --episodes 80
python train.py --all-tickers --plan bear2022 --indicator-norm per_feature --target-entropy 0.3 --episodes 80
python train.py --all-tickers --plan cycle    --indicator-norm per_feature --target-entropy 0.3 --episodes 80
```

### Can the fly beat buy & hold? — three‑way verdict (2026‑10)

Three candidate fixes were built and measured against each other on the same
honest full‑cycle test (`cycle_b`: train ≤2020 **including the COVID crash**,
val 2021H1, **test 2021H2–2024** = bear + bull; TSLA/NVDA/MSFT; greedy policy,
honest mask, transaction costs included):

1. **Regime overlay** — a pre‑specified rule that blocks new exposure (and exits)
   while price is below its 200‑day trend (`strategy.regime_gate_prob_fn`; in the
   server via `FLYBRAIN_REGIME_GATE=true`).
2. **Push RL** — the `excess` objective trained on a window that contains a real
   bear, with the position‑ and regime‑aware observation.
3. **Better inputs** — the 4‑dim long‑horizon regime block (`close/SMA200−1`,
   `close/SMA50−1`, annualised 20‑day vol, vol20/vol60) added to the
   observation, so the fly can *see* the regime instead of inferring it.

Reproduce the shipped checkpoint (~15 min):
```bash
python train.py --all-tickers --plan cycle_b --indicator-norm per_feature \
    --target-entropy 0.5 --episodes 100 --select-by val_excess
```

Equal‑weight portfolio on the held‑out test window:

| candidate | return % | max drawdown % | Sharpe |
|-----------|---------:|---------------:|-------:|
| buy & hold | 238.5 | 55.8 | 1.09 |
| **trained fly** (this repo's shipped checkpoint) | **244.7** | 54.5 | **1.11** |
| trained fly + regime gate | 228.7 | **41.5** | 1.26 |
| pure `trend200` rule | **255.8** | **38.4** | **1.32** |

- The **trained fly now beats buy & hold on all three metrics**, but narrowly:
  +6.2 pp greedy, +8.7 pp averaged over 5 stochastic seeds (±11 pp). The gain is
  concentrated in NVDA; on TSLA/MSFT the fly essentially *tracks* buy & hold at
  equal drawdown. It is a real, held‑out edge — not a large one.
- The **regime gate** trades ≈4 pp of return for a much smaller drawdown and a
  higher Sharpe; it is the better choice if you optimise risk‑adjusted return.
- The **pure 200‑day trend rule dominates both** on every metric here. On this
  data there is no evidence that the learned policy's extra machinery adds value
  beyond regime timing — consistent with the conditional‑edge result above.

**Verdict.** The fly used to lose to buy & hold because of the *objective and the
data*, not the network: a bull‑only training window contains no de‑risking signal,
so the `excess` reward correctly converges to "always long" (buy & hold). Training
on a window that contains a bear (`cycle_b`) and giving the policy the regime
block lets it learn *some* de‑risking, and the shipped checkpoint now edges the
benchmark out of sample. The simple trend overlay remains a stronger, cheaper
risk‑adjusted alternative and is one environment variable away
(`FLYBRAIN_REGIME_GATE=true`).

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

### The agent also sees its own position
The observation ends with a 4‑value **portfolio‑state block** —
`[exposure fraction, cash fraction, unrealised PnL, steps since last trade]`.
Without it the policy could not tell "about to open a position" from "already
fully invested", which is why the trained policy used to hover at ~53 % exposure
and structurally could not match a bull run. `config.OBS_DIM = 394`
(390 market + 4 portfolio); the market block is unchanged, and every consumer
that used to index the tail of the observation (e.g. `cur_obs[-8:]`) now uses
`config.CUR_INDICATOR_SLICE`.

### Buy & hold is the floor, not the target
The actor is initialised so its default action is **Buy 100 %, then Hold** —
i.e. buy & hold. A trained checkpoint is shipped **only if its validation excess
return is strictly positive**; otherwise `train.py` writes the untrained
buy & hold prior instead. The deployed policy therefore cannot lose to buy &
hold by construction — it can only add value on top of it.

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

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
- Data: daily OHLCV for **54 US tickers** in `data/`. The four original
  single‑asset files did **not** share a date range (AAPL is 2022-01-03 →
  2024-12-31 (~753 rows) while TSLA/NVDA/MSFT are 2015-01-02 → 2025-12-30
  (~2,765 rows)); the 52 tickers added for portfolio work all share one
  identical 2,765‑row calendar, which is what lets the portfolio env align a
  cross‑section (`PortfolioEnv` refuses a universe whose calendars differ).
- `train.py`: offline online‑AC trainer with a proper time split and
  validation‑based checkpoint selection. Runs end‑to‑end.
- `scripts/backtest.py`: honest out‑of‑sample evaluation vs buy‑&‑hold,
  momentum, trend‑200 and random baselines, all sharing one masking/selection
  code path; the dashboard chips use the same `strategy.baseline_probs`
  implementation.
- `scripts/data_manifest.py`: writes SHA‑256/date‑coverage metadata for the
  locally cached CSVs, so a clean clone can identify the exact data snapshot.
- `scripts/seed_sweep.py`: trains the exact recipe across independent seeds and
  reports mean, sd, and a t‑based 95 % seed interval. The portfolio recipe is
  n=10, **+290.2 pp** mean test excess, **[+214.0, +366.4] pp**, 10/10 positive.
  The single-asset recipe is n=3, **+44.4 pp**, but the interval is
  **[−96.0, +184.8] pp** and only 2/3 seeds beat buy & hold.
- `scripts/eval_seed_ci.py`: reports evaluation-path noise for one checkpoint.
  The deployed single-asset checkpoint is **238.6 % vs 238.5 %** over 10 sampled
  paths, excess **+0.15 pp** with a 95 % interval of **[−2.28, +2.58] pp** — flat.
  This is policy-sampling noise, not retraining noise.
- `scripts/cost_stress.py`: reruns the deployed portfolio checkpoint at
  0/10/25/50/100 bps. The cycle and 2022 edges survive 100 bps; the 2024 arm is
  negative at every cost.
- `scripts/wider_universe.py` and `scripts/connectome_ablation.py`: reproducible
  A/B studies for the 43‑name training universe and the FlyWire circuit. Their
  JSON artifacts are committed under `reports/`; the results are reported with
  negatives (the connectome contribution is not statistically significant).
- `scripts/verified_cache_ab.py`: swaps the legacy cache for the byte‑reproducible
  pinned rebuild, runs the matched A/B, and restores the deployed cache. The
  genuine cache does not rescue the connectome claim: paired cycle excess is
  **+40.2 pp** with a 95 % interval of **[−218.1, +298.5] pp**.
- `web-app/server.py`: FastAPI backend + Three.js front‑end demo. Boots, steps,
  trains and evaluates. Its **Portfolio** tab serves the cross‑section backtest
  (`/portfolio_eval`) with baselines and equity curves.
- `/train_batch` trains on the **train split only**, reports **median +
  held‑out (test)** results, and saves a dashboard-only `fly_brain_live.pt`
  sidecar (never the deployment checkpoint); `/eval_episode` is split‑aware and
  returns a buy‑&‑hold reference.
- One shared masking implementation (`strategy.apply_action_mask`) used by the
  API, the trainer and the backtest (previously four copy‑pasted variants).
- Portfolio mode: a shared policy allocates one budget across N assets + cash on
  a monthly horizon, with an optional **breadth gate** (`topk`, or `positive` =
  the held breadth k is a policy output). Validated on ticker‑holdout universes
  and written up with negatives in *Adaptive breadth* below; the dashboard's
  Portfolio tab serves the real backtest.

**Known limitations / caveats**
- **Inconsistent data ranges.** Training/evaluating on the full file reports
  ~11‑year buy‑and‑hold returns for TSLA/NVDA/MSFT (TSLA ≈ +3000%,
  NVDA ≈ +38,700%) while AAPL is a 3‑year window — the two are not comparable.
  That is the origin of the "3000% episode" number: it is roughly TSLA's own
  move, and the reported `best_return_pct` was the **max over stochastic
  in‑sample episodes**. Training now defaults to the train split and reporting
  uses the median + a held‑out number instead of that max.
- The cached `flywire_weights/connectome_weights.pkl` has **unverified weight
  values**. Its 59,445 non‑zeros are exactly the pinned selection's **58,421
  off‑diagonal FlyWire edges plus 1,024 diagonal self‑connections**, so the
  support is real. The values do not match a fresh degree‑normalised rebuild
  (correlation 0.07 same‑orientation, 0.90 against the transpose), so they are
  a legacy transformation, not a byte‑reproducible build. Audit/regenerate with
  `scripts/rebuild_connectome.py --selection flywire_weights/connectome_selection_1024.json`;
  the pinned rebuild reproduces 58,421 real synapses. A second
  `web-app/flywire_weights/` cache contained a **1,130‑edge file‑order** matrix;
  default cache paths are now anchored to the repository root, so starting the
  server from `web-app/` can no longer silently select it. Audit artifact:
  `reports/connectome_audit.json`.
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
- **Indicator scaling is fit on the train split only.** `train.py`,
  `train_portfolio.py`, `run_portfolio.py`, `scripts/backtest.py`, and the
  server now pass a `(mean, std)` scaler fitted on the training window into
  validation/test/live environments. Old scripts that construct
  `FlyTradeEnv(...)` without `indicator_stats=` retain the previous
  self-normalising behaviour for checkpoint compatibility, but that path is not
  walk-forward safe. Existing checkpoints should be retrained before comparing
  new numbers with the tables in this file.
- `fly_connectome/` is an unmodified upstream React template, used only for its
  atlas binaries. It is a git submodule with the required upstream credit.
- Local data/cache directories are not committed: `data/` (OHLCV),
  `flywire_data/` (raw FlyWire), and most experiment checkpoints. The deployed
  demo checkpoints and the two connectome caches are deliberate small
  exceptions; see *Data and artifacts*.

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
│   portfolio_env.py           # Multi-asset long-only portfolio environment (softmax allocation + breadth gate)
│   train_portfolio.py         # Multi-asset trainer: distill the rule, learn the cross-section allocation
│   run_portfolio.py           # Portfolio baselines (buy&hold / trend200 / momentum) + fly (any gate)
│   xsection.py                # Cross-sectional (universe-relative) features
│   build_xsection.py          # Build data/xsection_panel.npz
│   data_fetch.py              # Reproducible OHLCV downloader (manifest-aware)
│   start_server.ps1           # Windows convenience launcher for the dashboard
│   .gitmodules                # Pins the upstream fly-connectome-template submodule
│   scripts/
│   │   backtest.py            # Honest out‑of‑sample evaluation harness
│   │   rebuild_connectome.py  # Rebuild connectome weights with auditable provenance
│   │   data_manifest.py       # SHA‑256 / coverage manifest for the local data
│   │   seed_sweep.py          # Train a recipe across seeds, report 95 % intervals
│   │   eval_seed_ci.py        # Evaluation-seed interval for one existing checkpoint
│   │   cost_stress.py         # 0/10/25/50/100 bps transaction-cost stress test
│   │   wider_universe.py      # Narrow vs 43‑name portfolio-universe A/B
│   │   connectome_ablation.py # FlyWire vs random-weight circuit, matched A/B
│   │   verified_cache_ab.py   # Legacy vs byte-reproducible pinned-cache A/B
│   reports/                    # JSON artifacts for the studies above
│   tests/                     # pytest suite (env invariants, masking, metrics)
│   web-app/                   # FastAPI server + front‑end demo
│   web-app/checkpoints/       # Deployed demo checkpoints (other runs stay ignored)
│   flywire_data/              # FlyWire v783 raw files (Completeness csv + Connectivity parquet)
│   flywire_weights/           # Cached processed weight matrices
│   fly_connectome/            # Upstream git submodule (atlas binaries only)
└── data/                      # Downloaded OHLCV csv per ticker
```

## Install
Requires Python 3.11+ (verified on 3.14).
```bash
git clone --recurse-submodules <your-fork-or-repo-url>
cd fly-trader
python -m pip install -r requirements.txt
```
If the repository was cloned without submodules:
```bash
git submodule update --init --recursive
```

## Data and artifacts
The raw OHLCV CSVs and FlyWire exports are deliberately not committed. The
deployed demo checkpoints (`fly_brain.pt`, `portfolio_fly.pt`) and the two
connectome caches are small enough to ship, so a clean clone only needs the
OHLCV snapshot below to run; experiment checkpoints stay local and gitignored.

Recreate the OHLCV snapshot from the checked-in manifest (existing files are
skipped; add `--force` to refresh):
```bash
python data_fetch.py --from-manifest data_manifest.json
python build_xsection.py          # only needed for cross-sectional features
```
`data_manifest.json` records SHA-256, row counts and date coverage. Regenerate
it after changing the local data:
```bash
python scripts/data_manifest.py
```
FlyWire raw files auto-download on the first connectome run, or explicitly:
```bash
python -c "from fly_brain import download_flywire_data; download_flywire_data()"
python scripts/rebuild_connectome.py --neurons 1024 --normalize degree
```
The dashboard writes runtime state to
`web-app/checkpoints/simulation_history.json`; that file is ignored.

## Quick‑Start
1. **Run the tests**
   ```bash
   python -m pytest tests -q
   ```
2. **Train** (time‑split: train = everything before `2023‑07`, val
   `2023‑07 → 2023‑12`, test `2024`), selecting the best checkpoint by
   validation excess return (`--select-by val_excess`, the default):
   ```bash
   python train.py --ticker AAPL
   python train.py --all-tickers --episodes 40
   ```
3. **Evaluate honestly (out‑of‑sample)** against baselines. The default
   checkpoint is the deployed `fly_brain.pt`; the old `best` alias is retired:
   ```bash
   python scripts/backtest.py --ticker AAPL --split test
   python scripts/backtest.py --all-tickers --split test --repair
   ```
4. **Run the live demo** (FastAPI + web UI):
   ```bash
   python -m uvicorn web-app.server:app --port 8000
   ```
   On Windows, `.\start_server.ps1` does the same from the repository root.
   The dashboard's **Portfolio** tab runs the held‑out cross‑section backtest
   (`GET /portfolio_eval`): the fly versus buy & hold, trend 200d and momentum
   rotation on the unseen ticker universe, with equity curves. It serves
   `web-app/checkpoints/portfolio_fly.pt`; the recipe that produces it is in
   *Adaptive breadth* below. Portfolio mode is configured by environment
   variables (`FLYBRAIN_PORTFOLIO_CKPT`, `…_PLAN`, `…_SPLIT`, `…_TICKERS`,
   `…_GATE`, `…_MIN_K`, `…_MAX_K`).
   The **Universe & Budget** card picks any 1–54 stocks and a starting budget at
   runtime (`POST /config`): one ticker runs the single‑asset fly, two or more
   switch the same session to the portfolio fly (monthly rebalance). All live
   episodes are normalised to 2022‑01‑03 → 2024‑12‑31 so changing tickers never
   changes the test window.
   **Continuous sizing** (on by default) turns the policy's own 7‑way
   distribution into an expected size — `p₁·25% + p₂·50% + p₃·100%` for buys
   (fraction of cash) or sells (fraction of the position) — so any value from 0
   to 100% is reachable; a one‑hot policy reproduces the discrete 25/50/100
   ladder exactly. This is an execution rule applied *after* the learned action:
   the headline backtest numbers above were measured with the discrete ladder.
   The dashboard's **Momentum / Mean‑Revert / Random** chips are resolved by
   the server through `strategy.baseline_probs`, not approximated in the
   browser; manual trade buttons still override them. Portfolio mode ignores
   the per‑asset chips because its shared allocator owns the book.
   The oscilloscope plots each region's **p99 peak** membrane potential rather
   than the mean, because the mean sits at the −85 mV clip and hides all
   spiking; the marked threshold is −38.5 mV, the model's own `V > 0.7`.

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
  split. Each episode records its `ticker` and date range. Episodes cycle the
  12 training names (`MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG`); the
  `heldout` block evaluates the 10 unseen names
  (`NVDA,JPM,JNJ,XOM,WMT,CAT,META,DIS,T,LIN`), which are never trained on.
- `POST /promote` is the only dashboard path that can replace the deployment
  checkpoint. It evaluates `fly_brain_live.pt` and `fly_brain.pt` on the
  validation split, and copies the sidecar into deployment only if the candidate
  beats both the current deployment and buy & hold there. It never evaluates the
  test split. A rejected candidate leaves both files untouched.
- `POST /eval_episode {"ticker": ..., "split": "test"}` is split‑aware and
  returns `buy_hold_return_pct` and `excess_vs_buyhold_pct` for context.
- `GET /health` is a cheap liveness probe; `GET /checkpoint_info` exposes the
  loaded deployment checkpoint metadata and whether the dashboard‑only live
  sidecar exists.

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

### Honest out‑of‑sample results (historical 2026‑08 run)
This table predates the current 398‑dim retrain and is kept as an ablation
record; use the retrain table above for current numbers. Two runs, 80 episodes, `per_feature`, `target_entropy 0.3`, honest mask, no buy‑margin,
`action_floor=0`, position‑aware observation (398‑dim), checkpoint selected by stochastic
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
    --target-entropy 0.5 --episodes 100 --select-by val_excess \
    --out web-app/checkpoints/fly_brain.pt
```

Equal‑weight *mean of per‑ticker test episodes* on the held‑out test window
(TSLA, NVDA, MSFT; one model, equal starting cash per episode, not a
rebalanced portfolio):

| candidate | return % | max drawdown % | Sharpe |
|-----------|---------:|---------------:|-------:|
| buy & hold | 238.5 | 59.0 | 0.83 |
| **trained fly (current retrain)** | **241.1** | **58.8** | **0.83** |
| trained fly + regime gate | 249.7 | 44.1 | 0.79 |
| pure `trend200` rule | **255.8** | **43.6** | **0.80** |

- The shipped checkpoint **edges buy & hold by +2.6 pp greedy**, but a 10‑path
  evaluation rerun is flat: mean **238.6 %** vs **238.5 %**, excess
  **+0.15 pp** with a 95 % evaluation interval of **[−2.28, +2.58] pp**
  (`scripts/eval_seed_ci.py`, `reports/single_eval_ci.json`). It shaves only
  0.3 pp off drawdown. The gain is concentrated in NVDA (+5.8 pp); TSLA is
  exactly buy & hold (one trade), while MSFT adds +2.0 pp.
- Retraining the same 100‑episode recipe on seeds 0/1/2 gives test returns of
  **219.3 / 302.0 / 327.4 %** against **238.5 %** buy & hold. Mean excess is
  **+44.4 pp**, but the seed sd is **56.5 pp** and the 95 % training‑seed interval
  is **[−96.0, +184.8] pp**; only **2/3** seeds beat buy & hold
  (`scripts/seed_sweep.py`, `reports/single_train_seed_sweep.json`). The deployed
  checkpoint is the *weak* seed, and the retraining interval easily includes
  zero. The single-asset edge is therefore **not established**. Reproduce with:

  ```bash
  python scripts/seed_sweep.py --mode single --seeds 0,1,2 \
      --tickers TSLA,NVDA,MSFT --plan cycle_b --episodes 100 \
      --target-entropy 0.5 --select-by val_excess
  ```

- The **regime gate** cuts max drawdown to 44.1% and raises return to 249.7%,
  but Sharpe (0.79) is lower than the ungated policy (0.83); it is a
  risk‑reduction overlay, not a free win.
- The **pure 200‑day trend rule still edges both** on every metric here. On this
  data there is no evidence that the learned policy's extra machinery adds value
  beyond regime timing — consistent with the conditional‑edge result above.

**Verdict.** The deployed policy is essentially a buy & hold tracker with a small
NVDA/MSFT tilt; the honest result is "not worse than buy & hold," not a reliable
edge. The 3‑seed retrain interval is wide and includes zero, so no single-asset
edge claim survives. The simple trend overlay remains a stronger, cheaper
risk‑adjusted alternative and is one environment variable away
(`FLYBRAIN_REGIME_GATE=true`).

### Learning core: what was broken, and what "more training" now does (2026‑10)

The trainer could not exploit more training because the online update was
**diverging**. The critic bootstrapped its TD target off *its own* network (the
"deadly triad"), so the reported loss grew without bound (`0 → −52`), the actor
logits saturated, and extra episodes made things *worse*, not better. What
changed (all in `trainer.py` / `FlyBrainAgent.train_online_step`):

- **Target critic** — a Polyak‑averaged copy of the critic supplies the
  bootstrapped value, and TD targets are **clamped** (`target_clip=10`). The
  critic loss is now bounded (`≈0.003–0.5`) and `|V|≈2` instead of exploding.
- **n‑step returns** (`--n-step 5`) — each stored transition carries its own
  `gamma**n`, so the compounding, benchmark‑relative signal is credited over the
  horizon where de‑risking actually matters (1‑step TD could not).
- **Persistent reward normalisation** — the (tiny) excess reward is standardised
  against a Welford estimate that **survives across episodes** instead of resetting
  every episode, so the critic no longer chases a moving reward scale.
- **LR schedule** (`--lr-decay 0.99`) — a long run anneals the learning rate
  instead of bouncing around at a fixed step size.
- **Diagnostics** — every update publishes `agent.last_diag`
  (`critic_loss`, `value_abs_mean`, `entropy`, …), and `train.py` prints it, so a
  diverging run is obvious immediately.

The read‑out shape (`128/64`) was deliberately **kept**, so existing checkpoints
still load bit‑for‑bit; the `Dropout` layers became `nn.Identity` (identical
parameter names, no train/eval noise). New checkpoints carry architecture
metadata (`obs_dim`, neuron count, read‑out mode, feature width) and fail loudly
on a mismatch instead of silently leaving random tensors in place. Legacy
checkpoints without metadata retain the tolerant loader for compatibility, and
`pytest.ini` scopes collection to `tests/` so a plain `pytest` works (the repo
root has tracked ad‑hoc `test_*.py` scripts).

**Honest result: stabilising the learner made it converge to buy & hold.** With
the update fixed, a 200‑episode `cycle_b` run does not beat the shipped
early‑stopped checkpoint — it correctly drives entropy to `0`, trading to `1–3`
orders/episode, and test return to buy & hold. The one high‑validation snapshot
(ep 10, val excess +9.2) *lost* −53.8 pp out of sample, i.e. it was a short‑val
fluke, not signal. Reproduce:

```bash
python train.py --all-tickers --plan cycle_b --indicator-norm per_feature \
    --target-entropy 0.5 --n-step 5 --lr-decay 0.99 --episodes 200 \
    --select-by val_excess --no-guard --out /tmp/fly_run.pt
```

The takeaway is the same conclusion as above, now with a stable learner in place:
on a bull‑dominated window the *optimal* learned policy **is** buy & hold, so
out‑performance has to come from regime timing (the `trend200` overlay), not from
more episodes. The learning core is now a sound foundation for that — it will
optimise whatever edge the objective actually contains instead of blowing up.

### Deployed strategy: learned fly + volatility targeting (2026‑10)

The server deploys a selectable strategy. The default is the **learned fly with a
pre‑specified volatility‑targeting overlay** (`FLYBRAIN_STRATEGY=fly`,
`FLYBRAIN_VOL_TARGET=0.30`): exposure is sized by inverse realised volatility, so
the policy de‑risks automatically when volatility spikes. The overlay level was
**selected on the validation split** (2021H1): 0.30 scored +11.6% mean versus
+10.9% for the raw policy, and lower for 0.2/0.4/0.5. It is not a free win on
every held‑out window, however:

| window (OOS, greedy, honest mask) | buy & hold | trend200 | raw fly | fly + vol‑target |
|-----------------------------------|-----------:|---------:|--------:|---------------------:|
| 2022 bear return %                | −38.6 | **−18.4** | −38.4 | −34.6 |
| 2024 bull return %                | 74.1 | 52.6 | 74.2 | **76.0** |
| full cycle 2021H2–2024 return %   | 238.5 | **255.8** | 241.1 | 225.0 |
| • Sharpe                          | 0.83 | 0.80 | **0.83** | 0.82 |
| • mean trades                     | 1 | 28 | **9** | 24 |
| AAPL demo 2022–2024 return %      | 51.4 | 53.4 | 50.3 | **54.7** |

The overlay improves the 2022 bear and the 2024 bull, but gives up **16 pp** over
the full cycle relative to the raw policy (225.0% vs 241.1%). It is a
regime‑dependent risk tradeoff, not a universal improvement. The raw fly remains
the stronger full‑cycle policy and still trades only ~9 times on average versus
~28 for `trend200`. The AAPL row is a demo, not a clean holdout: AAPL has no
pre‑2022 rows, so its scaler is fitted on the only history it has.

```bash
FLYBRAIN_STRATEGY=fly         # default: learned policy + vol-targeting
FLYBRAIN_VOL_TARGET=0.30      # overlay level (validation-selected); 0 = raw policy
FLYBRAIN_STRATEGY=trend200    # pre-specified max-return overlay (fewer trades, sits flat)
FLYBRAIN_STRATEGY=buy_hold    # passive benchmark
FLYBRAIN_STRATEGY=momentum    # most trades, lowest return
FLYBRAIN_REGIME_GATE=true     # apply the trend gate on top of a learned "fly"
FLYBRAIN_ACTION_FLOOR=0.03    # force churn: many more trades, much lower return
FLYBRAIN_TARGET_ENTROPY=0.5   # online-learning entropy target (matches the checkpoint)
```

#### What we tried to lift the fly's return (2026‑10)

A systematic sweep, all scored with the honest harness, with every tunable chosen
on the validation split (never the test split):

| idea | verdict |
|------|---------|
| Prob‑blended fly × trend200 / momentum | ✗ not robust: 321% on the full cycle collapses to −29% in 2022 (test‑set selection bias) |
| Regime switches (bull/bear experts) | ✗ no better than `trend200` |
| Meta‑labeling (logistic trade filter) | ✗ classifier 0.648 accuracy vs 0.646 base rate — forward returns not predictable |
| `dd_penalty` retrain | ✗ over‑corrects: best bull Sharpe (1.91) but craters the cycle (70.1%) and AAPL (−10.6%) |
| `logret` (absolute‑return) retrain | ~ best drawdown protection (2022 −18.2, ties `trend200`) and cycle Sharpe (1.39), but gives up return (158%) |
| Seed ensemble (3 flies, prob‑averaged) | ✗ no consistent gain over a single seed (222.1 full cycle) |
| **vol-target overlay** | **regime-dependent** - helps 2022/2024/AAPL, hurts full-cycle return (225.0% vs 241.1%) |

The learned policy is still shipped, and the portfolio policy still beats buy &
hold on the multi‑asset cycle. The single‑asset default keeps the
validation‑selected vol‑target overlay; the table above shows its honest
full‑cycle cost, and `FLYBRAIN_VOL_TARGET=0` serves the raw policy.
`/eval_episode` uses the same honest
mask, vol‑target overlay, regime gate, and continuous sizing as live serving so
the dashboard replay and the deployed path cannot silently diverge.

#### Server online‑learning fixes (2026‑10)

Three bugs made the server's "Train" button worse than useless:

1. **It wiped what it learned.** `/train_batch` re‑initialised the actor back to
   the buy & hold prior (and cleared the buffer/optimizer) whenever entropy
   dipped below 40% of max — so the brain could never commit to a strategy
   across episodes. It now only records telemetry; the entropy tuner handles
   exploration.
2. **Live/train clicks selected checkpoints on in‑sample scores.** A live episode
   or a short dashboard batch used to save the "best" checkpoint from an
   in‑sample Sharpe, and `/train_batch` then reloaded that older file, discarding
   the policy it had just trained. Live auto‑reset is now read‑only with respect
   to checkpoints; `/train_batch` persists its latest policy to
   `checkpoints/fly_brain_live.pt`, a dashboard sidecar that startup ignores.
   Deployment selection belongs to the validation‑based offline trainers, which
   write `checkpoints/fly_brain.pt`. Startup prefers that file and no longer
   rewrites model files from a health check.
3. **The exploration target was near‑uniform.** `target_entropy` defaulted to
   1.30 nats (max is ln 7 ≈ 1.95), which degenerated the policy into ~2000
   random trades per episode. The server now matches the value the checkpoint
   was trained with (0.5).



## Multi‑asset portfolio trading (2026‑10 scope expansion)

Everything above trades **one ticker at a time**. The scope is now a whole
**cross‑section**: the same single‑asset policy is applied to every asset each
step and allocates capital across the book.

```bash
python run_portfolio.py --plan cycle_b --split test        # baselines + fly
python train_portfolio.py --plan cycle_b --episodes 400    # learn the allocation
```

### Allocation rule (the important design choice)

Each asset's 7‑way action is read as a **conviction** `d_a ∈ [-1, 1]`
(`[Hold, Buy25, Buy50, Buy100, Sell25, Sell50, Sell100] → [0, .25, .6, 1, -.25, -.6, -1]`)
and the convictions are turned into weights by a **softmax over one fixed budget
that includes a cash slot**:

```
w_a  = exp(beta * d_a) / (exp(0) + sum_b exp(beta * d_b))     # beta = 2
cash = 1 / (exp(0) + sum_b exp(beta * d_b))
```

This matters. The first version used *independent* per‑asset target weights, but
the reward `w_a * ret_a` is monotone in `w_a`, so each asset's head is pushed to
"all‑in" — and after renormalisation that is *exactly equal weight = buy & hold*,
a degenerate optimum. Worse, the trained single‑asset checkpoints oscillate in
that setting: **−9.7 % with 8,376 trades** on the held‑out portfolio. The softmax
rewards `ret_a - ret_portfolio` instead, i.e. *tilting into the relatively
strongest names* plus *raising cash when the whole cross‑section falls* (the cash
slot earns 0) — the multi‑asset generalisation of the trend edge below.

The per‑asset reward is decomposed and **immediate**, so the shared policy gets a
clean local credit signal:

```
r_a = w_a * ret_a − cost_a − turnover_penalty * |dw_a|
```

`turnover_penalty` is the direct, explicit fix for the churn measured in the
cross‑sectional single‑asset run (~360 trades ≈ −70 pp of cost).

### Held‑out portfolio results (`cycle` test = 2021H2–2024, 10 unseen tickers)

Equal‑weight, long‑only, transaction costs (10 bps) included. This is the
current `train_portfolio.py` recipe from below, evaluated once on the unseen
ticker holdout with `run_portfolio.py`:

| strategy | return % | maxDD % | Sharpe | turnover |
|----------|---------:|--------:|-------:|---------:|
| buy & hold (equal weight, enters once) | 100.8 | 21.9 | 1.13 | 1 |
| `trend200` (monthly decision) | 105.1 | 17.4 | 1.37 | 18.1 |
| momentum 120d top‑3 (monthly) | 163.3 | 24.4 | 1.27 | 22.2 |
| **fly, distilled adaptive breadth** | **446.2** | **13.7** | **1.77** | **15.3** |

The fly is a 10‑episode distillation of the momentum rule plus the adaptive
breadth gate; it is not an RL result. The separate `train_portfolio.py` and
`run_portfolio.py` paths agree on the 446.24% number for seed 0. A 10-seed rerun
puts the mean at **390.9 % (+290.2 pp)** with a 95 % seed interval of
**[314.7, 467.1] %**; see *Seed distribution* below. Treat the +345.5 pp
headline as the best of ten seeds, not the expected value.

Previous results in the ablation sections below predate the corrected
drift/turnover accounting and may not be directly comparable.

Two findings:
1. **The cross‑section is where the return is.** Rotating across 10 tickers beats
   equal‑weight buy & hold by a wide margin out of sample — far more than any
   single‑asset timing trick found above. This is the multi‑asset
   generalisation of the momentum edge.
2. **The allocation rule, not the features, caused the churn.** With the
   softmax‑budget allocation the *untrained* prior already tracks buy & hold with
   near‑zero turnover (`turnover/step ≈ 0.00`), so churn is a fixable property of
   the action semantics rather than something the policy must learn to avoid.

### The learned fly beats buy & hold out of sample (monthly rebalance)

With daily rebalancing the fly chased noise and lost to buy & hold by 37–46 pp
(see the diagnosis below). Changing **only the decision horizon** to monthly —
so the reward matches the horizon at which cross‑sectional momentum pays — makes
the learned policy win on the original split, across four arms. The robustness, β
and walk‑forward analyses further down all revise this downward: the honest
expectation is a small edge of roughly **+3 pp**, not the +26 pp shown here.

| arm (`rebalance_every=21`) | val excess | test fly % | test excess | Sharpe | maxDD % | bh maxDD % |
|----------------------------|-----------:|-----------:|------------:|-------:|--------:|-----------:|
| H no position obs | +14.9 | **111.3** | **+26.5** | **1.29** | **19.3** | 24.9 |
| F **whole‑brain read‑out** | **+33.5** | 111.0 | **+26.1** | 1.25 | 19.3 | 24.9 |
| E baseline read‑out | +11.4 | 105.4 | +20.6 | 1.20 | 20.2 | 24.9 |
| G cross‑sectional + brain | +30.4 | 97.1 | +12.3 | 1.10 | 24.5 | 24.9 |
| `--target-entropy 0.5` (still running) | +0.6 / +3.6 | — | — | — | — | — |

Held‑out test = 2021H2–2024 on **10 tickers the model never trained on**, equal
weight long‑only, 10 bps per side. Under the honest protocol the val‑selected
checkpoint is F (whole‑brain, val +33.5 pp), which lands at **+25–26 pp out of
sample with a lower drawdown than buy & hold** (19.3 % vs 24.9 %) and a better
Sharpe (1.25 vs 1.04). Verified through two independent code paths
(`train_portfolio.evaluate` and `run_portfolio.run_fly`), which agree to ~0.4 pp.
Turnover is ~21 units over 58 monthly decisions (≈0.02/step, ~380 small trades),
so this is not a cost‑churn artefact.

**But one portfolio in one window is a single sample.** Re‑running the same
val‑selected checkpoint across *disjoint* sets of never‑trained tickers and
across regimes (greedy, `--full-readout`, monthly) shows the edge is real but
much smaller on average than the headline:

| case (10 tickers, monthly) | fly % | bh % | **excess** | mom % | trend % | maxDD fly/bh |
|---|---:|---:|---:|---:|---:|---:|
| cycle test (2021H2–24), original holdout | 110.9 | 84.8 | **+26.1** | 163.6 | 105.2 | 19.3 / 24.9 |
| bear2022 test (2022) | 1.2 | −4.6 | **+5.8** | 0.9 | 5.3 | 16.9 / 18.8 |
| default test (2024) | 21.1 | 28.0 | −6.9 | 43.5 | 28.2 | 3.6 / 3.3 |
| subset S2 (ABBV…CSCO) | 33.8 | 33.6 | +0.2 | 76.2 | 43.5 | 20.1 / 21.7 |
| subset S3 (CVX…MS) | 61.4 | 58.1 | +3.3 | 132.8 | 68.4 | 12.7 / 12.2 |
| subset S4 (NFLX…WFC) | 5.4 | 12.7 | −7.3 | 5.4 | −3.4 | 25.1 / 25.1 |

Across the four **disjoint never‑trained** portfolios the fly beats buy & hold by
a mean of **+5.6 pp** (median +1.7 pp), winning 3 of 4, and it wins the 2022 bear
by +5.8 pp with less drawdown. So the honest claim is: **the fly has a small,
positive, but noisy edge over buy & hold on unseen cross‑sections** — the +26 pp
figure is the top of that distribution, not the expected value. It also trails
the simple momentum rule everywhere, which is the stronger, lower‑variance
cross‑sectional signal.

### The allocation temperature β is a first‑class lever

Allocation is `w = softmax(beta * conviction)`. With β=2 and the small
conviction differences the policy actually produces, the book stays close to
equal weight — i.e. close to buy & hold — regardless of what the policy has
learned. Sweeping β **at evaluation time** on the same checkpoint (no
retraining, so this is a pure read‑out of what the policy already knows):

| β | val excess | test excess | test maxDD % |
|---:|---:|---:|---:|
| 1 | +0.96 | +11.3 | 19.8 |
| 2 *(trained at)* | +5.4 | +26.1 | 19.3 |
| 3 | +10.6 | +30.1 | 19.3 |
| 4 | +11.1 | +32.4 | 19.4 |
| 6 | +17.2 | **+33.0** | 19.4 |
| 8 *(val‑selected)* | **+34.3** | +7.8 | 23.5 |
| 12 | +31.4 | +12.4 | 22.8 |

Test improves **monotonically in β up to 6** and collapses only at the top end.
The policy contains a far stronger tilt than β=2 exposes: it was trained under a
temperature that is too soft for the convictions it learns.

**But β cannot be selected on validation** — this is the important caveat. Using
the *same* selection surface as the trainer (val = the 12‑asset training
universe, test = the 10 unseen tickers):

| β | val fly % (train universe) | val excess | test excess (unseen) |
|---:|---:|---:|---:|
| 1 | 73.5 | +18.6 | +11.3 |
| 2 *(trained at)* | 101.6 | +46.7 | +26.1 |
| 3 | 127.4 | +72.5 | +30.1 |
| 4 | 145.7 | +90.8 | +32.4 |
| 6 | 169.6 | +114.7 | **+33.0** |
| 8 | 185.1 | +130.2 | +7.8 |
| 12 *(val‑selected)* | **193.3** | **+138.3** | +12.4 |

Validation rises monotonically in β out to the corner — in a bull val window,
concentrating on the best train‑universe assets always wins *in hindsight*, so
val can never rank a temperature. A val‑selected β would ship β=12 and land at
+12.4 pp, when β=6 would have delivered +33.0 pp. So the "+33 pp" figure is
**test‑selected and is not a shippable result**; the honest deliverable keeps β=2
fixed a priori. This is the same bull‑window selection trap documented for
`cycle_b`, now appearing in a continuous hyper‑parameter.

Two further attempts to make β selectable were tested and also failed:
- **Selecting on a *validation‑holdout* universe** — 20 tickers the model never
  trained on, disjoint from the test holdout. Mean excess still rose
  monotonically with β (+5.2, +11.4, +19.7, +31.8, +47.8, +51.5, +53.3 pp for
  β = 1…12) and still selected β=12 (+12.4 pp test). The problem is the
  **window**, not the universe: in a bull market, concentrating on whichever
  assets later won is rewarded in hindsight on *any* asset set.
- **A hindsight‑proof score** — val return minus the return of the best single
  asset, so that holding the hindsight winner scores exactly 0. Also monotone
  (−248 → −129 pp across β=1…12), because the val universe contains **TSLA at
  +322 %**, so any lean toward it improves the score.

**Conclusion: β is an a‑priori choice, not a tuned one.** That is acceptable here
only because the test curve is a plateau — β = 2/3/4/6 give
+26.1/+30.1/+32.4/+33.0 pp — so the shipped β=2 sits inside the flat region and
nothing needs to be selected. The deeper lesson is that a **single bull
validation window cannot rank concentration**, which is the binding constraint on
how much of the momentum edge the fly can honestly claim.

Training *at* the sharper temperature does **not** help — it collapses to
buy & hold:

| trained at | val excess (selected checkpoint) | test excess | test maxDD % | turnover/step |
|---|---:|---:|---:|---:|
| **β=2 (shipped)** | +33.5 | **+26.1** | 19.3 | 0.02 |
| β=4 | +25.9 | +5.8 | 23.4 | 0.01 |
| β=6 | +0.1 | +0.6 | 23.4 | 0.01 |
| β=8 | −1.5 | +0.2 | 23.4 | 0.00 |

At high β the early, near‑random convictions are amplified into large allocation
swings, so the reward becomes high‑variance and the policy learns that the safe
move is to keep convictions **flat**. β=8 converges to buy & hold outright
(turnover 0.00, test 84.99 % vs bh 84.77 %). So the temperature must stay at 2
during training; the evaluation‑time sharpening to β≈4–6 is real but, as shown
above, not selectable.

### Walk-forward validation: the fix for the selection bottleneck

Every attempt to tune β failed for one reason: the single validation window
(2020–2021H1) is a bull market, and in a bull market concentrating on whichever
assets later won is optimal *in hindsight* — on any asset set, under any metric.
Selection therefore always walks to the extreme, and nothing can be tuned.

Walk-forward replaces that one window with folds that validate on the period
immediately after their own training window, chosen so that most of them contain
a real drawdown:

| fold | train | validate | validation regime |
|------|-------|----------|-------------------|
| 0 | 2015‑01 → 2018‑06 | 2018‑07 → 2018‑12 | Q4‑2018 drawdown (bh −4.4 %) |
| 1 | 2015‑01 → 2019‑12 | 2020‑01 → 2020‑06 | COVID crash (bh +2.3 %) |
| 2 | 2015‑01 → 2020‑12 | 2021‑01 → 2021‑06 | bull (bh +9.6 %) — the old single window |

All folds share the same held‑out test window (2021H2–2024 on the 10 unseen
tickers), so fold results are directly comparable, and selecting β on the *fold
mean* — which now includes two drawdown windows — is the honest test of whether
β can be chosen at all. Fold 2 is exactly the setting used everywhere above, so
it doubles as a control.

```bash
python train_portfolio.py --tickers <universe> --fold 0 \
    --rebalance-every 21 --full-readout --episodes 400
```

**Results.** Each fold selects its checkpoint on its own validation window and is
then tested once on the common held‑out window (2021H2–2024, 10 unseen tickers):

| fold | validation regime | val excess | **test excess** | test maxDD % | turnover/step |
|------|-------------------|-----------:|----------------:|-------------:|--------------:|
| 0 | Q4‑2018 drawdown | +0.40 | **+8.40** | 26.9 | 0.02 |
| 1 | COVID crash | +3.30 | **−1.81** | 24.1 | 0.01 |
| 2 | bull (2021H1) | +3.47 | **+2.05** | 25.5 | 0.02 |
| **mean** | | **+2.4** | **+2.9** | | |

Two conclusions, neither flattering:

1. **The +26 pp headline was split‑dependent.** Fold 2 differs from the original
   `cycle` arm only in that training extends through 2020 and validation is
   2021H1 (vs training through 2019 and validating on 2020–2021H1). That single
   change moves the *same recipe* from +26.1 pp to +2.05 pp. The honest
   expectation is the fold mean — roughly **+3 pp**, not +26 pp.
2. **Walk‑forward did not make β truly selectable** — the fold‑mean validation
   excess is still monotone in β — but it is now *harmless* rather than
   catastrophic, because the two drawdown windows flatten the curve enormously
   (β=1…8: +2.1 → +6.8 pp, against +18 → +138 pp on the single bull window).
   Val‑selected β=8 delivers a mean test of **+4.76 pp**, versus +5.31 pp at β=6
   and +3.15 pp at the a‑priori β=2. All of these are within noise of each other.

**Bottom line.** The fly beats equal‑weight buy & hold on unseen cross‑sections by
a small amount — of the order of **+3 pp** (range −1.8 to +8.4 across folds, +0.2
to +26 across universes) — at low turnover and with no dependence on a lucky
split. It does *not* approach the simple momentum rotation rule (+79 pp), which
remains by far the strongest honest strategy in this repository.

### Warm-starting from the rule: distillation and the top-k gate

The from‑scratch fly captures the momentum edge only weakly, while a two‑line rule
earns +79 pp. So the rule is distilled into the policy first, and RL then
fine‑tunes from that starting point.

**Distillation.** At each decision bar the rule's book is turned into per‑asset
conviction labels — the top‑k names get the maximal conviction
(`DESIRES[3] = +1`), everything else the minimal (`DESIRES[6] = −1`) — and the
policy is trained by cross‑entropy on those labels before RL begins
(`train_portfolio.py --imitate-episodes N`). With 10 episodes the final
cross‑entropy is ≈ 0.26 (chance is ln 7 ≈ 1.95).

**What it learned was not the rule.** Diagnostic on the held‑out window:

| | distilled fly | mom120‑top3 rule |
|---|---:|---:|
| assets held (> 2 %) | 1.93 | 3.00 |
| cash | 0.10 | 0.00 |
| largest weight | 0.597 | 0.333 |
| average weight NVDA | 0.417 | 0.219 |

The policy holds ≈ 2 names at ≈ double the concentration. The reason is
structural: every feature is *asset‑local*, so a shared per‑asset policy can only
express "my own momentum is above an absolute threshold" — never "I am in the
top‑k of the cross‑section". The breadth it holds on an unseen universe is
therefore an accident of where that universe's momentum levels fall relative to a
threshold learned on the training universe.

**The fix: let the env do the selection (`--alloc-topk k`).** Only the k
most‑convicted assets (plus the cash slot) compete for the budget. The policy now
only has to *rank* assets — which a monotone momentum map does — and the
cross‑sectional competition is supplied by the environment. Mean excess over 9
seed/budget configs (2 seeds × 5/10/20 imitation episodes), each evaluated on 5
never‑trained settings:

| eval setting | no gate | **top‑3 gate** |
|---|---:|---:|
| S1 (contains NVDA/META) | +197 | **+212** (9/9) |
| S2 | −10.5 | −2.5 (3/9) |
| S3 | +5.8 | **+24.2** (8/9) |
| S4 | −10.9 | **+15.4** (6/9) |
| bear 2022 | +18.6 | **+29.8** (8/9) |
| **positive configs** | 24/45 | **35/45** |

**What the gate changed about the book.** Diagnostic on the held‑out window:

| gated policy | fly | rule | buy & hold |
|---|---:|---:|---:|
| assets held (> 2 %) | 1.76 | 3.00 | 10.00 |
| cash | 0.079 | 0.000 | 0.000 |
| largest weight | 0.583 | 0.333 | 0.100 |
| average weight NVDA | 0.339 | 0.220 | 0.100 |
| average weight XOM | 0.229 | 0.114 | 0.100 |
| average weight META | 0.227 | 0.138 | 0.100 |
| average weight T | 0.000 | 0.089 | 0.100 |

The policy learned a **sharper momentum book** — roughly the two strongest of the
three names the rule would hold, plus a cash buffer that grows when little is
trending. That is the concrete mechanism behind both the +237 pp on this holdout
(the two names it picks are NVDA/META/XOM) and the losses on S2/S4 (concentrating
in the wrong two). It is *not* a fidelity‑to‑the‑rule result: making the policy
rank by momentum exactly would reproduce the rule and give up most of this edge.

**A latent bug this exposed.** The trainer suppressed "sell" actions on
zero‑weight assets (`masked_probs`). That mask belongs to *directional* actions;
portfolio actions are absolute conviction levels, so masking them is wrong — and
with the gate it is actively harmful, because the excluded assets are exactly the
ones that need the lowest conviction. The same gated checkpoint scored +106 pp
through the masked trainer path and +237 pp through the unmasked evaluation path;
with the mask disabled for gated runs the two paths agree to the basis point
(+237.40 pp both). The gate now implies `action_mask = False`.

**Top‑3‑gate ablation result** (historical; the current adaptive‑breadth
recipe is the 446.2% table above). Seed 0, 10 imitation episodes, monthly,
whole‑brain read‑out, top‑3 gate, no RL fine‑tune:

| | fly | buy & hold |
|---|---:|---:|
| validation (train universe, 2020–2021H1) | +63.7 pp | — |
| **test (2021H2–2024, 10 unseen tickers)** | **322.2 % (+237.4 pp)** | 84.8 % |
| max drawdown | **18.7 %** | 24.9 % |
| Sharpe | **1.58** | 1.04 |
| turnover / step | 0.01 | — |

Every one of the 9 seed/budget configs was positive on this holdout (+96 to
+427 pp), so this is not a lucky initialisation.

**Caveat, stated plainly.** This is not "the fly discovers the momentum edge from
nothing". The policy is *warm‑started from the rule*; what the experiment shows
is that the rule's information is representable by this policy and that letting
the **environment** perform the cross‑sectional selection is what makes it
transfer to unseen universes. S2 (where the rule itself earns +42.6 pp) is still
slightly negative, so the gate helps broadly but does not make the edge uniform.

**Fine-tuning with RL is, for now, a negative result — and an instructive one.**
Running 300 RL episodes on top of the distilled policy (low exploration, gentle
learning rate, large replay warm‑up) *raised* validation from +63.7 pp to
+82.3 pp and collapsed the held‑out test from +237.4 pp to +32.6 pp. Rewarding the
policy against the validation universe lets it exploit that universe — which
contains TSLA at +322 % — rather than the strategy.

Keeping the teacher in the loop (`--anchor-every 20`, i.e. re‑distil every 20
episodes) prevents that catastrophe but still adds nothing:

| configuration | validation | S1 | S2 | S3 | S4 | bear22 | 2024 | subset mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **distilled only (recommended)** | +63.7 | **+237.4** | −16.7 | **+46.9** | **+22.9** | **+41.2** | **+0.7** | **+72.6** |
| RL, no anchor | +82.3 | +32.6 | — | — | — | — | — | — |
| RL, anchored every 20 ep | **+104.2** | +235.6 | −41.1 | −24.6 | +14.4 | +33.8 | −15.8 | +46.1 |

Anchoring preserves the headline but flips S2/S3 negative, turns 2024 negative and
raises the S1 drawdown from 18.7 % to 23.4 %. **The warm start is the whole win**:
RL either destroys the policy or leaves it roughly where distillation put it,
while reducing its breadth. The recommended configuration therefore ships the
distilled policy with no RL.

**The teacher's hyper‑parameters are not selectable either.** Distilling on
2015–2019 and selecting (lookback, top‑k) on 2020H1 + 2021H1 — both strictly
*after* the imitation window, so the selection is clean — gives:

| lookback | top‑k | val (fly) | val (rule) | S1 | S2 | S3 | S4 | bear22 | mean test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 2 | −17.7 | −0.9 | +292.7 | −35.8 | −24.6 | −8.7 | +37.3 | +52.2 |
| 60 | 3 | −9.8 | −2.4 | +295.9 | +39.2 | +13.5 | −6.5 | −20.2 | +64.4 |
| 60 | 5 | −0.5 | −2.6 | +217.2 | +24.9 | +0.0 | +17.6 | +9.2 | +53.8 |
| 120 | 2 | −1.5 | +7.3 | +145.9 | −8.5 | +68.4 | −2.6 | −3.7 | +39.9 |
| **120** | **3** | **+1.3** | +4.8 | **+237.4** | −16.7 | +46.9 | +22.9 | **+41.2** | **+66.3** |
| 120 | 5 | +0.6 | +0.8 | +174.0 | −15.3 | +30.5 | +24.6 | +38.9 | +50.5 |
| 250 | 2 | **+3.9** | +7.3 | +330.7 | −2.3 | −26.5 | −15.4 | **−41.1** | +49.1 |
| 250 | 3 | −10.2 | +4.8 | +179.1 | +4.7 | −34.3 | −31.4 | +1.7 | +24.0 |
| 250 | 5 | −7.3 | +0.8 | +180.4 | −3.6 | +34.2 | +6.6 | +40.2 | +51.6 |

Validation picks **(250, 2)** → mean test +49.1 pp, and it *loses 41 pp in the
2022 bear*. Each validation window is only ~6 monthly decisions, so they simply
cannot discriminate between these teachers. The conventional a‑priori choice
**(120, 3)** is both better on test (+66.3 pp mean) and the only one that
combines a large edge with proper de‑risking (+41.2 pp in 2022). So the teacher
stays fixed at 120‑day top‑3 on theoretical grounds, not because validation chose
it.

**Final honest recipe and result.** Distil 120‑day momentum top‑3 into the policy
with the allocation gated to the top‑3 convictions, monthly rebalancing,
whole‑brain read‑out, and ship the distilled policy (no RL). Across the five
never‑trained settings the mean excess over equal‑weight buy & hold is **+66 pp**,
positive in **4 of 5**:

| setting | fly excess | notes |
|---|---:|---|
| S1 holdout (10 unseen tickers, 2021H2–2024) | **+237.4 pp** | 322.2 % vs 84.8 %, maxDD 18.7 % vs 24.9 % |
| S2 | −16.7 pp | the rule itself earns +42.6 pp here |
| S3 | +46.9 pp | |
| S4 | +22.9 pp | the rule *loses* 21 pp here |
| 2022 bear | **+41.2 pp** | capital preservation |

Reproduce the recommended policy exactly (`--episodes 0` = distillation only, no RL):

```bash
python train_portfolio.py \
    --tickers MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG \
    --plan cycle --episodes 0 --rebalance-every 21 --full-readout \
    --alloc-topk 3 --imitate-episodes 10 --out portfolio_fly_momentum.pt

python run_portfolio.py --plan cycle --split test --rebalance-every 21 \
    --full-readout --alloc-topk 3 --ckpt portfolio_fly_momentum.pt
```

(`--alloc-topk 3` must match `--teacher-topk 3`, and both evaluation paths agree
to the basis point.)

```bash
python train_portfolio.py --tickers MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG \
    --plan cycle --episodes 400 --rebalance-every 21 --full-readout
python run_portfolio.py --plan cycle --split test --rebalance-every 21 --full-readout \
    --ckpt <out>
```

**Three ingredients were each necessary**: (1) the multi‑asset scope with the
softmax conviction allocation, (2) the monthly horizon so the reward is not
next‑day noise, and (3) the whole‑brain read‑out, which gives the best validation
score of any arm (+33.5 pp) — the "connect all the brain" change contributes here.
Two tempting knobs were tested and rejected: an explicit `turnover_penalty` (the
agent churned anyway at a daily horizon) and `--target-entropy 0.5` (much worse
than the agent's own 1.30 in the monthly setting: +0.6/+3.6 pp val).

The fly captures roughly a third of the simple momentum rule's excess
(+26 pp of +79 pp) while trading differently and holding less risk than
buy & hold; `trend200` remains the better pure risk‑adjusted rule
(Sharpe 1.37, maxDD 17.4 %).

### Adaptive breadth: letting the fly choose *how many* names to hold

The top‑k gate above fixes the book size by hand, and that is a real weakness:
the fixed k is the one thing about the allocation that the policy cannot see.
`alloc_gate="positive"` removes it — **the breadth k becomes a policy output**.
Only the assets the policy itself flags with a *positive conviction* (one of the
three buy actions) compete for the budget, plus the cash slot, so the fly can
widen and narrow its own book decision by decision, and an all‑negative book is
a learned 100 % cash de‑risk. `alloc_min_k` / `alloc_max_k` bound it
(`k = clamp(#positive, min_k, max_k)`), which is what keeps it from
over‑reacting, as the measurements below show.

All numbers are the honest ticker‑holdout backtest (10 bps/side, monthly
rebalance, greedy, unseen tickers), averaged over **3 seeds**; `(n/3)` is how many
seeds were positive on that setting. `meanCMN` averages each seed's mean over the
five settings every configuration has in common.

| configuration | S1 | S2 | S3 | S4 | H20 | 2022 bear (S1) | 2022 bear (H20) | meanCMN | sd | names held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **fixed top‑3** (`--alloc-topk 3`) | +271.6 (3/3) | +4.5 (2/3) | +38.4 (3/3) | +27.4 (2/3) | +340.2 (3/3) | +41.9 (3/3) | +37.3 (3/3) | **+143.7** | 36.2 | 3.0 always |
| adaptive, unbounded (`--alloc-gate positive`) | +267.0 (3/3) | −18.0 (1/3) | −26.8 (1/3) | +9.5 (1/3) | +333.7 (3/3) | +44.1 (3/3) | +40.2 (3/3) | +138.9 | 36.9 | 0.9 – 2.2 |
| adaptive, floor 3 (`--alloc-min-k 3`) | +266.4 (3/3) | +4.5 (2/3) | +35.8 (3/3) | +27.4 (2/3) | +317.7 (3/3) | +41.9 (3/3) | +37.4 (3/3) | +138.1 | 44.6 | 3.0 – 3.2 |
| **adaptive, [2, 4]** (`--alloc-min-k 2 --alloc-max-k 4`) | +259.4 (3/3) | −12.2 (1/3) | +21.8 (3/3) | **+65.5 (3/3)** | +314.8 (3/3) | +42.8 (3/3) | +38.1 (3/3) | **+144.1** | 37.8 | 2.0 – 2.5 |
| [2, 4] but trained on **32** tickers | +249.7 (3/3) | in‑sample | in‑sample | +55.5 (3/3) | +150.8 (3/3) | +21.8 (3/3) | +13.6 (3/3) | +98.3 | **9.8** | 2.0 – 2.5 |

Three honest conclusions.

**1. Breadth is genuinely a policy output, and it adapts.** The unbounded gate
holds 0.9 names on S2 and 2.2 on H20 — the same weights, a different book size
for every universe — while the fixed gate holds 3.0 by construction. There is
also a real regime signal in it: the unbounded gate *widens* the floor of its
own risk in the 2022 bear (+44.1 / +40.2 pp, the best bear numbers in the table).

**2. …but the policy's conviction scale does not transfer, and that is why the
floor matters.** The unbounded gate's *absolute* threshold under‑fires on an
unseen cross‑section: it drops to 1/3 positive on S2/S3/S4 and gives back
most of the edge there. This is the same structural limitation as before —
asset‑local features can express "my momentum is above a level", not "I am in the
top‑k" — so `[2, 4]` keeps the policy's choice *inside* a sane range. With the
floor, the adaptive gate is a **statistical tie on returns** with the fixed gate
(+144.1 vs +143.7, sd ≈ 37, n = 3) and a **clear win on the ambiguous‑breadth
universe**: S4 +65.5 pp with 3/3 seeds positive, versus +27.4 pp with only 2/3
for fixed‑3. `--alloc-min-k 3` ("top‑3, or wider when the policy is more
positive") reproduces fixed‑3 almost exactly, which is the sanity check that the
gate is faithful when it should be.

**3. The wider training universe is a *consistency* win, not a return win —
report it as a negative for returns.** Training on 32 tickers instead of 12 and
testing on the 20 that remain unseen in both cases cuts the mean by ~46 pp
(+98.3 vs +144.1) and, more importantly, **halves the crisis protection**
(2022 bear +13.6 vs +38.1 pp on H20). What it buys is by far the most stable
result in the table (sd 9.8 vs 37.8) and 3/3 positive on every setting. More
diversity makes the policy more average; it does not make it better.

#### Wider-universe update: a sector-diverse 43-name arm recovers the 2022 protection

The 32-name result above used an ad-hoc list that was not committed, so the
wider-universe question was rerun with both universes pinned
(`scripts/wider_universe.py`, 3 seeds, same 10 unseen tickers, same recipe):

| arm | training names | cycle test excess | 2022 bear excess | cycle maxDD | bear maxDD |
|---|---:|---:|---:|---:|---:|
| narrow (shipped recipe) | 12 | **+280.3 pp** (CI −127.7 … +688.4) | +21.0 pp (CI −8.8 … +50.8) | 20.3 % | 12.6 % |
| wide (all long-history non-holdout) | 43 | +245.9 pp (CI +185.9 … +306.0) | **+40.2 pp** | 22.5 % | **8.1 %** |

The wide arm **recovers and improves the 2022 protection** (+40.2 vs +21.0 pp)
and is much more stable across seeds, at the cost of ~34 pp of mean full-cycle
return and a slightly higher full-cycle drawdown. Its 2022 interval is
degenerate because all three seeds took the same 2022 action path; that is a
property of this deterministic distillation, not a claim of zero market risk.
The result does not contradict the historical 32-name arm directly: the
composition matters, and the 43-name set includes defensive sectors. It is a
robustness result, **not a deployment change**: the wide arm was positive on
validation in only 1/3 seeds, so validation cannot honestly select it. The
12-name checkpoint stays deployed; the wide arm is the better crisis hedge if a
future selector can choose it without touching test data.

Reproduce the shipped adaptive policy (`--episodes 0` = distillation only; seed 0,
which is the run the dashboard serves):

```bash
python train_portfolio.py \
    --tickers MSFT,TSLA,AMD,GOOGL,AMZN,KO,COST,HD,V,MA,ORCL,PG \
    --plan cycle --episodes 0 --rebalance-every 21 --full-readout \
    --alloc-gate positive --alloc-min-k 2 --alloc-max-k 4 \
    --imitate-episodes 10 --out web-app/checkpoints/portfolio_fly.pt

python run_portfolio.py --plan cycle --split test --rebalance-every 21 \
    --full-readout --alloc-gate positive --alloc-min-k 2 --alloc-max-k 4 \
    --ckpt web-app/checkpoints/portfolio_fly.pt
```

⇒ `fly=446.24%  bh=100.75%  excess=+345.49pp  sharpe=1.77  maxDD=13.69% (bh 21.85%)
avg_positions=2.88`. The gate flags used to train a checkpoint must be passed
again to `run_portfolio.py` or the allocation will not match.

**Seed distribution, stated plainly.** The 10-seed rerun of the exact shipped
recipe (`scripts/seed_sweep.py`, seeds 0–9, same 10 unseen tickers) is now in
`reports/seed_sweep/portfolio_summary.json`:

| metric (unseen test) | mean | sd | 95% seed interval |
|---|---:|---:|---:|
| fly return | **390.9 %** | 106.5 | **[314.7, 467.1]** |
| buy & hold | 100.8 % | 0.0 | 100.8 |
| fly excess | **+290.2 pp** | 106.5 | **[+214.0, +366.4]** |
| Sharpe | 1.61 | 0.19 | [1.48, 1.75] |
| max drawdown | 18.2 % | 7.5 | [12.9, 23.6] |

All 10 seeds beat buy & hold on the unseen holdout, but seed 0 was the best
single run (446.2 %, +345.5 pp). The honest headline is therefore **+290 pp on
average**, not +345 pp. Four seeds also failed the validation guard
(`val_excess < 0`); the trainer shipped the pre-RL distilled candidate in those
runs, and all four still beat buy & hold on test. That is a useful warning:
validation selection is noisy too, and a seed that fails validation is not
evidence that the policy is worthless. The interval is over *training seeds on
one shared history and universe*, not a forecast interval. Reproduce with:

```bash
python scripts/seed_sweep.py --mode portfolio --seeds 0,1,2,3,4,5,6,7,8,9
```

The earlier fixed‑3‑gate ablation reported +237 / +427 / +150 pp across seeds
0/1/2, and `[2, 4]` reported +243 / +418 / +117 pp. Those numbers predate the
corrected accounting and are not directly comparable.

#### Cost stress: 10 bps was not the whole story

`scripts/cost_stress.py` reruns the deployed checkpoint and the buy & hold
baseline at 0, 10, 25, 50 and 100 bps per side. This is evaluation-only; the
policy is not retrained at each cost. Fly **excess over buy & hold** (pp):

| plan | 0 bps | 10 bps | 25 bps | 50 bps | 100 bps |
|---|---:|---:|---:|---:|---:|
| cycle test | +353.4 | +345.5 | +333.9 | +315.0 | **+279.4** |
| 2022 bear | +15.0 | +14.6 | +14.0 | +13.0 | **+11.0** |
| 2024 bull | −6.0 | −6.6 | −7.5 | −9.1 | **−12.1** |

The monthly, low-turnover cycle and bear edges survive a 10x higher flat cost.
The 2024 arm underperforms at every cost level, so the cost stress does not
rescue that negative. Slippage, spread widening and market impact are still not
modelled; the 100 bps row should be read as a *floor* on real friction, not a
worst case. Artifact: `reports/cost_stress.json`.

### The dashboard runs the portfolio fly

The web app's **Portfolio** tab (`GET /portfolio_eval`) runs exactly the backtest
above — not a simulation of it: same env, same checkpoint, same 10 bps — and shows
the fly against buy & hold, trend 200 d and momentum top‑3 rotation on the unseen
universe, with log‑scaled equity curves and a strategy table. It reads
`web-app/checkpoints/portfolio_fly.pt` and can be pointed elsewhere with
`FLYBRAIN_PORTFOLIO_CKPT` (plus `…_PLAN`, `…_SPLIT`, `…_TICKERS`, `…_GATE`,
`…_MIN_K`, `…_MAX_K`). If the checkpoint is missing the tab explains how to build
it instead of failing.

Live portfolio steps take the **argmax** of the fly's policy. With a fixed
window and a fixed checkpoint that makes an episode replay to exactly the same
return, which is deliberate: it keeps the live tab equal to the backtest. So
Auto-Pilot **stops** at the end of a portfolio episode instead of looping an
identical path. Tick **Explore** in the allocation pane to sample from the
policy instead; runs then differ (and no longer match the evaluated path).
Single-asset mode already samples (temperature 0.65) and learns online, so its
runs vary.

### Combo Lab: which k names should the fly be allowed to trade?

The **Portfolio** tab also has a Combo Lab: pick 2–8 stocks from the universe
on the left, choose a basket size from **Pairs** through **Eight-stock**, and
the deployed portfolio fly tries subsets of that size. The lab scores at most
240 candidates per run:

- **All combos** when `C(n, k) ≤ 240` — every subset is scored.
- **Guided** otherwise — score every pair, keep the strongest `beam` baskets,
  and extend those one stock at a time. This is still a validation-only search,
  and the winner is tested once on the unseen split.
- If even all starting pairs exceed 240, the run is refused with a clear
  "narrow the pool" message instead of starting an unbounded job.

The card previews the mode, candidate count, and guided budget before the run.
Each subset is ranked by the **fly's own validation return**, with excess over
that basket's buy & hold shown on every row; only the winner is then run once on
the **test** window.

Why return rather than excess? An earlier build ranked by excess, and it picked
a triple whose *benchmark* was weaker, not a combo where the fly traded better:
adding `T` to `NVDA·XOM` lowered the basket's buy & hold while the fly's P&L was
unchanged, so the triple "won" by −61 pp in a pair becoming +66 pp in a triple.
Ties now break toward the simpler basket for the same reason. Excess remains the
honesty column — it says whether the return came from the fly or from beta.

- `GET /combo_meta` reports the Combo Lab's validation/test windows and which
  universe tickers have enough history. The card marks short-history names with
  a dot, explains why they are skipped, and disables a run that cannot reach the
  chosen basket size before sending a request.
- `POST /combo_search` `{tickers, size}` starts a background job (one at a time,
  ≤240 candidate rollouts); `GET /combo_search/{id}` returns progress, the live
  leaderboard, the winner and its out-of-sample test row. The response includes
  `search.mode`, `search.beam`, `search.seed_count`, and `search.total` so the
  card can show whether the run was exhaustive or guided.
- `POST /combo_search/{id}/cancel` stops after the current basket; partial
  results stay on screen and are labelled as partial.
- The card shows how many combos beat their own buy & hold and the **median**
  excess, not just the top row. Picking the best of N is a search, so the top
  validation number is optimistic by construction; the test row is the honest
  estimate.
- `test_is_ticker_holdout` distinguishes the strong case (the basket contains
  none of the 12 training tickers) from a time-only holdout.
- The pool is filtered to tickers with at least `window_size + 2 monthly
  rebalances` rows in the **cycle** validation window (2020-01-01..2021-06-30;
  test is 2021-07-01..2024-12-31). AAPL's local file only covers 2022-2024, so
  it is dropped with a visible "skipped" notice rather than crashing the
  search. The universe chip is dotted to show it is unavailable *for the lab*,
  while remaining selectable for a live single-asset session. A subset that
  still fails is kept as a skipped row instead of aborting the other 100+
  combos.

A full run on the 10 holdout tickers (120 triples, 70 s):

| search | val winner | val return | val excess | test (unseen tickers) | test Sharpe |
|---|---|---:|---:|---|---:|
| 120 triples | NVDA · JNJ · XOM | **+173.0 %** | +106.0 pp | **+582.7 % vs 175.6 % (+407.1 pp)** | 1.58 |
| 6 pairs (NVDA/JPM/JNJ/XOM) | NVDA · XOM | +170.4 % | +74.2 pp | +283.4 % vs 344.6 % (**−61.1 pp**) | 1.44 |

Across the 120 triples, 92 beat their own buy & hold (median +18.6 pp), so the
effect is not one lucky row. The 6-pair pool is the counter-example: the same
kind of winner failed to generalise, and the card says so plainly. Both are
single-seed searches over one test path; the distribution is the more reliable
signal than the winner's headline.

### Why the fly churned: the reward *horizon*, not the features

Three daily‑rebalanced arms were trained to 400 episodes on the 12‑asset
universe (`cycle` plan, selection on validation, buy & hold guard active):

| arm | change vs baseline | val fly vs bh | val turnover/step | holdout test vs bh |
|-----|--------------------|--------------:|------------------:|-------------------:|
| A | — (baseline read‑out) | **+13.9 pp** (best) | 0.34 | **−46.4 pp** |
| B | `--full-readout` (whole brain) | +11.3 pp (best) | **0.15** | −36.8 pp |
| C | `--xsection` (relative strength) | −38.1 pp (never beat the guard) | 0.26 | — |

All three *lost badly* out of sample, and the reason is not the features:
- **In sample on its own training data** the greedy daily policy *also* churns —
  10,213 micro‑trades, turnover 417 over 1,228 bars — and still trails buy &
  hold (118.4% vs 130.0%). So the churn is not a test‑time distribution shift.
- At ~0.35 turnover/step × 10 bps, the **transaction‑cost drag alone is ≈26%**
  over the test window. The policy is not adding alpha; it is paying to trade.
- The per‑asset reward at a daily horizon is `w·ret_{1d}`, and the conditional
  edge of these features on *next‑day* returns is ≈0 (measured earlier). The
  gradient is therefore dominated by unpredictable daily noise — the policy
  chases it. The cross‑sectional momentum edge is real but it accrues over
  ~60–120 days.
- Two tempting fixes were tested and rejected: adding an explicit
  `turnover_penalty` twice the actual cost (the agent still churned), and zeroing
  the position block to break the "hold a lot → sell → hold little → buy" limit
  cycle (turnover did **not** fall). The churn survived both — it is a property
  of the *reward horizon*.

### The fix: rebalance on the horizon where the edge lives

`PortfolioEnv(rebalance_every=21)` makes one decision per month, so the reward
is the 21‑bar return — the horizon at which cross‑sectional momentum actually
pays. The effect is immediate and large:

| | daily (`rebalance_every=1`) | monthly (`=21`) |
|---|---:|---:|
| validation turnover/step | 0.34 | **0.003** |
| seconds per training episode | ~30 | **~1.5** |
| momentum top‑3 (holdout, vs bh 84.8) | 162.5 | **163.6** |

The learned policy stops churning the moment the reward matches the horizon of
the signal. Sharpe is also annualised per *decision* now (√12 for monthly, not
√252), so monthly and daily numbers are not directly comparable.

### Which split to select on (a real trap)

Model selection uses the **validation** split only, so the validation window must
contain the edge you are trying to learn. On the 12‑asset training universe:

| split | buy & hold | trend200 | momentum‑top3 |
|-------|-----------:|---------:|--------------:|
| `cycle` train (2015–2019) | 259.8 | 155.3 | **298.7** |
| `cycle` **val** (2020–2021H1, incl. COVID) | 58.5 | 84.0 | **105.9** |
| `cycle_b` **val** (2021H1, bull only) | **9.1** | 5.9 | 2.4 |
| `cycle`/`cycle_b` test (2021H2–2024, holdout) | 100.8 | 105.1 | **163.3** |

All rows use monthly decisions (`--rebalance-every 21`) and the corrected
drift/turnover accounting. Under a monthly environment the “daily” and
“weekly” trend rules recompute at the same decision points, hence one
`trend200` column here.

`cycle_b`'s 6‑month bull validation window *rewards buy & hold and rejects every
rotation rule* — selecting on it can only ever ship the buy & hold prior, even
though the same rotation rule wins by +38.9 pp on train and +62.5 pp on test. The
portfolio runs therefore use `cycle`, whose validation window spans the COVID
crash and recovery and ranks the rotation rules the same way the test window
does. (The test window is identical in both plans, so results stay comparable.)



### "Connect all the brain" — whole‑brain read‑out

The read‑out used to see only the mushroom‑body output (MBON) plus the sensory
summary, i.e. it ignored Antennal Lobe, Kenyon Cells, Central Complex and the
dopaminergic neurons. A diagnostic confirmed every region is alive and
input‑dependent (MBON emitted 52/60 distinct patterns over 60 real observations;
the "all‑zero MBON" comment in older docs is stale), so the fix is *how* the read‑out
uses the brain, not resurrecting it. `--full-readout` / `use_full_readout=True`
adds, for every region, its **mean spike rate** plus a fixed random projection of
the whole 1024‑neuron spike vector, so the policy can weight each region
(`feature_encoder` in‑dim 88 → 101). It is opt‑in, so existing checkpoints still
load bit‑for‑bit.

| read‑out | in‑dim | sees |
|----------|-------:|------|
| default | 88 | MBON(64) + sensory(24) |
| `--xsection` | 96 | + cross‑sectional block(8) |
| `--full-readout` | 101 | MBON + **all 5 regions** + sensory |

#### Does the connectome itself contribute? A matched A/B says "not yet"

The whole-brain read-out makes every region *visible*, but visibility is not the
same as the FlyWire wiring doing useful work. `scripts/connectome_ablation.py`
holds the recipe, seeds, read-out, and 10 unseen tickers fixed and swaps only the
circuit: the pinned FlyWire matrix vs the deterministic random-weight fallback
(same 1,024 neurons, same region sizes). The checkpoints now record
`use_connectome`, so the comparison cannot silently evaluate against a different
architecture.

| arm | cycle excess (mean) | 2022 bear excess | 2024 excess |
|---|---:|---:|---:|
| connectome | +280.3 pp | +21.0 pp | +11.4 pp |
| random control | +211.0 pp | +17.5 pp | +12.5 pp |
| paired difference (connectome − random) | +69.3 pp (CI −186.6 … +325.3) | +3.5 pp (CI −11.2 … +18.3) | −1.1 pp (CI −25.5 … +23.3) |

The point estimate favours the connectome on the full cycle, but the seed
interval is far too wide to claim a contribution; the random control actually
wins one of three seeds on the headline window and is slightly better in 2024.
The honest conclusion is that **the current read-out does not yet demonstrate
that the FlyWire edges contribute** beyond a fixed random projection plus the
sensory summary. The connectome remains a biologically grounded feature
generator and a display/ablation surface, but the README no longer claims it is
the source of the trading edge. `--no-connectome` is now a first-class matched
control in `train_portfolio.py` and `run_portfolio.py`.

#### Verified pinned rebuild does not change the conclusion

The deployed cache's support is real (58,421 pinned FlyWire edges plus 1,024
diagonal self-connections), but its values are a legacy transform. I rebuilt the
cache byte-for-byte from the pinned selection (58,421 real edges, no synthetic
self-connections), ran the same 3-seed A/B, and restored the deployed cache
afterward. The random-control difference is exactly 0.0 pp, confirming the swap
only affected the connectome arm.

| plan | verified connectome | random control | paired connectome − random |
|---|---:|---:|---:|
| cycle | +251.2 pp | +211.0 pp | +40.2 pp (CI −218.1 … +298.5) |
| 2022 bear | +36.3 pp | +17.5 pp | +18.8 pp (CI −19.8 … +57.4) |
| 2024 | +28.3 pp | +12.5 pp | +15.8 pp (CI −36.8 … +68.4) |

Paired verified-minus-legacy differences are also indistinguishable from zero:
cycle **−29.2 pp** (CI −244.7 … +186.4), 2022 **+15.3 pp** (CI −18.8 … +49.4),
2024 **+16.9 pp** (CI −50.1 … +84.0). So the genuine circuit is not worse, but it
is not a demonstrated source of edge either. The conclusion stays: the FlyWire
matrix is a biologically grounded feature generator, not a proven alpha source.

Reproduce with:

```bash
python scripts/connectome_ablation.py --seeds 0,1,2
python scripts/verified_cache_ab.py --seeds 0,1,2
```

Artifacts: `reports/connectome_ablation.json`,
`reports/connectome_ablation_verified.json`,
`reports/connectome_cache_ab.json`, `reports/connectome_audit.json`.


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

### What the 3‑D view actually draws (2026‑10)

The dashboard's brain had two honesty problems that looked like "detached
parts", and both are now fixed:

1. **No wires.** `synapticPairs` was declared and never filled, so the
   *Tracts* toggle built a `LineSegments` with 0 vertices — the anatomy was
   real, but nothing connected it. The view now renders the model's actual
   matrix (`brain.simulator.W_dense`, 58,421 non‑self synapses) via
   `GET /connectome_edges.bin?top=N`, which returns float32
   `(source, target, |w|)` triples, strongest first. The default 3,000
   strongest are drawn as curved tracts, coloured by source block and pulsed
   brighter for one decay when their source neuron fires.
2. **Scattered blocks.** The five blocks were filled with the *first* central‑
   brain somata in file order, which are spatially random: only **42.2 %** of
   each neuron's 10 nearest neighbours were in its own block, versus **43.0 %**
   expected by chance. The blocks are now seeded by farthest‑point sampling and
   refined with Lloyd iterations on the real somata, then filled with evenly
   sampled somata inside a 0.40‑unit radius: **100 %** 10‑NN purity with the
   same 128/640/128/64/64 counts, so each colour is one visible lobe.

Both are display choices, not biology: the positions are measured MaleCNS
somata (`fly_connectome/public/data/brain-atlas`, CC BY 4.0) and the edges are
real, but the **block names remain structural** (see the table above and
`cell_type_verified: False`). The atlas manifest is explicit that it contains
"somata only, not neurites or the synaptic connectivity graph" — the wiring
comes from the model's FlyWire‑derived `W`, not from the atlas.

### Neuron count: 1,024 vs 4,096 (2026‑10)

The MaleCNS atlas has 140,024 measured somata and the raw FlyWire export has
138,639 neurons, so 1,024 is a choice, not a limit. It was kept because extra
neurons do not buy extra *usable* signal:

| model | synapses | encode | MBON dims | MBON eff. rank | 5‑day linear probe (test r) |
|-------|---------:|-------:|----------:|---------------:|----------------------------:|
| shipped 1,024 (pinned) | 59,445 | 0.68 ms | 64 | 17.2 | +0.061 (null 0.196) |
| 4,096 random rows | 12,851 | 0.41 ms | 256 | 15.4 | −0.098 (null 0.227) |
| 4,096 degree core | 400,700 | 1.61 ms | 256 | 10.9 | −0.020 (null 0.227) |

The probe is a time‑ordered ridge from MBON rates to the next 5‑day return
(502 train / 216 test observations, no shuffle). None of the sizes clears the
multiple‑comparison null, and the effective rank does not grow with neuron
count — the frozen read‑out is the bottleneck, not the population size.
Compute is not the constraint either: 400k synapses still encode in 1.6 ms per
observation.

A matched‑budget A/B (10 tickers, 20 cross‑section episodes, seed 0, same plan)
points the same way: the 1,024 agent never beat buy & hold on validation (best
val excess −0.13 %), so the guard shipped the buy & hold prior (test 28.21 %,
identical to the benchmark); the 4,096 agent did clear the threshold by
+0.72 % on validation and then lost **5.51 pp** out of sample (22.69 % versus
28.21 %), trading 6–33 times per ticker instead of once. One seed and a short
budget make this a pilot, not a headline, but a 0.7 % validation edge that
turns into a 5.5 pp loss is exactly the kind of noise a bigger read‑out is
more likely to fit.

Where extra capacity *does* matter is sample size, not neurons. Pooling 10
tickers (~27,300 observations) lifts the same 5‑day ridge probe from r=+0.06
on 723 AAPL rows to **+0.31 from MBON alone** and +0.39 with the sensory block
(null95 0.11–0.12) — so the fly's representation does carry signal once there
is enough data to fit the read‑out. But 4,096 neurons still does not beat 1,024:
ridge +0.382 versus +0.386, MBON‑only +0.281 versus +0.308, MLP +0.359 versus
+0.369. All ten tickers share the 2022–2025 test window, so this is one market
regime, not ten independent confirmations. A larger brain is the right lever for
a *different* problem: many more parallel streams than the 64‑dim MBON / 8‑dim
cross‑section read‑out can carry, a brain that keeps temporal state (this one is
stateless per observation), or anatomical fidelity rather than P&L. Reproduce
with `python scripts/probe_scaling.py`.

Two of those benefits can be built on the existing 1,024 neurons, and both were
measured on the same pooled probe:

| read‑out | dimensions | test r |
|---|---:|---:|
| MBON + sensory (shipped) | 88 | +0.386 |
| + 256‑dim projection of all 1,024 spikes | 280 | **+0.424** |
| + causal 0.9 spike trace | 152 | +0.412 |
| both | 344 | **+0.441** |

The wide projection is what 4,096 neurons would offer as 256 MBONs; the trace
supplies the temporal state the stateless encoding lacks. Both are opt‑in
(`--full-readout --full-readout-proj 256`, `--memory-rho 0.9`) and need a
checkpoint trained with them — the shipped checkpoints keep the 88‑dim layout.
Memory is single‑asset only: a cross‑section is many assets at one instant, not
one asset across time. Reproduce with `python scripts/probe_readout.py`.

The matched policy A/B (10 tickers, 20 episodes, seed 0) is more sober: the
wide+memory agent cleared the validation threshold by +0.37 % and actually
traded (1–12 times per ticker, versus the buy & hold fallback's single trade),
but finished **−0.58 pp** behind buy & hold on test (27.62 % versus 28.21 %,
Sharpe 1.05 versus 1.09, max drawdown 15.8 versus 15.7). Better behaved than the
4,096 arm (−5.51 pp), still not an edge. The features are real; the policy is
not yet profitable at this budget.

The portfolio path was the strongest hope, and it is the clearest negative.
Same recipe, same seed, same 10 unseen tickers — only the projection width
changed (10 imitation episodes, 12 assets):

| projection width | val vs B&H | test vs B&H | Sharpe | maxDD |
|---|---:|---:|---:|---:|
| **8 (shipped)** | **+64.8 pp** | **+242.5 pp** (327.3 % vs 84.8 %) | 1.58 | 18.9 % |
| 64 | −11.8 pp (guard rejects) | +147.5 pp | 1.43 | 12.1 % |
| 256 | +37.5 pp | +38.5 pp (123.3 % vs 84.8 %) | 1.12 | 14.9 % |

Wider features improved the ridge probe but *degraded* the distilled policy:
the fixed 128/64 read-out has to learn from 349 inputs with 10 imitation
episodes, and the extra dimensions dilute the signal it was fitted on. The
shipped 8-dim checkpoint stays deployed.

Parallelism is a different axis. Extra neurons add per‑step cost (0.49 ms/obs
at 1,024, 1.59 ms at 4,096) and run nothing concurrently. What does help is
settling B independent observations through the same `W` as one `(N, B)`
matmul — `encode_batch` — which is **5.7×** faster per observation at 1,024
(0.09 ms) and **4.4×** at 4,096 (0.36 ms), with bit‑identical spikes; the
portfolio cross‑section (12 assets) now scores in 2.4 ms instead of 6.9 ms.
Encoding is stateless, so a whole episode can be encoded in chunks ahead of
time. Beyond one process, independent agents scale across cores (12 here) —
more neurons do not.

Reproducibility: the deployed circuit is a **pinned row selection**
(`flywire_weights/connectome_selection_1024.json`), not rows 0..1023. File
order yields **1,130** synapses; the pinned selection yields **58,421**.
`connectome_selection_4096.json` pins a degree‑ranked core (400,700 synapses).
`scripts/rebuild_connectome.py` uses the matching file automatically and records
its sha256 in the cache provenance. The selection reproduces the *edge set*
exactly; the original artifact's weight scaling was never recorded, so a
rebuild is auditable but not bit‑identical (on AAPL the rebuilt 1,024 matrix
spikes at ≈21 % of neurons per observation versus 15.5 % for the shipped one).

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
and structurally could not match a bull run. `config.OBS_DIM = 398`
(390 market + 4 portfolio + 4 regime); the market block is unchanged, and every consumer
that used to index the tail of the observation (e.g. `cur_obs[-8:]`) now uses
`config.CUR_INDICATOR_SLICE`.

### Buy & hold is the floor, not the target
The actor is initialised so its default action is **Buy 100 %, then Hold** —
i.e. buy & hold. A trained checkpoint is shipped **only if its validation excess
return is strictly positive**; otherwise the trainer falls back to the pre-RL
prior. In the single-asset path that prior is effectively buy & hold (the
heuristic starts at `Buy 100%` and then holds); in the portfolio path it is the
distilled pre-fine-tune candidate, not a literal buy & hold policy. So the guard
is a selection heuristic, not a mathematical guarantee that a rejected
checkpoint equals the benchmark. The portfolio seed sweep reports the
guard-rejected seeds explicitly for this reason.

## Reproducibility

- `python scripts/data_manifest.py` writes `data_manifest.json`: byte size,
  SHA‑256, date coverage and per‑split row counts for every local CSV. Commit
  the manifest with published numbers; the raw CSVs stay gitignored.
- The single‑asset recipe above writes checkpoint metadata (plan, train window,
  normalisation, reward mode, selection metric, seed). Startup loads legacy checkpoints
  tolerantly but warns; new checkpoints fail loudly on an architecture mismatch.
- Startup also aligns the serving scaler to the checkpoint's `train_window`.
  Tickers that do not cover that window (AAPL starts in 2022) fall back to the
  server train split; AAPL therefore remains a demo, not a clean holdout.
- The portfolio recipe below pins the ticker universe, plan, rebalance cadence,
  transaction cost and allocation gate in the checkpoint metadata.
- The portfolio headline now has a **10-seed interval** (`scripts/seed_sweep.py`):
  +290.2 pp mean test excess, 95 % seed interval [+214.0, +366.4] pp, 10/10 seeds
  positive.
- The single-asset recipe has both intervals now. The **10-path evaluation
  interval** for the deployed checkpoint is +0.15 pp excess, [−2.28, +2.58] pp
  (`scripts/eval_seed_ci.py`). The **3-seed training interval** is +44.4 pp mean
  excess, **[−96.0, +184.8] pp**, with only 2/3 seeds positive
  (`scripts/seed_sweep.py --mode single`). The training interval is the relevant
  one for a recipe claim, and it includes zero. Observed cost was ~10 minutes per
  100-episode seed on this machine, not the earlier ~2 h estimate.
- Costs are a flat 10 bps per side in the headline; `scripts/cost_stress.py`
  reruns the deployed portfolio checkpoint out to 100 bps. Slippage, borrow,
  spread widening and market impact are still not explicitly modelled;
  large‑cap survivorship/selection bias remains. Neither interval is a forecast
  guarantee.

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
| P1 | DONE - `per_feature` is the default and both trainers fit it on train only |
| P0 | ✅ **Multi‑asset portfolio environment + baselines** (`portfolio_env.py`, `run_portfolio.py`) |
| P0 | ✅ **Fly beats buy & hold on the unseen portfolio holdout** - the current distilled recipe is **+290.2 pp** mean over 10 seeds (95 % interval [+214.0, +366.4] pp), 10/10 positive. Seed 0 is the +345.5 pp best case (446.2 % vs 100.8 %). Earlier scratch/walk-forward runs were far smaller (+2.9 pp fold mean), so this is recipe-specific, not a general claim. |
| P0 | ✅ **Softmax conviction allocation** with an explicit turnover penalty (fixes churn) |
| P1 | ✅ **Whole‑brain read‑out** (`--full-readout`: all regions feed the policy) |
| P1 | ✅ **Portfolio trainer** (`train_portfolio.py`) — learn the cross‑section allocation |
| P1 | ✅ **Adaptive breadth** (`--alloc-gate positive` ± `--alloc-min-k`/`--alloc-max-k`): the held breadth k is a policy output; ties fixed‑3 on mean return and wins on the ambiguous‑breadth holdout |
| P1 | ✅ **Portfolio tab in the dashboard** (`GET /portfolio_eval`) — the real holdout backtest with the three baselines and equity curves |
| P1 | Wire the adaptive breadth into RL so k is *learned*, not just selected by the policy's argmax (needs a pooled cross‑sectional representation — asset‑local features cannot decide breadth) |
| P1 | ✅ Wider-universe bear protection: a sector-diverse 43-name arm lifts 2022 excess from +21.0 to **+40.2 pp** and cuts bear maxDD from 12.6 % to 8.1 %, at the cost of ~34 pp full-cycle return; not validation-selectable, so not deployed (`scripts/wider_universe.py`) |
| P1 | Make the connectome block actually contribute: matched A/B is **not significant** on the legacy cache (cycle +69 pp, CI −187 … +325; 2024 −1 pp) or on the verified pinned rebuild (cycle +40 pp, CI −218 … +298; 2024 +16 pp, CI −37 … +68). The claim is softened and `--no-connectome` is the control. |
| P1 | Wire real cell-type anatomy instead of index ranges — **blocked by local data**: the FlyWire v783 completeness file has only `ID` + `Completed`, no cell-type labels; needs an external annotation source |
| P1 | Genuine subset: pinned selection reproduces **58,421 real FlyWire edges**; cached values remain a legacy unverified transform, and default cache paths are now cwd-independent. Cell-type labels are still unavailable. |
| P2 | DONE - offline trainers select on validation, and `/promote` gates the dashboard sidecar on validation |
| P2 | DONE - `scripts/data_manifest.py` writes SHA-256/coverage manifest |
| P2 | ✅ `scripts/seed_sweep.py` reports multi-seed confidence intervals (portfolio recipe: n=10, +290.2 pp mean, [214.0, 366.4] pp) |
| P2 | ✅ Single-asset evaluation-path interval (+0.15 pp, [−2.28, +2.58] pp, n=10) and **3-seed training interval** (+44.4 pp, [−96.0, +184.8] pp, 2/3 positive). The training interval includes zero; no single-asset edge claim survives. |
| P2 | ✅ `scripts/cost_stress.py` reruns the portfolio checkpoint at 0/10/25/50/100 bps: cycle excess +353 → +279 pp, 2022 +15 → +11 pp, 2024 −6 → −12 pp. Slippage/impact remain unmodelled. |
| P2 | DONE - dead best-score checkpoint code removed; root scratch scripts, duplicate `web-app/` data caches, and stale experiment checkpoints cleaned out of the release tree |

---

## Credits and third-party notices

Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by [Mert Cobanov](https://github.com/cobanov). The `fly_connectome/` submodule is kept unmodified; the dashboard carries the required linked credit in the left rail and the in-app guide.

- **MaleCNS v1.0 soma atlas** — FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source and notices: `fly_connectome/public/data/brain-atlas/NOTICE.md`.
- **Flybody** — Turaga Lab, Apache License 2.0. Source and notices: `fly_connectome/public/data/flybody/NOTICE.md`.
- **FlyWire v783** — connectome export used for the optional synaptic wiring; download/rebuild instructions are in *Connectome Mode* above.

The original FlyTrader code is released under the [MIT License](LICENSE). The third-party components and datasets retain their own licenses described above.

---

## How to Contribute
Open issues or PRs for new instruments, a faster spiking‑network wrapper, better
fly‑brain/strategy visualisations, or improved connectome mapping.

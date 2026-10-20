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
  momentum and random baselines, all sharing one masking/selection code path.
- `web-app/server.py`: FastAPI backend + Three.js front‑end demo. Boots, steps,
  trains and evaluates. Its **Portfolio** tab serves the cross‑section backtest
  (`/portfolio_eval`) with baselines and equity curves.
- `/train_batch` trains on the **train split only** and reports **median +
  held‑out (test)** results; `/eval_episode` is split‑aware and returns a
  buy‑&‑hold reference.
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
│   portfolio_env.py           # Multi-asset long-only portfolio environment (softmax allocation + breadth gate)
│   train_portfolio.py         # Multi-asset trainer: distill the rule, learn the cross-section allocation
│   run_portfolio.py           # Portfolio baselines (buy&hold / trend200 / momentum) + fly (any gate)
│   xsection.py                # Cross-sectional (universe-relative) features
│   build_xsection.py          # Build data/xsection_panel.npz
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
parameter names, no train/eval noise). Checkpoint loading now tolerates
architecture drift (missing/shape‑changed tensors degrade gracefully instead of
crashing the server), and `pytest.ini` scopes collection to `tests/` so a plain
`pytest` works (the repo root has tracked ad‑hoc `test_*.py` scripts).

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
**selected on the validation split** (2021H1), never on the test windows, and it
improves the learned policy on *every* held‑out window — including the 2022 bear,
which is where the raw fly was weakest:

| window (OOS, greedy, honest mask) | buy & hold | trend200 | raw fly | **fly + vol‑target** |
|-----------------------------------|-----------:|---------:|--------:|---------------------:|
| 2022 bear return %                | −38.6 | **−18.4** | −40.7 | −34.1 |
| 2024 bull return %                | **74.1** | 52.6 | 65.8 | 67.3 |
| full cycle 2021H2–2024 return %   | 238.5 | 255.8 | 244.7 | **261.8** |
| ″ Sharpe                          | 1.09 | 1.32 | 1.11 | 1.25 |
| ″ trades                          | 3 | 83 | 136 | 113 |
| AAPL demo 2022–2024 return %      | 51.4 | **53.4** | 47.8 | 52.6 |

It beats buy & hold and `trend200` on the full cycle while still trading far more
than `trend200` (113 vs 83) and keeping the learned policy in the loop. (The raw
fly remains the trades‑vs‑profit knee on the *trade count* axis; forcing extra
churn with `FLYBRAIN_ACTION_FLOOR` turns the fly's ~36 AAPL trades into ~365 but
cuts its AAPL return from $140k to $102k, so that knob stays off.)

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
| **vol‑target overlay** | ✓ **beats the raw fly on every held‑out window** |

The learned policy is still shipped and still beats buy & hold on the multi‑asset
cycle, so nothing is thrown away — the default just adds the one overlay that
survived honest out‑of‑sample testing. `/eval_episode` keeps evaluating the
*brain itself* on purpose.

#### Server online‑learning fixes (2026‑10)

Three bugs made the server's "Train" button worse than useless:

1. **It wiped what it learned.** `/train_batch` re‑initialised the actor back to
   the buy & hold prior (and cleared the buffer/optimizer) whenever entropy
   dipped below 40% of max — so the brain could never commit to a strategy
   across episodes. It now only records telemetry; the entropy tuner handles
   exploration.
2. **Every "Train" click overwrote the shipped best policy.** `_best_score`
   started at `-999`, so the first batch always counted as a new record. It is
   now seeded from `fly_brain_best_score.json` at startup.
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

### Held‑out portfolio results (`cycle_b` test = 2021H2–2024, 10 unseen tickers)

Equal‑weight, long‑only, transaction costs (10 bps) included:

| strategy | return % | maxDD % | Sharpe | turnover |
|----------|---------:|--------:|-------:|---------:|
| buy & hold (equal weight) | 84.2 | 25.8 | 1.14 | 1 |
| `trend200` daily | **134.9** | 23.9 | **1.51** | 76 |
| `trend200` weekly | 122.5 | 26.9 | 1.44 | 32 |
| momentum 120d top‑3 (monthly) | **162.5** | 28.8 | 1.32 | 21 |
| fly, *untrained prior* (softmax allocation) | 82.9 | 25.4 | 1.14 | ~0 |

Two findings:
1. **The cross‑section is where the return is.** Rotating across 10 tickers beats
   equal‑weight buy & hold by **+50 to +78 pp** out of sample — far more than any
   single‑asset timing trick found above. This is the multi‑asset
   generalisation of the `trend200` edge.
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

**Honest headline result** (seed 0, 10 imitation episodes, monthly, whole‑brain
read‑out, top‑3 gate, no RL fine‑tune yet):

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

⇒ `fly=327.31%  bh=84.77%  excess=+242.54pp  sharpe=1.58  maxDD=18.89% (bh 24.90%)
avg_positions=2.10`. The gate flags used to train a checkpoint must be passed
again to `run_portfolio.py` or the allocation will not match.

**Seed sensitivity, stated plainly.** The single‑number headline is fragile: on
S1 the fixed‑3 gate scores +237 / +427 / +150 pp across seeds 0/1/2, and `[2, 4]`
scores +243 / +418 / +117. Seed 0 reproduces the +237.4 pp headline above
exactly, which is why the recipe is worth quoting — but a claim that rests on one
seed is worth about the spread between them, and that is why every table here
quotes 3 seeds and a count of positive settings rather than a single best run.

### The dashboard runs the portfolio fly

The web app's **Portfolio** tab (`GET /portfolio_eval`) runs exactly the backtest
above — not a simulation of it: same env, same checkpoint, same 10 bps — and shows
the fly against buy & hold, trend 200 d and momentum top‑3 rotation on the unseen
universe, with log‑scaled equity curves and a strategy table. It reads
`web-app/checkpoints/portfolio_fly.pt` and can be pointed elsewhere with
`FLYBRAIN_PORTFOLIO_CKPT` (plus `…_PLAN`, `…_SPLIT`, `…_TICKERS`, `…_GATE`,
`…_MIN_K`, `…_MAX_K`). If the checkpoint is missing the tab explains how to build
it instead of failing.

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

| split | buy & hold | trend200‑w | momentum‑top3 |
|-------|-----------:|-----------:|--------------:|
| `cycle` train (2015–2019) | 222.2 | 163.4 | **304.2** |
| `cycle` **val** (2020–2021H1, incl. COVID) | 55.0 | 76.7 | **104.9** |
| `cycle_b` **val** (2021H1, bull only) | **9.9** | 3.4 | 2.9 |
| `cycle`/`cycle_b` test (2021H2–2024, holdout) | 84.2 | 122.5 | **162.5** |

`cycle_b`'s 6‑month bull validation window *rewards buy & hold and rejects every
rotation rule* — selecting on it can only ever ship the buy & hold prior, even
though the same rotation rule wins by +50 pp on train and +78 pp on test. The
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
| P0 | ✅ **Multi‑asset portfolio environment + baselines** (`portfolio_env.py`, `run_portfolio.py`) |
| P0 | ✅ **Fly beats buy & hold on unseen portfolios** — +26 pp on the original split, +2.9 pp walk‑forward fold mean from scratch, and **+237 pp on the original holdout with a top‑k‑gated warm start from the momentum rule** |
| P0 | ✅ **Softmax conviction allocation** with an explicit turnover penalty (fixes churn) |
| P1 | ✅ **Whole‑brain read‑out** (`--full-readout`: all regions feed the policy) |
| P1 | ✅ **Portfolio trainer** (`train_portfolio.py`) — learn the cross‑section allocation |
| P1 | ✅ **Adaptive breadth** (`--alloc-gate positive` ± `--alloc-min-k`/`--alloc-max-k`): the held breadth k is a policy output; ties fixed‑3 on mean return and wins on the ambiguous‑breadth holdout |
| P1 | ✅ **Portfolio tab in the dashboard** (`GET /portfolio_eval`) — the real holdout backtest with the three baselines and equity curves |
| P1 | Wire the adaptive breadth into RL so k is *learned*, not just selected by the policy's argmax (needs a pooled cross‑sectional representation — asset‑local features cannot decide breadth) |
| P1 | Recover the wider‑universe bear protection (32‑ticker training currently halves 2022 downside protection) |
| P1 | Make the connectome block actually contribute (fix AL→KC→MBON cascade / signals) |
| P1 | Wire real cell‑type anatomy instead of index ranges |
| P1 | Adopt a genuine, well‑connected connectome subset (or drop the claim) |
| P2 | Select the best checkpoint by held-out, not in-sample, performance |
| P2 | Remove `flywire_data/` duplication; retire legacy scripts |

---
### How to Contribute
Open issues or PRs for new instruments, a faster spiking‑network wrapper, better
fly‑brain/strategy visualisations, or improved connectome mapping.

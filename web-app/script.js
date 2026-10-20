/* ═══════════════════════════════════════════════════════════════
   FlyTrader – script.js (v0.5)
   MaleCNS 1024-LIF Drosophila Connectome Visualizer & RL Trader
   ═══════════════════════════════════════════════════════════════ */

const API = 'http://127.0.0.1:8000';

/* ─────────────────────────────────────────────
   State
───────────────────────────────────────────── */
let N_NEURONS = 1024; // Updated from server /brain_status at boot
let portfolioHistory = [100000];
let rewardHistory    = [];
let spikeData = new Array(N_NEURONS).fill(0);
let spikeData1024 = spikeData; // Alias for backward compatibility
let voltageHistory   = { al: [], kc: [], cx: [], dan: [], mbon: [] };
let autoPilot        = false;
let autoTimer        = null;
let stepCount        = 0;
let lastAction       = 0;
let currentDopamine  = 0.0;
let totalEpisodes    = 0;
let lastTrainData    = null; // Stores last /train_batch response for the lab
let learningHistory  = [];  // Persistent learning history from /learning_history
let brainNeedsUpdate = true; // Drive 3-D brain refresh from the render loop

/* ─────────────────────────────────────────────
   DOM refs
───────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const qPortfolio   = $('portfolio-val');
const qReturn      = $('return-val');
const qSharpe      = $('sharpe-val');
const qDrawdown    = $('drawdown-val');
const qWinrate     = $('winrate-val');
const qStep        = $('step-val');
const qStatus      = $('status-text');
const qDot         = document.querySelector('.dot');
const qSpikeRate   = $('spike-rate');
const qTradeLog    = $('trade-log');
const qSpeedSlider = $('speed-slider');
const qSpeedVal    = $('speed-val');
const qAutoBtn     = $('auto-btn');
const qAvgReward   = $('avg-reward');
const qLastReward  = $('last-reward');
const qActivityBars = $('activity-bars');
const qDopaminePill = $('dopamine-pill');
const qDopamineVal  = $('dopamine-val');
const qBrainBadge   = $('brain-badge');

// Live session (universe / budget / sizing) and portfolio-mode colours
let sessionMode = 'single';
const LIVE_PF_COLORS = ['#3d8eff', '#00e5ff', '#8c52ff', '#34d399', '#ffb703',
                        '#f87171', '#f59e0b', '#a78bfa', '#22d3ee', '#4ade80'];
const _pfColorMap = {};

// Neuropil meters
const qRateAL   = $('rate-al');
const qValAL    = $('val-al');
const qRateKC   = $('rate-kc');
const qValKC    = $('val-kc');
const qRateCX   = $('rate-cx');
const qValCX    = $('val-cx');
const qRateDAN  = $('rate-dan');
const qValDAN   = $('val-dan');
const qRateMBON = $('rate-mbon');
const qValMBON  = $('val-mbon');

/* ─────────────────────────────────────────────
   Tab Switching
───────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = $('tab-' + tab);
      if (panel) panel.classList.add('active');
      // Redraw canvases that may have been hidden
      if (tab === 'stats') { drawRewardChartBig(); fetchLearningHistory(); fetchSimulationHistory(); }
      if (tab === 'sims')  { fetchSimulationHistory(); }
      if (tab === 'portfolio') { fetchPortfolioEval(); }
    });
  });
}

/* ─────────────────────────────────────────────
   API helpers
───────────────────────────────────────────── */
async function apiPing() {
  try {
    const r = await fetch(`${API}/ping`);
    if (r.ok) setStatus('live', 'API Live');
    else       setStatus('error', 'API Error');
  } catch {
    setStatus('error', 'API Offline');
  }
}

async function apiReset() {
  try {
    const r = await fetch(`${API}/reset`);
    const d = await r.json();
    portfolioHistory = [d.portfolio];
    rewardHistory    = [];
    stepCount        = 0;
    voltageHistory   = { al: [], kc: [], cx: [], dan: [], mbon: [] };
    qTradeLog.innerHTML = '';
    updatePortfolioCard(d.portfolio, 0, 0, 0);
    qSharpe.textContent   = '—';
    qDrawdown.textContent = '—';
    qWinrate.textContent  = '—';
    qStep.textContent     = '0';
    // Reset also re-labels the counter: in a portfolio session the total is
    // the number of monthly decisions, not the single-asset bar count.
    const tbStep = $('tb-step');
    if (tbStep) tbStep.textContent = `0 / ${d.total_steps || d.n_steps || 723}`;
    spikeData = new Array(N_NEURONS).fill(0);
    spikeData1024 = spikeData;
    currentDopamine       = 0.0;
    updateDopamineUI(0.0);
    drawPortfolioChart();
    drawRewardChart();
    drawHeatmap(spikeData);
    updateActivityBars(spikeData, null);
    updateBrainViz(spikeData1024, 0.0);
    drawOscilloscope();
    updateNeuronCountDisplay();
    toast('Episode reset ↺', '');
  } catch(e) {
    setStatus('error', 'API Offline');
  }
}

async function apiStep(manualAction = -1) {
  const policy = document.querySelector('input[name="policy"]:checked')?.value || 'brain';
  const learning = $('learning-toggle')?.checked ?? true;

  let action = manualAction;
  if (manualAction === -1) {
    if (policy === 'brain') {
      action = -1; // Tell server Fly Brain to deliberate
    } else if (policy === 'momentum') {
      action = Math.random() < 0.2 ? 0 : lastAction || 1;
    } else if (policy === 'mean-revert') {
      action = lastAction === 1 ? 2 : lastAction === 2 ? 1 : 1;
    } else {
      action = Math.floor(Math.random() * 3);
    }
  }

  try {
    const r = await fetch(`${API}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, policy, learning })
    });
    if (!r.ok) {
      const errText = await r.text();
      console.error('API Error:', r.status, errText);
      setStatus('error', `API ${r.status}`);
      return null;
    }
    const d = await r.json();
    handleStepResult(d);
    return d;
  } catch (err) {
    console.error('Fetch error:', err);
    setStatus('error', 'API Offline');
    return null;
  }
}

/* ─────────────────────────────────────────────
   Universe / budget / sizing
───────────────────────────────────────────── */
function selectedTickers() {
  return [...document.querySelectorAll('#ticker-chips .ticker-chip.on')]
    .map(b => b.dataset.ticker);
}

function updateUniverseHint() {
  const hint = $('uni-hint');
  if (!hint) return;
  const sel = selectedTickers();
  hint.textContent = sel.length === 0 ? 'Pick at least one stock'
    : sel.length === 1 ? `${sel[0]} · single-asset fly`
    : `${sel.length} stocks · portfolio fly (monthly rebalance)`;
}

function syncBudgetChips() {
  const v = String(Math.round(Number($('budget-input')?.value) || 0));
  document.querySelectorAll('.budget-chip').forEach(c =>
    c.classList.toggle('active', c.dataset.budget === v));
}

function applySessionMode(mode) {
  sessionMode = mode === 'portfolio' ? 'portfolio' : 'single';
  const single = $('delib-single');
  const pf = $('delib-portfolio');
  if (single) single.hidden = (sessionMode !== 'single');
  if (pf) pf.hidden = (sessionMode !== 'portfolio');
  // A basket is traded by the portfolio brain; the manual single-asset
  // override and the per-asset policy chips do not apply.
  document.querySelectorAll('.action-btn').forEach(b => {
    b.disabled = sessionMode !== 'single';
    b.style.opacity = sessionMode === 'single' ? '' : '.45';
  });
}

function renderPortfolioAllocation(d) {
  const stack = $('pf-delib-stack');
  const legend = $('pf-delib-legend');
  const dec = $('pf-delib-decision');
  if (!stack || !legend) return;

  const holdings = Array.isArray(d.holdings) ? d.holdings : [];
  const cash = Number(d.cash_weight) || 0;
  const parts = holdings.filter(h => h.weight > 0.0005)
    .map(h => ({ ticker: h.ticker, weight: h.weight, cash: false }));
  if (cash > 0.0005) parts.push({ ticker: 'Cash', weight: cash, cash: true });

  parts.forEach(h => {
    if (!_pfColorMap[h.ticker]) {
      _pfColorMap[h.ticker] = h.cash
        ? '#4b5b74'
        : LIVE_PF_COLORS[Object.keys(_pfColorMap).length % LIVE_PF_COLORS.length];
    }
  });

  stack.innerHTML = parts.map(h =>
    `<div class="delib-seg" style="flex-grow:${h.weight};background:${_pfColorMap[h.ticker]}"></div>`
  ).join('');

  legend.innerHTML = parts.map(h =>
    `<span class="dl-item"><i class="dl-dot" style="background:${_pfColorMap[h.ticker]}"></i>` +
    `<span class="dl-name">${h.ticker}</span>` +
    `<span class="dl-pct">${(h.weight * 100).toFixed(1)}%</span></span>`
  ).join('');

  const n = parts.filter(p => !p.cash).length;
  if (dec) {
    dec.textContent = `${n} held · ${(cash * 100).toFixed(1)}% cash`;
    dec.className = 'delib-decision';
  }
}

function applyConfigToUI(cfg) {
  const s = cfg.session || {};
  const active = new Set(s.tickers || []);
  const chips = $('ticker-chips');
  if (chips && !chips.childElementCount) {
    chips.innerHTML = (cfg.available_tickers || [])
      .map(t => `<button type="button" class="ticker-chip" data-ticker="${t}">${t}</button>`)
      .join('');
    chips.querySelectorAll('.ticker-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('on');
        updateUniverseHint();
      });
    });
  }
  if (chips) {
    chips.querySelectorAll('.ticker-chip').forEach(b =>
      b.classList.toggle('on', active.has(b.dataset.ticker)));
  }
  const bi = $('budget-input');
  if (bi && s.budget) bi.value = Math.round(s.budget);
  const dt = $('dynamic-sizing-toggle');
  if (dt) dt.checked = !!s.dynamic_sizing;

  applySessionMode(s.mode || 'single');
  updateUniverseHint();
  syncBudgetChips();
}

async function fetchConfig() {
  try {
    const r = await fetch(`${API}/config`);
    if (!r.ok) return null;
    const d = await r.json();
    applyConfigToUI(d);
    return d;
  } catch {
    return null;
  }
}

async function applyUniverse() {
  const btn = $('apply-universe-btn');
  const status = $('uni-status');
  const tickers = selectedTickers();
  if (!tickers.length) {
    status.textContent = 'Pick at least one stock first.';
    status.className = 'uni-status err';
    return;
  }
  const budget = Number($('budget-input')?.value) || 100000;
  const dynamic = $('dynamic-sizing-toggle')?.checked ?? true;
  const label = btn.textContent;

  btn.disabled = true;
  btn.textContent = 'Building…';
  status.textContent = tickers.length > 1
    ? `Building the portfolio env (reads ${tickers.length} CSVs)…`
    : 'Building…';
  status.className = 'uni-status';

  try {
    const r = await fetch(`${API}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers, budget, dynamic_sizing: dynamic })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
    applyConfigToUI(d);
    await apiReset();
    status.textContent = d.session.mode === 'portfolio'
      ? `Applied · ${d.session.tickers.length} stocks · ${d.n_steps} monthly rebalances`
      : `Applied · ${d.session.tickers[0]} · ${d.n_steps} steps · $${Math.round(d.session.budget).toLocaleString('en-US')}`;
    status.className = 'uni-status ok';
    toast('Universe applied', '');
  } catch (e) {
    status.textContent = String(e.message || e);
    status.className = 'uni-status err';
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function initUniverseControls() {
  const chips = $('ticker-chips');
  if (chips) {
    chips.addEventListener('click', e => {
      if (e.target.classList.contains('ticker-chip')) updateUniverseHint();
    });
  }
  document.querySelectorAll('.budget-chip').forEach(c => {
    c.addEventListener('click', () => {
      const bi = $('budget-input');
      if (bi) bi.value = c.dataset.budget;
      syncBudgetChips();
    });
  });
  const bi = $('budget-input');
  if (bi) bi.addEventListener('input', syncBudgetChips);
  const apply = $('apply-universe-btn');
  if (apply) apply.addEventListener('click', applyUniverse);
}

async function apiTrainBatch() {
  const btn = $('train-batch-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span><span>Evolving Neural Synapses…</span>`;

  try {
    const r = await fetch(`${API}/train_batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ episodes: 25 })
    });
    const d = await r.json();
    if (d.status === 'success') {
      lastTrainData = d;
      totalEpisodes = d.total_episodes;
      qBrainBadge.textContent = `🧠 Gen ${Math.floor(totalEpisodes / 25) + 1} · Ep ${totalEpisodes}`;
      toast(`⚡ Trained 25 Episodes! Median: ${d.median_return_pct > 0 ? '+' : ''}${d.median_return_pct}% · Held-out: ${(d.heldout && d.heldout.fly_excess_vs_buyhold_pct !== undefined) ? (d.heldout.fly_excess_vs_buyhold_pct > 0 ? '+' : '') + d.heldout.fly_excess_vs_buyhold_pct + '% vs buy&hold' : 'n/a'}`, 'buy-toast');
      // Auto-open the lab to show results
      openLabModal(d);
      await apiReset();
      await fetchLearningHistory();
      await fetchSimulationHistory();
    }
  } catch(e) {
    toast('Training failed: API offline', 'sell-toast');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span><span>Evolve Fly Brain (Train 25 Eps)</span>`;
    // Re-enable simulations button if we have data
    const simsBtn = $('train-sims-btn');
    if (simsBtn && totalEpisodes > 0) simsBtn.disabled = false;
  }
}

async function apiRunSimulations() {
  const btn = $('train-sims-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">🔄</span><span>Running 5 Simulations…</span>`;
  
  try {
    let totalResults = null;
    // Run 5 simulations of 25 episodes each (server caps at 100 per call)
    for (let i = 0; i < 5; i++) {
      btn.innerHTML = `<span class="btn-icon">🔄</span><span>Simulation ${i+1}/5…</span>`;
      const r = await fetch(`${API}/train_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes: 25 })
      });
      const d = await r.json();
      if (d.status === 'success') {
        totalResults = d;
        // Update progress
        totalEpisodes = d.total_episodes;
        qBrainBadge.textContent = `🧠 Gen ${Math.floor(totalEpisodes / 25) + 1} · Ep ${totalEpisodes}`;
        lastTrainData = d;
      } else {
        throw new Error('Training failed');
      }
      // Small delay between simulations
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    if (totalResults) {
      toast(`⚡ Completed 5 Simulations! Median: ${totalResults.median_return_pct > 0 ? '+' : ''}${totalResults.median_return_pct}% · Held-out: ${(totalResults.heldout && totalResults.heldout.fly_excess_vs_buyhold_pct !== undefined) ? (totalResults.heldout.fly_excess_vs_buyhold_pct > 0 ? '+' : '') + totalResults.heldout.fly_excess_vs_buyhold_pct + '% vs buy&hold' : 'n/a'}`, 'buy-toast');
      // Auto-open the lab to show results
      openLabModal(totalResults);
      await apiReset();
      await fetchLearningHistory();
      await fetchSimulationHistory();
    }
  } catch(e) {
    toast('Simulations failed: API offline', 'sell-toast');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">🔄</span><span>Run 5 Simulations</span>`;
  }
}

/* ─────────────────────────────────────────────
   Handle step result
───────────────────────────────────────────── */
function handleStepResult(d) {
  if (!d || d.step === undefined) return;
  stepCount = d.step;

  if (d.history && d.history.length > 0) portfolioHistory = d.history;
  const reward = typeof d.reward === 'number' && isFinite(d.reward) ? d.reward : 0;
  rewardHistory.push(reward);
  if (rewardHistory.length > 200) rewardHistory.shift();

  spikeData = d.spikes_full || new Array(N_NEURONS).fill(0);
  spikeData1024 = spikeData; // Alias for backward compatibility
  lastAction      = d.action_idx;
  currentDopamine = d.dopamine || 0.0;
  updateNeuronCountDisplay();

  const sharpe = typeof d.sharpe === 'number' && isFinite(d.sharpe) ? d.sharpe.toFixed(3) : '0.000';
  const drawdown = typeof d.drawdown === 'number' && isFinite(d.drawdown) ? d.drawdown.toFixed(1) : '0.0';
  const winrate = typeof d.win_rate === 'number' && isFinite(d.win_rate) ? d.win_rate.toFixed(1) : '0.0';
  const totalReturn = typeof d.total_return === 'number' && isFinite(d.total_return) ? d.total_return : 0;
  const portfolio = typeof d.portfolio === 'number' && isFinite(d.portfolio) ? d.portfolio : 100000;

  // Update cards
  applySessionMode(d.mode || 'single');
  updatePortfolioCard(portfolio, totalReturn, reward, d.action_idx);
  qSharpe.textContent   = sharpe;
  qDrawdown.textContent = drawdown + '%';
  qWinrate.textContent  = winrate + '%';
  qStep.textContent     = d.step;

  if (d.mode === 'portfolio') renderPortfolioAllocation(d);

  // Topbar chips
  const tbSharpe   = $('tb-sharpe');   if (tbSharpe)   tbSharpe.textContent   = sharpe;
  const tbDrawdown = $('tb-drawdown'); if (tbDrawdown) tbDrawdown.textContent = drawdown + '%';
  const tbWinrate  = $('tb-winrate');  if (tbWinrate)  tbWinrate.textContent  = winrate + '%';
  const tbStep     = $('tb-step');     if (tbStep)     tbStep.textContent     = `${d.step} / ${d.n_steps || 723}`;

  // Stats tab mirrors
  const perfSharpe   = $('perf-sharpe');   if (perfSharpe)   perfSharpe.textContent   = sharpe;
  const perfDrawdown = $('perf-drawdown'); if (perfDrawdown) perfDrawdown.textContent = drawdown + '%';
  const perfWinrate  = $('perf-winrate');  if (perfWinrate)  perfWinrate.textContent  = winrate + '%';

  // Dopamine & Action Probabilities
  updateDopamineUI(currentDopamine);
  if (d.action_probs && d.action_probs.length === 7) {
    updateDeliberationBars(d.action_probs, d.action_idx, d);
  }

  // Voltages (Vm) – accumulate for oscilloscope. The trace uses the per-region
  // peak (99th percentile) because the mean sits at the -85mV clip; the textual
  // read-outs keep the mean.
  if (d.voltages) {
    const vg = d.voltages_peak || d.voltages;
    const push = (arr, val) => { arr.push(val); if (arr.length > 120) arr.shift(); };
    if (vg.al   !== undefined) push(voltageHistory.al,   vg.al);
    if (vg.kc   !== undefined) push(voltageHistory.kc,   vg.kc);
    if (vg.cx   !== undefined) push(voltageHistory.cx,   vg.cx);
    if (vg.dan  !== undefined) push(voltageHistory.dan,  vg.dan);
    if (vg.mbon !== undefined) push(voltageHistory.mbon, vg.mbon);
    updateVmLabels(d.voltages);
  }

  // Neuropil rates
  if (d.regions) {
    updateNeuropilMeters(d.regions);
    qSpikeRate.textContent = `Spike rate: ${d.regions.spike_rate_pct}% (${d.regions.total_active} active)`;
  }

  // Reward stats
  const avg = rewardHistory.reduce((a, b) => a + b, 0) / (rewardHistory.length || 1);
  qAvgReward.textContent  = avg.toFixed(5);
  qLastReward.textContent = d.reward.toFixed(5);
  // Mirror to other tabs
  const paR = $('perf-avg-reward'); if (paR) paR.textContent = avg.toFixed(5);
  const arC = $('avg-reward-ctrl'); if (arC) arC.textContent = avg.toFixed(5);
  const lrC = $('last-reward-ctrl'); if (lrC) lrC.textContent = d.reward.toFixed(5);

  // HUD badges
  const hudDa = $('hud-dopamine');
  const hudVm = $('hud-voltage');
  if (hudDa) hudDa.textContent = `DA: ${currentDopamine > 0 ? '+' : ''}${currentDopamine.toFixed(3)}`;
  if (hudVm && d.voltages?.dan !== undefined) {
    hudVm.textContent = `Vm: ${d.voltages.dan.toFixed(1)} mV`;
  }

  // 3-D Brain & 2-D Charts
  drawPortfolioChart();
  drawRewardChart();
  drawHeatmap(spikeData);
  updateActivityBars(spikeData, d.regions);
  brainNeedsUpdate = true;
  drawOscilloscope();

  // Trade log
  if (d.trades && d.trades.length > 0) {
    const trade = d.trades[d.trades.length - 1];
    if (trade.step === d.step) appendTrade(trade);
  }

  // Toast on trades — 7-action space: 1-3 = Buy, 4-6 = Sell
  if (d.action_idx >= 1 && d.action_idx <= 3) toast(`▲ ${d.action} @ $${d.portfolio.toLocaleString('en-US')}`, 'buy-toast');
  if (d.action_idx >= 4 && d.action_idx <= 6) toast(`▼ ${d.action} @ $${d.portfolio.toLocaleString('en-US')}`, 'sell-toast');

  if (d.done) {
    if (autoPilot) {
      // Auto-loop: reset and continue learning!
      toast('🔄 Episode done — auto-looping for continued learning…', '');
      // We return true to signal done so scheduleNext won't double-schedule
      setTimeout(async () => {
        if (autoPilot) {
          await apiReset();
          scheduleNext();
        }
      }, 800);
    } else {
      toast('🏁 Episode complete! Press Auto-Pilot to continue learning.', '');
    }
  }
}

function updateVmLabels(vg) {
  const pairs = [
    ['vm-al',   vg.al],
    ['vm-kc',   vg.kc],
    ['vm-cx',   vg.cx],
    ['vm-dan',  vg.dan],
    ['vm-mbon', vg.mbon],
  ];
  pairs.forEach(([id, val]) => {
    const el = $(id);
    if (el && val !== undefined) {
      el.textContent = `${val.toFixed(1)} mV`;
      // Color: near threshold (-55) → amber, below resting (-70) → dim
      if (val > -58) {
        el.style.color = '#f59e0b';
      } else if (val > -65) {
        el.style.color = '#7dd3fc';
      } else {
        el.style.color = 'var(--txt-dim)';
      }
    }
  });
}

function updateDopamineUI(da) {
  qDopamineVal.textContent = (da > 0 ? '+' : '') + da.toFixed(3);
  if (da < -0.05) {
    qDopaminePill.classList.add('negative');
  } else {
    qDopaminePill.classList.remove('negative');
  }
}

function updateDeliberationBars(probs, activeAction, payload) {
  const rows = [
    ['hold',    'seg-hold',    'pct-hold',    'dl-hold'],
    ['buy25',   'seg-buy25',   'pct-buy25',   'dl-buy25'],
    ['buy50',   'seg-buy50',   'pct-buy50',   'dl-buy50'],
    ['buy100',  'seg-buy100',  'pct-buy100',  'dl-buy100'],
    ['sell25',  'seg-sell25',  'pct-sell25',  'dl-sell25'],
    ['sell50',  'seg-sell50',  'pct-sell50',  'dl-sell50'],
    ['sell100', 'seg-sell100', 'pct-sell100', 'dl-sell100'],
  ];

  rows.forEach(([, segId, pctId, dlId], i) => {
    const p = Math.max(0, Number(probs[i]) || 0);
    const seg = $(segId);
    const pct = $(pctId);
    const dl  = $(dlId);
    if (seg) seg.style.flexGrow = String(p);
    if (pct) pct.textContent = p * 100 >= 1 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`;
    if (dl)  dl.classList.toggle('is-active', activeAction === i);
  });

  const dec = $('delib-decision');
  if (!dec) return;

  const sizePct = payload && Number.isFinite(payload.size_pct) ? payload.size_pct : null;
  const dynamic = !!(payload && payload.dynamic);

  if (dynamic && sizePct !== null && activeAction >= 1 && activeAction <= 3) {
    dec.textContent = `BUY ${sizePct.toFixed(1)}% of cash`;
    dec.className = 'delib-decision buy';
  } else if (dynamic && sizePct !== null && activeAction >= 4 && activeAction <= 6) {
    dec.textContent = `SELL ${sizePct.toFixed(1)}% of position`;
    dec.className = 'delib-decision sell';
  } else if (payload && payload.action_label) {
    dec.textContent = payload.action_label.toUpperCase();
    dec.className = 'delib-decision' + (activeAction >= 1 && activeAction <= 3 ? ' buy'
                                      : activeAction >= 4 ? ' sell' : '');
  } else {
    dec.textContent = 'HOLD';
    dec.className = 'delib-decision';
  }
}

function updateNeuropilMeters(r) {
  qRateAL.style.width   = `${Math.min(100, r.antennal_lobe * 2.5)}%`;
  qValAL.textContent    = `${r.antennal_lobe}%`;

  qRateKC.style.width   = `${Math.min(100, r.kenyon_cells * 3.5)}%`;
  qValKC.textContent    = `${r.kenyon_cells}%`;

  qRateCX.style.width   = `${Math.min(100, r.central_complex * 2.5)}%`;
  qValCX.textContent    = `${r.central_complex}%`;

  qRateDAN.style.width  = `${Math.min(100, r.dopamine_dan * 2.5)}%`;
  qValDAN.textContent   = `${r.dopamine_dan}%`;

  qRateMBON.style.width = `${Math.min(100, r.mbon_output * 2.5)}%`;
  qValMBON.textContent  = `${r.mbon_output}%`;
}

function updatePortfolioCard(val, ret, reward, action) {
  const fmt = val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const sign = ret >= 0 ? '+' : '';
  const retStr = `${sign}${ret.toFixed(2)}%`;
  const cls = ret >= 0 ? 'pos' : 'neg';

  qPortfolio.textContent = `$${fmt}`;
  qReturn.textContent = retStr;
  qReturn.className = `card-sub ${cls}`;

  // Topbar chips
  const tbPort = $('tb-portfolio');
  const tbRet  = $('tb-return');
  if (tbPort) tbPort.textContent = `$${val.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  if (tbRet)  { tbRet.textContent = retStr; tbRet.className = `stat-chip-val ${cls}`; }

  // Stats tab mirrors
  const perfPort = $('perf-portfolio');
  const perfRet  = $('perf-return');
  if (perfPort) perfPort.textContent = `$${fmt}`;
  if (perfRet)  { perfRet.textContent = retStr; perfRet.className = `perf-val ${cls}`; }
}

/* ─────────────────────────────────────────────
   Oscilloscope (Membrane Potential Vm trace)
   Drawn at devicePixelRatio for crispness, with a real mV scale and an
   auto-range so five regions stay separable instead of collapsing into
   one flat line near -70mV.
───────────────────────────────────────────── */
function drawOscilloscope() {
  const canvas = $('oscilloscope-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || canvas.width;
  const H = canvas.clientHeight || canvas.height;
  if (!W || !H) return;
  canvas.width  = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const traces = [
    { key: 'al',   arr: voltageHistory.al,   color: '#00e5ff', label: 'AL' },
    { key: 'kc',   arr: voltageHistory.kc,   color: '#8c52ff', label: 'KC' },
    { key: 'cx',   arr: voltageHistory.cx,   color: '#ffb703', label: 'CX' },
    { key: 'dan',  arr: voltageHistory.dan,  color: '#ff3366', label: 'DAN' },
    { key: 'mbon', arr: voltageHistory.mbon, color: '#00f59b', label: 'MBON' },
  ];

  // Auto-range over the live samples, with a floor on the span so a quiet
  // stretch does not get amplified into noise.
  let lo = Infinity, hi = -Infinity;
  traces.forEach(({ arr }) => {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isFinite(v) && v !== 0) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  });
  if (!Number.isFinite(lo)) { lo = -80; hi = -50; }
  const span = Math.max(12, hi - lo);
  const mid = (lo + hi) / 2;
  lo = mid - span / 2 - 1.5;
  hi = mid + span / 2 + 1.5;

  const padL = 30, padR = 4, padT = 4, padB = 2;
  const plotW = Math.max(10, W - padL - padR);
  const plotH = Math.max(10, H - padT - padB);
  const yFor = v => padT + ((hi - v) / (hi - lo)) * plotH;

  ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  // Horizontal grid + mV labels.
  for (let i = 0; i <= 3; i++) {
    const v = hi - (i / 3) * (hi - lo);
    const y = yFor(v);
    ctx.strokeStyle = 'rgba(255,255,255,0.055)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = 'rgba(109,127,156,0.9)';
    ctx.fillText(v.toFixed(0), padL - 4, y);
  }

  // Spike threshold: the model fires when V > 0.7, which maps to -38.5 mV.
  if (-38.5 <= hi && -38.5 >= lo) {
    const y = yFor(-38.5);
    ctx.strokeStyle = 'rgba(245,158,11,0.55)';
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Resting potential reference.
  if (-70 <= hi && -70 >= lo) {
    const y = yFor(-70);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.setLineDash([2, 5]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Traces.
  traces.forEach(({ arr, color, key }) => {
    const last = arr.length ? arr[arr.length - 1] : null;
    const valEl = $(`osc-val-${key}`);
    if (valEl) valEl.textContent = Number.isFinite(last) && last !== 0 ? `${last.toFixed(1)} mV` : '—';
    if (arr.length < 2) return;

    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = padL + (i / (arr.length - 1)) * plotW;
      const y = yFor(arr[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.globalAlpha = 0.85;
    ctx.shadowColor = color;
    ctx.shadowBlur = 3;
    ctx.stroke();
    ctx.restore();
  });
}

/* ─────────────────────────────────────────────
   3-D Drosophila MaleCNS Biological Brain & Connectome
   MaleCNS v1.0 (HHMI Janelia / Cambridge / Google Research)
───────────────────────────────────────────── */
let threeScene, threeCamera, threeRenderer;
let atlasPointsMesh, instancedMesh, synapseLines, haloPoints, brainShellMesh;
let currentView = 'biological';
let currentRegionFilter = 'all';
let showTracts = false; // By default NO random lines: pure genuine fruit fly brain anatomy!
let isDragging = false, prevMouseX = 0, prevMouseY = 0;
let camPhi = 0.08, camTheta = 0.0, camRadius = 5.8;
let circuitAtlasIndices = new Int32Array(N_NEURONS);
let atlasPointsMaterial = null;
let brainCenter = new THREE.Vector3(0, 0, 0);

const dummy = new THREE.Object3D();
let activeCoords = new Float32Array(N_NEURONS * 3);  // agent somas embedded in Central Brain
let neuronRegions = new Uint8Array(N_NEURONS);       // 0: AL, 1: KC, 2: CX, 3: DAN, 4: MBON
const synapticPairs = [];                              // Bundled anatomical fascicles

// 1024-LIF Region colors
const REGION_COLORS = [
  new THREE.Color(0x00f0ff), // AL: Radiant Electric Cyan
  new THREE.Color(0xa855f7), // KC: Vivid Deep Orchid
  new THREE.Color(0xf59e0b), // CX: Brilliant Amber Gold
  new THREE.Color(0xff2a6d), // DAN: High-Intensity Crimson
  new THREE.Color(0x10b981), // MBON: Neon Emerald
];

const REGION_NAMES = ['AL', 'KC', 'CX', 'DAN', 'MBON'];
const REGION_HEX   = ['#00f0ff', '#a855f7', '#f59e0b', '#ff2a6d', '#10b981'];

// MaleCNS Real Anatomical Groups:
// Group 0: Optic lobes (90,807 neurons, #81b5c8 - crystalline cyan/blue)
// Group 1: Central brain (32,170 neurons, #cfcac0 - luminous ivory/pearl)
// Group 2: Descending neurons (1,312 neurons, #dfb672 - amber gold)
const MALECNS_GROUP_COLORS = [
  new THREE.Color(0x76b5d4), // Optic: Soft crystalline cyan/blue
  new THREE.Color(0xdad5cb), // Central brain: Luminous pearl/ivory
  new THREE.Color(0xe0b468), // Descending: Rich amber gold
];

let maleCnsPositions = null; // Float32Array of all 124,289 brain somata
let maleCnsGroups = null;    // Uint8Array of group IDs (0: optic, 1: central, 2: descending)
let maleCnsCount = 0;

/* ── Asynchronously Load Genuine MaleCNS v1.0 Brain Atlas ── */
async function loadMaleCnsAtlas() {
  try {
    const [posBuf, grpBuf] = await Promise.all([
      fetch('/data/brain-atlas/positions.bin').then(r => {
        if (!r.ok) throw new Error('Cannot load positions.bin');
        return r.arrayBuffer();
      }),
      fetch('/data/brain-atlas/groups.bin').then(r => {
        if (!r.ok) throw new Error('Cannot load groups.bin');
        return r.arrayBuffer();
      })
    ]);

    const rawPos = new Float32Array(posBuf);
    // groups.bin is uint8 (1 byte per entry, 140,024 somata)
    const rawGrp = new Uint8Array(grpBuf);
    const nTotal = rawGrp.length;

    // Filter to brain-only somata (groups 0: optic, 1: central, 2: descending, omitting VNC/other)
    const validIndices = [];
    for (let i = 0; i < nTotal; i++) {
      if (rawGrp[i] < 3) validIndices.push(i);
    }

    maleCnsCount = validIndices.length;
    maleCnsPositions = new Float32Array(maleCnsCount * 3);
    maleCnsGroups = new Uint8Array(maleCnsCount);

    // 1. Transform raw coordinates to native Drosophila orientation (x, -y, -z)
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let k = 0; k < maleCnsCount; k++) {
      const origIdx = validIndices[k];
      const tx =  rawPos[origIdx * 3 + 0];
      const ty = -rawPos[origIdx * 3 + 1];
      const tz = -rawPos[origIdx * 3 + 2];

      maleCnsPositions[k * 3 + 0] = tx;
      maleCnsPositions[k * 3 + 1] = ty;
      maleCnsPositions[k * 3 + 2] = tz;
      maleCnsGroups[k] = rawGrp[origIdx];

      if (tx < minX) minX = tx; if (tx > maxX) maxX = tx;
      if (ty < minY) minY = ty; if (ty > maxY) maxY = ty;
      if (tz < minZ) minZ = tz; if (tz > maxZ) maxZ = tz;
    }

    // 2. Uniform scaling & center precisely at origin (0, 0, 0)
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const maxExtent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    const scale = 5.0 / maxExtent;

    for (let k = 0; k < maleCnsCount; k++) {
      maleCnsPositions[k * 3 + 0] = (maleCnsPositions[k * 3 + 0] - cx) * scale;
      maleCnsPositions[k * 3 + 1] = (maleCnsPositions[k * 3 + 1] - cy) * scale;
      maleCnsPositions[k * 3 + 2] = (maleCnsPositions[k * 3 + 2] - cz) * scale;
    }

    brainCenter.set(0, 0, 0);
    buildMaleCnsPointsMesh();
    console.log(`MaleCNS v1.0 biological brain loaded successfully: ${maleCnsCount.toLocaleString()} measured somata.`);
  } catch (err) {
    console.warn('Could not load MaleCNS binary atlas, using procedural anatomical fallback:', err);
  }
}

/* ── Build 124,289 Biological MaleCNS Somata Points Cloud ── */
function buildMaleCnsPointsMesh() {
  if (!threeScene || !maleCnsPositions) return;

  if (atlasPointsMesh) {
    threeScene.remove(atlasPointsMesh);
    atlasPointsMesh.geometry.dispose();
    atlasPointsMesh.material.dispose();
  }
  if (brainShellMesh) {
    threeScene.remove(brainShellMesh);
    brainShellMesh.geometry.dispose();
    brainShellMesh.material.dispose();
    brainShellMesh = null;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(maleCnsPositions, 3));

  const colors = new Float32Array(maleCnsCount * 3);
  const groupsAttr = new Float32Array(maleCnsCount);
  const activityAttr = new Float32Array(maleCnsCount);

  for (let i = 0; i < maleCnsCount; i++) {
    const grp = maleCnsGroups[i];
    groupsAttr[i] = grp;
    const col = MALECNS_GROUP_COLORS[grp] || MALECNS_GROUP_COLORS[1];
    colors[i * 3 + 0] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('group', new THREE.BufferAttribute(groupsAttr, 1));
  geometry.setAttribute('activity', new THREE.BufferAttribute(activityAttr, 1));

  atlasPointsMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
      uFilterGroup: { value: -1.0 }
    },
    vertexShader: `
      attribute vec3 color;
      attribute float group;
      attribute float activity;
      varying vec3 vColor;
      varying float vActivity;
      varying float vGroup;
      uniform float uPixelRatio;
      uniform float uFilterGroup;

      void main() {
        vColor = color;
        vActivity = activity;
        vGroup = group;
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPos;

        float ptSize = (1.1 + activity * 2.5) * uPixelRatio;
        if (uFilterGroup >= 0.0 && uFilterGroup <= 2.0) {
          if (abs(group - uFilterGroup) > 0.1) {
            ptSize *= 0.6;
          } else {
            ptSize *= 1.35;
          }
        }
        gl_PointSize = ptSize;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vActivity;
      varying float vGroup;
      uniform float uFilterGroup;

      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float r = length(uv);
        if (r > 0.5) discard;
        float disc = 1.0 - smoothstep(0.18, 0.5, r);

        vec3 col = vColor;
        float alpha = 0.44 * disc;

        if (uFilterGroup >= 0.0 && uFilterGroup <= 2.0) {
          if (abs(vGroup - uFilterGroup) > 0.1) {
            alpha *= 0.12;
          } else {
            alpha = min(1.0, alpha * 2.2);
          }
        }

        if (vActivity > 0.02) {
          vec3 flare = mix(vec3(0.0, 0.95, 1.0), vec3(1.0, 1.0, 1.0), smoothstep(0.4, 1.0, vActivity));
          col = mix(col, flare, vActivity);
          alpha = max(alpha, (0.45 + 0.55 * vActivity) * disc);
        }

        gl_FragColor = vec4(col, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending
  });

  atlasPointsMesh = new THREE.Points(geometry, atlasPointsMaterial);
  threeScene.add(atlasPointsMesh);
}

/* ── Generate Anatomical Coordinates for 1024-LIF Somas in Central Brain ── */
function generateEmbeddedAgentPositions() {
  const useAtlas = maleCnsPositions && maleCnsCount > 0;
  const rng = (seed) => {
    let s = seed % 2147483647;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  };
  const rand = rng(711923);

  if (useAtlas) {
    const centralSomata = [];
    for (let k = 0; k < maleCnsCount; k++) {
      if (maleCnsGroups[k] === 1) centralSomata.push(k);
    }

    const nCentral = centralSomata.length;
    // Map the 5 functional regions to Central Brain somata:
    // AL (128), KC (640), CX (128), DAN (64), MBON (64)
    const alIndices   = centralSomata.slice(0, 128);
    const kcIndices   = centralSomata.slice(128, 128 + 640);
    const cxIndices   = centralSomata.slice(768, 768 + 128);
    const danIndices  = centralSomata.slice(896, 896 + 64);
    const mbonIndices = centralSomata.slice(960, 960 + 64);
    const ordered = alIndices.concat(kcIndices, cxIndices, danIndices, mbonIndices);

    for (let i = 0; i < N_NEURONS && i < ordered.length; i++) {
      const idx = ordered[i];
      circuitAtlasIndices[i] = idx;
      activeCoords[i * 3 + 0] = maleCnsPositions[idx * 3 + 0];
      activeCoords[i * 3 + 1] = maleCnsPositions[idx * 3 + 1];
      activeCoords[i * 3 + 2] = maleCnsPositions[idx * 3 + 2];

      if      (i < 128) neuronRegions[i] = 0; // AL
      else if (i < 768) neuronRegions[i] = 1; // KC
      else if (i < 896) neuronRegions[i] = 2; // CX
      else if (i < 960) neuronRegions[i] = 3; // DAN
      else              neuronRegions[i] = 4; // MBON
    }
  } else {
    for (let i = 0; i < N_NEURONS; i++) {
      activeCoords[i * 3 + 0] = (rand() - 0.5) * 4.0;
      activeCoords[i * 3 + 1] = (rand() - 0.5) * 2.0;
      activeCoords[i * 3 + 2] = (rand() - 0.5) * 1.5;
      neuronRegions[i] = (i < 128) ? 0 : (i < 768) ? 1 : (i < 896) ? 2 : (i < 960) ? 3 : 4;
    }
  }
}

/* ── Generate Soft Radial Point Glow Sprite ── */
function createGlowSpriteTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
  gradient.addColorStop(0.25, 'rgba(255, 255, 255, 0.7)');
  gradient.addColorStop(0.55, 'rgba(180, 220, 255, 0.25)');
  gradient.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

/* ── Initialize Three.js Scene ── */
async function initThree() {
  const wrap = $('brain-canvas-wrap');
  if (!wrap) return;

  const W = Math.max(wrap.clientWidth, 320);
  const H = Math.max(wrap.clientHeight, 330);

  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(46, W / H, 0.1, 500);
  updateCamera();

  threeRenderer = new THREE.WebGLRenderer({
    canvas: $('brain-canvas'),
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance'
  });
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  threeRenderer.setSize(W, H);

  threeScene.add(new THREE.AmbientLight(0x162438, 1.8));

  const p1 = new THREE.PointLight(0x00f0ff, 2.5, 140);
  p1.position.set(28, 30, 35);
  threeScene.add(p1);

  const p2 = new THREE.PointLight(0xa855f7, 2.2, 140);
  p2.position.set(-28, -20, 28);
  threeScene.add(p2);

  await loadMaleCnsAtlas();
  generateEmbeddedAgentPositions();
  buildAgentSomas();
  threeRenderer.setAnimationLoop(threeAnimate);
  setupOrbitControls(wrap);
}

/* ── Build 1024 Spiking Decision Somas (Glowing Neural Dots) ── */
function buildAgentSomas() {
  if (instancedMesh) {
    threeScene.remove(instancedMesh);
    if (instancedMesh.geometry) instancedMesh.geometry.dispose();
    if (instancedMesh.material) instancedMesh.material.dispose();
    instancedMesh = null;
  }
  if (haloPoints) {
    threeScene.remove(haloPoints);
    if (haloPoints.geometry) haloPoints.geometry.dispose();
    if (haloPoints.material) haloPoints.material.dispose();
    haloPoints = null;
  }

  // Pure pinpoint glowing neural dots (no bulky 3D balls)
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(activeCoords, 3));

  const colors = new Float32Array(N_NEURONS * 3);
  const sizes = new Float32Array(N_NEURONS);
  const alphas = new Float32Array(N_NEURONS);

  for (let i = 0; i < N_NEURONS; i++) {
    const reg = neuronRegions[i];
    const baseCol = REGION_COLORS[reg] || REGION_COLORS[0];
    colors[i * 3 + 0] = baseCol.r;
    colors[i * 3 + 1] = baseCol.g;
    colors[i * 3 + 2] = baseCol.b;
    sizes[i] = 1.6;   // resting pinpoint dot
    alphas[i] = 0.22; // resting dim opacity
  }

  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  geom.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
    },
    vertexShader: `
      attribute vec3 color;
      attribute float size;
      attribute float alpha;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uPixelRatio;

      void main() {
        vColor = color;
        vAlpha = alpha;
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPos;
        gl_PointSize = size * uPixelRatio * (8.0 / -mvPos.z);
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float r = length(uv);
        if (r > 0.5) discard;
        float core = 1.0 - smoothstep(0.0, 0.38, r);
        float halo = 1.0 - smoothstep(0.15, 0.5, r);
        vec3 col = mix(vColor, vec3(1.0, 1.0, 1.0), core * 0.45);
        float a = vAlpha * (core * 0.75 + halo * 0.25);
        gl_FragColor = vec4(col, a);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  instancedMesh = new THREE.Points(geom, mat);
  threeScene.add(instancedMesh);

  if (showTracts) buildSynapticLines();
}

/* ── Build Bundled Neuropil Fascicles ── */
function buildSynapticLines() {
  if (synapseLines) {
    threeScene.remove(synapseLines);
    synapseLines.geometry.dispose();
    synapseLines.material.dispose();
    synapseLines = null;
  }

  if (!showTracts) return;

  const nPairs = synapticPairs.length;
  const linePositions = new Float32Array(nPairs * 6);
  const lineColors    = new Float32Array(nPairs * 6);

  for (let p = 0; p < nPairs; p++) {
    const [src, tgt] = synapticPairs[p];
    const off = p * 6;

    linePositions[off + 0] = activeCoords[src * 3 + 0];
    linePositions[off + 1] = activeCoords[src * 3 + 1];
    linePositions[off + 2] = activeCoords[src * 3 + 2];

    linePositions[off + 3] = activeCoords[tgt * 3 + 0];
    linePositions[off + 4] = activeCoords[tgt * 3 + 1];
    linePositions[off + 5] = activeCoords[tgt * 3 + 2];

    const srcCol = REGION_COLORS[neuronRegions[src]];
    const tgtCol = REGION_COLORS[neuronRegions[tgt]];

    lineColors[off + 0] = srcCol.r;
    lineColors[off + 1] = srcCol.g;
    lineColors[off + 2] = srcCol.b;

    lineColors[off + 3] = tgtCol.r;
    lineColors[off + 4] = tgtCol.g;
    lineColors[off + 5] = tgtCol.b;
  }

  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeom.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

  const lineMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  synapseLines = new THREE.LineSegments(lineGeom, lineMat);
  threeScene.add(synapseLines);
}

/* ── Live Neural Activity Update Loop ── */
function updateBrainViz(spikes, dopamine) {
  if (!activeCoords) return;

  const palette = $('colour-mode')?.value || 'anatomical';

  // 1. Update genuine 124,289 Drosophila somata activity
  if (atlasPointsMesh && atlasPointsMesh.geometry) {
    const actAttr = atlasPointsMesh.geometry.getAttribute('activity');
    if (actAttr) {
      const actArr = actAttr.array;
      for (let k = 0; k < actArr.length; k++) {
        actArr[k] *= 0.88;
      }
      if (circuitAtlasIndices && circuitAtlasIndices.length > 0) {
        for (let i = 0; i < N_NEURONS; i++) {
          const somaIdx = circuitAtlasIndices[i];
          if (somaIdx >= 0 && somaIdx < actArr.length) {
            if (spikeData1024 && spikeData1024[i] > 0) {
              actArr[somaIdx] = 1.0;
            }
            if (neuronRegions[i] === 3 && Math.abs(dopamine) > 0.02) {
              actArr[somaIdx] = Math.min(1.0, actArr[somaIdx] + Math.abs(dopamine) * 1.8);
            }
          }
        }
      }
      actAttr.needsUpdate = true;
    }

    if (atlasPointsMaterial && atlasPointsMaterial.uniforms) {
      let filterVal = -1.0;
      if (currentRegionFilter === 'optic') filterVal = 0.0;
      else if (currentRegionFilter === 'central') filterVal = 1.0;
      else if (currentRegionFilter === 'descending') filterVal = 2.0;
      else if (currentRegionFilter === 'agent') filterVal = 3.0;
      if (atlasPointsMaterial.uniforms.uFilterGroup) {
        atlasPointsMaterial.uniforms.uFilterGroup.value = filterVal;
      }
    }
  }

  // 2. Update circuit somas dots (sharp pinpoint dots that flare up when firing)
  if (instancedMesh && instancedMesh.geometry) {
    const colAttr = instancedMesh.geometry.getAttribute('color');
    const sizeAttr = instancedMesh.geometry.getAttribute('size');
    const alphaAttr = instancedMesh.geometry.getAttribute('alpha');

    if (colAttr && sizeAttr && alphaAttr) {
      const colArr = colAttr.array;
      const sizeArr = sizeAttr.array;
      const alphaArr = alphaAttr.array;

      for (let i = 0; i < N_NEURONS; i++) {
        const isSpike = spikeData1024 && spikeData1024[i] > 0;
        const reg = neuronRegions[i];
        const baseCol = REGION_COLORS[reg] || REGION_COLORS[0];

        if (isSpike) {
          // Firing: bright glowing pinpoint dot!
          sizeArr[i] = 4.2;
          alphaArr[i] = 1.0;
          colArr[i * 3 + 0] = Math.min(1.0, baseCol.r * 0.4 + 0.7);
          colArr[i * 3 + 1] = Math.min(1.0, baseCol.g * 0.4 + 0.7);
          colArr[i * 3 + 2] = Math.min(1.0, baseCol.b * 0.4 + 0.7);
        } else if (reg === 3 && Math.abs(dopamine) > 0.03) {
          // DAN dopaminergic flare
          sizeArr[i] = 4.8;
          alphaArr[i] = 0.95;
          if (dopamine > 0) {
            colArr[i * 3 + 0] = 0.1; colArr[i * 3 + 1] = 1.0; colArr[i * 3 + 2] = 0.5;
          } else {
            colArr[i * 3 + 0] = 1.0; colArr[i * 3 + 1] = 0.2; colArr[i * 3 + 2] = 0.3;
          }
        } else {
          // Resting: tiny subtle dot
          sizeArr[i] = 1.6;
          alphaArr[i] = 0.22;
          colArr[i * 3 + 0] = baseCol.r;
          colArr[i * 3 + 1] = baseCol.g;
          colArr[i * 3 + 2] = baseCol.b;
        }
      }

      colAttr.needsUpdate = true;
      sizeAttr.needsUpdate = true;
      alphaAttr.needsUpdate = true;
    }
  }
}

/* ── Camera & Controls ── */
function updateCamera() {
  const x = camRadius * Math.sin(camTheta) * Math.cos(camPhi);
  const y = camRadius * Math.sin(camPhi);
  const z = camRadius * Math.cos(camTheta) * Math.cos(camPhi);
  threeCamera.position.set(x, y, z);
  threeCamera.lookAt(0, 0, 0);
}

function setupOrbitControls(wrap) {
  wrap.addEventListener('mousedown', e => {
    isDragging = true;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
  });

  window.addEventListener('mouseup', () => isDragging = false);

  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - prevMouseX;
    const dy = e.clientY - prevMouseY;
    camTheta -= dx * 0.006;
    camPhi   = Math.max(-Math.PI * 0.44, Math.min(Math.PI * 0.44, camPhi + dy * 0.006));
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
    updateCamera();
  });

  wrap.addEventListener('wheel', e => {
    e.preventDefault();
    camRadius = Math.max(2.5, Math.min(16.0, camRadius + e.deltaY * 0.005));
    updateCamera();
  }, { passive: false });

  $('recenter-cam-btn')?.addEventListener('click', () => {
    camPhi = 0.08; camTheta = 0.0; camRadius = 5.8;
    updateCamera();
  });

  $('toggle-tracts-btn')?.addEventListener('click', () => {
    showTracts = !showTracts;
    const btn = $('toggle-tracts-btn');
    if (btn) btn.textContent = showTracts ? '⚡ Tracts: ON' : '⚡ Tracts: OFF';
    buildSynapticLines();
  });
}

function threeAnimate() {
  if (brainNeedsUpdate) {
    updateBrainViz(spikeData1024, currentDopamine);
    brainNeedsUpdate = false;
  }
  if (!isDragging && threeScene) {
    camTheta += 0.0008;
    updateCamera();
  }
  threeRenderer.render(threeScene, threeCamera);
}

/* ─────────────────────────────────────────────
   2-D Charts (Portfolio & Reward)
───────────────────────────────────────────── */
function drawPortfolioChart() {
  const canvas = $('port-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);
  if (portfolioHistory.length < 2) return;

  const data = portfolioHistory;
  const minP = Math.min(...data) * 0.995;
  const maxP = Math.max(...data) * 1.005;
  const range = maxP - minP || 1;

  const getX = i => (i / (data.length - 1)) * W;
  const getY = v => H - ((v - minP) / range) * (H - 24) - 12;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let s = 1; s <= 3; s++) {
    const gy = (H / 4) * s;
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
  }

  // Gradient fill under Fly Trader
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  const lastVal = data[data.length - 1];
  const isProfit = lastVal >= data[0];
  grad.addColorStop(0, isProfit ? 'rgba(34, 197, 94, 0.35)' : 'rgba(239, 68, 68, 0.3)');
  grad.addColorStop(1, 'rgba(0,0,0,0.0)');

  ctx.beginPath();
  ctx.moveTo(getX(0), getY(data[0]));
  for (let i = 1; i < data.length; i++) ctx.lineTo(getX(i), getY(data[i]));
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Fly Trader Line
  ctx.beginPath();
  ctx.moveTo(getX(0), getY(data[0]));
  for (let i = 1; i < data.length; i++) ctx.lineTo(getX(i), getY(data[i]));
  ctx.strokeStyle = isProfit ? '#22c55e' : '#ef4444';
  ctx.lineWidth = 2.2;
  ctx.stroke();

  // Buy & Hold baseline line
  const bh = data[0];
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.moveTo(0, getY(bh));
  ctx.lineTo(W, getY(bh));
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.setLineDash([]);

  // Current value label
  const cY = getY(lastVal);
  ctx.fillStyle = isProfit ? '#22c55e' : '#ef4444';
  ctx.font = `500 10px 'JetBrains Mono', monospace`;
  ctx.fillText(`$${lastVal.toLocaleString('en-US', { maximumFractionDigits: 0 })}`, W - 90, Math.max(14, Math.min(H - 4, cY - 4)));
}

function drawRewardChart() {
  _drawRewardOnCanvas($('reward-canvas'));
  _drawRewardOnCanvas($('reward-canvas-big'));
  _drawRewardOnCanvas($('reward-canvas-ctrl'));
}

function drawRewardChartBig() {
  _drawRewardOnCanvas($('reward-canvas-big'));
}

function _drawRewardOnCanvas(canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W < 2 || H < 2) return;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);
  if (rewardHistory.length < 2) return;

  const data = rewardHistory;
  const maxAbs = Math.max(...data.map(Math.abs), 0.005);

  const getX = i => (i / (data.length - 1)) * W;
  const getY = v => H / 2 - (v / maxAbs) * (H / 2 - 6);

  // Zero axis
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  // Positive fill
  const posGrad = ctx.createLinearGradient(0, 0, 0, H / 2);
  posGrad.addColorStop(0, 'rgba(34, 211, 238, 0.28)');
  posGrad.addColorStop(1, 'rgba(34, 211, 238, 0.0)');
  ctx.beginPath();
  ctx.moveTo(getX(0), H / 2);
  data.forEach((v, i) => { ctx.lineTo(getX(i), getY(Math.max(0, v))); });
  ctx.lineTo(W, H / 2);
  ctx.fillStyle = posGrad;
  ctx.fill();

  // Reward sparkline
  ctx.beginPath();
  ctx.moveTo(getX(0), getY(data[0]));
  for (let i = 1; i < data.length; i++) ctx.lineTo(getX(i), getY(data[i]));
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

function drawHeatmap(spikes) {
  const canvas = $('heatmap-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Draw a 32×32 heatmap: region-colored with spike glow
  // Limited to 1024 neurons to match 32×32 grid
  const maxN = Math.min(N_NEURONS, 1024);
  const cellW = W / 32;
  const cellH = H / 32;

  for (let i = 0; i < maxN; i++) {
    const row = Math.floor(i / 32);
    const col = i % 32;
    const isSpike = spikes[i] > 0;
    const reg = neuronRegions[i];
    const hex = REGION_HEX[reg];

    const x = col * cellW;
    const y = row * cellH;

    if (isSpike) {
      // Glow effect for spiking neurons
      const grd = ctx.createRadialGradient(x + cellW/2, y + cellH/2, 0, x + cellW/2, y + cellH/2, cellW * 1.2);
      grd.addColorStop(0, hex + 'ff');
      grd.addColorStop(0.5, hex + '88');
      grd.addColorStop(1, hex + '00');
      ctx.fillStyle = grd;
      ctx.fillRect(Math.max(0, x - cellW * 0.2), Math.max(0, y - cellH * 0.2), cellW * 1.4, cellH * 1.4);
    } else {
      ctx.fillStyle = hex + '2a';
      ctx.fillRect(x, y, cellW, cellH);
    }
  }

  // Region boundary lines
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 0.5;
  // AL: 0-127 (rows 0-3), KC: 128-767 (rows 4-23), CX: 768-895 (rows 24-27), DAN: 896-959 (rows 28-29), MBON: 960-1023 (rows 30-31)
  [4, 24, 28, 30].forEach(row => {
    ctx.beginPath();
    ctx.moveTo(0, row * cellH);
    ctx.lineTo(W, row * cellH);
    ctx.stroke();
  });

  // Region labels on side
  ctx.font = `bold 7px 'JetBrains Mono', monospace`;
  const regionLabels = [
    { label: 'AL', startRow: 0, color: REGION_HEX[0] },
    { label: 'KC', startRow: 4, color: REGION_HEX[1] },
    { label: 'CX', startRow: 24, color: REGION_HEX[2] },
    { label: 'DAN', startRow: 28, color: REGION_HEX[3] },
    { label: 'MBON', startRow: 30, color: REGION_HEX[4] },
  ];
  regionLabels.forEach(({ label, startRow, color }) => {
    ctx.fillStyle = color + 'cc';
    ctx.fillText(label, 2, startRow * cellH + 8);
  });
}

/* ─────────────────────────────────────────────
   Activity Bars — Rich per-region display
───────────────────────────────────────────── */
function getRegionBoundaries() {
  // Proportional anatomical partitions scaled to N_NEURONS
  // Default 1024: AL=128, KC=640, CX=128, DAN=64, MBON=64
  if (N_NEURONS === 1024) return [128, 768, 896, 960, 1024];
  const scale = N_NEURONS / 1024;
  return [
    Math.round(128 * scale),   // AL
    Math.round(768 * scale),   // KC
    Math.round(896 * scale),   // CX
    Math.round(960 * scale),   // DAN
    N_NEURONS                  // MBON (remainder)
  ];
}

function getRegionSizes() {
  const b = getRegionBoundaries();
  return [b[0], b[1] - b[0], b[2] - b[1], b[3] - b[2], b[4] - b[3]];
}

function updateActivityBars(spikes, regions) {
  qActivityBars.innerHTML = '';

  // Count spikes per region
  const regionCounts = [0, 0, 0, 0, 0];
  const regionSizes  = getRegionSizes();
  for (let i = 0; i < N_NEURONS; i++) {
    if (spikes[i] > 0) regionCounts[neuronRegions[i]]++;
  }

  const totalSpikes = regionCounts.reduce((a, b) => a + b, 0);

  if (totalSpikes === 0) {
    qActivityBars.innerHTML = `<span class="no-spike-msg">Awaiting spike bursts…<br><small>Start Auto-Pilot to begin</small></span>`;
    return;
  }

  // One bar per region
  regionCounts.forEach((count, reg) => {
    if (count === 0) return;
    const pct = Math.round(count / regionSizes[reg] * 100);
    const color = REGION_HEX[reg];
    const name = ['Antennal Lobe', 'Kenyon Cells', 'Central Complex', 'DAN (Dopamine)', 'MBON (Valence)'][reg];
    const div = document.createElement('div');
    div.className = 'a-row';
    div.innerHTML = `
      <span class="a-label" style="color:${color}">${REGION_NAMES[reg]}</span>
      <div class="a-bar-wrap">
        <div class="a-bar" style="width:${Math.min(100, pct * 2.5)}%; background:${color}55; border-right: 2px solid ${color}"></div>
      </div>
      <span class="a-val">${count}<span style="color:var(--txt-dim);font-size:0.62rem"> / ${regionSizes[reg]}</span></span>
    `;
    div.title = `${name}: ${count} of ${regionSizes[reg]} neurons firing (${pct}%)`;
    qActivityBars.appendChild(div);
  });

  // Summary
  const summaryDiv = document.createElement('div');
  summaryDiv.className = 'a-summary';
  summaryDiv.innerHTML = `<span class="a-total">${totalSpikes} <span style="color:var(--txt-dim)">/ ${N_NEURONS} neurons active</span></span>`;
  qActivityBars.appendChild(summaryDiv);
}

// Update all elements that display neuron count
function updateNeuronCountDisplay() {
  // Update elements that use data-neuron-count attribute
  document.querySelectorAll('[data-neuron-count]').forEach(el => {
    const originalText = el.textContent || el.innerText;
    // Replace 1024 with current N_NEURONS in the text
    const newText = originalText.replace(/1024/g, N_NEURONS.toString());
    if (el.textContent !== undefined) {
      el.textContent = newText;
    } else {
      el.innerText = newText;
    }
  });
  
  // Update brain badge to include N_NEURONS
  if (qBrainBadge) {
    qBrainBadge.textContent = `🧠 Gen ${Math.floor(totalEpisodes / 25) + 1} · Ep ${totalEpisodes} · ${N_NEURONS}-LIF Neurons`;
  }
  
  // Update activity bars summary if it exists
  const activitySummary = qActivityBars.querySelector('.a-summary');
  if (activitySummary) {
    const totalSpikes = spikeData.filter(v => v > 0).length;
    activitySummary.innerHTML = `<span class="a-total">${totalSpikes} <span style="color:var(--txt-dim)">/ ${N_NEURONS} neurons active</span></span>`;
  }
}

/* ─────────────────────────────────────────────
   Evolution Lab Modal
───────────────────────────────────────────── */
function openLabModal(data) {
  const modal = $('evolution-modal');
  if (!modal) return;
  modal.classList.add('open');

  if (data) populateLab(data);
}

function closeLabModal() {
  const modal = $('evolution-modal');
  if (modal) modal.classList.remove('open');
}

function populateLab(d) {
  // KPI cards
  const avgRet = d.avg_return_pct || 0;
  const bestRet = d.best_return_pct || 0;
  const medianRet = d.median_return_pct;
  const p = d.plasticity || {};
  const ve = d.valence_evolution || {};

  const labAvgRet = $('lab-avg-ret');
  if (labAvgRet) {
    labAvgRet.textContent = `${avgRet >= 0 ? '+' : ''}${avgRet.toFixed(2)}%`;
    labAvgRet.className = `kpi-value ${avgRet >= 0 ? 'pos' : 'neg'}`;
  }
  const labBestRet = $('lab-best-ret');
  if (labBestRet) {
    // Best is the MAX over stochastic in-sample episodes; median is the honest one.
    const medTxt = (medianRet === undefined || medianRet === null)
      ? '' : `Median: ${medianRet >= 0 ? '+' : ''}${medianRet.toFixed(2)}% · `;
    labBestRet.textContent = `${medTxt}Best: ${bestRet >= 0 ? '+' : ''}${bestRet.toFixed(2)}%`;
  }

  // Held-out (out-of-sample) result — this is the number that matters.
  const ho = d.heldout || {};
  const labHo = $('lab-heldout-ret');
  if (labHo) {
    const v = ho.mean_fly_return_pct;
    if (v === undefined) {
      labHo.textContent = '—';
      labHo.className = 'kpi-value';
    } else {
      labHo.textContent = `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
      labHo.className = `kpi-value ${v >= 0 ? 'pos' : 'neg'}`;
    }
  }
  const labHoSub = $('lab-heldout-sub');
  if (labHoSub) {
    if (ho.fly_excess_vs_buyhold_pct === undefined) {
      labHoSub.textContent = 'vs buy & hold: —';
    } else {
      const x = ho.fly_excess_vs_buyhold_pct;
      const range = ho.date_range ? `${ho.date_range[0]}..${ho.date_range[1]}` : '';
      labHoSub.textContent = `vs buy & hold: ${x >= 0 ? '+' : ''}${x.toFixed(2)}% (${range})`;
    }
  }

  const labLtp = $('lab-ltp-ltd');
  if (labLtp) labLtp.textContent = `${p.ltp_count || 0} LTP / ${p.ltd_count || 0} LTD`;

  const labLoss = $('lab-loss');
  if (labLoss) labLoss.textContent = (d.avg_loss || 0).toFixed(5);

  const labEp = $('lab-total-episodes');
  if (labEp) labEp.textContent = d.total_episodes || 0;

  // Episode chart
  if (d.episodes_data && d.episodes_data.length > 0) {
    drawEvolutionChart(d.episodes_data);
  }

  // Plasticity grid
  if (p.deltas && p.deltas.length > 0) {
    drawPlasticityGrid(p.deltas, p.weights_before, p.weights_after);
  }

  // Valence comparison
  if (ve.before && ve.after) {
    updateValenceComparison(ve.before, ve.after);
  }
}

function drawEvolutionChart(episodes) {
  const canvas = $('evo-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 800;
  const H = canvas.clientHeight || 200;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const n = episodes.length;
  if (n < 1) return;

  const returns  = episodes.map(e => e.return_pct);
  const winrates = episodes.map(e => e.win_rate);
  const losses   = episodes.map(e => e.loss * 100); // scale for display

  const allVals = [...returns, ...winrates, ...losses];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const rangeV = (maxV - minV) || 1;

  const padL = 10, padR = 10, padT = 14, padB = 20;
  const cW = W - padL - padR;
  const cH = H - padT - padB;

  const gx = (i) => padL + (i / Math.max(1, n - 1)) * cW;
  const gy = (v) => padT + cH - ((v - minV) / rangeV) * cH;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let s = 1; s <= 3; s++) {
    const y = padT + (cH / 4) * s;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }

  // Zero line
  const zeroY = gy(0);
  if (zeroY >= padT && zeroY <= padT + cH) {
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw Return % (green/red)
  function drawLine(data, color, lineW = 2) {
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = gx(i), y = gy(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.stroke();

    // Dots at each episode
    data.forEach((v, i) => {
      ctx.beginPath();
      ctx.arc(gx(i), gy(v), 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawLine(returns, '#22c55e', 2.5);
  drawLine(winrates, '#8c52ff', 1.8);
  drawLine(losses, '#ff3366', 1.4);

  // Episode numbers along bottom
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = `9px 'JetBrains Mono', monospace`;
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 10))) {
    ctx.fillText(`${episodes[i].episode}`, gx(i) - 4, H - 4);
  }
}

function drawPlasticityGrid(deltas, weightsBefore, weightsAfter) {
  const grid = $('plasticity-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const maxDelta = Math.max(...deltas.map(Math.abs), 0.0001);

  deltas.slice(0, 64).forEach((d, i) => {
    const cell = document.createElement('div');
    cell.className = 'p-cell';
    const intensity = Math.min(1, Math.abs(d) / maxDelta);
    if (d > 0.0001) {
      cell.style.background = `rgba(34, 197, 94, ${0.15 + intensity * 0.8})`;
      cell.title = `Synapse ${i}: +${d.toFixed(4)} (LTP)`;
    } else if (d < -0.0001) {
      cell.style.background = `rgba(239, 68, 68, ${0.15 + intensity * 0.8})`;
      cell.title = `Synapse ${i}: ${d.toFixed(4)} (LTD)`;
    } else {
      cell.style.background = `rgba(100, 120, 160, 0.15)`;
      cell.title = `Synapse ${i}: stable`;
    }
    grid.appendChild(cell);
  });
}

function updateValenceComparison(before, after) {
  const actions = ['hold', 'buy', 'sell'];
  const beforeIds = ['vb-hold', 'vb-buy', 'vb-sell'];
  const afterIds  = ['va-hold', 'va-buy', 'va-sell'];

  before.forEach((p, i) => {
    const el = $(beforeIds[i]);
    if (el) {
      el.style.width = `${Math.round(p * 100)}%`;
      const parent = el.parentElement;
      if (parent) {
        const span = parent.nextElementSibling;
        if (span) span.textContent = `${Math.round(p * 100)}%`;
      }
    }
  });

  after.forEach((p, i) => {
    const el = $(afterIds[i]);
    if (el) {
      el.style.width = `${Math.round(p * 100)}%`;
      const parent = el.parentElement;
      if (parent) {
        const span = parent.nextElementSibling;
        if (span) span.textContent = `${Math.round(p * 100)}%`;
      }
    }
  });
}

function appendTrade(trade) {
  // Derive a clean single-word CSS class from the action string
  const rawCls = trade.action.toLowerCase();
  const cls = rawCls.startsWith('buy') ? 'buy' : rawCls.startsWith('sell') ? 'sell' : 'hold';
  const div = document.createElement('div');
  div.className = `trade-item ${cls}`;
  const px = (trade.price === null || trade.price === undefined || trade.price === 0)
    ? '' : `<span class="tp">$${trade.price}</span>`;
  div.innerHTML = `
    <span class="ta ${cls}">${trade.action}</span>
    ${px}
    <span class="tp">→ $${trade.portfolio.toLocaleString('en-US')}</span>
  `;
  qTradeLog.prepend(div);
  // Cap trade log to 30 entries
  while (qTradeLog.children.length > 30) qTradeLog.removeChild(qTradeLog.lastChild);
}

function toast(msg, cls) {
  const el = document.createElement('div');
  el.className = `toast ${cls}`;
  el.textContent = msg;
  $('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 2800);
}

function setStatus(state, text) {
  qStatus.textContent = text;
  qDot.className = 'dot ' + (state === 'live' ? 'live' : state === 'error' ? 'error' : '');
}

/* ─────────────────────────────────────────────
    Portfolio (cross-section) tab
    One shared policy allocates the whole book across N assets + cash on a
    monthly rebalance; the numbers are the honest ticker-holdout backtest.
 ───────────────────────────────────────────── */
let portfolioData = null;

const PF_COLORS = {
  fly:             '#00e5ff',
  'buy&hold':      '#22c55e',
  trend200:        '#ffb703',
  'mom120-top3-m': '#8c52ff',
};
const PF_LABELS = {
  fly:             'Fly',
  'buy&hold':      'Buy & Hold',
  trend200:        'Trend 200d',
  'mom120-top3-m': 'Momentum top-3',
};
const PF_ORDER = ['buy&hold', 'trend200', 'mom120-top3-m', 'fly'];

async function fetchPortfolioEval(refresh = false) {
  const cfg = $('portfolio-config');
  if (cfg) cfg.textContent = 'running held-out backtest…';
  try {
    const r = await fetch(`${API}/portfolio_eval${refresh ? '?refresh=true' : ''}`);
    const d = await r.json();
    portfolioData = d;          // the chart reads this module-level state
    renderPortfolio(d);
  } catch (e) {
    console.warn('Portfolio eval failed:', e);
    renderPortfolio({ available: false, error: String(e) });
  }
}

function _pfKpi(id, text, cls) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className = 'sim-kpi-val' + (cls ? ' ' + cls : '');
}

function _pfPct(v) { return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`; }
function _pfPct0(v) { return `${v >= 0 ? '+' : ''}${Math.round(v)}%`; }

function renderPortfolio(d) {
  const table = $('portfolio-table');
  const cfg   = $('portfolio-config');

  if (!d || d.available === false) {
    ['pf-fly', 'pf-bh', 'pf-exc', 'pf-sharpe', 'pf-dd', 'pf-pos']
      .forEach(id => _pfKpi(id, '–'));
    const c = $('portfolio-equity-chart');
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    const leg = $('portfolio-legend'); if (leg) leg.innerHTML = '';
    if (cfg) cfg.textContent = (d && d.checkpoint)
      ? `no checkpoint at checkpoints/${d.checkpoint}`
      : 'portfolio backtest unavailable';
    if (table) {
      const recipe = (d && d.recipe)
        ? `<div class="pf-recipe">train it first:<br>${d.recipe}</div>` : '';
      const err = (d && d.error) ? `<div class="pf-error">${d.error}</div>` : '';
      table.innerHTML = err + recipe;
    }
    return;
  }

  const s  = d.strategies;
  const fl = s.fly, bh = s['buy&hold'];

  if (cfg) {
    const gate = d.gate.mode === 'positive'
      ? `${d.gate.mode} [${d.gate.min_k},${d.gate.max_k}]` : d.gate.mode;
    cfg.innerHTML =
      `<span class="pf-dim">${d.plan}/${d.split} · ` +
      `${d.date_range[0].slice(0, 7)} → ${d.date_range[1].slice(0, 7)}</span><br>` +
      `<b>${d.n_assets}</b> unseen tickers · monthly · ${d.cost_bps} bps/side<br>` +
      `gate <b>${gate}</b> · ` +
      `${d.full_readout ? 'whole-brain read-out' : 'MBON read-out'}`;
  }

  const exc = d.fly_excess_vs_buyhold_pp;
  _pfKpi('pf-fly', _pfPct(fl.return_pct), fl.return_pct >= 0 ? 'pos' : 'neg');
  _pfKpi('pf-bh', _pfPct(bh.return_pct), bh.return_pct >= 0 ? 'pos' : 'neg');
  _pfKpi('pf-exc', _pfPct(exc), exc >= 0 ? 'pos' : 'neg');
  _pfKpi('pf-sharpe', fl.sharpe.toFixed(2));
  _pfKpi('pf-dd', `${fl.max_drawdown_pct.toFixed(1)}%`);
  _pfKpi('pf-pos', fl.avg_positions.toFixed(1));

  if (table) {
    const live = PF_ORDER.filter(n => s[n]);
    const best = Math.max(...live.map(n => s[n].return_pct));
    const head = `<div class="pf-row head"><span>Strategy</span>` +
      `<span class="pf-num">Return</span><span class="pf-num">Max DD</span>` +
      `<span class="pf-num">Sharpe</span><span class="pf-num">Names</span></div>`;
    const rows = live.map(n => {
      const v = s[n];
      const cls = (n === 'fly' ? ' pf-fly-row' : '') +
                  (v.return_pct === best ? ' pf-best-row' : '');
      return `<div class="pf-row${cls}">` +
        `<span class="pf-name">${PF_LABELS[n] || n}</span>` +
        `<span class="pf-num ${v.return_pct >= 0 ? 'pf-pos' : 'pf-neg'}">${_pfPct(v.return_pct)}</span>` +
        `<span class="pf-num">${v.max_drawdown_pct.toFixed(1)}%</span>` +
        `<span class="pf-num">${v.sharpe.toFixed(2)}</span>` +
        `<span class="pf-num">${v.avg_positions.toFixed(1)}</span></div>`;
    }).join('');
    table.innerHTML = head + rows;
  }

  drawPortfolioEquityChart();
}

function drawPortfolioEquityChart() {
  const canvas = $('portfolio-equity-chart');
  if (!canvas || !portfolioData || !portfolioData.available) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 280;
  const H = canvas.clientHeight || 170;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const s = portfolioData.strategies;
  const names = PF_ORDER.filter(n => s[n] && s[n].curve && s[n].curve.length > 1);
  if (!names.length) return;
  // Log scale: the fly's held-out curve spans orders of magnitude, so a linear
  // axis would flatten every other strategy onto one line.
  const curves = names.map(n => s[n].curve.map(v => Math.log10(Math.max(v, 1e-3))));
  const n = Math.max(...curves.map(c => c.length));
  const flat = curves.reduce((a, c) => a.concat(c), []);
  const ymin = Math.min(...flat), ymax = Math.max(...flat);

  const padL = 40, padR = 8, padT = 10, padB = 16;
  const cW = W - padL - padR, cH = H - padT - padB;
  const gx = i => padL + (i / (n - 1)) * cW;
  const gy = v => padT + cH - ((v - ymin) / ((ymax - ymin) || 1)) * cH;

  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillStyle = '#6d7f9c';
  ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {
    const v = ymin + (ymax - ymin) * t / 4;
    const y = gy(v);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(_pfPct0((Math.pow(10, v) - 1) * 100), padL - 5, y + 3);
  }

  // break-even reference: equity = 1 (0% return)
  const y0 = gy(0);
  if (y0 > padT && y0 < padT + cH) {
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(W - padR, y0); ctx.stroke();
  }

  const dates = portfolioData.curve_dates || [];
  if (dates.length) {
    ctx.textAlign = 'left';
    ctx.fillText(dates[0].slice(0, 7), padL, H - 4);
    ctx.textAlign = 'right';
    ctx.fillText(dates[dates.length - 1].slice(0, 7), W - padR, H - 4);
  }

  names.forEach((name, k) => {
    const c = curves[k];
    ctx.beginPath();
    c.forEach((v, i) => { const x = gx(i), y = gy(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = PF_COLORS[name] || '#888';
    ctx.lineWidth = name === 'fly' ? 2.0 : 1.1;
    ctx.globalAlpha = name === 'fly' ? 1 : 0.75;
    ctx.stroke();
    ctx.globalAlpha = 1;
  });

  const leg = $('portfolio-legend');
  if (leg) {
    leg.innerHTML = names.map(nm =>
      `<span class="pf-legend-item"><span class="pf-legend-swatch" ` +
      `style="background:${PF_COLORS[nm]}"></span>${PF_LABELS[nm] || nm}</span>`
    ).join('');
  }
}

/* ─────────────────────────────────────────────
    Learning History Fetch & Charts
 ───────────────────────────────────────────── */
async function fetchLearningHistory() {
  try {
    const r = await fetch(`${API}/learning_history`);
    if (!r.ok) return;
    const d = await r.json();
    learningHistory = d.history || [];
    totalEpisodes = d.total_episodes || 0;
    const el = $('learn-total-episodes');
    if (el) el.textContent = totalEpisodes;
    drawLearningHistoryChart();
  } catch(e) {
    console.warn('Learning history fetch failed:', e);
  }
}

function drawLearningHistoryChart() {
  const canvas = $('learn-history-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 400;
  const H = canvas.clientHeight || 160;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const data = learningHistory.slice(-50);
  const n = data.length;
  if (n < 2) return;

  const returns  = data.map(e => e.return_pct);
  const sharpe   = data.map(e => e.sharpe);
  const winrate  = data.map(e => e.win_rate);

  const allVals = [...returns, ...sharpe, ...winrate];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const rangeV = (maxV - minV) || 1;

  const padL = 28, padR = 8, padT = 12, padB = 18;
  const cW = W - padL - padR;
  const cH = H - padT - padB;

  const gx = (i) => padL + (i / (n - 1)) * cW;
  const gy = (v) => padT + cH - ((v - minV) / rangeV) * cH;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let s = 1; s <= 3; s++) {
    const y = padT + (cH / 4) * s;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }

  // Return % (green)
  ctx.beginPath();
  data.forEach((e, i) => {
    const x = gx(i), y = gy(e.return_pct);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.8;
  ctx.stroke();

  // Sharpe (cyan)
  ctx.beginPath();
  data.forEach((e, i) => {
    const x = gx(i), y = gy(e.sharpe);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.4;
  ctx.stroke();

  // Win rate (purple)
  ctx.beginPath();
  data.forEach((e, i) => {
    const x = gx(i), y = gy(e.win_rate);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#8c52ff';
  ctx.lineWidth = 1.2;
  ctx.stroke();

  // Episode numbers
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = `9px 'JetBrains Mono', monospace`;
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 6))) {
    ctx.fillText(`${data[i].episode}`, gx(i) - 4, H - 4);
  }
}

/* ─────────────────────────────────────────────
    Simulation Runs History
 ───────────────────────────────────────────── */
let simulationHistory = [];

async function fetchSimulationHistory() {
  try {
    const r = await fetch(`${API}/simulation_history`);
    if (!r.ok) return;
    const d = await r.json();
    simulationHistory = d.runs || [];
    renderSimulationHistoryUI();
  } catch(e) {
    console.warn('Simulation history fetch failed:', e);
  }
}

async function apiClearSimulationHistory() {
  try {
    const r = await fetch(`${API}/simulation_history/clear`, { method: 'POST' });
    if (r.ok) {
      toast('Simulation history cleared', 'buy-toast');
      await fetchSimulationHistory();
    }
  } catch(e) {
    toast('Failed to clear history', 'sell-toast');
  }
}

function renderSimulationHistoryUI() {
  const badge = $('sim-tab-badge');
  if (badge) badge.textContent = simulationHistory.length;

  // Aggregate KPIs
  const totalRuns = simulationHistory.length;
  const totalEps = simulationHistory.reduce((sum, r) => sum + (r.episodes || 0), 0);
  // Aggregate the HELD-OUT metrics (the trustworthy ones), not the in-sample max.
  const hoRet = simulationHistory
    .map(r => (r.heldout || {}).mean_fly_return_pct)
    .filter(v => v !== undefined && !isNaN(v));
  const hoExcess = simulationHistory
    .map(r => (r.heldout || {}).fly_excess_vs_buyhold_pct)
    .filter(v => v !== undefined && !isNaN(v));

  const heldoutMean = hoRet.length > 0 ? (hoRet.reduce((a, b) => a + b, 0) / hoRet.length) : 0;
  const excessMean = hoExcess.length > 0 ? (hoExcess.reduce((a, b) => a + b, 0) / hoExcess.length) : 0;

  const aggRuns = $('sim-agg-runs');
  if (aggRuns) aggRuns.textContent = totalRuns;
  const aggEps = $('sim-agg-episodes');
  if (aggEps) aggEps.textContent = totalEps;
  const aggBest = $('sim-agg-best');
  if (aggBest) {
    aggBest.textContent = `${heldoutMean >= 0 ? '+' : ''}${heldoutMean.toFixed(1)}%`;
    aggBest.className = `sim-kpi-val ${heldoutMean >= 0 ? 'pos' : 'neg'}`;
  }
  const aggAvg = $('sim-agg-avg');
  if (aggAvg) {
    aggAvg.textContent = `${excessMean >= 0 ? '+' : ''}${excessMean.toFixed(1)}%`;
    aggAvg.className = `sim-kpi-val ${excessMean >= 0 ? 'pos' : 'neg'}`;
  }

  // Render Full Run List in #tab-sims
  const listEl = $('sim-runs-list');
  if (listEl) {
    if (simulationHistory.length === 0) {
      listEl.innerHTML = `
        <div class="sim-empty-state">
          <span style="font-size:1.6rem;opacity:.6">🧪</span>
          <p>No simulations run yet.</p>
          <p class="sim-empty-sub">Click <b>"Run 5 Simulations"</b> or <b>"Evolve Fly Brain"</b> to generate training history.</p>
        </div>`;
    } else {
      listEl.innerHTML = '';
      // Show newest runs first
      [...simulationHistory].reverse().forEach((run) => {
        const card = document.createElement('div');
        card.className = 'sim-card';
        const isPos = (run.median_return_pct !== undefined ? run.median_return_pct : run.avg_return_pct || 0) >= 0;
        const bestRet = run.best_return_pct || 0;
        const medianRet = (run.median_return_pct !== undefined && run.median_return_pct !== null)
          ? run.median_return_pct : (run.avg_return_pct || 0);
        const ho = run.heldout || {};
        const hoRet = ho.mean_fly_return_pct;
        const hoExcess = ho.fly_excess_vs_buyhold_pct;
        const epData = run.episodes_data || [];

        card.innerHTML = `
          <div class="sim-card-header">
            <div class="sim-run-tag">
              <span class="sim-run-num">Run #${run.run}</span>
              <span class="sim-timestamp">${run.timestamp || ''}</span>
            </div>
            <span class="sim-return-badge ${isPos ? 'pos' : 'neg'}">
              ${isPos ? '+' : ''}${medianRet.toFixed(2)}% <span style="opacity:.6;font-weight:400;font-size:.85em">median</span>
            </span>
          </div>
          <div class="sim-meta-row">
            <div class="sim-meta-item">
              <span class="sim-meta-lbl">Episodes</span>
              <span class="sim-meta-val">${run.episodes}</span>
            </div>
            <div class="sim-meta-item">
              <span class="sim-meta-lbl">Median</span>
              <span class="sim-meta-val ${medianRet >= 0 ? 'pos' : 'neg'}">${medianRet >= 0 ? '+' : ''}${medianRet.toFixed(1)}%</span>
            </div>
            <div class="sim-meta-item">
              <span class="sim-meta-lbl">Best Ep</span>
              <span class="sim-meta-val ${bestRet >= 0 ? 'pos' : 'neg'}">${bestRet >= 0 ? '+' : ''}${bestRet.toFixed(1)}%</span>
            </div>
            <div class="sim-meta-item">
              <span class="sim-meta-lbl">Held-out</span>
              <span class="sim-meta-val ${(hoRet || 0) >= 0 ? 'pos' : 'neg'}">${hoRet === undefined ? '—' : (hoRet >= 0 ? '+' : '') + hoRet.toFixed(1) + '%'}</span>
            </div>
            <div class="sim-meta-item">
              <span class="sim-meta-lbl">vs Buy&amp;Hold</span>
              <span class="sim-meta-val ${(hoExcess || 0) >= 0 ? 'pos' : 'neg'}">${hoExcess === undefined ? '—' : (hoExcess >= 0 ? '+' : '') + hoExcess.toFixed(1) + '%'}</span>
            </div>
          </div>
          <div class="sim-card-actions">
            <button class="sim-btn-inspect" data-run-id="${run.run}">
              🔬 Inspect in Lab
            </button>
            ${epData.length > 0 ? `<button class="sim-toggle-btn" data-run-toggle="${run.run}">▼ Show Episodes (${epData.length})</button>` : ''}
          </div>
          ${epData.length > 0 ? `
            <div class="sim-episodes-table-wrap" id="sim-episodes-${run.run}" style="display:none">
              <table class="sim-episodes-table">
                <thead>
                  <tr>
                    <th>Ep</th>
                    <th>Ticker</th>
                    <th>Period</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Win%</th>
                    <th>Trades</th>
                  </tr>
                </thead>
                <tbody>
                  ${epData.map(e => `
                    <tr>
                      <td>#${e.episode}</td>
                      <td>${e.ticker || '—'}</td>
                      <td style="opacity:.7;font-size:.85em">${(e.start_date && e.end_date) ? e.start_date + ' → ' + e.end_date : (e.steps || '—')}</td>
                      <td style="color:${e.return_pct >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:600">
                        ${e.return_pct >= 0 ? '+' : ''}${e.return_pct}%
                      </td>
                      <td>${e.sharpe}</td>
                      <td>${e.win_rate}%</td>
                      <td>${e.trades}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          ` : ''}
        `;
        listEl.appendChild(card);
      });

      // Attach inspect handlers
      listEl.querySelectorAll('.sim-btn-inspect').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const rId = parseInt(btn.dataset.runId);
          const found = simulationHistory.find(r => r.run === rId);
          if (found) {
            openLabModal(found);
          }
        });
      });

      // Attach table toggle handlers
      listEl.querySelectorAll('.sim-toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const rId = btn.dataset.runToggle;
          const wrap = $(`sim-episodes-${rId}`);
          if (wrap) {
            const isHidden = wrap.style.display === 'none';
            wrap.style.display = isHidden ? 'block' : 'none';
            btn.textContent = isHidden ? `▲ Hide Episodes` : `▼ Show Episodes (${wrap.querySelectorAll('tbody tr').length})`;
          }
        });
      });
    }
  }

  // Render Mini List in #tab-stats
  const statsList = $('stats-sim-list');
  if (statsList) {
    if (simulationHistory.length === 0) {
      statsList.innerHTML = `<div class="sim-mini-empty">No simulation runs recorded.</div>`;
    } else {
      statsList.innerHTML = '';
      [...simulationHistory].slice(-5).reverse().forEach(run => {
        const item = document.createElement('div');
        item.className = 'sim-mini-item';
        const medRet = (run.median_return_pct !== undefined && run.median_return_pct !== null)
          ? run.median_return_pct : (run.avg_return_pct || 0);
        const isPos = medRet >= 0;
        const hoX = (run.heldout || {}).fly_excess_vs_buyhold_pct;
        item.innerHTML = `
          <span><b>Run #${run.run}</b> <small style="color:var(--txt-dim)">(${run.episodes} eps${hoX !== undefined ? ' · held-out ' + (hoX >= 0 ? '+' : '') + hoX.toFixed(1) + '% vs B&H' : ''})</small></span>
          <span style="color:${isPos ? 'var(--green)' : 'var(--red)'};font-weight:600">
            ${isPos ? '+' : ''}${medRet.toFixed(2)}%
          </span>
        `;
        item.addEventListener('click', () => {
          openLabModal(run);
        });
        statsList.appendChild(item);
      });
    }
  }
}
function startAuto() {
  autoPilot = true;
  qAutoBtn.textContent = '⏹ Stop Auto-Pilot';
  qAutoBtn.classList.add('running');
  scheduleNext();
}

function stopAuto() {
  autoPilot = false;
  clearTimeout(autoTimer);
  qAutoBtn.textContent = '▶ Start Auto-Pilot';
  qAutoBtn.classList.remove('running');
}

function scheduleNext() {
  if (!autoPilot) return;
  const rawDelay = parseInt(qSpeedSlider.value);
  const delay = isNaN(rawDelay) || rawDelay < 16 ? 400 : rawDelay;
  autoTimer = setTimeout(async () => {
    if (!autoPilot) return;
    const d = await apiStep(-1);
    // If done, handleStepResult manages the auto-reset + reschedule.
    // Only self-schedule here when NOT done.
    if (d && !d.done && autoPilot) scheduleNext();
  }, delay);
}

/* ─────────────────────────────────────────────
    Event Listeners
───────────────────────────────────────────── */
/* Buttons are set up in boot() after DOM ready */

$('reset-btn').addEventListener('click', () => {
  stopAuto();
  apiReset();
});

$('auto-btn').addEventListener('click', () => {
  autoPilot ? stopAuto() : startAuto();
});

$('portfolio-run-btn')?.addEventListener('click', () => fetchPortfolioEval(true));

$('train-batch-btn').addEventListener('click', () => {
  stopAuto();
  apiTrainBatch();
});

$('train-sims-btn').addEventListener('click', () => {
  stopAuto();
  apiRunSimulations();
});

qSpeedSlider.addEventListener('input', () => {
  qSpeedVal.textContent = qSpeedSlider.value + ' ms';
});

$('view-mode').addEventListener('change', e => {
  currentView = e.target.value;
  if (currentView === 'biological') {
    showTracts = false;
    if (atlasPointsMesh) atlasPointsMesh.visible = true;
    if (instancedMesh) instancedMesh.visible = true;
    if (brainShellMesh) brainShellMesh.visible = true;
  } else if (currentView === 'connectome') {
    showTracts = true;
    if (atlasPointsMesh) atlasPointsMesh.visible = true;
    if (instancedMesh) instancedMesh.visible = true;
    if (brainShellMesh) brainShellMesh.visible = true;
  } else if (currentView === 'tracts') {
    showTracts = true;
    if (atlasPointsMesh) atlasPointsMesh.visible = false;
    if (instancedMesh) instancedMesh.visible = false;
    if (brainShellMesh) brainShellMesh.visible = false;
  } else if (currentView === 'circuit') {
    showTracts = false;
    if (atlasPointsMesh) atlasPointsMesh.visible = false;
    if (instancedMesh) instancedMesh.visible = true;
    if (brainShellMesh) brainShellMesh.visible = true;
  }
  const tractBtn = $('toggle-tracts-btn');
  if (tractBtn) tractBtn.textContent = showTracts ? '⚡ Tracts: ON' : '⚡ Tracts: OFF';
  buildSynapticLines();
  updateBrainViz(spikeData1024, currentDopamine);
});

$('colour-mode').addEventListener('change', () => {
  updateBrainViz(spikeData1024, currentDopamine);
});

// Evolution Lab open/close
['open-lab-btn', 'open-lab-top-btn'].forEach(id => {
  const el = $(id);
  if (el) el.addEventListener('click', () => openLabModal(lastTrainData));
});

['close-lab-btn', 'close-lab-footer-btn'].forEach(id => {
  const el = $(id);
  if (el) el.addEventListener('click', closeLabModal);
});

const trainAgainBtn = $('train-again-btn');
if (trainAgainBtn) {
  trainAgainBtn.addEventListener('click', () => {
    closeLabModal();
    stopAuto();
    apiTrainBatch();
  });
}

// Close modal on backdrop click
$('evolution-modal')?.addEventListener('click', e => {
  if (e.target === $('evolution-modal')) closeLabModal();
});

// Policy chip selector styling
document.querySelectorAll('input[name="policy"]').forEach(radio => {
  radio.addEventListener('change', () => {
    document.querySelectorAll('.policy-chip').forEach(c => c.classList.remove('active'));
    radio.closest('.policy-chip').classList.add('active');
  });
});

// Neuropil filter buttons
document.querySelectorAll('.np-filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.np-filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentRegionFilter = btn.dataset.region || 'all';
    updateBrainViz(spikeData1024, currentDopamine);
  });
});

/* ─────────────────────────────────────────────
   Resize handling
───────────────────────────────────────────── */
function handleResize() {
  const wrap = $('brain-canvas-wrap');
  if (!wrap || !threeRenderer) return;
  const W = Math.max(wrap.clientWidth, 320);
  const H = Math.max(wrap.clientHeight, 270);
  threeCamera.aspect = W / H;
  threeCamera.updateProjectionMatrix();
  threeRenderer.setSize(W, H);
  drawPortfolioChart();
  drawPortfolioEquityChart();
  drawRewardChart();
  drawLearningHistoryChart();
  drawOscilloscope();
}
window.addEventListener('resize', handleResize);

/* ─────────────────────────────────────────────
   Boot
───────────────────────────────────────────── */
async function boot() {
  initTabs();
  await apiPing();
  await fetchLearningHistory();
  await fetchSimulationHistory();

  // Clear sims button & View all link setup
  const clearSimsBtn = $('btn-clear-sims');
  if (clearSimsBtn) {
    clearSimsBtn.addEventListener('click', apiClearSimulationHistory);
  }
  const viewAllSimsLink = $('link-view-all-sims');
  if (viewAllSimsLink) {
    viewAllSimsLink.addEventListener('click', () => {
      const tabBtn = document.querySelector('.tab-btn[data-tab="sims"]');
      if (tabBtn) tabBtn.click();
    });
  }
  // Fetch neuron count from server BEFORE building the 3-D brain, so the
  // activeCoords / neuronRegions arrays and InstancedMesh are sized correctly.
  try {
    const r = await fetch(`${API}/brain_status`);
    if (r.ok) {
      const d = await r.json();
      N_NEURONS = d.N_NEURONS || 1024;
      activeCoords = new Float32Array(N_NEURONS * 3);
      neuronRegions = new Uint8Array(N_NEURONS);
    }
  } catch {}
  initThree();
  initUniverseControls();
  await fetchConfig();
  await apiReset();
  updateNeuronCountDisplay();
// Initialize simulations button state
   const simsBtn = $('train-sims-btn');
   if (simsBtn) simsBtn.disabled = totalEpisodes === 0;

   // Set up action button event listeners (after DOM is ready)
   ['btn-buy25','btn-buy50','btn-buy100','btn-hold','btn-sell25','btn-sell50','btn-sell100'].forEach(id => {
      const btn = $(id);
      if (btn) {  // Check if button exists before adding event listener
        btn.addEventListener('click', () => {
          const action = parseInt(btn.dataset.action);
          apiStep(action);
        });
      }
   });
 }

// Start once DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

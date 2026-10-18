import gymnasium as gym
import math
import numpy as np
import pandas as pd
from pathlib import Path


def _load_ohlcv(path: Path) -> pd.DataFrame:
    """Load a yfinance-saved CSV robustly regardless of header format."""
    raw = pd.read_csv(path, header=0)
    # yfinance multi-header: first col is 'Price', rows 0='Ticker', 1='Date'
    # Real data starts at row 2. Detect this by checking first column name.
    first_col = raw.columns[0]
    if first_col == 'Price':
        # Skip the two metadata rows, rename first col to Date
        raw = raw.iloc[2:].reset_index(drop=True)
        raw = raw.rename(columns={first_col: 'Date'})
    elif first_col.lower() not in ('date', 'datetime'):
        raw = raw.rename(columns={first_col: 'Date'})

    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return raw


class FlyTradeEnv(gym.Env):
    """Gymnasium environment for stock trading using the FlyTrader model.

    Observation: A vector of recent OHLCV values plus technical indicators.
    Action: Discrete space -> 0: Hold, 1: Buy 25%, 2: Buy 50%, 3: Buy 100%, 4: Sell 25%, 5: Sell 50%, 6: Sell 100%.
    Reward: Risk-adjusted return (log-return + drawdown penalty + Sharpe bonus + volatility penalty).
    Transaction cost: 0.1% of trade value, deducted from each buy/sell.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path: Path, window_size: int = 30, initial_cash: float = 1e5, transaction_cost: float = 0.001, stop_loss: float = 0.05, take_profit: float = 0.15, enable_sl_tp: bool = False, start_date=None, end_date=None, indicator_norm: str = "per_feature", reward_mode: str = "shaped", dd_penalty: float = 0.0, use_xsection: bool = False):
        super().__init__()
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.data = _load_ohlcv(data_path)
        required = ["Open", "High", "Low", "Close", "Volume"]
        assert all(col in self.data.columns for col in required), "Missing OHLCV columns"
        # Compute the long-horizon regime block on the FULL history *before*
        # slicing, otherwise a 200-day trend is undefined for the first ~10
        # months of a test slice.
        regime_full = self._compute_regime_block(self.data)
        # Cross-sectional block: also computed on the FULL history then sliced,
        # so universe-relative ranks are defined from the first bar of a test
        # window. Loaded from the precomputed panel (build_xsection.py).
        self.use_xsection = bool(use_xsection)
        xsec_full = None
        if self.use_xsection:
            from config import DATA_DIR
            from xsection import default_panel_path, features_for_ticker, load_panel
            ticker = Path(data_path).stem.replace("_daily", "")
            panel = load_panel(default_panel_path(DATA_DIR))
            xsec_full = features_for_ticker(panel, ticker, self.data.index)
        # Optional inclusive date slicing, used for train/validation/test splits.
        keep = np.ones(len(self.data), dtype=bool)
        if start_date is not None:
            keep &= np.asarray(self.data.index >= pd.to_datetime(start_date))
        if end_date is not None:
            keep &= np.asarray(self.data.index <= pd.to_datetime(end_date))
        self.regime = regime_full[keep]
        self.xsec = xsec_full[keep] if xsec_full is not None else None
        self.data = self.data[keep].copy()
        assert len(self.data) > window_size + 1, (
            f"Not enough rows ({len(self.data)}) after date slicing "
            f"[{start_date} .. {end_date}] for window_size={window_size}"
        )
        self.start_date = start_date
        self.end_date = end_date
        self.n_steps = len(self.data) - self.window_size
        self._compute_indicators()
        n_ohlcv = window_size * len(required)
        n_ind = window_size * 8  # RSI, MACD, BB width, ATR, trend strength, vol regime, SMA ratio, EMA trend
        n_port = 4  # exposure, cash fraction, unrealised PnL, steps-since-trade
        n_regime = 4  # dist SMA200, dist SMA50, ann. vol20, vol20/vol60
        n_xsec = int(self.xsec.shape[1]) if self.xsec is not None else 0
        self.n_xsec = n_xsec
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_ohlcv + n_ind + n_xsec + n_port + n_regime,), dtype=np.float32)
        # 7 actions: Hold, Buy 25%, Buy 50%, Buy 100%, Sell 25%, Sell 50%, Sell 100%
        self.action_space = gym.spaces.Discrete(7)
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.enable_sl_tp = enable_sl_tp
        assert indicator_norm in ("block", "per_feature"), "indicator_norm must be 'block' or 'per_feature'"
        self.indicator_norm = indicator_norm
        assert reward_mode in ("shaped", "excess", "logret"), \
            "reward_mode must be 'shaped', 'excess' or 'logret'"
        self.reward_mode = reward_mode
        # Drawdown-sensitive term: penalises the *increase* in the agent's
        # own peak-to-trough drawdown at each step. Zero by default (pure
        # objective); a positive value is what makes de-risking pay, which is
        # the only known way for a long-only daily strategy to beat passive
        # exposure over a full cycle.
        self.dd_penalty = float(dd_penalty)
        self.reset()

    # Fractional trade sizes for each action
    BUY_FRACTIONS = [0.0, 0.25, 0.50, 1.0, 0.0, 0.0, 0.0]
    SELL_FRACTIONS = [0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 1.0]

    @staticmethod
    def _compute_regime_block(df) -> np.ndarray:
        """Long-horizon trend/volatility context, one row per bar (no look-ahead).

        Columns: [close/SMA200 - 1, close/SMA50 - 1, ann. vol20, vol20/vol60].
        Rows before the lookback is available are neutral (0 trend, 0 vol).
        """
        close = df["Close"].values.astype(np.float64)
        n = len(close)
        out = np.zeros((n, 4), dtype=np.float32)
        if n == 0:
            return out
        rets = np.zeros(n, dtype=np.float64)
        rets[1:] = np.diff(close) / (close[:-1] + 1e-12)

        def trailing_mean(k):
            res = np.full(n, np.nan)
            if n >= k:
                c = np.cumsum(np.insert(close, 0, 0.0))
                res[k - 1:] = (c[k:] - c[:-k]) / k
            return res

        def trailing_vol(k):
            res = np.zeros(n, dtype=np.float64)
            if n >= k:
                c1 = np.cumsum(np.insert(rets, 0, 0.0))
                c2 = np.cumsum(np.insert(rets ** 2, 0, 0.0))
                for i in range(k - 1, n):
                    m = (c1[i + 1] - c1[i + 1 - k]) / k
                    m2 = (c2[i + 1] - c2[i + 1 - k]) / k
                    res[i] = math.sqrt(max(0.0, m2 - m * m)) * math.sqrt(252.0)
            return res

        sma50, sma200 = trailing_mean(50), trailing_mean(200)
        vol20, vol60 = trailing_vol(20), trailing_vol(60)
        trend200 = np.where(np.isnan(sma200) | (sma200 <= 1e-9), 0.0,
                            close / (sma200 + 1e-12) - 1.0)
        trend50 = np.where(np.isnan(sma50) | (sma50 <= 1e-9), 0.0,
                           close / (sma50 + 1e-12) - 1.0)
        out[:, 0] = np.clip(trend200, -1.0, 1.0)
        out[:, 1] = np.clip(trend50, -1.0, 1.0)
        out[:, 2] = np.clip(vol20, 0.0, 3.0)
        out[:, 3] = np.clip(vol20 / (vol60 + 1e-9), 0.0, 3.0)
        return out

    def _compute_indicators(self):
        close = self.data["Close"].values.astype(np.float64)
        high = self.data["High"].values.astype(np.float64)
        low = self.data["Low"].values.astype(np.float64)
        vol = self.data["Volume"].values.astype(np.float64)
        n = len(close)

        # --- RSI ---
        rsi = np.full(n, 50.0, dtype=np.float64)
        deltas = np.diff(close, prepend=close[0])
        for i in range(1, n):
            win_gains = deltas[max(0, i-13):i+1]
            gain = np.sum(win_gains[win_gains > 0])
            loss = -np.sum(win_gains[win_gains < 0])
            if loss > 0:
                rs = gain / loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
            else:
                rsi[i] = 100.0 if gain > 0 else 50.0

        # --- MACD ---
        macd_line = np.full(n, 0.0, dtype=np.float64)
        signal_line = np.full(n, 0.0, dtype=np.float64)
        for i in range(n):
            e12 = close[max(0, i-11):i+1]
            e26 = close[max(0, i-25):i+1]
            if len(e12) >= 2:
                macd_line[i] = np.mean(e12[-12:]) - np.mean(e26[-26:]) if len(e26) >= 26 else np.mean(e12[-12:])
            if i >= 8:
                signal_line[i] = np.mean(macd_line[max(0, i-8):i+1])
        macd_hist = macd_line - signal_line

        # --- Bollinger Band Width ---
        bb_width = np.full(n, 0.0, dtype=np.float64)
        for i in range(n):
            win = close[max(0, i-19):i+1]
            if len(win) >= 5:
                m = np.mean(win)
                s = np.std(win)
                bb_width[i] = (m + 2*s - (m - 2*s)) / (m + 1e-9)

        # --- ATR ---
        atr = np.full(n, 0.0, dtype=np.float64)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(n):
            win = tr[max(0, i-13):i+1]
            atr[i] = np.mean(win) if len(win) > 0 else 0.0

        # --- Market Regime Features ---
        # Trend strength: slope of recent price action
        trend_strength = np.full(n, 0.0, dtype=np.float64)
        for i in range(n):
            win = close[max(0, i-19):i+1]
            if len(win) >= 5:
                x = np.arange(len(win))
                slope = np.polyfit(x, win, 1)[0]
                trend_strength[i] = slope / (np.mean(win) + 1e-9)

        # Volatility regime: recent vs long-term volatility
        vol_regime = np.full(n, 0.0, dtype=np.float64)
        for i in range(n):
            short_vol = np.std(close[max(0, i-4):i+1]) if i >= 4 else np.std(close[:i+1])
            long_vol = np.std(close[max(0, i-19):i+1]) if i >= 19 else np.std(close[:i+1])
            vol_regime[i] = short_vol / (long_vol + 1e-9) if long_vol > 0 else 1.0

        # --- Trend Indicators ---
        # SMA(10) and SMA(20) ratio
        sma10 = np.full(n, 0.0, dtype=np.float64)
        sma20 = np.full(n, 0.0, dtype=np.float64)
        sma_ratio = np.full(n, 1.0, dtype=np.float64)
        for i in range(n):
            if i >= 9:
                sma10[i] = np.mean(close[i-9:i+1])
            if i >= 19:
                sma20[i] = np.mean(close[i-19:i+1])
            if sma20[i] > 1e-9:
                sma_ratio[i] = sma10[i] / sma20[i]

        # EMA(10) trend
        ema10 = np.full(n, 0.0, dtype=np.float64)
        ema_trend = np.full(n, 0.0, dtype=np.float64)
        if n > 0:
            ema10[0] = close[0]
            alpha = 2.0 / (10 + 1)
            for i in range(1, n):
                ema10[i] = alpha * close[i] + (1 - alpha) * ema10[i-1]
            for i in range(n):
                if ema10[i] > 1e-9:
                    ema_trend[i] = (close[i] - ema10[i]) / ema10[i]

        self.indicators = np.stack([rsi, macd_hist, bb_width, atr, trend_strength, vol_regime, sma_ratio, ema_trend], axis=1).astype(np.float32)
        # Per-indicator statistics for the 'per_feature' normalisation mode.
        # NOTE: computed over the whole (sliced) series. For strict walk-forward
        # correctness these should come from the training slice only.
        self.ind_mean = self.indicators.mean(axis=0).astype(np.float64)
        self.ind_std = self.indicators.std(axis=0).astype(np.float64)

    def _market_observation(self, idx: int) -> np.ndarray:
        """[OHLCV window, indicator window, cross-sectional block] for bar ``idx``.

        This is the *asset-local* part of the observation: everything that
        depends only on this ticker's own history. The portfolio env reuses it
        and supplies its own portfolio block.
        """
        window = self.data.iloc[idx : idx + self.window_size][["Open", "High", "Low", "Close", "Volume"]]
        norm = window["Close"].iloc[0]
        ohlcv = (window / norm).values.flatten().astype(np.float32)
        ind = self.indicators[idx : idx + self.window_size].astype(np.float64)
        if self.indicator_norm == "per_feature":
            # Normalise each indicator independently (RSI vs ATR vs ema_trend
            # live on very different scales). Keeps the trailing per-step
            # indicators interpretable, unlike the legacy block z-score.
            ind = (ind - self.ind_mean) / (self.ind_std + 1e-9)
            ind = np.clip(ind, -5.0, 5.0)
            ind = ind.flatten()
        else:
            ind = ind.flatten()
            ind_mean = np.mean(ind)
            ind_std = np.std(ind)
            if ind_std > 1e-6:
                ind = (ind - ind_mean) / ind_std
        ridx = min(idx + self.window_size - 1, len(self.regime) - 1)
        # Cross-sectional context for the *current* bar (empty block when off).
        if self.xsec is not None and ridx >= 0 and len(self.xsec):
            xsec = self.xsec[ridx]
        else:
            xsec = np.zeros(self.n_xsec, dtype=np.float32)
        return np.concatenate([ohlcv, ind, np.asarray(xsec, dtype=np.float32)])

    def _regime_observation(self, idx: int) -> np.ndarray:
        """Long-horizon regime context for bar ``idx`` (final 4 obs values)."""
        ridx = min(idx + self.window_size - 1, len(self.regime) - 1)
        return (self.regime[ridx] if ridx >= 0 and len(self.regime)
                else np.zeros(4, dtype=np.float32))

    def _get_observation(self, idx: int) -> np.ndarray:
        # Asset-local features (OHLCV + indicators + optional cross-section).
        market = self._market_observation(idx)
        # Portfolio-state block: lets the policy condition on its own position.
        price_idx = min(idx + self.window_size - 1, len(self.data) - 1)
        price = float(self.data.iloc[price_idx]["Close"])
        position_value = self.shares * price
        total = self.cash + position_value
        exposure = position_value / total if total > 1e-9 else 0.0
        cash_frac = self.cash / total if total > 1e-9 else 1.0
        if self.shares > 0 and self.entry_price > 0:
            unreal = float(np.clip(price / self.entry_price - 1.0, -1.0, 1.0))
        else:
            unreal = 0.0
        recency = float(np.clip((self.current_step - self.last_trade_step) / 30.0, 0.0, 1.0))
        port = np.array([exposure, cash_frac, unreal, recency], dtype=np.float32)
        regime = self._regime_observation(idx)
        return np.concatenate([market, port, np.asarray(regime, dtype=np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_history = []
        self.entry_price = 0.0  # Track entry price for stop-loss/take-profit
        self.highest_price = 0.0  # Track highest price since entry
        self.last_trade_step = -10  # Track last trade step for overtrading penalty
        self.steps_in_cash = 0  # Track consecutive steps holding 100% cash
        # --- Buy & hold benchmark (used by reward_mode='excess') ---
        # The benchmark buys 100% at the first tradable close and holds forever.
        self.benchmark_entry_price = None
        self.benchmark_value = self.initial_cash
        self.prev_benchmark_value = self.initial_cash
        # Peak equity, for the optional drawdown penalty.
        self.peak_value = self.initial_cash
        obs = self._get_observation(self.current_step)
        return obs, {}

    def _apply_action(self, action: int, price: float, confidence: float = 1.0):
        """Execute fractional trade based on action (0-6) and confidence (0-1)."""
        if self.cash > 0 and action >= 1 and action <= 3:
            fraction = self.BUY_FRACTIONS[action] * confidence
            amount = self.cash * fraction
            if amount > 0:
                self.shares += amount / price
                self.cash -= amount
                self.cash -= amount * self.transaction_cost
        elif self.shares > 0 and action >= 4:
            fraction = self.SELL_FRACTIONS[action]
            amount = self.shares * fraction
            if amount > 0:
                self.cash += amount * price
                self.shares -= amount
                self.cash -= amount * price * self.transaction_cost
            # Reset tracking if fully sold
            if self.shares <= 1e-9:
                self.entry_price = 0.0
                self.highest_price = 0.0

    def step(self, action: int, confidence: float = 1.0):
        if self.current_step >= self.n_steps:
            idx = len(self.data) - 1
            price = float(self.data.iloc[idx]["Close"])
            pv = float(self.cash + self.shares * price)
            return np.zeros_like(self.observation_space.low), 0.0, True, False, {"portfolio_value": pv}

        # Execute action at the close price of the current step
        idx = min(self.current_step + self.window_size - 1, len(self.data) - 1)
        price = float(self.data.iloc[idx]["Close"])
        
        if self.cash > 0 and action >= 1 and action <= 3:
            fraction = self.BUY_FRACTIONS[action] * confidence
            max_amount = self.cash / (1 + self.transaction_cost)  # Account for transaction cost
            amount = self.cash * fraction
            if amount > max_amount:
                amount = max_amount
            if amount > 0:
                self.shares += amount / price
                self.cash -= amount * (1 + self.transaction_cost)
        elif self.shares > 0 and action >= 4:
            fraction = self.SELL_FRACTIONS[action]
            amount = self.shares * fraction
            if amount > 0:
                sell_value = amount * price
                self.cash += sell_value * (1 - self.transaction_cost)
                self.shares -= amount
            # Reset tracking if fully sold
            if self.shares <= 1e-9:
                self.entry_price = 0.0
                self.highest_price = 0.0

        # Track entry price and highest price for stop-loss/take-profit
        if self.shares > 0 and self.entry_price == 0.0:
            self.entry_price = price
        if self.shares > 0 and price > self.highest_price:
            self.highest_price = price

        # Check stop-loss and take-profit (only if enabled)
        stop_loss_triggered = False
        take_profit_triggered = False
        if self.enable_sl_tp and self.shares > 0 and self.entry_price > 0:
            drawdown_from_entry = (self.entry_price - price) / self.entry_price
            gain_from_entry = (price - self.entry_price) / self.entry_price
            if drawdown_from_entry >= self.stop_loss:
                stop_loss_triggered = True
            elif gain_from_entry >= self.take_profit:
                take_profit_triggered = True

        # Auto-execute stop-loss or take-profit
        if stop_loss_triggered or take_profit_triggered:
            sell_value = self.shares * price
            self.cash += sell_value
            self.shares = 0.0
            self.cash -= sell_value * self.transaction_cost
            self.entry_price = 0.0
            self.highest_price = 0.0

        # Advance
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_obs = self._get_observation(self.current_step) if not done else np.zeros_like(self.observation_space.low)

        # Compute portfolio value
        portfolio_value = float(self.cash + self.shares * price)
        self.portfolio_history.append(portfolio_value)

        # --- Buy & hold benchmark (buy 100% at the first tradable close, hold) ---
        if self.benchmark_entry_price is None:
            self.benchmark_entry_price = price
        self.benchmark_value = self.initial_cash * (price / (self.benchmark_entry_price + 1e-12))

        prev_pv = self.portfolio_history[-2] if len(self.portfolio_history) > 1 else self.initial_cash
        log_ret = float(np.log((portfolio_value + 1e-9) / (prev_pv + 1e-9)))
        bench_log_ret = float(np.log((self.benchmark_value + 1e-9) / (self.prev_benchmark_value + 1e-9)))
        self.prev_benchmark_value = self.benchmark_value

        # --- Reward ---
        # 'excess' compares the agent's per-step return against the passive
        # buy & hold benchmark. A pure hold yields ~0 reward, so the only way to
        # earn positive reward is to *time* the market better than the benchmark.
        # This is the objective that matters for "beat buy & hold".
        if self.reward_mode == "excess":
            reward = (log_ret - bench_log_ret) * 100.0
        else:
            reward = log_ret * 100.0

        # Optional drawdown penalty: charge the *deepening* of the equity
        # drawdown at this step. Being flat during a slide costs nothing (the
        # drawdown does not deepen), so this is exactly the signal that teaches
        # a long-only agent to step aside ahead of / during deep drawdowns.
        if self.dd_penalty > 0.0:
            prev_peak = self.peak_value
            new_peak = max(prev_peak, portfolio_value)
            dd_before = (prev_peak - prev_pv) / (prev_peak + 1e-12)
            dd_after = (new_peak - portfolio_value) / (new_peak + 1e-12)
            reward -= self.dd_penalty * max(0.0, dd_after - dd_before) * 100.0
            self.peak_value = new_peak
        else:
            self.peak_value = max(self.peak_value, portfolio_value)

        if self.reward_mode == "shaped":
            # --- Legacy hand-coded regime guidance (kept for back-compat) ---
            # Observation layout: 150 OHLCV + 240 indicator features (8 x 30)
            # Last features = current-step indicators: [rsi, macd_hist, bb_width,
            #   atr, trend_strength, vol_regime, sma_ratio, ema_trend]
            obs_flat = self._get_observation(
                min(self.current_step, len(self.data) - self.window_size - 1))
            # ema_trend is the last feature of the *market* block (the portfolio
            # block is appended after it), so index by the market width, not [-1].
            n_market = self.window_size * 13  # 5 OHLCV + 8 indicators per bar
            ema_trend = float(obs_flat[n_market - 1]) if len(obs_flat) >= n_market else 0.0

            # 1. Position Entry (Buying)
            if action in [1, 2, 3]:
                # Neutralize entry fee impact from immediate reward so the RL
                # gradient does not penalize taking action to open positions
                reward = max(reward, 0.0)
                if ema_trend > 0.005:
                    reward += 0.25 if self.shares <= 1e-6 else 0.10
                elif ema_trend > -0.005:
                    reward += 0.08
                elif ema_trend < -0.015:
                    reward -= 0.20

            # 2. Position Management (Holding)
            if action == 0:
                if self.shares > 0 and ema_trend > 0.01:
                    reward += 0.06
                elif self.shares <= 1e-6 and ema_trend < -0.015:
                    reward += 0.03
                elif self.shares > 0 and ema_trend < -0.01:
                    reward -= 0.10

            # 3. Capital Protection (Selling)
            if action in [4, 5, 6]:
                if self.shares > 0 and ema_trend < -0.01:
                    reward += 0.15
                elif self.shares > 0 and ema_trend > 0.03:
                    reward -= 0.02

            # 4. Anti-churn overtrading penalty
            if action != 0:
                steps_since_last_trade = self.current_step - self.last_trade_step
                if steps_since_last_trade < 2:
                    reward -= 0.10
                self.last_trade_step = self.current_step

            # 5. Cash inactivity penalty
            if self.shares <= 1e-6:
                self.steps_in_cash = getattr(self, 'steps_in_cash', 0) + 1
                if ema_trend > 0.002:
                    reward -= 0.05
                elif self.steps_in_cash > 2:
                    reward -= 0.02
            else:
                self.steps_in_cash = 0

        # Terminal bonus
        if done:
            if self.reward_mode == "excess":
                # Difference between agent and benchmark total return.
                excess_ret = ((portfolio_value - self.initial_cash)
                              - (self.benchmark_value - self.initial_cash)) / self.initial_cash
                reward += 0.3 * excess_ret
            elif self.reward_mode == "shaped":
                total_return = (portfolio_value - self.initial_cash) / self.initial_cash
                reward += 0.3 * total_return

        # Safety
        if not np.isfinite(reward):
            reward = 0.0
        info = {
            "portfolio_value": float(portfolio_value),
            "benchmark_value": float(self.benchmark_value),
        }
        return next_obs, reward, done, False, info

    def render(self, mode="human"):
        if mode == "human":
            if self.portfolio_history:
                print(f"Step {self.current_step}: Portfolio = ${self.portfolio_history[-1]:.2f}")
        else:
            pass

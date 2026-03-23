"""
Continuous Swing Environment (V7) — Continuous Position Control MDP

GMGP1 redesign: continuous SAC agent controls position fraction in [-1, 1].
Multi-scale OHLCV features (no LOB), DSR reward, sticky deadband, fee curriculum.

Action Space: Box(-1, 1, shape=(1,)) — target position fraction
  -1.0 = full short, 0.0 = flat, +1.0 = full long

Reward: Differential Sharpe Ratio (DSR) of per-bar PnL minus transaction costs.

Key differences from SwingScalperEnv (V6):
  - Continuous actions (position sizing) vs binary direction
  - Flat state possible (action ≈ 0)
  - Sticky deadband prevents micro-churn
  - Multi-scale OHLCV obs (no LOB features)
  - DSR reward (not raw PnL)
  - Fee curriculum support
"""
import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler

logger = logging.getLogger(__name__)


class ContinuousSwingEnv(gym.Env):
    """Continuous position control with multi-scale OHLCV features and DSR reward."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_handler: Optional["MultiScaleOHLCVHandler"] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # Core config
        self.initial_balance = float(config.get("initial_balance", 100000.0))
        self.window_size = int(config.get("window_size", 30))
        features_per_scale = int(config.get("features_per_scale", 7))

        # Fees (supports curriculum — can be updated at runtime)
        self.taker_fee = float(config.get("taker_fee", 0.0))

        # Deadband: ignore position changes smaller than this
        self.deadband_threshold = float(config.get("deadband_threshold", 0.25))

        # Slippage model: flat bps per unit of position change
        # V7 uses OHLCV (no volume), so slippage is a flat rate, not volume-dependent.
        # Set slippage_base_bps > 0 for production backtests (e.g., 0.3 for Gold).
        self.slippage_base_bps = max(0.0, float(config.get("slippage_base_bps", 0.0)))

        # ATR-based position capping
        self.atr_cap_percentile = float(config.get("atr_cap_percentile", 90))
        self.atr_cap_max_position = float(config.get("atr_cap_max_position", 0.5))

        # Gap detection: zero out returns exceeding 3x ATR/price (session gaps, rolls)
        # Default off — enable for futures with trading halts (Gold, ES).
        self.gap_detection = bool(config.get("gap_detection", False))
        self._gap_atr_mult = float(config.get("gap_atr_mult", 3.0))

        # Reward config
        reward_cfg = config.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "dsr")
        self.dsr_eta = float(reward_cfg.get("dsr_eta", 0.001))
        self.dsr_scale = float(reward_cfg.get("dsr_scale", 1.0))

        # Episode config
        self.episode_length = int(config.get("episode_length", 1000))
        self.random_start = bool(config.get("random_start", True))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.30))
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct

        # Scales from handler (must be set before obs space construction)
        self._scales = config.get("scales", [3, 15, 60])

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Build obs space dynamically from scales config
        obs_spaces = {}
        for i in range(len(self._scales)):
            obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, features_per_scale), dtype=np.float32
            )
        obs_spaces["private"] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # State variables
        self.current_position = 0.0
        self.prev_close = 0.0
        self.current_close = 0.0
        self.current_atr = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.cumulative_fees = 0.0
        self.trade_count = 0
        self.current_step = 0
        self._episode_end = 0

        # DSR state
        self._dsr_A = 0.0  # EMA of returns
        self._dsr_B = 0.0  # EMA of squared returns
        self._dsr_warmup = 0

        # ATR tracking for private state
        self._atr_buffer = []
        self._atr_rolling_mean = 0.0

        # Time encoding cache — avoid np.datetime64 computation every step
        self._cached_time_ptr = -1
        self._cached_time_sin = 0.0
        self._cached_time_cos = 1.0

        # Current observations
        self._current_obs = None

        # (_scales already set above, before obs space construction)

    def set_fees(self, taker_fee: float):
        """Runtime fee update for curriculum learning."""
        self.taker_fee = taker_fee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_position = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.cumulative_fees = 0.0
        self.trade_count = 0
        self.current_step = 0
        self.prev_close = 0.0
        self.current_close = 0.0
        self.current_atr = 0.0

        # DSR state reset
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._dsr_warmup = 0

        # ATR tracking
        self._atr_buffer = []
        self._atr_rolling_mean = 0.0

        # Reset time cache
        self._cached_time_ptr = -1

        if self.handler:
            self.handler.reset()

            # Random start
            data_len = getattr(self.handler, '_len', 0)
            ws = getattr(self.handler, 'window_size', self.window_size)
            if self.episode_length > 0 and data_len > 0 and self.random_start:
                max_start = max(ws, data_len - self.episode_length - ws)
                start_idx = self.np_random.integers(ws, max_start)
                self.handler._ptr = start_idx

            self._episode_end = self.episode_length if self.episode_length > 0 else 0

            # Read first step to initialize prices
            first = self.handler.step()
            if first is not None:
                self.current_close = first["close"]
                self.prev_close = self.current_close
                self.current_atr = first["atr"]
                self._current_obs = self._extract_obs(first)
            else:
                self._current_obs = self._empty_obs()
        else:
            self._current_obs = self._empty_obs()

        return self._get_observation(), {}

    def step(self, action):
        raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
        target_position = np.clip(raw_action, -1.0, 1.0)

        self.current_step += 1

        # 1. Advance data
        step_data = self.handler.step() if self.handler else None

        terminated = False
        truncated = False

        if self.handler and step_data is None:
            truncated = True
            return self._get_observation(), 0.0, terminated, truncated, self._make_info(0.0, False)

        # Save previous close
        self.prev_close = self.current_close

        # 2. Update market state
        if step_data is not None:
            self.current_close = step_data["close"]
            self.current_atr = step_data["atr"]
            self._current_obs = self._extract_obs(step_data)

            # Update ATR rolling stats
            self._atr_buffer.append(self.current_atr)
            if len(self._atr_buffer) > 200:
                self._atr_buffer = self._atr_buffer[-200:]
            self._atr_rolling_mean = np.mean(self._atr_buffer)

        # 3. Compute position delta with deadband
        delta = target_position - self.current_position
        traded = False

        if abs(delta) < self.deadband_threshold:
            delta = 0.0
        else:
            # ATR cap: reduce max position in high-vol regimes
            # FIX R4-AUD-10: Skip during warmup — percentile unreliable with <50 samples
            if self._atr_rolling_mean > 0 and len(self._atr_buffer) >= 50:
                atr_pct = sorted(self._atr_buffer)
                p90_idx = int(len(atr_pct) * self.atr_cap_percentile / 100.0)
                p90_idx = min(p90_idx, len(atr_pct) - 1)
                if self.current_atr > atr_pct[p90_idx]:
                    target_position = np.clip(
                        target_position,
                        -self.atr_cap_max_position,
                        self.atr_cap_max_position
                    )
                    delta = target_position - self.current_position
                    if abs(delta) < self.deadband_threshold:
                        delta = 0.0

            if delta != 0.0:
                self.current_position += delta
                self.current_position = np.clip(self.current_position, -1.0, 1.0)
                traded = True
                self.trade_count += 1

        # 4. Compute PnL
        # FIX R5-AUD-01: Compute price_return once, reuse for both reward and equity update
        pnl_bps = 0.0
        price_return = 0.0
        if self.prev_close > 0:
            price_return = (self.current_close - self.prev_close) / self.prev_close
            # Gap detection: zero out returns from session gaps / contract rolls
            # that exceed gap_atr_mult × ATR/price. These are non-tradeable artifacts.
            if (self.gap_detection and self._atr_rolling_mean > 1e-12
                    and abs(price_return) > self._gap_atr_mult * self._atr_rolling_mean / self.prev_close):
                price_return = 0.0
            pnl_bps = self.current_position * price_return * 10000.0

        # Transaction cost (fee + slippage)
        tc_bps = 0.0
        if traded:
            tc_bps = self.taker_fee * 10000.0 * abs(delta)
            if self.slippage_base_bps > 0:
                tc_bps += self.slippage_base_bps * abs(delta)
            self.cumulative_fees += tc_bps

        # Step return
        R_t = pnl_bps - tc_bps

        # 5. Compute reward
        if self.reward_mode == "dsr":
            reward = self._compute_dsr(R_t)
        else:
            # Raw PnL mode
            reward = float(np.clip(R_t, -50.0, 50.0))

        # 6. Update equity
        # FIX R2-AUD-05: Use current equity (not initial_balance) so PnL compounds correctly.
        # Without this, drawdown recovery is inflated and long backtests diverge from reality.
        equity_delta = self.current_position * price_return * self.equity
        if traded:
            fee_frac = self.taker_fee + self.slippage_base_bps / 10000.0
            equity_delta -= fee_frac * abs(delta) * self.equity
        self.equity += equity_delta
        self.peak_equity = max(self.peak_equity, self.equity)

        # 7. Termination — peak-based drawdown stop
        # FIX R4-AUD-07: Compare against peak_equity (not initial_balance) so a 30%
        # drawdown from the portfolio high triggers termination, regardless of absolute level.
        if self.equity < self._stop_loss_threshold * self.peak_equity:
            terminated = True

        if self._episode_end > 0 and self.current_step >= self._episode_end:
            truncated = True

        obs = self._get_observation()
        info = self._make_info(reward, traded)

        return obs, reward, terminated, truncated, info

    def _compute_dsr(self, R_t: float) -> float:
        """Differential Sharpe Ratio (Moody & Saffell 2001)."""
        delta_A = R_t - self._dsr_A
        delta_B = R_t * R_t - self._dsr_B

        prev_A, prev_B = self._dsr_A, self._dsr_B
        prev_variance = max(prev_B - prev_A ** 2, 0.0)

        self._dsr_A += self.dsr_eta * delta_A
        self._dsr_B += self.dsr_eta * delta_B
        self._dsr_warmup += 1

        if self._dsr_warmup > 1 and prev_variance > 1e-16:
            denom = prev_variance ** 1.5
            dsr = (prev_B * delta_A - 0.5 * prev_A * delta_B) / denom
            return float(np.clip(dsr * self.dsr_scale, -10.0, 10.0))

        return 0.0

    def _extract_obs(self, step_data: Dict) -> Dict[str, np.ndarray]:
        """Extract scale arrays from handler step data (positional keys)."""
        obs = {}
        for i in range(len(self._scales)):
            key = f"scale_{i}"
            if key in step_data:
                arr = step_data[key]
                # Avoid redundant copy if already float32 (handler typically returns f32)
                obs[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        return obs

    def _get_private_state(self) -> np.ndarray:
        """Build 5-dim private state vector.

        [current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]
        """
        # 1. Current position [-1, 1]
        pos = self.current_position

        # 2. Unrealized PnL normalized — not applicable for continuous
        # (position changes continuously, use recent return as proxy)
        pnl_proxy = 0.0
        if self.prev_close > 0 and self.current_close > 0:
            ret_bps = (self.current_close - self.prev_close) / self.prev_close * 10000.0
            pnl_proxy = float(np.clip(self.current_position * ret_bps / 100.0, -1.0, 1.0))

        # 3-4. Time encoding (cached — only recompute when pointer advances)
        if self._current_obs and hasattr(self, 'handler') and self.handler:
            ptr = getattr(self.handler, '_ptr', 0)
            if ptr != self._cached_time_ptr:
                ts = getattr(self.handler, '_base_timestamps', None)
                if ts is not None and ptr > 0 and ptr <= len(ts):
                    ts_val = ts[ptr - 1]
                    dt = np.datetime64(ts_val, 'ns')
                    minutes = (dt - dt.astype('datetime64[D]')).astype('timedelta64[m]').astype(int)
                    self._cached_time_sin = float(np.sin(2 * np.pi * minutes / 1440.0))
                    self._cached_time_cos = float(np.cos(2 * np.pi * minutes / 1440.0))
                self._cached_time_ptr = ptr
            time_sin = self._cached_time_sin
            time_cos = self._cached_time_cos
        else:
            time_sin, time_cos = 0.0, 1.0

        # 5. ATR ratio
        if self._atr_rolling_mean > 1e-12:
            atr_ratio = float(np.clip(self.current_atr / self._atr_rolling_mean, 0.0, 3.0)) / 3.0
        else:
            atr_ratio = 0.5

        return np.array([pos, pnl_proxy, time_sin, time_cos, atr_ratio], dtype=np.float32)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build full observation dict."""
        obs = {}
        n_scales = len(self._scales)
        features_per_scale = int(self.config.get("features_per_scale", 7))

        if self._current_obs:
            for i in range(n_scales):
                key = f"scale_{i}"
                if key in self._current_obs:
                    # No .copy() needed — SyncVectorEnv stacks (copies) all env obs,
                    # and _current_obs is overwritten on next handler.step().
                    obs[key] = self._current_obs[key]
                else:
                    obs[key] = np.zeros((self.window_size, features_per_scale), dtype=np.float32)
        else:
            for i in range(n_scales):
                obs[f"scale_{i}"] = np.zeros((self.window_size, features_per_scale), dtype=np.float32)

        obs["private"] = self._get_private_state()
        return obs

    def _empty_obs(self) -> Dict[str, np.ndarray]:
        features_per_scale = int(self.config.get("features_per_scale", 7))
        return {
            f"scale_{i}": np.zeros((self.window_size, features_per_scale), dtype=np.float32)
            for i in range(len(self._scales))
        }

    def _make_info(self, reward: float, traded: bool) -> Dict:
        drawdown_pct = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        return {
            "portfolio_value": self.equity,
            "position": self.current_position,
            "traded": traded,
            "trade_count": self.trade_count,
            "cumulative_fees": self.cumulative_fees,
            "drawdown_pct": drawdown_pct,
            "reward_total": reward,
            "reward_nav": reward,
            "taker_fee": self.taker_fee,
        }

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Pos: {self.current_position:.3f}, "
            f"Equity: {self.equity:.2f}, Trades: {self.trade_count}, "
            f"Fee: {self.taker_fee:.5f}"
        )

    def close(self):
        super().close()

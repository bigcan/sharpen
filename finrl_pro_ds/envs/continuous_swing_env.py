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

        # ATR-based position capping
        self.atr_cap_percentile = float(config.get("atr_cap_percentile", 90))
        self.atr_cap_max_position = float(config.get("atr_cap_max_position", 0.5))

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

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            "scale_3m": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, features_per_scale), dtype=np.float32
            ),
            "scale_15m": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, features_per_scale), dtype=np.float32
            ),
            "scale_1h": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, features_per_scale), dtype=np.float32
            ),
            "private": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            ),
        })

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

        # Current observations
        self._current_obs = None

        # Scales from handler
        self._scales = config.get("scales", [3, 15, 60])

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
            if self._atr_rolling_mean > 0:
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
        pnl_bps = 0.0
        if self.prev_close > 0:
            price_return = (self.current_close - self.prev_close) / self.prev_close
            pnl_bps = self.current_position * price_return * 10000.0

        # Transaction cost
        tc_bps = 0.0
        if traded:
            tc_bps = self.taker_fee * 10000.0 * abs(delta)
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
        if self.prev_close > 0:
            price_return = (self.current_close - self.prev_close) / self.prev_close
            equity_delta = self.current_position * price_return * self.initial_balance
        else:
            equity_delta = 0.0
        if traded:
            equity_delta -= self.taker_fee * abs(delta) * self.initial_balance
        self.equity += equity_delta
        self.peak_equity = max(self.peak_equity, self.equity)

        # 7. Termination
        if self.equity < self._stop_loss_threshold * self.initial_balance:
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

    @staticmethod
    def _scale_key(scale: int) -> str:
        """Map scale minutes to observation key: 60 → 'scale_1h', else 'scale_{n}m'."""
        if scale == 60:
            return "scale_1h"
        return f"scale_{scale}m"

    def _extract_obs(self, step_data: Dict) -> Dict[str, np.ndarray]:
        """Extract scale arrays from handler step data."""
        obs = {}
        for scale in self._scales:
            src_key = f"scale_{scale}m"   # handler uses raw minutes
            dst_key = self._scale_key(scale)  # obs space uses 1h
            if src_key in step_data:
                obs[dst_key] = step_data[src_key].astype(np.float32)
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

        # 3-4. Time encoding
        if self._current_obs and hasattr(self, 'handler') and self.handler:
            ts = getattr(self.handler, '_base_timestamps', None)
            ptr = getattr(self.handler, '_ptr', 0)
            if ts is not None and ptr > 0 and ptr <= len(ts):
                ts_val = ts[ptr - 1]
                # Convert to minute of day
                dt = np.datetime64(ts_val, 'ns')
                minutes = (dt - dt.astype('datetime64[D]')).astype('timedelta64[m]').astype(int)
                time_sin = float(np.sin(2 * np.pi * minutes / 1440.0))
                time_cos = float(np.cos(2 * np.pi * minutes / 1440.0))
            else:
                time_sin, time_cos = 0.0, 1.0
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

        if self._current_obs:
            for key in ["scale_3m", "scale_15m", "scale_1h"]:
                if key in self._current_obs:
                    obs[key] = self._current_obs[key].copy()
                else:
                    obs[key] = np.zeros((self.window_size, 7), dtype=np.float32)
        else:
            for key in ["scale_3m", "scale_15m", "scale_1h"]:
                obs[key] = np.zeros((self.window_size, 7), dtype=np.float32)

        obs["private"] = self._get_private_state()
        return obs

    def _empty_obs(self) -> Dict[str, np.ndarray]:
        return {
            self._scale_key(s): np.zeros((self.window_size, 7), dtype=np.float32)
            for s in self._scales
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

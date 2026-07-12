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
import logging
from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.dsr import DSRCalculator

if TYPE_CHECKING:
    from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler

logger = logging.getLogger(__name__)


class ContinuousSwingEnv(gym.Env):
    """Continuous position control with multi-scale OHLCV features and DSR reward."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config: dict[str, Any], data_handler: Optional["MultiScaleOHLCVHandler"] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # Core config
        self.initial_balance = float(config.get("initial_balance", 100000.0))
        self.window_size = int(config.get("window_size", 30))
        # FE-04: fall back to the forwarded features: block so a config that
        # only declares features.features_per_scale doesn't silently get the
        # legacy 7 here while the handler emits 8.
        features_per_scale = int(
            config.get(
                "features_per_scale",
                config.get("features", {}).get("features_per_scale", 7),
            ),
        )
        self._features_per_scale = features_per_scale

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

        # Leverage cap (default 1.0 = current behavior). Scales action to
        # target_position ∈ [-max_leverage, +max_leverage]. ATR cap remains
        # absolute (acts as hard safety in high-vol regimes regardless of leverage).
        self.max_leverage = float(config.get("max_leverage", 1.0))

        # Gap detection: zero out returns exceeding 3x ATR/price (session gaps, rolls)
        # Default off — enable for futures with trading halts (Gold, ES).
        self.gap_detection = bool(config.get("gap_detection", False))
        self._gap_atr_mult = float(config.get("gap_atr_mult", 3.0))

        # N2 (PF-XCHECK dual-equity): when set, mark a SECOND equity curve at the
        # bar (H+L)/2 midpoint, in lockstep with the close-marked equity, for the
        # Stage 2.5-R sensitivity audit's mid-vs-close PF cross-check. SHADOW-ONLY
        # (SENS-4): equity_mid never feeds reward / termination / ATR / obs, so the
        # close-marked trajectory is byte-identical with the flag on or off (SENS-5).
        # Default OFF → zero behavior change for training / live / existing backtests.
        self.record_dual_equity = bool(config.get("record_dual_equity", False))

        # Reward config
        reward_cfg = config.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "dsr")
        self.dsr_eta = float(reward_cfg.get("dsr_eta", 0.001))
        self.dsr_scale = float(reward_cfg.get("dsr_scale", 1.0))

        # Path 2: Regime-adaptive DSR
        self._regime_adaptive_dsr = bool(reward_cfg.get("regime_adaptive_dsr", False))
        raw_mults = reward_cfg.get("regime_eta_multipliers", None)
        self._regime_eta_mults: dict[int, float] | None = None
        if raw_mults and self._regime_adaptive_dsr:
            # Config uses string keys; convert to int vol-regime codes
            name_to_code = {"LOW_VOL": 0, "NORMAL_VOL": 1, "HIGH_VOL": 2}
            self._regime_eta_mults = {
                name_to_code.get(k, int(k)): float(v) for k, v in raw_mults.items()
            }

        # Episode config
        self.episode_length = int(config.get("episode_length", 1000))
        self.random_start = bool(config.get("random_start", True))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.30))
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct

        # v6: Hard risk constraints (AlphaSeek-informed). 0 = disabled (backward compat).
        self.stop_loss_bps = float(config.get("stop_loss_bps", 0))
        self.max_holding_bars = int(config.get("max_holding_bars", 0))

        # Scales from handler (must be set before obs space construction)
        self._scales = config.get("scales", [3, 15, 60])

        # v6: Observation mode
        self._obs_mode = config.get("obs_mode", "window")
        self._summary_feature_indices = config.get("summary_feature_indices", [0, 1, 2, 6, 7])

        # PRISM L1: Regime features injected into private state
        self._prism_enabled = bool(config.get("prism_enabled", False))
        self._prism_dim = 13  # GAHMM probs (6) + composite (1) + Chronos (6)
        self._private_dim = 5 + (self._prism_dim if self._prism_enabled else 0)

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # Build obs space dynamically from scales config
        obs_spaces = {}
        if self._obs_mode == "summary_stats":
            # v6: Each scale returns (n_feat * 3,) summary stats
            n_summary = len(self._summary_feature_indices) * 3
            for i in range(len(self._scales)):
                obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(n_summary,), dtype=np.float32,
                )
        else:
            for i in range(len(self._scales)):
                obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.window_size, features_per_scale), dtype=np.float32,
                )
        obs_spaces["private"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._private_dim,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # State variables
        self.current_position = 0.0
        self.prev_close = 0.0
        self.current_close = 0.0
        self.current_atr = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        # N2 dual-equity shadow state (only advanced when record_dual_equity).
        self.prev_mid = 0.0
        self.current_mid = 0.0
        self.equity_mid = self.initial_balance
        self.peak_equity_mid = self.initial_balance
        self.cumulative_fees = 0.0
        self.trade_count = 0
        self.current_step = 0
        self._episode_end = 0

        # v6: Position tracking for hard risk constraints
        self._entry_equity = self.initial_balance
        self._position_direction = 0   # +1 long, -1 short, 0 flat
        self._bars_in_position = 0

        # DSR state
        self._dsr = DSRCalculator(
            eta=self.dsr_eta,
            scale=self.dsr_scale,
            regime_eta_multipliers=self._regime_eta_mults,
            regime_blend=float(reward_cfg.get("regime_blend", 0.5)),
        )

        # ATR tracking for private state
        self._atr_buffer = []
        self._atr_rolling_mean = 0.0

        # Current regime code from PRISM (-1 = unknown)
        self._current_regime_code = -1

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

        # N2 dual-equity shadow reset
        self.prev_mid = 0.0
        self.current_mid = 0.0
        self.equity_mid = self.initial_balance
        self.peak_equity_mid = self.initial_balance

        # v6: Position tracking reset
        self._entry_equity = self.initial_balance
        self._position_direction = 0
        self._bars_in_position = 0

        # DSR state reset
        self._dsr.reset()

        # Regime code reset
        self._current_regime_code = -1

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
                if self.record_dual_equity and "high" in first and "low" in first:
                    self.current_mid = 0.5 * (first["high"] + first["low"])
                    self.prev_mid = self.current_mid
                self._current_obs = self._extract_obs(first)
            else:
                self._current_obs = self._empty_obs()
        else:
            self._current_obs = self._empty_obs()

        return self._get_observation(), {}

    def step(self, action):
        raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
        target_position = np.clip(raw_action, -1.0, 1.0) * self.max_leverage

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
        if self.record_dual_equity:
            self.prev_mid = self.current_mid

        # 2. Update market state
        if step_data is not None:
            self.current_close = step_data["close"]
            if self.record_dual_equity and "high" in step_data and "low" in step_data:
                self.current_mid = 0.5 * (step_data["high"] + step_data["low"])
            self.current_atr = step_data["atr"]
            self._current_obs = self._extract_obs(step_data)
            # RCRP + Path 2: Track current regime code from handler
            self._current_regime_code = step_data.get("regime_code", -1)

            # Update ATR rolling stats
            self._atr_buffer.append(self.current_atr)
            if len(self._atr_buffer) > 200:
                self._atr_buffer = self._atr_buffer[-200:]
            self._atr_rolling_mean = np.mean(self._atr_buffer)

        # 3. Compute position delta with deadband
        position_before = self.current_position  # v6: track for correct fee computation
        delta = target_position - self.current_position
        traded = False

        # B5 fix: scale deadband by max_leverage so trade frequency is invariant
        # to the leverage knob. Without this, deadband=0.25 in scaled-position
        # space means the agent crosses it 1/L as often (high-L → more trades →
        # superlinear fee drag → PF artifact unrelated to alpha).
        # ATR cap stays absolute (it's a safety constraint, not a knob).
        effective_deadband = self.deadband_threshold * self.max_leverage

        if abs(delta) < effective_deadband:
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
                        self.atr_cap_max_position,
                    )
                    delta = target_position - self.current_position
                    if abs(delta) < effective_deadband:
                        delta = 0.0

            if delta != 0.0:
                self.current_position += delta
                self.current_position = np.clip(
                    self.current_position, -self.max_leverage, self.max_leverage,
                )
                traded = True
                self.trade_count += 1

        # 3b. v6: Track position direction for hard risk constraints
        new_direction = (1 if self.current_position > 0.01
                         else (-1 if self.current_position < -0.01 else 0))
        if new_direction != self._position_direction:
            self._entry_equity = self.equity
            self._position_direction = new_direction
            self._bars_in_position = 0

        # 3c. v6: Position-level stop-loss — force-flat if position loses too much
        if self.stop_loss_bps > 0 and self._position_direction != 0:
            loss_bps = ((self._entry_equity - self.equity)
                        / max(self._entry_equity, 1e-9) * 10000.0)
            if loss_bps > self.stop_loss_bps:
                self.current_position = 0.0
                self._position_direction = 0
                self._bars_in_position = 0
                traded = True
                self.trade_count += 1

        # 3d. v6: Max holding timer — force-flat after N bars
        if self.max_holding_bars > 0 and self._position_direction != 0:
            self._bars_in_position += 1
            if self._bars_in_position >= self.max_holding_bars:
                self.current_position = 0.0
                self._position_direction = 0
                self._bars_in_position = 0
                traded = True
                self.trade_count += 1
        elif self._position_direction == 0:
            self._bars_in_position = 0

        # 4. Compute PnL
        # FIX R5-AUD-01: Compute price_return once, reuse for both reward and equity update
        pnl_bps = 0.0
        price_return = 0.0
        gap_fired = False  # N2: shared gap mask, applied to BOTH close and mid returns
        if self.prev_close > 0:
            price_return = (self.current_close - self.prev_close) / self.prev_close
            # Gap detection: zero out returns from session gaps / contract rolls
            # that exceed gap_atr_mult × ATR/price. These are non-tradeable artifacts.
            if (self.gap_detection and self._atr_rolling_mean > 1e-12
                    and abs(price_return) > self._gap_atr_mult * self._atr_rolling_mean / self.prev_close):
                price_return = 0.0
                gap_fired = True
            pnl_bps = self.current_position * price_return * 10000.0

        # Transaction cost (fee + slippage)
        # v6: Use total position change (covers forced close from stop-loss / max-holding)
        total_delta = abs(self.current_position - position_before)
        tc_bps = 0.0
        if traded and total_delta > 1e-9:
            tc_bps = self.taker_fee * 10000.0 * total_delta
            if self.slippage_base_bps > 0:
                tc_bps += self.slippage_base_bps * total_delta
            self.cumulative_fees += tc_bps

        # Step return
        R_t = pnl_bps - tc_bps

        # 5. Compute reward
        if self.reward_mode == "dsr":
            # Path 2: Set regime context before DSR compute (vol_regime = code % 3)
            if self._regime_adaptive_dsr and self._current_regime_code >= 0:
                self._dsr.set_regime_context(self._current_regime_code % 3)
            reward = self._dsr.compute(R_t)
        else:
            # Raw PnL mode
            reward = float(np.clip(R_t, -50.0, 50.0))

        # 6. Update equity
        # FIX R2-AUD-05: Use current equity (not initial_balance) so PnL compounds correctly.
        # Without this, drawdown recovery is inflated and long backtests diverge from reality.
        # N2: fee_frac hoisted out of the `if` (same value, used by both equity curves);
        # the not-traded close result is unchanged because the subtraction stays guarded.
        fee_frac = self.taker_fee + self.slippage_base_bps / 10000.0
        equity_delta = self.current_position * price_return * self.equity
        if traded and total_delta > 1e-9:
            equity_delta -= fee_frac * total_delta * self.equity
        self.equity += equity_delta
        self.peak_equity = max(self.peak_equity, self.equity)

        # 6b. N2 PF-XCHECK shadow: mark a second equity curve at the bar (H+L)/2
        # midpoint, in lockstep with the close-marked curve. Reuses current_position,
        # total_delta, fee_frac, and the gap mask BY REFERENCE (MATH-N2-a) so fees are
        # common-mode and the ONLY difference vs the close curve is the marking price.
        # SHADOW-ONLY (SENS-4): never read by reward / termination / ATR / obs, so the
        # close-marked trajectory is byte-identical with the flag on or off (SENS-5).
        if self.record_dual_equity:
            mid_return = 0.0
            if self.prev_mid > 0 and not gap_fired:
                mid_return = (self.current_mid - self.prev_mid) / self.prev_mid
            equity_mid_delta = self.current_position * mid_return * self.equity_mid
            if traded and total_delta > 1e-9:
                equity_mid_delta -= fee_frac * total_delta * self.equity_mid
            self.equity_mid += equity_mid_delta
            self.peak_equity_mid = max(self.peak_equity_mid, self.equity_mid)

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

    def _extract_obs(self, step_data: dict) -> dict[str, np.ndarray]:
        """Extract scale arrays from handler step data (positional keys)."""
        obs = {}
        for i in range(len(self._scales)):
            key = f"scale_{i}"
            if key in step_data:
                arr = step_data[key]
                # Avoid redundant copy if already float32 (handler typically returns f32)
                obs[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        # PRISM L1: Pass through regime features for private state
        if "prism" in step_data:
            obs["prism"] = step_data["prism"]
        return obs

    def _get_private_state(self) -> np.ndarray:
        """Build 5-dim private state vector.

        [current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]
        """
        # 1. Current position normalized to [-1, 1] regardless of max_leverage.
        # B1 fix: at max_leverage>1, raw position can range to ±max_leverage.
        # Reporting raw values causes train/eval distribution shift across
        # different leverages and breaks cross-L policy comparison. Divide by
        # max_leverage so the agent sees position-as-fraction-of-cap, the same
        # contract regardless of leverage knob.
        _lev = max(self.max_leverage, 1e-9)
        pos = self.current_position / _lev

        # 2. Unrealized PnL normalized — not applicable for continuous
        # (position changes continuously, use recent return as proxy).
        # Same B1 normalization: scale by 1/max_leverage so the proxy is
        # invariant to the leverage knob.
        pnl_proxy = 0.0
        if self.prev_close > 0 and self.current_close > 0:
            ret_bps = (self.current_close - self.prev_close) / self.prev_close * 10000.0
            pnl_proxy = float(np.clip((self.current_position / _lev) * ret_bps / 100.0, -1.0, 1.0))

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

        base = np.array([pos, pnl_proxy, time_sin, time_cos, atr_ratio], dtype=np.float32)

        # PRISM L1: Append regime features from handler step data
        if self._prism_enabled:
            if self._current_obs and "prism" in self._current_obs:
                prism = self._current_obs["prism"]
            else:
                prism = np.zeros(self._prism_dim, dtype=np.float32)
            return np.concatenate([base, prism])

        return base

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Build full observation dict."""
        obs = {}
        n_scales = len(self._scales)
        features_per_scale = self._features_per_scale

        if self._current_obs:
            for i in range(n_scales):
                key = f"scale_{i}"
                if key in self._current_obs:
                    obs[key] = self._current_obs[key]
                else:
                    obs[key] = self._zero_scale(features_per_scale)
        else:
            for i in range(n_scales):
                obs[f"scale_{i}"] = self._zero_scale(features_per_scale)

        obs["private"] = self._get_private_state()
        return obs

    def _zero_scale(self, features_per_scale: int) -> np.ndarray:
        """Return zero array matching current obs_mode shape."""
        if self._obs_mode == "summary_stats":
            n_summary = len(self._summary_feature_indices) * 3
            return np.zeros((n_summary,), dtype=np.float32)
        return np.zeros((self.window_size, features_per_scale), dtype=np.float32)

    def _empty_obs(self) -> dict[str, np.ndarray]:
        return {
            f"scale_{i}": self._zero_scale(self._features_per_scale)
            for i in range(len(self._scales))
        }

    def _make_info(self, reward: float, traded: bool) -> dict:
        drawdown_pct = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        return {
            "portfolio_value": self.equity,
            # N2 PF-XCHECK: (H+L)/2-marked shadow equity; None when not recording.
            "portfolio_value_mid": self.equity_mid if self.record_dual_equity else None,
            "position": self.current_position,
            "traded": traded,
            "trade_count": self.trade_count,
            "cumulative_fees": self.cumulative_fees,
            "drawdown_pct": drawdown_pct,
            "reward_total": reward,
            "reward_nav": reward,
            "taker_fee": self.taker_fee,
            "regime_code": self._current_regime_code,
        }

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Pos: {self.current_position:.3f}, "
            f"Equity: {self.equity:.2f}, Trades: {self.trade_count}, "
            f"Fee: {self.taker_fee:.5f}",
        )

    # --- PropFirmWrapper compatibility ---

    @property
    def timestamps(self) -> Optional[np.ndarray]:
        """Expose handler timestamps for PropFirmWrapper EOD boundary detection."""
        if self.handler and hasattr(self.handler, '_base_timestamps'):
            return self.handler._base_timestamps
        return None

    @property
    def step_idx(self) -> int:
        """Expose current data pointer for PropFirmWrapper."""
        if self.handler:
            return getattr(self.handler, '_ptr', 0)
        return 0

    def close(self):
        super().close()

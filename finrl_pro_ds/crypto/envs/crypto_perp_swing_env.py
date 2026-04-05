"""Crypto Perpetual Futures Swing Environment (Sync-2H).

Hybrid of ContinuousSwingEnv V7 (MDP structure) and CryptoPerpEnv (perp mechanics).
SAC-only, DSR reward, multi-scale obs, vol-regime scaling, deadband, fee curriculum,
hard risk constraints.

Action Space: Box(-1, 1, shape=(n_assets,)) -- per-asset target position weight.
Observation: Dict with scale keys (summary_stats or window mode) + private state.
Reward: Differential Sharpe Ratio (DSR) of portfolio return minus costs.

Key design choices ported from GMGP1/GMGP2:
  - DSR reward (self-normalizing, natural curriculum)
  - Per-asset deadband (prevents micro-churn)
  - Vol-regime exposure scaling (adaptive leverage)
  - Fee curriculum (train with zero fees first)
  - Hard risk constraints (per-asset stop-loss + max holding timer)

Perp mechanics preserved from CryptoPerpEnv:
  - Funding rates (UTC 00/08/16, applied BEFORE rebalance -- C3)
  - Gross exposure enforcement (proportional scaling + net short floor)
  - Volume-dependent slippage model
  - Fixed-notional PnL tracking (C1 invariant)
  - Asset availability mask (C2)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.dsr import DSRCalculator

if TYPE_CHECKING:
    from finrl_pro_ds.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler

logger = logging.getLogger(__name__)


class CryptoPerpSwingEnv(gym.Env):
    """Multi-asset crypto perpetual futures with GMGP1/GMGP2-proven MDP structure."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict[str, Any],
        data_handler: Optional["MultiScaleCryptoHandler"] = None,
    ):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # --- Core dimensions ---
        self.n_assets = int(config.get("n_assets", 10))
        self.initial_balance = float(config.get("initial_balance", 100000.0))
        self.window_size = int(config.get("window_size", 30))

        # --- Fee curriculum (runtime-updatable) ---
        self.taker_fee = float(config.get("taker_fee", 0.0))

        # --- Deadband: per-asset, skip trade if |delta| < threshold ---
        self.deadband_threshold = float(config.get("deadband_threshold", 0.03))

        # --- Slippage model (volume-dependent, from CryptoPerpEnv) ---
        self.slippage_base_bps = float(config.get("slippage_base_bps", 3.0))
        self.slippage_impact_bps = float(config.get("slippage_impact_bps", 15.0))

        # --- Exposure constraints ---
        self.max_gross_exposure = float(config.get("max_gross_exposure", 1.0))
        self.max_net_short_exposure = float(config.get("max_net_short_exposure", -0.50))

        # --- Vol-regime scaling ---
        vol_cfg = config.get("vol_scaling", {})
        self._vol_scaling_enabled = bool(vol_cfg.get("enabled", True))
        self._high_vol_threshold = float(vol_cfg.get("high_vol_threshold", 1.5))
        self._high_vol_scale = float(vol_cfg.get("high_vol_scale", 0.3))
        self._low_vol_threshold = float(vol_cfg.get("low_vol_threshold", 0.7))
        self._low_vol_scale = float(vol_cfg.get("low_vol_scale", 1.3))
        self._vol_warmup_bars = int(vol_cfg.get("warmup_bars", 100))

        # --- Hard risk constraints (from GMGP2) ---
        self.stop_loss_bps = float(config.get("stop_loss_bps", 0))
        self.max_holding_bars = int(config.get("max_holding_bars", 0))

        # --- Reward ---
        reward_cfg = config.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "dsr")
        self.dsr_eta = float(reward_cfg.get("dsr_eta", 0.001))
        self.dsr_scale = float(reward_cfg.get("dsr_scale", 1.0))

        # --- Episode config ---
        self.episode_length = int(config.get("episode_length", 720))
        self.random_start = bool(config.get("random_start", True))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.20))
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct

        # --- Circuit breaker (env-level) ---
        self.circuit_breaker_threshold = float(config.get("circuit_breaker_threshold", 0.1))

        # --- Observation mode ---
        self._scales = config.get("scales", [1, 4, 24])
        self._obs_mode = config.get("obs_mode", "summary_stats")
        features_per_scale = int(config.get("features_per_scale", 8))
        self._summary_feature_indices = config.get("summary_feature_indices", [0, 1, 2, 6, 7])
        n_summary_per_asset = len(self._summary_feature_indices) * 3

        # --- Private state dimension ---
        # [gross_exposure, net_exposure, margin_pct, portfolio_atr_ratio,
        #  drawdown_pct, time_sin, time_cos, pos_0..pos_N-1]
        self._private_dim = 7 + self.n_assets

        # --- Spaces ---
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32,
        )

        obs_spaces = {}
        if self._obs_mode == "summary_stats":
            for i in range(len(self._scales)):
                obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_assets * n_summary_per_asset,), dtype=np.float32,
                )
        else:
            for i in range(len(self._scales)):
                obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_assets, self.window_size, features_per_scale),
                    dtype=np.float32,
                )
        obs_spaces["private"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._private_dim,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # --- Pre-allocated state arrays ---
        self.positions = np.zeros(self.n_assets, dtype=np.float64)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        self.entry_notionals = np.zeros(self.n_assets, dtype=np.float64)

        # Per-asset risk tracking
        self._entry_equities = np.zeros(self.n_assets, dtype=np.float64)
        self._bars_in_position = np.zeros(self.n_assets, dtype=np.int32)

        # DSR calculator
        self._dsr = DSRCalculator(eta=self.dsr_eta, scale=self.dsr_scale)

        # Vol tracking
        self._atr_buffer: list[float] = []
        self._portfolio_atr_mean = 0.0

        # Time encoding cache
        self._cached_time_ptr = -1
        self._cached_time_sin = 0.0
        self._cached_time_cos = 1.0

        # Current step data
        self._current_obs: dict[str, np.ndarray] | None = None
        self._current_close = np.zeros(self.n_assets, dtype=np.float64)
        self._current_atr = np.zeros(self.n_assets, dtype=np.float64)
        self._current_funding = np.zeros(self.n_assets, dtype=np.float64)
        self._current_volume = np.zeros(self.n_assets, dtype=np.float64)
        self._prev_close = np.zeros(self.n_assets, dtype=np.float64)

        # Scalar state
        self.margin_balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.cumulative_fees = 0.0
        self.cumulative_funding = 0.0
        self.trade_count = 0
        self.current_step = 0
        self._episode_end = 0

        # Funding mask (built on first step when timestamps available)
        self._funding_mask: np.ndarray | None = None

    def set_fees(self, taker_fee: float):
        """Runtime fee update for curriculum learning."""
        self.taker_fee = taker_fee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset positions
        self.positions[:] = 0.0
        self.entry_prices[:] = 0.0
        self.entry_notionals[:] = 0.0
        self._entry_equities[:] = 0.0
        self._bars_in_position[:] = 0

        # Reset scalar state
        self.margin_balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.cumulative_fees = 0.0
        self.cumulative_funding = 0.0
        self.trade_count = 0
        self.current_step = 0

        # Reset DSR
        self._dsr.reset()

        # Reset vol tracking
        self._atr_buffer = []
        self._portfolio_atr_mean = 0.0
        self._cached_time_ptr = -1

        # Reset price state
        self._current_close[:] = 0.0
        self._current_atr[:] = 0.0
        self._prev_close[:] = 0.0

        if self.handler:
            self.handler.reset()

            # Build funding mask from handler timestamps
            if self._funding_mask is None and hasattr(self.handler, "_base_timestamps"):
                ts = self.handler._base_timestamps
                if len(ts) > 0:
                    epochs = ts.astype("datetime64[s]").astype("int64")
                    hours = (epochs % 86400) // 3600
                    self._funding_mask = np.isin(hours, [0, 8, 16])

            # Random start
            data_len = getattr(self.handler, "_len", 0)
            ws = getattr(self.handler, "window_size", self.window_size)
            if self.episode_length > 0 and data_len > 0 and self.random_start:
                max_start = max(ws, data_len - self.episode_length - ws)
                if max_start > ws:
                    start_idx = self.np_random.integers(ws, max_start)
                    self.handler._ptr = start_idx

            self._episode_end = self.episode_length if self.episode_length > 0 else 0

            # Read first step to initialize prices
            first = self.handler.step()
            if first is not None:
                self._current_close = first["close"].copy()
                self._prev_close = self._current_close.copy()
                self._current_atr = first["atr"].copy()
                self._current_funding = first["funding_rate"].copy()
                self._current_volume = first["volume"].copy()
                self._current_obs = self._extract_obs(first)
            else:
                self._current_obs = self._empty_obs()
        else:
            self._current_obs = self._empty_obs()

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).ravel()
        target_weights = np.clip(action, -1.0, 1.0)

        self.current_step += 1

        # 1. Advance data
        step_data = self.handler.step() if self.handler else None

        terminated = False
        truncated = False

        if self.handler and step_data is None:
            truncated = True
            return self._get_observation(), 0.0, terminated, truncated, self._make_info(0.0)

        # Save previous close
        self._prev_close = self._current_close.copy()

        # 2. Update market state
        if step_data is not None:
            self._current_close = step_data["close"].copy()
            self._current_atr = step_data["atr"].copy()
            self._current_funding = step_data["funding_rate"].copy()
            self._current_volume = step_data["volume"].copy()
            self._current_obs = self._extract_obs(step_data)

            # Update portfolio ATR for vol-regime scaling
            portfolio_atr = float(np.mean(self._current_atr))
            self._atr_buffer.append(portfolio_atr)
            if len(self._atr_buffer) > 200:
                self._atr_buffer = self._atr_buffer[-200:]
            self._portfolio_atr_mean = float(np.mean(self._atr_buffer))

        # 3. Asset availability mask (C2)
        asset_available = self._current_close > 1e-10
        target_weights[~asset_available] = 0.0

        # 4. Enforce gross exposure with vol-regime scaling
        effective_gross = self._get_effective_gross_exposure()
        target_weights = self._enforce_gross_exposure(target_weights, effective_gross)

        # 5. Per-asset deadband
        delta = target_weights - self.positions
        deadband_mask = np.abs(delta) < self.deadband_threshold
        target_weights[deadband_mask] = self.positions[deadband_mask]

        # 6. Calculate portfolio value BEFORE price move
        unrealized_before = self._calc_unrealized_pnl(self._prev_close)
        portfolio_value_before = max(
            self.margin_balance + float(unrealized_before.sum()),
            self.initial_balance * 0.001,
        )

        # 7. Apply funding (C3: BEFORE rebalance, on OLD positions)
        funding_cost = 0.0
        ptr = getattr(self.handler, "_ptr", 0) - 1
        if (self._funding_mask is not None
                and 0 <= ptr < len(self._funding_mask)
                and self._funding_mask[ptr]):
            funding_cost = self._apply_funding(self._current_close)
            self.cumulative_funding += funding_cost
            self.margin_balance -= funding_cost

        # 8. Execute rebalance
        old_positions = self.positions.copy()
        delta_weights = target_weights - old_positions
        abs_delta = np.abs(delta_weights)

        # 9. Transaction costs
        total_fees, total_slippage = self._calc_transaction_costs(
            abs_delta, self._current_close, portfolio_value_before,
        )

        # 10. Realize PnL on closed/reduced positions
        realized_this_step = self._realize_pnl(
            old_positions, delta_weights, self._current_close,
        )

        # 11. Update positions
        self.positions = old_positions + delta_weights
        traded_mask = abs_delta >= 1e-8
        self.trade_count += int(traded_mask.sum())

        # 12. Update entry prices/notionals for new/increased positions
        self._update_entry_prices(
            old_positions, delta_weights, self._current_close, portfolio_value_before,
        )

        # 13. Deduct costs, credit realized PnL
        self.cumulative_fees += total_fees + total_slippage
        self.margin_balance -= (total_fees + total_slippage)
        self.margin_balance += realized_this_step

        # 14. Liquidation check
        if self.margin_balance < 0:
            forced_pnl = float(self._calc_unrealized_pnl(self._current_close).sum())
            self.margin_balance += forced_pnl
            self.margin_balance = max(self.margin_balance, 0.0)
            self.positions[:] = 0.0
            self.entry_prices[:] = 0.0
            self.entry_notionals[:] = 0.0
            self._bars_in_position[:] = 0

        # 15. Hard risk constraints (per-asset)
        self._apply_hard_constraints(self._current_close, portfolio_value_before)

        # 16. Compute portfolio value
        new_unrealized = self._calc_unrealized_pnl(self._current_close)
        self.equity = self.margin_balance + float(new_unrealized.sum())
        self.peak_equity = max(self.peak_equity, self.equity)

        # 17. Compute return and reward
        if portfolio_value_before > 1e-6:
            step_return = (self.equity - portfolio_value_before) / portfolio_value_before
        else:
            step_return = 0.0

        # DSR on portfolio return in bps
        R_t = step_return * 10000.0
        if self.reward_mode == "dsr":
            reward = self._dsr.compute(R_t)
        else:
            reward = float(np.clip(R_t, -50.0, 50.0))

        # 18. Termination checks
        # Peak-based drawdown (R4-AUD-07)
        if self.equity < self._stop_loss_threshold * self.peak_equity:
            terminated = True

        # Circuit breaker (absolute floor)
        if self.equity < self.circuit_breaker_threshold * self.initial_balance:
            terminated = True

        if self._episode_end > 0 and self.current_step >= self._episode_end:
            truncated = True

        obs = self._get_observation()
        info = self._make_info(reward)

        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Exposure / constraints
    # -----------------------------------------------------------------------

    def _get_effective_gross_exposure(self) -> float:
        """Compute effective gross exposure limit based on vol regime."""
        if not self._vol_scaling_enabled:
            return self.max_gross_exposure
        if len(self._atr_buffer) < self._vol_warmup_bars:
            return self.max_gross_exposure

        current_atr = float(np.mean(self._current_atr))
        if self._portfolio_atr_mean < 1e-12:
            return self.max_gross_exposure

        vol_ratio = current_atr / self._portfolio_atr_mean

        if vol_ratio > self._high_vol_threshold:
            multiplier = self._high_vol_scale
        elif vol_ratio < self._low_vol_threshold:
            multiplier = self._low_vol_scale
        else:
            multiplier = 1.0

        return self.max_gross_exposure * multiplier

    def _enforce_gross_exposure(
        self, raw_action: np.ndarray, max_gross: float,
    ) -> np.ndarray:
        """Enforce sum(|weights|) <= max_gross and net short floor."""
        result = raw_action.copy()

        gross = np.abs(result).sum()
        if gross > max_gross:
            result *= max_gross / gross

        # Net short floor (H3 fix from CryptoPerpEnv)
        net = result.sum()
        if net < self.max_net_short_exposure:
            short_mask = result < 0
            if short_mask.any():
                short_sum = result[short_mask].sum()
                long_sum = result[~short_mask].sum()
                target_short = self.max_net_short_exposure - long_sum
                if short_sum < -1e-8:
                    scale = min(target_short / short_sum, 1.0)
                    result[short_mask] *= scale

        return result

    def _apply_hard_constraints(
        self, current_price: np.ndarray, portfolio_value: float,
    ):
        """Per-asset stop-loss and max holding timer (from GMGP2)."""
        for i in range(self.n_assets):
            if np.abs(self.positions[i]) < 1e-8:
                self._bars_in_position[i] = 0
                continue

            self._bars_in_position[i] += 1

            # Per-asset stop-loss
            if self.stop_loss_bps > 0 and self.entry_notionals[i] > 1e-8:
                price_ratio = current_price[i] / max(self.entry_prices[i], 1e-10) - 1.0
                asset_pnl = np.sign(self.positions[i]) * self.entry_notionals[i] * price_ratio
                loss_bps = -asset_pnl / max(self.entry_notionals[i], 1e-9) * 10000.0
                if loss_bps > self.stop_loss_bps:
                    # Force flat — realize PnL
                    realized = float(asset_pnl)
                    self.margin_balance += realized
                    self.positions[i] = 0.0
                    self.entry_prices[i] = 0.0
                    self.entry_notionals[i] = 0.0
                    self._bars_in_position[i] = 0
                    self.trade_count += 1
                    continue

            # Max holding timer
            if self.max_holding_bars > 0 and self._bars_in_position[i] >= self.max_holding_bars:
                # Force flat — realize PnL
                price_ratio = current_price[i] / max(self.entry_prices[i], 1e-10) - 1.0
                realized = float(
                    np.sign(self.positions[i]) * self.entry_notionals[i] * price_ratio,
                )
                self.margin_balance += realized
                self.positions[i] = 0.0
                self.entry_prices[i] = 0.0
                self.entry_notionals[i] = 0.0
                self._bars_in_position[i] = 0
                self.trade_count += 1

    # -----------------------------------------------------------------------
    # PnL calculations (vectorized, from CryptoPerpEnv)
    # -----------------------------------------------------------------------

    def _calc_unrealized_pnl(self, current_price: np.ndarray) -> np.ndarray:
        """Per-asset unrealized PnL using fixed entry notionals (C1)."""
        pnl = np.zeros(self.n_assets, dtype=np.float64)
        active = (np.abs(self.positions) > 1e-8) & (self.entry_notionals > 1e-8)
        if active.any():
            price_ratio = current_price[active] / (self.entry_prices[active] + 1e-10) - 1.0
            pnl[active] = np.sign(self.positions[active]) * self.entry_notionals[active] * price_ratio
        return pnl

    def _realize_pnl(
        self,
        old_positions: np.ndarray,
        delta_weights: np.ndarray,
        current_price: np.ndarray,
    ) -> float:
        """Realize PnL for reduced/closed positions (vectorized, H2 fix)."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        has_position = abs_old >= 1e-8
        has_entry = np.abs(self.entry_prices) >= 1e-10
        has_notional = self.entry_notionals >= 1e-8
        active = has_position & has_entry & has_notional

        if not active.any():
            return 0.0

        closed_fraction = np.zeros(self.n_assets, dtype=np.float64)

        # Position flip — fully close old
        flipped = active & (np.sign(old_positions) != np.sign(new_positions)) & (abs_new > 1e-8)
        closed_fraction[flipped] = 1.0

        # Position reduction (same sign, smaller magnitude)
        reduced = active & ~flipped & (abs_new < abs_old)
        if reduced.any():
            closed_fraction[reduced] = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]

        price_change = current_price / (self.entry_prices + 1e-10) - 1.0
        pnl = np.sign(old_positions) * closed_fraction * self.entry_notionals * price_change
        return float(pnl.sum())

    def _update_entry_prices(
        self,
        old_positions: np.ndarray,
        delta_weights: np.ndarray,
        current_price: np.ndarray,
        portfolio_value: float,
    ):
        """Update entry prices/notionals (vectorized, H2 fix, C7 fix)."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        closed = abs_new < 1e-8
        self.entry_prices[closed] = 0.0
        self.entry_notionals[closed] = 0.0

        from_flat = ~closed & (abs_old < 1e-8)
        self.entry_prices[from_flat] = current_price[from_flat]
        self.entry_notionals[from_flat] = abs_new[from_flat] * portfolio_value
        self._entry_equities[from_flat] = self.equity

        flipped = ~closed & ~from_flat & (np.sign(old_positions) != np.sign(new_positions))
        self.entry_prices[flipped] = current_price[flipped]
        self.entry_notionals[flipped] = abs_new[flipped] * portfolio_value
        self._entry_equities[flipped] = self.equity

        increased = ~closed & ~from_flat & ~flipped & (abs_new > abs_old)
        if increased.any():
            added_notional = np.abs(delta_weights[increased]) * portfolio_value
            old_notional = self.entry_notionals[increased]
            new_notional = old_notional + added_notional
            self.entry_prices[increased] = (
                self.entry_prices[increased] * old_notional
                + current_price[increased] * added_notional
            ) / (new_notional + 1e-10)
            self.entry_notionals[increased] = new_notional

        reduced = ~closed & ~from_flat & ~flipped & ~increased & (abs_new < abs_old)
        if reduced.any():
            closed_frac = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]
            self.entry_notionals[reduced] *= (1.0 - closed_frac)

    # -----------------------------------------------------------------------
    # Transaction costs
    # -----------------------------------------------------------------------

    def _calc_transaction_costs(
        self,
        abs_delta: np.ndarray,
        price: np.ndarray,
        portfolio_value: float,
    ) -> tuple[float, float]:
        """Fees + volume-dependent slippage (from CryptoPerpEnv)."""
        active = abs_delta >= 1e-8
        if not active.any():
            return 0.0, 0.0

        notionals = abs_delta[active] * portfolio_value
        total_fees = float(np.sum(notionals) * self.taker_fee)

        # Volume-dependent slippage (prev-bar volume, no look-ahead)
        hourly_vols = self._current_volume[active]
        volume_ratios = np.where(
            hourly_vols > 1e-6,
            notionals / hourly_vols,
            1.0,
        )
        slippage_bps = self.slippage_base_bps + self.slippage_impact_bps * volume_ratios
        total_slippage = float(np.sum(notionals * slippage_bps * 1e-4))

        return total_fees, total_slippage

    # -----------------------------------------------------------------------
    # Funding rate
    # -----------------------------------------------------------------------

    def _apply_funding(self, price: np.ndarray) -> float:
        """Apply funding to open positions (vectorized, from CryptoPerpEnv)."""
        active = (np.abs(self.positions) >= 1e-8) & (np.abs(self.entry_prices) >= 1e-10)
        if not active.any():
            return 0.0
        current_notional = self.entry_notionals[active] * (
            price[active] / (self.entry_prices[active] + 1e-10)
        )
        funding_cost = np.sign(self.positions[active]) * current_notional * self._current_funding[active]
        return float(funding_cost.sum())

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _extract_obs(self, step_data: dict) -> dict[str, np.ndarray]:
        """Extract scale arrays from handler step data."""
        obs = {}
        for i in range(len(self._scales)):
            key = f"scale_{i}"
            if key in step_data:
                arr = step_data[key]
                obs[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        return obs

    def _get_private_state(self) -> np.ndarray:
        """Build private state vector.

        [gross_exposure, net_exposure, margin_pct, portfolio_atr_ratio,
         drawdown_pct, time_sin, time_cos, pos_0..pos_N-1]
        """
        abs_pos = np.abs(self.positions)
        gross = float(abs_pos.sum())
        net = float(self.positions.sum())
        margin_pct = float(np.clip(self.margin_balance / max(self.initial_balance, 1e-9), 0.0, 2.0))

        # Portfolio ATR ratio
        if self._portfolio_atr_mean > 1e-12:
            current_atr = float(np.mean(self._current_atr))
            atr_ratio = float(np.clip(current_atr / self._portfolio_atr_mean, 0.0, 3.0)) / 3.0
        else:
            atr_ratio = 0.5

        # Drawdown
        drawdown_pct = float(1.0 - self.equity / max(self.peak_equity, 1e-9))
        drawdown_pct = float(np.clip(drawdown_pct, 0.0, 1.0))

        # Time encoding
        if self.handler and hasattr(self.handler, "_ptr"):
            ptr = self.handler._ptr
            if ptr != self._cached_time_ptr:
                ts = getattr(self.handler, "_base_timestamps", None)
                if ts is not None and ptr > 0 and ptr <= len(ts):
                    ts_val = ts[ptr - 1]
                    dt = np.datetime64(ts_val, "ns")
                    minutes = (dt - dt.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
                    self._cached_time_sin = float(np.sin(2 * np.pi * minutes / 1440.0))
                    self._cached_time_cos = float(np.cos(2 * np.pi * minutes / 1440.0))
                self._cached_time_ptr = ptr

        base = np.array(
            [gross, net, margin_pct, atr_ratio, drawdown_pct,
             self._cached_time_sin, self._cached_time_cos],
            dtype=np.float32,
        )

        # Append per-asset positions (already in [-1, 1])
        return np.concatenate([base, self.positions.astype(np.float32)])

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Build full observation dict."""
        obs = {}
        n_scales = len(self._scales)

        if self._current_obs:
            for i in range(n_scales):
                key = f"scale_{i}"
                if key in self._current_obs:
                    obs[key] = self._current_obs[key]
                else:
                    obs[key] = self._zero_scale()
        else:
            for i in range(n_scales):
                obs[f"scale_{i}"] = self._zero_scale()

        obs["private"] = self._get_private_state()
        return obs

    def _zero_scale(self) -> np.ndarray:
        """Zero array matching current obs_mode shape."""
        if self._obs_mode == "summary_stats":
            n_summary = len(self._summary_feature_indices) * 3
            return np.zeros((self.n_assets * n_summary,), dtype=np.float32)
        features_per_scale = int(self.config.get("features_per_scale", 8))
        return np.zeros((self.n_assets, self.window_size, features_per_scale), dtype=np.float32)

    def _empty_obs(self) -> dict[str, np.ndarray]:
        return {
            f"scale_{i}": self._zero_scale()
            for i in range(len(self._scales))
        }

    def _make_info(self, reward: float) -> dict:
        abs_pos = np.abs(self.positions)
        drawdown_pct = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        return {
            "portfolio_value": self.equity,
            "margin_balance": self.margin_balance,
            "gross_exposure": float(abs_pos.sum()),
            "net_exposure": float(self.positions.sum()),
            "drawdown_pct": float(drawdown_pct),
            "trade_count": self.trade_count,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_funding": self.cumulative_funding,
            "reward_total": reward,
            "taker_fee": self.taker_fee,
            "n_long": int((self.positions > 0.01).sum()),
            "n_short": int((self.positions < -0.01).sum()),
        }

    def render(self, mode="human"):
        abs_pos = np.abs(self.positions)
        print(
            f"Step {self.current_step}: equity={self.equity:.2f}, "
            f"gross={abs_pos.sum():.3f}, trades={self.trade_count}",
        )

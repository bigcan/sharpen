"""Cryptocurrency Perpetual Futures Trading Environment (sync-1H).

Gymnasium-compatible environment for trading USDT-margined perpetual futures
with long/short positions, funding rate mechanics, and no-leverage constraint.

Key differences from ProStockEnv:
- Signed positions: positive = long, negative = short
- No leverage: sum(|weights|) ≤ 1.0 enforced via proportional scaling
- Funding rates applied at UTC-aligned timestamps (00:00, 08:00, 16:00)
- Volume-dependent slippage model
- Margin balance tracking (initial + realized PnL + unrealized PnL - fees - funding)
- Sortino-focused reward with turnover penalty
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.obs_guard import sanitize_obs

logger = logging.getLogger(__name__)

# NAN-01: ceiling on the OBSERVED participation ratio (order notional / bar volume)
# used by the cost-to-rebalance feature. Bounds the obs only — the cost actually
# CHARGED in the step path is deliberately left unbounded so economics are unchanged.
MAX_OBS_PARTICIPATION = 1.0e4


class CryptoPerpEnv(gym.Env):
    """Perpetual futures environment for the Synapse Crypto 1H strategy.

    Observation space (962 dims for 20 assets × 44 features):
        [margin_balance_pct]                           # 1
        + [per_asset_features × n_assets]              # n_assets × tech_dim
        + [current_position_per_asset]                 # n_assets (signed)
        + [unrealized_pnl_per_asset]                   # n_assets
        + [funding_rate_per_asset]                     # n_assets
        + [cost_to_rebalance_per_asset]                # n_assets
        + [portfolio_concentration (ENB)]              # 1

    Action space:
        Continuous [-1, 1] per asset. Negative = short, positive = long.
        Gross exposure ≤ 1.0 enforced via proportional scaling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        price_ary: np.ndarray,          # (T, n_assets) — close prices
        tech_ary: np.ndarray,            # (T, n_assets * tech_dim)
        funding_rate_ary: np.ndarray,    # (T, n_assets) — funding rates per bar
        volume_ary: np.ndarray,          # (T, n_assets) — hourly volume (quote)
        timestamps: np.ndarray,          # (T,) — UTC timestamps (int64 epoch seconds)
        initial_capital: float = 100_000.0,
        maker_fee_pct: float = 0.0002,   # 2 bps
        taker_fee_pct: float = 0.0005,   # 5 bps
        slippage_base_bps: float = 3.0,
        slippage_impact_bps: float = 15.0,
        max_gross_exposure: float = 1.0,
        max_net_short_exposure: float = -0.50,
        turnover_penalty: float = 0.002,
        reward_type: str = "sortino_softmax",
        sortino_window: int = 168,       # 1 week for downside deviation
        reward_scaling: float = 1.0,
        reward_clip_range: tuple = (-5.0, 5.0),
        min_trade_pct: float = 0.005,    # Skip dust trades < 0.5%
        circuit_breaker_threshold: float = 0.1,
        enable_trade_log: bool = False,
        action_ema_alpha: float = 0.0,
        random_start: bool = False,
        random_start_pct: float = 0.1,
        long_only: bool = False,
    ) -> None:
        super().__init__()

        # --- Validate inputs ---
        assert price_ary.ndim == 2, "price_ary must be (T, n_assets)"
        assert tech_ary.ndim == 2, "tech_ary must be (T, n_assets * tech_dim)"
        assert funding_rate_ary.ndim == 2, "funding_rate_ary must be (T, n_assets)"
        assert volume_ary.ndim == 2, "volume_ary must be (T, n_assets)"
        T, n_assets = price_ary.shape
        assert tech_ary.shape[0] == T
        assert funding_rate_ary.shape == (T, n_assets)
        assert volume_ary.shape == (T, n_assets)
        assert timestamps.shape == (T,)
        assert tech_ary.shape[1] % n_assets == 0

        # --- Store data arrays ---
        self.price_ary = price_ary.astype(np.float64)
        self.tech_ary = tech_ary.astype(np.float32)
        self.funding_rate_ary = funding_rate_ary.astype(np.float64)
        self.volume_ary = volume_ary.astype(np.float64)
        self.timestamps = timestamps.astype(np.int64)

        # --- Dimensions ---
        self.n_assets = n_assets
        self.tech_dim = tech_ary.shape[1] // n_assets
        self.max_step = T - 1

        # --- Environment parameters ---
        self.initial_capital = float(initial_capital)
        self.maker_fee_pct = float(maker_fee_pct)
        self.taker_fee_pct = float(taker_fee_pct)
        self.slippage_base_bps = float(slippage_base_bps)
        self.slippage_impact_bps = float(slippage_impact_bps)
        self.max_gross_exposure = float(max_gross_exposure)
        self.max_net_short_exposure = float(max_net_short_exposure)
        self.turnover_penalty = float(turnover_penalty)
        self.reward_type = reward_type
        self.sortino_window = sortino_window
        self.reward_scaling = float(reward_scaling)
        self.reward_clip_range = reward_clip_range
        self.min_trade_pct = float(min_trade_pct)
        self.circuit_breaker_threshold = float(circuit_breaker_threshold)
        self.enable_trade_log = enable_trade_log
        self.action_ema_alpha = float(action_ema_alpha)
        self.random_start = random_start
        self.random_start_pct = float(random_start_pct)
        self.long_only = long_only
        if self.long_only:
            self.max_net_short_exposure = 0.0

        # --- Pre-compute UTC funding hours for each bar ---
        # Funding applies at 00:00, 08:00, 16:00 UTC
        self._funding_mask = self._build_funding_mask()

        # --- Observation & action spaces ---
        # obs = margin_pct(1) + tech(n*td) + pos(n) + upnl(n) + fr(n) + cost(n) + enb(1)
        self.obs_dim = 1 + (self.n_assets * self.tech_dim) + 4 * self.n_assets + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        action_low = 0.0 if self.long_only else -1.0
        self.action_space = gym.spaces.Box(
            low=action_low, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # --- Pre-compute asset availability mask (C2: zero-price protection) ---
        self._asset_available = self.price_ary > 1e-10  # (T, n_assets)

        # --- Pre-allocated working arrays (OPT: avoid per-step allocations) ---
        self._obs_buffer = np.empty(self.obs_dim, dtype=np.float32)
        self._pnl_buffer = np.zeros(n_assets, dtype=np.float64)
        self._closed_frac_buffer = np.zeros(n_assets, dtype=np.float64)
        self._cost_buffer = np.zeros(n_assets, dtype=np.float32)

        # --- Cached step results (OPT: avoid recomputation in _get_obs) ---
        self._cached_unrealized: np.ndarray | None = None
        self._cached_portfolio_value: float = 0.0
        self._cached_abs_positions: np.ndarray | None = None

        # --- Runtime state (initialized in reset) ---
        self.step_idx = 0
        self.margin_balance = 0.0
        self.positions = np.zeros(n_assets, dtype=np.float64)  # signed weights
        self.entry_prices = np.zeros(n_assets, dtype=np.float64)
        self.entry_notionals = np.zeros(n_assets, dtype=np.float64)  # C1: fixed notional at entry
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.cumulative_funding = 0.0
        self.portfolio_values = []
        self.returns_history = []
        self.trade_log = [] if enable_trade_log else None

    def _build_funding_mask(self) -> np.ndarray:
        """Pre-compute which bars align with funding times (00, 08, 16 UTC)."""
        # H4 fix: Vectorized instead of Python loop over 35K+ bars
        hours = (self.timestamps % 86400) // 3600
        return np.isin(hours, [0, 8, 16])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.random_start and self.max_step > 0:
            max_offset = int(self.max_step * self.random_start_pct)
            self.step_idx = int(self.np_random.integers(0, max_offset)) if max_offset > 0 else 0
        else:
            self.step_idx = 0
        self.margin_balance = self.initial_capital
        self.positions[:] = 0.0
        self.entry_prices[:] = 0.0
        self.entry_notionals[:] = 0.0
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.cumulative_funding = 0.0
        self.portfolio_values = [self.initial_capital]
        self.returns_history = []
        if self.enable_trade_log:
            self.trade_log = []

        # Clear cached step results
        self._cached_unrealized = None
        self._cached_portfolio_value = 0.0
        self._cached_abs_positions = None

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if hasattr(action, 'cpu'):
            action = action.cpu()
        action = np.asarray(action, dtype=np.float64).ravel().clip(-1.0, 1.0)
        if self.long_only:
            action = action.clip(0.0, 1.0)

        # F3: Action EMA smoothing — blend raw action toward current position
        # to reduce churn. alpha=0 disables; alpha=1 uses raw action fully.
        if self.action_ema_alpha > 0.0:
            alpha = self.action_ema_alpha
            action = alpha * action + (1.0 - alpha) * self.positions

        # --- Enforce no-leverage constraint via proportional scaling ---
        target_weights = self._enforce_gross_exposure(action)

        # --- Advance to next bar ---
        self.step_idx += 1
        price = self.price_ary[self.step_idx]

        # --- C2: Zero out actions for assets with invalid/zero prices ---
        target_weights[~self._asset_available[self.step_idx]] = 0.0
        prev_price = self.price_ary[self.step_idx - 1]

        # --- Calculate portfolio value BEFORE this bar's price move ---
        # Use prev_price to capture true pre-bar state so price-driven
        # returns on held positions are visible in the reward signal.
        unrealized_at_prev = self._calc_unrealized_pnl(prev_price)
        portfolio_value_before = max(
            self.margin_balance + unrealized_at_prev.sum(),
            self.initial_capital * 0.001,  # Floor to prevent negative notionals
        )

        # --- C3 fix: Apply funding rate BEFORE rebalance on OLD positions ---
        # On real exchanges, funding settles on positions held during the
        # funding period, then the agent rebalances afterward.
        funding_cost = 0.0
        if self._funding_mask[self.step_idx]:
            funding_cost = self._apply_funding(price)
            self.cumulative_funding += funding_cost
            self.margin_balance -= funding_cost

        # --- Execute rebalance: compute deltas and apply costs ---
        old_positions = self.positions.copy()
        delta_weights = target_weights - old_positions

        # OPT: Compute abs(delta) once, reuse in dust filter + transaction costs
        abs_delta = np.abs(delta_weights)

        # Filter dust trades
        dust_mask = abs_delta < self.min_trade_pct
        delta_weights[dust_mask] = 0.0
        abs_delta[dust_mask] = 0.0  # Keep abs_delta in sync

        # Calculate transaction costs (fees + slippage)
        total_fees, total_slippage = self._calc_transaction_costs_fast(
            delta_weights, abs_delta, price, portfolio_value_before,
        )

        # Realize PnL on closed/reduced positions
        realized_this_step = self._realize_pnl(
            old_positions, delta_weights, price,
        )

        # Update positions
        self.positions = old_positions + delta_weights

        # Update entry prices and notionals for new/increased positions
        self._update_entry_prices(old_positions, delta_weights, price, portfolio_value_before)

        # Deduct costs from margin
        self.cumulative_fees += total_fees + total_slippage
        self.realized_pnl += realized_this_step
        self.margin_balance -= (total_fees + total_slippage)
        self.margin_balance += realized_this_step

        # Clamp margin balance: if margin hits zero the portfolio is liquidated
        if self.margin_balance < 0:
            # R6 fix: Force-close all positions and realize remaining PnL
            # before zeroing margin. Without this, unrealized PnL on liquidated
            # positions was silently discarded, corrupting the portfolio value.
            forced_pnl = float(self._calc_unrealized_pnl(price).sum())
            self.realized_pnl += forced_pnl
            self.margin_balance += forced_pnl
            # If still negative, the loss exceeds the portfolio — clamp to 0
            self.margin_balance = max(self.margin_balance, 0.0)
            self.positions = np.zeros(self.n_assets, dtype=np.float64)
            self.entry_prices = np.zeros(self.n_assets, dtype=np.float64)
            self.entry_notionals = np.zeros(self.n_assets, dtype=np.float64)

        # --- Calculate new portfolio value ---
        new_unrealized = self._calc_unrealized_pnl(price)
        unrealized_sum = float(new_unrealized.sum())
        portfolio_value = self.margin_balance + unrealized_sum
        self.portfolio_values.append(portfolio_value)
        # OPT: Trim to prevent unbounded growth
        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-500:]

        # --- Calculate reward ---
        if portfolio_value_before > 1e-6:
            step_return = (portfolio_value - portfolio_value_before) / portfolio_value_before
        else:
            step_return = 0.0
        self.returns_history.append(step_return)
        # OPT: Trim to prevent unbounded growth — only need sortino_window for reward
        if len(self.returns_history) > self.sortino_window * 2:
            self.returns_history = self.returns_history[-self.sortino_window:]

        reward = self._calc_reward(step_return, abs_delta, portfolio_value_before)

        # --- Trade logging ---
        if self.enable_trade_log and self.trade_log is not None:
            abs_old = np.abs(old_positions).sum()
            abs_delta_sum = float(abs_delta.sum()) + 1e-10
            for i in range(self.n_assets):
                if abs_delta[i] >= self.min_trade_pct:
                    self.trade_log.append({
                        "step": self.step_idx,
                        "asset": i,
                        "delta_weight": float(delta_weights[i]),
                        "price": float(price[i]),
                        "notional": float(abs_delta[i] * portfolio_value_before),
                        "fee": float(total_fees * abs_delta[i] / abs_delta_sum),
                        "funding": float(funding_cost * abs(old_positions[i]) / (abs_old + 1e-10)) if abs_old > 0 else 0.0,
                    })

        # --- OPT: Cache results for _get_obs() reuse ---
        abs_pos = np.abs(self.positions)
        self._cached_unrealized = new_unrealized
        self._cached_portfolio_value = portfolio_value
        self._cached_abs_positions = abs_pos

        # --- Check termination ---
        # Gymnasium semantics: terminated = MDP terminal state (circuit breaker),
        # truncated = time limit reached. SB3 bootstraps from truncated states
        # but not terminated states, so this distinction matters for value learning.
        circuit_triggered = portfolio_value < self.circuit_breaker_threshold * self.initial_capital
        terminated = circuit_triggered
        truncated = self.step_idx >= self.max_step

        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "margin_balance": self.margin_balance,
            "unrealized_pnl": unrealized_sum,
            "realized_pnl": self.realized_pnl,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_funding": self.cumulative_funding,
            "step_return": step_return,
            "gross_exposure": float(abs_pos.sum()),
            "net_exposure": float(self.positions.sum()),
            "n_long": int((self.positions > self.min_trade_pct).sum()),
            "n_short": int((self.positions < -self.min_trade_pct).sum()),
            "circuit_triggered": circuit_triggered,
            "funding_applied": self._funding_mask[self.step_idx],
            "position": self.positions.copy(),
        }

        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Constraint enforcement
    # -----------------------------------------------------------------------
    def _enforce_gross_exposure(self, raw_action: np.ndarray) -> np.ndarray:
        """Enforce sum(|weights|) ≤ max_gross_exposure and net short floor.

        Preserves relative magnitudes and signs (conviction-aware).
        H3 fix: Also enforces max_net_short_exposure during backtest
        (previously only enforced by risk manager in paper/live mode).
        """
        result = raw_action.copy()

        # Enforce gross exposure ceiling
        gross = np.abs(result).sum()
        if gross > self.max_gross_exposure:
            result *= self.max_gross_exposure / gross

        # H3 fix: Enforce net short floor (always active, not just paper/live)
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

    # -----------------------------------------------------------------------
    # PnL calculations
    # -----------------------------------------------------------------------
    def _calc_unrealized_pnl(self, current_price: np.ndarray) -> np.ndarray:
        """Calculate per-asset unrealized PnL using fixed entry notionals.

        Uses the notional value locked at position entry, not the dynamic
        margin balance. This prevents PnL on one position from being
        artificially affected by margin changes from other positions.

        PnL = sign(position) * entry_notional * (current_price / entry_price - 1)
        """
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
        """Realize PnL for positions being reduced or closed.

        Uses fixed entry notionals to compute realized PnL. The closed
        fraction of the entry notional determines the realized amount.

        H2 fix: Fully vectorized (was per-asset Python loop).
        """
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        has_position = abs_old >= 1e-8
        has_entry = np.abs(self.entry_prices) >= 1e-10
        has_notional = self.entry_notionals >= 1e-8
        active = has_position & has_entry & has_notional

        if not active.any():
            return 0.0

        # OPT: Reuse pre-allocated buffer
        closed_fraction = self._closed_frac_buffer
        closed_fraction[:] = 0.0

        # Case 1: Position flip — fully close old
        flipped = active & (np.sign(old_positions) != np.sign(new_positions)) & (abs_new > 1e-8)
        closed_fraction[flipped] = 1.0

        # Case 2: Position reduction (same sign, smaller magnitude)
        reduced = active & ~flipped & (abs_new < abs_old)
        if reduced.any():
            closed_fraction[reduced] = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]

        # Compute PnL: sign(old) * closed_fraction * entry_notional * (price/entry - 1)
        price_change = current_price / (self.entry_prices + 1e-10) - 1.0
        pnl = np.sign(old_positions) * closed_fraction * self.entry_notionals * price_change
        return float(pnl.sum())

    def _update_entry_prices(
        self,
        old_positions: np.ndarray,
        delta_weights: np.ndarray,
        current_price: np.ndarray,
        portfolio_value: float,
    ) -> None:
        """Update entry prices and notionals for position changes.

        Tracks fixed entry notionals so PnL is independent of margin changes.
        For increased positions, compute weighted average entry price/notional.
        For reduced positions, scale down notional proportionally.
        For new positions (from flat or flipped), set fresh entry.

        IMPORTANT (C7 fix): On a position flip, the old entry_notional is used
        by _realize_pnl() BEFORE this method is called, so the realized PnL on
        the closed portion is already computed correctly. We only set the NEW
        entry_notional here for the flipped direction.

        H2 fix: Fully vectorized (was per-asset Python loop).
        """
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        # Case 1: Position closed (new weight ~ 0)
        closed = abs_new < 1e-8
        self.entry_prices[closed] = 0.0
        self.entry_notionals[closed] = 0.0

        # Case 2: New position from flat (old weight ~ 0, new weight significant)
        from_flat = ~closed & (abs_old < 1e-8)
        self.entry_prices[from_flat] = current_price[from_flat]
        self.entry_notionals[from_flat] = abs_new[from_flat] * portfolio_value

        # Case 3: Flipped direction (sign changed, not from flat, not closed)
        flipped = ~closed & ~from_flat & (np.sign(old_positions) != np.sign(new_positions))
        self.entry_prices[flipped] = current_price[flipped]
        self.entry_notionals[flipped] = abs_new[flipped] * portfolio_value

        # Case 4: Position increased (same sign, larger magnitude)
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

        # Case 5: Position reduced (same sign, smaller magnitude)
        reduced = ~closed & ~from_flat & ~flipped & ~increased & (abs_new < abs_old)
        if reduced.any():
            closed_frac = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]
            self.entry_notionals[reduced] *= (1.0 - closed_frac)
            # entry_price stays the same for the remaining portion

    # -----------------------------------------------------------------------
    # Transaction costs
    # -----------------------------------------------------------------------
    def _calc_transaction_costs_fast(
        self,
        delta_weights: np.ndarray,
        abs_delta: np.ndarray,
        price: np.ndarray,
        portfolio_value: float,
    ) -> tuple[float, float]:
        """Calculate fees and slippage for the rebalance.

        OPT: Accepts pre-computed abs_delta to avoid redundant np.abs().
        Vectorized for training performance (called millions of times by SB3).

        Returns:
            (total_fees, total_slippage) in quote currency.
        """
        active = abs_delta >= 1e-8
        if not active.any():
            return 0.0, 0.0

        notionals = abs_delta[active] * portfolio_value

        # Fee: use taker fee as conservative estimate for backtesting
        total_fees = float(np.sum(notionals) * self.taker_fee_pct)

        # Volume-dependent slippage: base + impact * (order_size / hourly_volume)
        # Use previous bar's volume to avoid look-ahead bias (current bar
        # volume is unknowable at bar open when the trade is executed).
        vol_idx = max(self.step_idx - 1, 0)
        hourly_vols = self.volume_ary[vol_idx, active]
        volume_ratios = np.where(
            hourly_vols > 1e-6,
            notionals / hourly_vols,
            1.0,  # Max slippage if no volume
        )
        slippage_bps = self.slippage_base_bps + self.slippage_impact_bps * volume_ratios
        total_slippage = float(np.sum(notionals * slippage_bps * 1e-4))

        return total_fees, total_slippage

    # -----------------------------------------------------------------------
    # Funding rate
    # -----------------------------------------------------------------------
    def _apply_funding(self, price: np.ndarray) -> float:
        """Apply funding rate to all open positions.

        Funding is charged on the current notional value of each position
        (entry_notional × price_ratio), matching real exchange mechanics.
        Long pays funding when rate > 0; short receives (and vice versa).

        Vectorized for training performance (called millions of times by SB3).
        """
        funding_rates = self.funding_rate_ary[self.step_idx]
        active = (np.abs(self.positions) >= 1e-8) & (np.abs(self.entry_prices) >= 1e-10)
        if not active.any():
            return 0.0
        current_notional = self.entry_notionals[active] * (price[active] / self.entry_prices[active])
        funding_cost = np.sign(self.positions[active]) * current_notional * funding_rates[active]
        return float(funding_cost.sum())

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------
    def _calc_reward(
        self,
        step_return: float,
        abs_delta: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """Calculate reward based on the configured reward type.

        OPT: Accepts pre-computed abs_delta to avoid redundant np.abs().

        sortino_softmax: Sortino-focused reward with turnover penalty.
        simple: Raw return with turnover penalty.
        """
        if self.reward_type == "sortino_softmax":
            reward = self._sortino_reward(step_return)
        else:
            reward = step_return * self.reward_scaling

        # Turnover penalty (OPT: reuse pre-computed abs_delta)
        if portfolio_value > 1e-6:
            turnover = float(abs_delta.sum())
            reward -= turnover * self.turnover_penalty

        clip_lo, clip_hi = self.reward_clip_range
        return max(clip_lo, min(clip_hi, reward))

    def _sortino_reward(self, step_return: float) -> float:
        """Sortino-focused reward using true downside deviation.

        Uses sqrt(mean(min(r, 0)^2)) over ALL observations (positive returns
        contribute zero), matching the standard Sortino formula used in
        statistics.py, arbitrator.py, and automl evaluation.
        """
        if len(self.returns_history) < 2:
            return step_return * self.reward_scaling

        # Use recent returns for downside deviation
        window = min(self.sortino_window, len(self.returns_history))
        recent = np.array(self.returns_history[-window:])

        # True downside deviation: sqrt(sum(min(r, 0)^2) / (n - 1))
        downside_sq = np.minimum(recent, 0.0) ** 2
        n = len(downside_sq)
        dd = float(np.sqrt(np.sum(downside_sq) / max(n - 1, 1)))

        if dd < 1e-8 or not np.isfinite(dd):
            return step_return * self.reward_scaling

        # Sortino-style: return / downside_deviation
        sortino_signal = step_return / dd
        return sortino_signal * self.reward_scaling

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector.

        OPT: Uses cached unrealized PnL, portfolio value, and abs(positions)
        from step() to avoid redundant computation. Fills pre-allocated buffer
        in-place to avoid np.concatenate allocation.
        """
        # OPT: Reuse cached values from step() when available
        if self._cached_unrealized is not None:
            unrealized = self._cached_unrealized
            portfolio_value = self._cached_portfolio_value
            abs_pos = self._cached_abs_positions
            # Clear cache (one-shot — reset() calls _get_obs without step())
            self._cached_unrealized = None
            self._cached_abs_positions = None
        else:
            # Fallback for reset() path where step() hasn't run yet
            price = self.price_ary[self.step_idx]
            unrealized = self._calc_unrealized_pnl(price)
            portfolio_value = self.margin_balance + unrealized.sum()
            abs_pos = np.abs(self.positions)

        inv_cap = 1.0 / self.initial_capital
        buf = self._obs_buffer
        n = self.n_assets
        td = self.tech_dim

        # 1. Margin balance as % of initial capital
        buf[0] = portfolio_value * inv_cap

        # 2. Technical features (already flattened: n_assets * tech_dim, already float32)
        off = 1
        tech_flat = self.tech_ary[self.step_idx]
        buf[off:off + n * td] = tech_flat
        off += n * td

        # 3. Current positions (signed weights) — float64→float32 copy
        buf[off:off + n] = self.positions
        off += n

        # 4. Unrealized PnL per asset (normalized by initial capital)
        buf[off:off + n] = unrealized * inv_cap
        off += n

        # 5. Funding rates per asset
        buf[off:off + n] = self.funding_rate_ary[self.step_idx]
        off += n

        # 6. Cost to rebalance: estimated txn cost if fully rebalanced to zero
        cost_buf = self._cost_buffer
        cost_buf[:] = 0.0
        active_pos = abs_pos > 1e-8
        if active_pos.any():
            pv_for_cost = max(portfolio_value, self.initial_capital * 0.01)
            obs_vol_idx = max(self.step_idx - 1, 0)
            notionals = abs_pos[active_pos] * pv_for_cost
            hourly_vols = self.volume_ary[obs_vol_idx, active_pos]
            # NAN-01: `> 1e-6` only catches a volume of EXACTLY ~zero; a tiny-but-nonzero
            # volume bar divides an O(capital) notional by ~0 and emits ~1e10 — finite in
            # float64 (so `sanitize_obs`'s isfinite scan never sees it) but `inf` under fp16
            # AMP, which LayerNorm then turns into a per-row NaN in the actor's `loc`.
            vol_ratios = np.where(hourly_vols > 1e-6, notionals / hourly_vols, 1.0)
            np.minimum(vol_ratios, MAX_OBS_PARTICIPATION, out=vol_ratios)
            slip_bps = self.slippage_base_bps + self.slippage_impact_bps * vol_ratios
            costs = notionals * (self.taker_fee_pct + slip_bps * 1e-4)
            cost_buf[active_pos] = costs / (self.initial_capital + 1e-6)
        buf[off:off + n] = cost_buf
        off += n

        # 7. Portfolio concentration (Effective Number of Bets = 1/HHI)
        total_w = float(abs_pos.sum())
        if total_w > 1e-8:
            w_norm = abs_pos / total_w
            hhi = float((w_norm ** 2).sum())
            enb = (1.0 / hhi if hhi > 1e-8 else float(n)) / n
        else:
            enb = 0.0
        buf[off] = enb

        # NaN/inf guard + hard float16-safe magnitude bound (NAN-01).
        return sanitize_obs(buf)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def render(self):
        """Render environment state (human-readable summary)."""
        summary = self.get_portfolio_summary()
        logger.info(
            f"Step {summary['step']:>5d} | "
            f"PV {summary['portfolio_value']:>12,.2f} | "
            f"Ret {summary['total_return']:>+8.2%} | "
            f"Gross {summary['gross_exposure']:.2f} | "
            f"Pos {summary['n_positions']}",
        )

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio state summary."""
        price = self.price_ary[self.step_idx]
        unrealized = self._calc_unrealized_pnl(price)
        portfolio_value = self.margin_balance + unrealized.sum()

        abs_w = np.abs(self.positions)
        return {
            "step": self.step_idx,
            "portfolio_value": portfolio_value,
            "margin_balance": self.margin_balance,
            "unrealized_pnl": float(unrealized.sum()),
            "realized_pnl": self.realized_pnl,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_funding": self.cumulative_funding,
            "gross_exposure": float(abs_w.sum()),
            "net_exposure": float(self.positions.sum()),
            "n_positions": int((abs_w > self.min_trade_pct).sum()),
            "n_long": int((self.positions > self.min_trade_pct).sum()),
            "n_short": int((self.positions < -self.min_trade_pct).sum()),
            "total_return": portfolio_value / self.initial_capital - 1.0,
        }

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

logger = logging.getLogger(__name__)


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
        turnover_penalty: float = 0.002,
        reward_type: str = "sortino_softmax",
        sortino_window: int = 168,       # 1 week for downside deviation
        reward_scaling: float = 1.0,
        reward_clip_range: tuple = (-5.0, 5.0),
        min_trade_pct: float = 0.005,    # Skip dust trades < 0.5%
        circuit_breaker_threshold: float = 0.1,
        enable_trade_log: bool = False,
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
        self.turnover_penalty = float(turnover_penalty)
        self.reward_type = reward_type
        self.sortino_window = sortino_window
        self.reward_scaling = float(reward_scaling)
        self.reward_clip_range = reward_clip_range
        self.min_trade_pct = float(min_trade_pct)
        self.circuit_breaker_threshold = float(circuit_breaker_threshold)
        self.enable_trade_log = enable_trade_log

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
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # --- Pre-compute asset availability mask (C2: zero-price protection) ---
        self._asset_available = self.price_ary > 1e-10  # (T, n_assets)

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

        self.step_idx = 0
        self.margin_balance = self.initial_capital
        self.positions = np.zeros(self.n_assets, dtype=np.float64)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        self.entry_notionals = np.zeros(self.n_assets, dtype=np.float64)
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.cumulative_funding = 0.0
        self.portfolio_values = [self.initial_capital]
        self.returns_history = []
        if self.enable_trade_log:
            self.trade_log = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).ravel().clip(-1.0, 1.0)

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

        # Filter dust trades
        dust_mask = np.abs(delta_weights) < self.min_trade_pct
        delta_weights[dust_mask] = 0.0

        # Calculate transaction costs (fees + slippage)
        total_fees, total_slippage = self._calc_transaction_costs(
            delta_weights, price, portfolio_value_before
        )

        # Realize PnL on closed/reduced positions
        realized_this_step = self._realize_pnl(
            old_positions, delta_weights, price
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
        portfolio_value = self.margin_balance + new_unrealized.sum()
        self.portfolio_values.append(portfolio_value)

        # --- Calculate reward ---
        if portfolio_value_before > 1e-6:
            step_return = (portfolio_value - portfolio_value_before) / portfolio_value_before
        else:
            step_return = 0.0
        self.returns_history.append(step_return)

        reward = self._calc_reward(step_return, delta_weights, portfolio_value_before)

        # --- Trade logging ---
        if self.enable_trade_log and self.trade_log is not None:
            abs_old = np.abs(old_positions).sum()
            for i in range(self.n_assets):
                if abs(delta_weights[i]) >= self.min_trade_pct:
                    self.trade_log.append({
                        "step": self.step_idx,
                        "asset": i,
                        "delta_weight": float(delta_weights[i]),
                        "price": float(price[i]),
                        "notional": float(abs(delta_weights[i]) * portfolio_value_before),
                        "fee": float(total_fees * abs(delta_weights[i]) / (np.abs(delta_weights).sum() + 1e-10)),
                        "funding": float(funding_cost * abs(old_positions[i]) / (abs_old + 1e-10)) if abs_old > 0 else 0.0,
                    })

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
            "unrealized_pnl": float(new_unrealized.sum()),
            "realized_pnl": self.realized_pnl,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_funding": self.cumulative_funding,
            "step_return": step_return,
            "gross_exposure": float(np.abs(self.positions).sum()),
            "net_exposure": float(self.positions.sum()),
            "n_long": int((self.positions > self.min_trade_pct).sum()),
            "n_short": int((self.positions < -self.min_trade_pct).sum()),
            "circuit_triggered": circuit_triggered,
            "funding_applied": self._funding_mask[self.step_idx],
        }

        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Constraint enforcement
    # -----------------------------------------------------------------------
    def _enforce_gross_exposure(self, raw_action: np.ndarray) -> np.ndarray:
        """Enforce sum(|weights|) ≤ max_gross_exposure via proportional scaling.

        Preserves relative magnitudes and signs (conviction-aware).
        """
        gross = np.abs(raw_action).sum()
        if gross > self.max_gross_exposure:
            return raw_action * (self.max_gross_exposure / gross)
        return raw_action.copy()

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
        """
        realized = 0.0
        new_positions = old_positions + delta_weights

        for i in range(self.n_assets):
            old_w = old_positions[i]
            new_w = new_positions[i]

            if abs(old_w) < 1e-8:
                continue  # No existing position to realize

            # Determine closed fraction of the position
            if np.sign(old_w) != np.sign(new_w) and abs(new_w) > 1e-8:
                # Position flip: fully close old
                closed_fraction = 1.0
            elif abs(new_w) < abs(old_w):
                # Position reduction
                closed_fraction = (abs(old_w) - abs(new_w)) / abs(old_w)
            else:
                continue  # Position increased — no realization

            # PnL of closed portion using entry notional
            if abs(self.entry_prices[i]) > 1e-10 and self.entry_notionals[i] > 1e-8:
                price_change = current_price[i] / self.entry_prices[i] - 1.0
                realized += (
                    np.sign(old_w) * closed_fraction * self.entry_notionals[i] * price_change
                )

        return realized

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
        """
        new_positions = old_positions + delta_weights

        for i in range(self.n_assets):
            old_w = old_positions[i]
            new_w = new_positions[i]

            if abs(new_w) < 1e-8:
                # Position closed
                self.entry_prices[i] = 0.0
                self.entry_notionals[i] = 0.0
                continue

            if abs(old_w) < 1e-8:
                # New position from flat
                self.entry_prices[i] = current_price[i]
                self.entry_notionals[i] = abs(new_w) * portfolio_value
            elif np.sign(old_w) != np.sign(new_w):
                # Flipped direction: _realize_pnl already used old notional.
                # Set fresh entry for the new direction only.
                self.entry_prices[i] = current_price[i]
                self.entry_notionals[i] = abs(new_w) * portfolio_value
            elif abs(new_w) > abs(old_w):
                # Position increased — weighted average entry
                added_notional = abs(delta_weights[i]) * portfolio_value
                old_notional = self.entry_notionals[i]
                new_notional = old_notional + added_notional
                self.entry_prices[i] = (
                    self.entry_prices[i] * old_notional + current_price[i] * added_notional
                ) / (new_notional + 1e-10)
                self.entry_notionals[i] = new_notional
            elif abs(new_w) < abs(old_w):
                # Position reduced — scale down notional proportionally
                closed_fraction = (abs(old_w) - abs(new_w)) / abs(old_w)
                self.entry_notionals[i] *= (1.0 - closed_fraction)
                # entry_price stays the same for the remaining portion

    # -----------------------------------------------------------------------
    # Transaction costs
    # -----------------------------------------------------------------------
    def _calc_transaction_costs(
        self,
        delta_weights: np.ndarray,
        price: np.ndarray,
        portfolio_value: float,
    ) -> tuple[float, float]:
        """Calculate fees and slippage for the rebalance.

        Vectorized for training performance (called millions of times by SB3).

        Returns:
            (total_fees, total_slippage) in quote currency.
        """
        abs_delta = np.abs(delta_weights)
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
        delta_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """Calculate reward based on the configured reward type.

        sortino_softmax: Sortino-focused reward with turnover penalty.
        simple: Raw return with turnover penalty.
        """
        if self.reward_type == "sortino_softmax":
            reward = self._sortino_reward(step_return)
        else:
            reward = step_return * self.reward_scaling

        # Turnover penalty
        if portfolio_value > 1e-6:
            turnover = float(np.abs(delta_weights).sum())
            reward -= turnover * self.turnover_penalty

        return float(np.clip(reward, *self.reward_clip_range))

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
        """Build the observation vector."""
        price = self.price_ary[self.step_idx]
        unrealized = self._calc_unrealized_pnl(price)
        portfolio_value = self.margin_balance + unrealized.sum()

        # 1. Margin balance as % of initial capital
        margin_pct = np.array(
            [portfolio_value / self.initial_capital], dtype=np.float32
        )

        # 2. Technical features (already flattened: n_assets * tech_dim)
        tech_flat = self.tech_ary[self.step_idx]
        if not np.isfinite(tech_flat).all():
            tech_flat = np.nan_to_num(tech_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Current positions (signed weights)
        positions = self.positions.astype(np.float32)

        # 4. Unrealized PnL per asset (normalized by initial capital)
        unrealized_norm = (unrealized / self.initial_capital).astype(np.float32)

        # 5. Funding rates per asset
        funding = self.funding_rate_ary[self.step_idx].astype(np.float32)

        # 6. Cost to rebalance: estimated txn cost if fully rebalanced to zero
        # Use previous bar's volume to avoid look-ahead bias
        # Vectorized for training performance.
        obs_vol_idx = max(self.step_idx - 1, 0)
        pv_for_cost = max(portfolio_value, self.initial_capital * 0.01)
        abs_pos = np.abs(self.positions)
        active_pos = abs_pos > 1e-8
        cost_to_rebal = np.zeros(self.n_assets, dtype=np.float32)
        if active_pos.any():
            notionals = abs_pos[active_pos] * pv_for_cost
            hourly_vols = self.volume_ary[obs_vol_idx, active_pos]
            vol_ratios = np.where(hourly_vols > 1e-6, notionals / hourly_vols, 1.0)
            slip_bps = self.slippage_base_bps + self.slippage_impact_bps * vol_ratios
            costs = notionals * (self.taker_fee_pct + slip_bps * 1e-4)
            cost_to_rebal[active_pos] = costs / (self.initial_capital + 1e-6)

        # 7. Portfolio concentration (Effective Number of Bets = 1/HHI)
        abs_weights = np.abs(self.positions)
        total_w = abs_weights.sum()
        if total_w > 1e-8:
            w_norm = abs_weights / total_w
            hhi = float((w_norm ** 2).sum())
            enb = 1.0 / hhi if hhi > 1e-8 else float(self.n_assets)
        else:
            enb = 0.0  # No positions
        enb_arr = np.array([enb / self.n_assets], dtype=np.float32)  # Normalize to [0, 1]

        return np.concatenate([
            margin_pct,
            tech_flat,
            positions,
            unrealized_norm,
            funding,
            cost_to_rebal,
            enb_arr,
        ]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def render(self):
        """Render environment state (human-readable summary)."""
        summary = self.get_portfolio_summary()
        print(
            f"Step {summary['step']:>5d} | "
            f"PV {summary['portfolio_value']:>12,.2f} | "
            f"Ret {summary['total_return']:>+8.2%} | "
            f"Gross {summary['gross_exposure']:.2f} | "
            f"Pos {summary['n_positions']}"
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

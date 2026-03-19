"""Delta-Neutral Funding Rate Arbitrage Environment.

Gymnasium-compatible environment for spot-perp funding rate arbitrage.
The agent allocates capital across arb pairs (long spot + short perp or
reverse) to collect funding payments while maintaining delta neutrality.

Key mechanics:
- Dual-leg positions: each pair has both a spot and perp leg
- Funding rate settlement at UTC 00/08/16 on existing positions
- Capital constraint: spot_notional + perp_margin ≤ max_gross_exposure
- Delta neutrality enforced by reward penalty (not hard constraint)
- Circuit breaker on portfolio drawdown
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class FundingArbEnv(gym.Env):
    """Delta-neutral funding rate arbitrage environment.

    Observation space (~326 dims for 20 assets):
        [portfolio_value_pct]                1
        [tech_features]                      n_assets × tech_dim (currently 15)
        [current_arb_weights]                n_assets (signed)
        [unrealized_basis_pnl]               n_assets
        [cumulative_funding]                 n_assets (normalized)
        [cost_to_exit]                       n_assets
        [time_to_funding]                    1
        [total_delta_pct]                    1
        [margin_usage_pct]                   1
        [capital_deployed_pct]               1
        [portfolio_concentration]            1

    Action space:
        Box(-1, 1, shape=(n_assets,)) — per-asset arb allocation weight.
        action[i] > 0 → standard arb (long spot + short perp)
        action[i] < 0 → reverse arb (short spot + long perp)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        spot_price_ary: np.ndarray,       # (T, n_assets) — spot close prices
        perp_price_ary: np.ndarray,       # (T, n_assets) — perp close prices
        funding_rate_ary: np.ndarray,     # (T, n_assets) — funding rates per bar
        spot_volume_ary: np.ndarray,      # (T, n_assets) — spot hourly volume
        perp_volume_ary: np.ndarray,      # (T, n_assets) — perp hourly volume
        tech_ary: np.ndarray,             # (T, n_assets * 12) — arb features
        timestamps: np.ndarray,           # (T,) — UTC epoch seconds
        initial_capital: float = 100_000.0,
        spot_taker_fee_pct: float = 0.0001,   # 1 bp Binance spot
        perp_taker_fee_pct: float = 0.0005,   # 5 bp Binance futures
        perp_margin_rate: float = 0.05,       # 5% margin for perp leg
        slippage_base_bps: float = 2.0,
        slippage_impact_bps: float = 10.0,
        spot_borrow_rate_hourly: float = 8.33e-6,  # ~0.02%/day ÷ 24h (Binance margin)
        max_gross_exposure: float = 0.80,
        deadband_threshold: float = 0.01,
        # Legacy penalty params -- unused since PV-return reward (S198)
        lambda_basis: float = 0.0,
        lambda_delta: float = 100.0,
        lambda_turnover: float = 0.003,
        reward_scaling: float = 10000.0,
        reward_clip_range: tuple = (-5.0, 5.0),
        circuit_breaker_threshold: float = 0.05,
        action_repeat: int = 1,
        enable_trade_log: bool = False,
    ) -> None:
        super().__init__()

        # --- Validate inputs ---
        assert spot_price_ary.ndim == 2, "spot_price_ary must be (T, n_assets)"
        assert perp_price_ary.ndim == 2, "perp_price_ary must be (T, n_assets)"
        T, n_assets = spot_price_ary.shape
        assert perp_price_ary.shape == (T, n_assets)
        assert funding_rate_ary.shape == (T, n_assets)
        assert spot_volume_ary.shape == (T, n_assets)
        assert perp_volume_ary.shape == (T, n_assets)
        assert tech_ary.shape[0] == T
        assert tech_ary.shape[1] % n_assets == 0
        assert timestamps.shape == (T,)

        # --- Store data arrays ---
        self.spot_price_ary = spot_price_ary.astype(np.float64)
        self.perp_price_ary = perp_price_ary.astype(np.float64)
        self.funding_rate_ary = funding_rate_ary.astype(np.float64)
        self.spot_volume_ary = spot_volume_ary.astype(np.float64)
        self.perp_volume_ary = perp_volume_ary.astype(np.float64)
        self.tech_ary = tech_ary.astype(np.float32)
        self.timestamps = timestamps.astype(np.int64)

        # --- Dimensions ---
        self.n_assets = n_assets
        self.tech_dim = tech_ary.shape[1] // n_assets
        self.max_step = T - 1

        # --- Environment parameters ---
        self.initial_capital = float(initial_capital)
        self.spot_taker_fee_pct = float(spot_taker_fee_pct)
        self.perp_taker_fee_pct = float(perp_taker_fee_pct)
        self.perp_margin_rate = float(perp_margin_rate)
        self.slippage_base_bps = float(slippage_base_bps)
        self.slippage_impact_bps = float(slippage_impact_bps)
        self.spot_borrow_rate_hourly = float(spot_borrow_rate_hourly)
        self.max_gross_exposure = float(max_gross_exposure)
        self.deadband_threshold = float(deadband_threshold)
        self.lambda_basis = float(lambda_basis)
        self.lambda_delta = float(lambda_delta)
        self.lambda_turnover = float(lambda_turnover)
        self.reward_scaling = float(reward_scaling)
        self.reward_clip_range = reward_clip_range
        self.circuit_breaker_threshold = float(circuit_breaker_threshold)
        self.action_repeat = max(1, int(action_repeat))
        self.enable_trade_log = enable_trade_log

        # --- Pre-compute funding mask (UTC 00/08/16) ---
        self._funding_mask = self._build_funding_mask()

        # --- Pre-compute hours to next funding settlement for each bar ---
        self._hours_to_funding = self._build_hours_to_funding()

        # --- Observation & action spaces ---
        # obs = pv_pct(1) + tech(n*12) + weights(n) + basis_pnl(n)
        #     + cum_funding(n) + cost_exit(n) + time_to_fund(1)
        #     + delta_pct(1) + margin_pct(1) + capital_pct(1) + concentration(1)
        self.obs_dim = 1 + (n_assets * self.tech_dim) + 4 * n_assets + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_assets,), dtype=np.float32,
        )

        # --- Pre-compute asset availability mask ---
        self._asset_available = (
            (self.spot_price_ary > 1e-10) & (self.perp_price_ary > 1e-10)
        )

        # --- Runtime state (initialized in reset) ---
        self.step_idx = 0
        self.margin_balance = 0.0
        self.arb_weights = np.zeros(n_assets, dtype=np.float64)
        self.spot_entry_prices = np.zeros(n_assets, dtype=np.float64)
        self.perp_entry_prices = np.zeros(n_assets, dtype=np.float64)
        self.spot_notionals = np.zeros(n_assets, dtype=np.float64)
        self.perp_notionals = np.zeros(n_assets, dtype=np.float64)
        self.cumulative_funding = np.zeros(n_assets, dtype=np.float64)
        self.cumulative_fees = 0.0
        self.cumulative_borrow_costs = 0.0
        self.portfolio_values: list[float] = []
        self.trade_log: list[dict] | None = [] if enable_trade_log else None

    # -------------------------------------------------------------------
    # Pre-computation
    # -------------------------------------------------------------------
    def _build_funding_mask(self) -> np.ndarray:
        """Pre-compute which bars align with funding times (00, 08, 16 UTC)."""
        hours = (self.timestamps % 86400) // 3600
        return np.isin(hours, [0, 8, 16])

    def _build_hours_to_funding(self) -> np.ndarray:
        """Pre-compute hours until next funding settlement for each bar."""
        hours_in_day = (self.timestamps % 86400) // 3600
        # Funding at 0, 8, 16. Next funding hour (vectorized):
        result = np.where(
            hours_in_day < 8, 8 - hours_in_day,
            np.where(hours_in_day < 16, 16 - hours_in_day, 24 - hours_in_day),
        ).astype(np.float64)
        return result / 8.0  # normalize to [0, 1]

    # -------------------------------------------------------------------
    # Reset / Step
    # -------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.step_idx = 0
        self.margin_balance = self.initial_capital
        self.arb_weights = np.zeros(self.n_assets, dtype=np.float64)
        self.spot_entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        self.perp_entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        self.spot_notionals = np.zeros(self.n_assets, dtype=np.float64)
        self.perp_notionals = np.zeros(self.n_assets, dtype=np.float64)
        self.cumulative_funding = np.zeros(self.n_assets, dtype=np.float64)
        self.cumulative_fees = 0.0
        self.cumulative_borrow_costs = 0.0
        self.portfolio_values = [self.initial_capital]
        if self.enable_trade_log:
            self.trade_log = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Execute action then hold for action_repeat bars. Accumulates reward."""
        action = np.asarray(action, dtype=np.float64).ravel().clip(-1.0, 1.0)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for repeat_idx in range(self.action_repeat):
            # Only apply the action on the first bar; hold on subsequent bars
            if repeat_idx == 0:
                bar_action = action
            else:
                bar_action = self.arb_weights.copy()  # hold current positions

            r, term, trunc, bar_info = self._step_one_bar(bar_action)
            total_reward += r
            info = bar_info
            terminated = term
            truncated = trunc
            if terminated or truncated:
                break

        obs = self._get_obs()
        return obs, float(total_reward), terminated, truncated, info

    def _step_one_bar(self, action: np.ndarray):
        """Execute one bar of simulation. Returns (reward, terminated, truncated, info)."""
        # Advance to next bar
        self.step_idx += 1
        spot_price = self.spot_price_ary[self.step_idx]
        perp_price = self.perp_price_ary[self.step_idx]

        # Zero out actions for unavailable assets
        action = action.copy()
        action[~self._asset_available[self.step_idx]] = 0.0

        # Portfolio value before this bar
        portfolio_value_before = self._get_portfolio_value(
            self.spot_price_ary[self.step_idx - 1],
            self.perp_price_ary[self.step_idx - 1],
        )
        portfolio_value_before = max(portfolio_value_before, self.initial_capital * 0.001)

        # --- Step 1: Apply funding on OLD positions at settlement bars ---
        funding_earned_this_step = np.zeros(self.n_assets, dtype=np.float64)
        if self._funding_mask[self.step_idx]:
            funding_earned_this_step = self._apply_funding(perp_price)

        # --- Step 1b: Apply spot borrowing costs on reverse-arb positions ---
        borrow_cost = self._apply_spot_borrow_costs(spot_price)

        # --- FARB-02 fix: snapshot pre-trade basis PnL for reward penalty ---
        pre_trade_basis_pnl = self._calc_total_unrealized_basis_pnl(
            spot_price, perp_price
        )

        # --- Step 2: Apply deadband and capital constraints ---
        target_weights = self._apply_deadband(action, self.arb_weights)
        target_weights = self._enforce_capital_constraint(target_weights)

        # --- Step 3: Execute trades (close/open both legs) ---
        old_weights = self.arb_weights.copy()
        delta_weights = target_weights - old_weights

        # Calculate transaction costs on BOTH legs
        spot_fees, spot_slippage = self._calc_leg_costs(
            delta_weights, spot_price, self.spot_volume_ary,
            self.spot_taker_fee_pct, portfolio_value_before,
        )
        perp_fees, perp_slippage = self._calc_leg_costs(
            delta_weights, perp_price, self.perp_volume_ary,
            self.perp_taker_fee_pct, portfolio_value_before,
        )
        total_fees = spot_fees + perp_fees
        total_slippage = spot_slippage + perp_slippage

        # Realize basis PnL on closed/reduced positions
        realized_basis_pnl = self._realize_basis_pnl(
            old_weights, delta_weights, spot_price, perp_price
        )

        # Update positions and entry prices
        self.arb_weights = old_weights + delta_weights
        self._update_entries(old_weights, delta_weights, spot_price, perp_price, portfolio_value_before)

        # Update margin
        self.cumulative_fees += total_fees + total_slippage
        self.margin_balance -= (total_fees + total_slippage)
        self.margin_balance += realized_basis_pnl

        # Clamp margin
        if self.margin_balance < 0:
            forced_pnl = self._calc_total_unrealized_basis_pnl(spot_price, perp_price)
            self.margin_balance += forced_pnl
            self.margin_balance = max(self.margin_balance, 0.0)
            self.arb_weights = np.zeros(self.n_assets, dtype=np.float64)
            self.spot_entry_prices = np.zeros(self.n_assets, dtype=np.float64)
            self.perp_entry_prices = np.zeros(self.n_assets, dtype=np.float64)
            self.spot_notionals = np.zeros(self.n_assets, dtype=np.float64)
            self.perp_notionals = np.zeros(self.n_assets, dtype=np.float64)

        # --- Step 4: Compute new portfolio value ---
        portfolio_value = self._get_portfolio_value(spot_price, perp_price)
        self.portfolio_values.append(portfolio_value)

        # --- Step 5: Compute reward ---
        # PV-return reward: directly optimize portfolio value growth.
        # Costs, funding, and basis risk are all reflected in PV changes.
        # No penalty engineering needed — what grows PV is good.
        funding_total = float(funding_earned_this_step.sum())
        net_delta = self._calc_net_delta(spot_price, perp_price, portfolio_value)
        turnover = float(np.abs(delta_weights).sum())

        pv_return = (portfolio_value - portfolio_value_before) / (portfolio_value_before + 1e-10)
        raw_reward = self.reward_scaling * pv_return
        reward = float(np.clip(raw_reward, *self.reward_clip_range))

        # --- Step 6: Termination ---
        circuit_triggered = portfolio_value < self.circuit_breaker_threshold * self.initial_capital
        terminated = circuit_triggered
        truncated = self.step_idx >= self.max_step

        # --- Trade log ---
        if self.enable_trade_log and self.trade_log is not None:
            for i in range(self.n_assets):
                if abs(delta_weights[i]) >= self.deadband_threshold:
                    self.trade_log.append({
                        "step": self.step_idx,
                        "asset": i,
                        "delta_weight": float(delta_weights[i]),
                        "spot_price": float(spot_price[i]),
                        "perp_price": float(perp_price[i]),
                        "funding_earned": float(funding_earned_this_step[i]),
                    })

        info = {
            "portfolio_value": portfolio_value,
            "margin_balance": self.margin_balance,
            "total_funding_earned": float(self.cumulative_funding.sum()),
            "step_funding": funding_total,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_borrow_costs": self.cumulative_borrow_costs,
            "step_borrow_cost": borrow_cost,
            "net_delta": net_delta,
            "gross_exposure": float(np.abs(self.arb_weights).sum()),
            "n_active_pairs": int((np.abs(self.arb_weights) > self.deadband_threshold).sum()),
            "circuit_triggered": circuit_triggered,
            "funding_applied": self._funding_mask[self.step_idx],
        }

        return reward, terminated, truncated, info

    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------
    def _apply_deadband(
        self, target: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Apply deadband: ignore small changes in allocation."""
        result = target.copy()
        small_change = np.abs(target - current) < self.deadband_threshold
        result[small_change] = current[small_change]
        # Also zero out very small target positions
        result[np.abs(result) < self.deadband_threshold] = 0.0
        return result

    def _enforce_capital_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Enforce total capital deployment ≤ max_gross_exposure.

        Each pair uses |w| × (1 + margin_rate) of capital:
        spot_notional = |w| × PV, perp_margin = |w| × PV × margin_rate
        """
        result = weights.copy()
        capital_factor = 1.0 + self.perp_margin_rate  # ~1.05
        gross_capital = np.abs(result).sum() * capital_factor
        if gross_capital > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross_capital
            result *= scale
        return result

    # -------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------
    def _apply_funding(self, perp_price: np.ndarray) -> np.ndarray:
        """Apply funding rate to open perp positions.

        Standard arb (weight > 0): short perp → receive funding when rate > 0
        Reverse arb (weight < 0): long perp → receive funding when rate < 0

        The perp side sign is opposite to the arb weight:
        - arb_weight > 0 means short perp → perp_sign = -1
        - arb_weight < 0 means long perp → perp_sign = +1

        Funding cost = perp_sign × perp_notional_current × funding_rate
        Positive funding_cost means we PAY; negative means we RECEIVE.
        For standard arb (short perp) with positive funding rate:
          cost = -1 × notional × positive_rate = negative → we receive!
        """
        funding_rates = self.funding_rate_ary[self.step_idx]
        active = np.abs(self.arb_weights) >= 1e-8
        earned = np.zeros(self.n_assets, dtype=np.float64)

        if not active.any():
            return earned

        # Current perp notional (mark-to-market)
        perp_notional_current = np.where(
            np.abs(self.perp_entry_prices) > 1e-10,
            self.perp_notionals * (perp_price / (self.perp_entry_prices + 1e-10)),
            0.0,
        )

        # Perp position sign: opposite of arb weight
        perp_sign = -np.sign(self.arb_weights)

        # funding_cost: positive = pay, negative = receive
        funding_cost = perp_sign * perp_notional_current * funding_rates
        # earned = -funding_cost (what we receive is the negative of what we pay)
        earned[active] = -funding_cost[active]

        self.cumulative_funding += earned
        self.margin_balance += float(earned.sum())

        return earned

    def _apply_spot_borrow_costs(self, spot_price: np.ndarray) -> float:
        """Deduct spot margin interest on reverse-arb (short spot) positions.

        Reverse arb (arb_weight < 0) borrows spot to sell short.
        Cost = spot_notional_current × spot_borrow_rate_hourly per bar (1h).
        Deducted from margin_balance every bar.
        """
        # Reverse arb = arb_weight < 0 (short spot leg)
        short_spot = self.arb_weights < -1e-8
        if not short_spot.any() or self.spot_borrow_rate_hourly <= 0:
            return 0.0

        # Mark-to-market spot notional for short-spot positions
        spot_notional_current = np.where(
            np.abs(self.spot_entry_prices) > 1e-10,
            self.spot_notionals * (spot_price / (self.spot_entry_prices + 1e-10)),
            0.0,
        )

        borrow_cost = float((spot_notional_current[short_spot] * self.spot_borrow_rate_hourly).sum())
        self.cumulative_borrow_costs += borrow_cost
        self.margin_balance -= borrow_cost
        return borrow_cost

    def _calc_expected_funding_per_bar(self, perp_price: np.ndarray) -> float:
        """Calculate expected funding income per bar for reward shaping.

        Spreads the 8-hourly funding signal across all bars proportionally.
        Standard arb (weight > 0, short perp) earns when funding > 0.
        Uses currently known funding rate (no look-ahead).
        """
        active = np.abs(self.arb_weights) >= 1e-8
        if not active.any():
            return 0.0

        funding_rates = self.funding_rate_ary[self.step_idx]

        # Mark-to-market perp notional
        perp_notional_current = np.where(
            np.abs(self.perp_entry_prices) > 1e-10,
            self.perp_notionals * (perp_price / (self.perp_entry_prices + 1e-10)),
            0.0,
        )

        # Perp sign: opposite of arb weight (short perp for standard arb)
        perp_sign = -np.sign(self.arb_weights)
        # Earned = -cost; divide by 8 to pro-rate from 8-hourly to per-bar
        expected = -perp_sign * perp_notional_current * funding_rates / 8.0
        return float(expected[active].sum())

    # -------------------------------------------------------------------
    # Transaction costs
    # -------------------------------------------------------------------
    def _calc_leg_costs(
        self,
        delta_weights: np.ndarray,
        price: np.ndarray,
        volume_ary: np.ndarray,
        fee_pct: float,
        portfolio_value: float,
    ) -> tuple[float, float]:
        """Calculate fees and slippage for one leg (spot or perp)."""
        abs_delta = np.abs(delta_weights)
        active = abs_delta >= 1e-8
        if not active.any():
            return 0.0, 0.0

        notionals = abs_delta[active] * portfolio_value
        total_fees = float(np.sum(notionals) * fee_pct)

        vol_idx = max(self.step_idx - 1, 0)
        hourly_vols = volume_ary[vol_idx, active]
        volume_ratios = np.ones_like(notionals)
        np.divide(notionals, hourly_vols, out=volume_ratios, where=hourly_vols > 1e-6)
        slippage_bps = self.slippage_base_bps + self.slippage_impact_bps * volume_ratios
        total_slippage = float(np.sum(notionals * slippage_bps * 1e-4))

        return total_fees, total_slippage

    # -------------------------------------------------------------------
    # PnL
    # -------------------------------------------------------------------
    def _realize_basis_pnl(
        self,
        old_weights: np.ndarray,
        delta_weights: np.ndarray,
        spot_price: np.ndarray,
        perp_price: np.ndarray,
    ) -> float:
        """Realize basis PnL on closed/reduced arb positions.

        For standard arb (w > 0): long spot + short perp
          spot_pnl = +notional × (spot_now / spot_entry - 1)
          perp_pnl = -notional × (perp_now / perp_entry - 1)  (short)

        The basis PnL is the net of both legs.
        """
        new_weights = old_weights + delta_weights
        abs_old = np.abs(old_weights)
        abs_new = np.abs(new_weights)

        has_position = abs_old >= 1e-8
        has_entry = (np.abs(self.spot_entry_prices) >= 1e-10) & (np.abs(self.perp_entry_prices) >= 1e-10)
        active = has_position & has_entry

        if not active.any():
            return 0.0

        closed_fraction = np.zeros(self.n_assets, dtype=np.float64)

        # Position flip: fully close old
        flipped = active & (np.sign(old_weights) != np.sign(new_weights)) & (abs_new > 1e-8)
        closed_fraction[flipped] = 1.0

        # Full close
        closed_full = active & (abs_new < 1e-8)
        closed_fraction[closed_full] = 1.0

        # Reduction
        reduced = active & ~flipped & ~closed_full & (abs_new < abs_old)
        if reduced.any():
            closed_fraction[reduced] = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]

        # Compute PnL per pair
        # Spot leg: sign(old_weight) × spot_notional × (spot_now/spot_entry - 1)
        spot_pnl = (
            np.sign(old_weights) * closed_fraction * self.spot_notionals
            * (spot_price / (self.spot_entry_prices + 1e-10) - 1.0)
        )
        # Perp leg: -sign(old_weight) × perp_notional × (perp_now/perp_entry - 1)
        perp_pnl = (
            -np.sign(old_weights) * closed_fraction * self.perp_notionals
            * (perp_price / (self.perp_entry_prices + 1e-10) - 1.0)
        )

        return float((spot_pnl + perp_pnl).sum())

    def _calc_total_unrealized_basis_pnl(
        self, spot_price: np.ndarray, perp_price: np.ndarray
    ) -> float:
        """Calculate total unrealized basis PnL across all positions."""
        active = (np.abs(self.arb_weights) >= 1e-8)
        if not active.any():
            return 0.0

        spot_pnl = (
            np.sign(self.arb_weights) * self.spot_notionals
            * np.where(
                np.abs(self.spot_entry_prices) > 1e-10,
                spot_price / (self.spot_entry_prices + 1e-10) - 1.0,
                0.0,
            )
        )
        perp_pnl = (
            -np.sign(self.arb_weights) * self.perp_notionals
            * np.where(
                np.abs(self.perp_entry_prices) > 1e-10,
                perp_price / (self.perp_entry_prices + 1e-10) - 1.0,
                0.0,
            )
        )
        return float((spot_pnl[active] + perp_pnl[active]).sum())

    def _calc_net_delta(
        self, spot_price: np.ndarray, perp_price: np.ndarray, portfolio_value: float
    ) -> float:
        """Calculate net directional exposure as fraction of portfolio value.

        For perfect delta-neutral: spot notional = perp notional, opposite signs → delta ≈ 0.
        Basis divergence creates residual delta.
        """
        if portfolio_value < 1e-6:
            return 0.0

        active = np.abs(self.arb_weights) >= 1e-8
        if not active.any():
            return 0.0

        # Spot exposure (market value)
        spot_current = np.where(
            np.abs(self.spot_entry_prices) > 1e-10,
            self.spot_notionals * (spot_price / (self.spot_entry_prices + 1e-10)),
            0.0,
        )
        spot_signed = np.sign(self.arb_weights) * spot_current

        # Perp exposure (market value)
        perp_current = np.where(
            np.abs(self.perp_entry_prices) > 1e-10,
            self.perp_notionals * (perp_price / (self.perp_entry_prices + 1e-10)),
            0.0,
        )
        perp_signed = -np.sign(self.arb_weights) * perp_current  # opposite sign

        net_exposure = float((spot_signed[active] + perp_signed[active]).sum())
        return net_exposure / portfolio_value

    # -------------------------------------------------------------------
    # Entry tracking
    # -------------------------------------------------------------------
    def _update_entries(
        self,
        old_weights: np.ndarray,
        delta_weights: np.ndarray,
        spot_price: np.ndarray,
        perp_price: np.ndarray,
        portfolio_value: float,
    ) -> None:
        """Update entry prices and notionals for position changes."""
        new_weights = old_weights + delta_weights
        abs_old = np.abs(old_weights)
        abs_new = np.abs(new_weights)

        # Closed positions
        closed = abs_new < 1e-8
        self.spot_entry_prices[closed] = 0.0
        self.perp_entry_prices[closed] = 0.0
        self.spot_notionals[closed] = 0.0
        self.perp_notionals[closed] = 0.0

        # New from flat
        from_flat = ~closed & (abs_old < 1e-8)
        self.spot_entry_prices[from_flat] = spot_price[from_flat]
        self.perp_entry_prices[from_flat] = perp_price[from_flat]
        self.spot_notionals[from_flat] = abs_new[from_flat] * portfolio_value
        self.perp_notionals[from_flat] = abs_new[from_flat] * portfolio_value

        # Flipped direction
        flipped = ~closed & ~from_flat & (np.sign(old_weights) != np.sign(new_weights))
        self.spot_entry_prices[flipped] = spot_price[flipped]
        self.perp_entry_prices[flipped] = perp_price[flipped]
        self.spot_notionals[flipped] = abs_new[flipped] * portfolio_value
        self.perp_notionals[flipped] = abs_new[flipped] * portfolio_value

        # Increased position
        increased = ~closed & ~from_flat & ~flipped & (abs_new > abs_old)
        if increased.any():
            added = np.abs(delta_weights[increased]) * portfolio_value
            old_spot_n = self.spot_notionals[increased]
            new_spot_n = old_spot_n + added
            self.spot_entry_prices[increased] = (
                self.spot_entry_prices[increased] * old_spot_n
                + spot_price[increased] * added
            ) / (new_spot_n + 1e-10)
            self.spot_notionals[increased] = new_spot_n

            old_perp_n = self.perp_notionals[increased]
            new_perp_n = old_perp_n + added
            self.perp_entry_prices[increased] = (
                self.perp_entry_prices[increased] * old_perp_n
                + perp_price[increased] * added
            ) / (new_perp_n + 1e-10)
            self.perp_notionals[increased] = new_perp_n

        # Reduced position
        reduced = ~closed & ~from_flat & ~flipped & ~increased & (abs_new < abs_old)
        if reduced.any():
            closed_frac = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]
            self.spot_notionals[reduced] *= (1.0 - closed_frac)
            self.perp_notionals[reduced] *= (1.0 - closed_frac)

    # -------------------------------------------------------------------
    # Portfolio value
    # -------------------------------------------------------------------
    def _get_portfolio_value(
        self, spot_price: np.ndarray, perp_price: np.ndarray
    ) -> float:
        """Total portfolio value = margin_balance (includes realized PnL, funding, fees, borrow costs) + unrealized basis PnL."""
        unrealized = self._calc_total_unrealized_basis_pnl(spot_price, perp_price)
        return max(self.margin_balance + unrealized, 0.0)

    # -------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        spot_price = self.spot_price_ary[self.step_idx]
        perp_price = self.perp_price_ary[self.step_idx]
        portfolio_value = self._get_portfolio_value(spot_price, perp_price)

        # 1. Portfolio value as % of initial capital
        pv_pct = np.array(
            [portfolio_value / self.initial_capital], dtype=np.float32
        )

        # 2. Technical features
        tech_flat = self.tech_ary[self.step_idx]
        if not np.isfinite(tech_flat).all():
            tech_flat = np.nan_to_num(tech_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Current arb weights
        weights = self.arb_weights.astype(np.float32)

        # 4. Unrealized basis PnL per asset (normalized)
        basis_pnl = self._calc_per_asset_basis_pnl(spot_price, perp_price)
        basis_pnl_norm = (basis_pnl / (self.initial_capital + 1e-10)).astype(np.float32)

        # 5. Cumulative funding per asset (normalized)
        cum_fund_norm = (self.cumulative_funding / (self.initial_capital + 1e-10)).astype(np.float32)

        # 6. Cost to exit each position
        cost_to_exit = self._calc_cost_to_exit(spot_price, perp_price, portfolio_value)

        # 7. Time to next funding (scalar, normalized [0,1])
        time_to_fund = np.array(
            [self._hours_to_funding[self.step_idx]], dtype=np.float32
        )

        # 8. Total delta %
        net_delta = self._calc_net_delta(spot_price, perp_price, portfolio_value)
        delta_pct = np.array([net_delta], dtype=np.float32)

        # 9. Margin usage %
        perp_margin_used = float(self.perp_notionals.sum()) * self.perp_margin_rate
        margin_pct = np.array(
            [perp_margin_used / (portfolio_value + 1e-10)], dtype=np.float32
        )

        # 10. Capital deployed %
        capital_deployed = float(self.spot_notionals.sum()) + perp_margin_used
        capital_pct = np.array(
            [capital_deployed / (portfolio_value + 1e-10)], dtype=np.float32
        )

        # 11. Portfolio concentration (ENB)
        abs_w = np.abs(self.arb_weights)
        total_w = abs_w.sum()
        if total_w > 1e-8:
            w_norm = abs_w / total_w
            hhi = float((w_norm ** 2).sum())
            enb = 1.0 / hhi if hhi > 1e-8 else float(self.n_assets)
        else:
            enb = 0.0
        enb_arr = np.array([enb / self.n_assets], dtype=np.float32)

        return np.concatenate([
            pv_pct,
            tech_flat,
            weights,
            basis_pnl_norm,
            cum_fund_norm,
            cost_to_exit,
            time_to_fund,
            delta_pct,
            margin_pct,
            capital_pct,
            enb_arr,
        ]).astype(np.float32)

    def _calc_per_asset_basis_pnl(
        self, spot_price: np.ndarray, perp_price: np.ndarray
    ) -> np.ndarray:
        """Per-asset unrealized basis PnL."""
        pnl = np.zeros(self.n_assets, dtype=np.float64)
        active = np.abs(self.arb_weights) >= 1e-8
        if not active.any():
            return pnl

        spot_pnl = (
            np.sign(self.arb_weights) * self.spot_notionals
            * np.where(
                np.abs(self.spot_entry_prices) > 1e-10,
                spot_price / (self.spot_entry_prices + 1e-10) - 1.0,
                0.0,
            )
        )
        perp_pnl = (
            -np.sign(self.arb_weights) * self.perp_notionals
            * np.where(
                np.abs(self.perp_entry_prices) > 1e-10,
                perp_price / (self.perp_entry_prices + 1e-10) - 1.0,
                0.0,
            )
        )
        pnl[active] = (spot_pnl + perp_pnl)[active]
        return pnl

    def _calc_cost_to_exit(
        self, spot_price: np.ndarray, perp_price: np.ndarray, portfolio_value: float
    ) -> np.ndarray:
        """Estimated cost to close each arb position (both legs)."""
        cost = np.zeros(self.n_assets, dtype=np.float32)
        active = np.abs(self.arb_weights) >= 1e-8
        if not active.any() or portfolio_value < 1e-6:
            return cost

        abs_w = np.abs(self.arb_weights[active])
        notionals = abs_w * portfolio_value

        # Spot leg costs
        vol_idx = max(self.step_idx - 1, 0)
        spot_vols = self.spot_volume_ary[vol_idx, active]
        spot_ratios = np.ones_like(notionals)
        np.divide(notionals, spot_vols, out=spot_ratios, where=spot_vols > 1e-6)
        spot_slip = self.slippage_base_bps + self.slippage_impact_bps * spot_ratios
        spot_cost = notionals * (self.spot_taker_fee_pct + spot_slip * 1e-4)

        # Perp leg costs
        perp_vols = self.perp_volume_ary[vol_idx, active]
        perp_ratios = np.ones_like(notionals)
        np.divide(notionals, perp_vols, out=perp_ratios, where=perp_vols > 1e-6)
        perp_slip = self.slippage_base_bps + self.slippage_impact_bps * perp_ratios
        perp_cost = notionals * (self.perp_taker_fee_pct + perp_slip * 1e-4)

        cost[active] = ((spot_cost + perp_cost) / (self.initial_capital + 1e-6)).astype(np.float32)
        return cost

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------
    def render(self):
        """Render environment state."""
        summary = self.get_portfolio_summary()
        logger.info(
            f"Step {summary['step']:>5d} | "
            f"PV {summary['portfolio_value']:>12,.2f} | "
            f"Fund {summary['total_funding_earned']:>8,.2f} | "
            f"Pairs {summary['n_active_pairs']}"
        )

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio state summary."""
        spot_price = self.spot_price_ary[self.step_idx]
        perp_price = self.perp_price_ary[self.step_idx]
        pv = self._get_portfolio_value(spot_price, perp_price)

        return {
            "step": self.step_idx,
            "portfolio_value": pv,
            "margin_balance": self.margin_balance,
            "total_funding_earned": float(self.cumulative_funding.sum()),
            "cumulative_fees": self.cumulative_fees,
            "net_delta": self._calc_net_delta(spot_price, perp_price, pv),
            "gross_exposure": float(np.abs(self.arb_weights).sum()),
            "n_active_pairs": int((np.abs(self.arb_weights) > self.deadband_threshold).sum()),
            "total_return": pv / self.initial_capital - 1.0,
        }

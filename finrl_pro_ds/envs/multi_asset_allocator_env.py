"""Multi-Asset Allocator Environment (cross-sectional / time-series momentum).

Phase 2 of the cross-sectional pivot (S553-cont-33). Generalizes the battle-tested
``CryptoPerpEnv`` allocator into a cross-asset momentum allocator that the RL agent
must drive to BEAT a frozen linear core out-of-sample (or we ship the linear rule).

Design (see ``.agent/artifacts/multi_asset_allocator_architecture.md``):
  - **Action = signed conviction in [-1, 1]^N** (ADR-2). The env owns the *structure*
    the linear falsification proved essential: per-asset causal vol-targeting and the
    leverage cap. The agent spends capacity on *combination / sizing-modulation*, not
    on re-learning vol-scaling. The conviction that reproduces the validated linear
    weight is ``trend_conviction`` (see ``features/cross_asset_signals.py``).
  - **Vol-scaling lives in the env**: ``w_i = clip(conviction_i *
    clip(target_vol/realized_vol_i[t], <= lev_cap), -lev_cap, +lev_cap)`` —
    byte-matching ``xsec_momentum_falsification.vol_scaled_weights``. ``vol_ary`` is a
    dedicated CAUSAL realized-vol input (uses returns ``<= t-1``); decoupling it from
    ``tech_ary`` ordering keeps live-parity robust.
  - **Accounting copied verbatim** from ``CryptoPerpEnv``: fixed entry-notional PnL
    (SHORT-ACCT — shorts never accrue notional_debt), vectorized realize/cost paths,
    proportional gross-exposure cap. Funding generalizes to a per-bar ``carry`` accrual
    (v1 ships carry = 0, TSMOM-only; ADR-6).
  - **Reward = DSR(portfolio step return) - turnover_penalty * sum|Δw|** (reuses the
    shared Moody-Saffell ``DSRCalculator``); ``reward_type`` falls back to a Sortino or
    simple return signal.

Invariants preserved: LEAK-1 (within-window renorm done upstream in the array builder),
LEAK-2 (signals causal + ``vol_ary`` shifted + weights applied T+1), SHORT-ACCT,
MARGIN-CFG (``max_gross_exposure`` config-bound), no ``print`` (logging only).

This is a NEW class; ``CryptoPerpEnv`` is untouched (ADR-1).
"""

from __future__ import annotations

import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from finrl_pro_ds.envs.dsr import DSRCalculator
from finrl_pro_ds.envs.obs_guard import sanitize_obs

logger = logging.getLogger(__name__)

# NAN-01: ceiling on the OBSERVED participation ratio (order notional / bar dollar
# volume) used by the cost-to-unwind feature. Bounds the obs only — the cost actually
# CHARGED in `_calc_costs` is deliberately left unbounded so economics are unchanged.
MAX_OBS_PARTICIPATION = 1.0e4


class MultiAssetAllocatorEnv(gym.Env):
    """Cross-asset momentum allocator over N instruments.

    Observation (``1 + N*tech_dim + 4*N + 1`` dims):
        [portfolio_value_pct]                          # 1
        + [per_asset_signal_features × N]              # N × tech_dim (incl. baseline_weight)
        + [current_position_per_asset]                 # N (signed weight)
        + [unrealized_pnl_per_asset]                   # N (÷ initial capital)
        + [carry_per_asset]                            # N
        + [cost_to_rebalance_per_asset]                # N
        + [portfolio_concentration (ENB)]              # 1

    Action:
        Continuous signed conviction per asset in ``[-1, 1]`` (``[0, 1]`` if
        ``allow_short=False``). The env vol-scales conviction into target weights.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        price_ary: np.ndarray,           # (T, N) — close prices
        tech_ary: np.ndarray,            # (T, N * tech_dim) — per-asset causal signals
        vol_ary: np.ndarray,             # (T, N) — CAUSAL realized vol (annualized, <= t-1)
        carry_ary: np.ndarray,           # (T, N) — per-bar carry return to a long unit
        volume_ary: np.ndarray,          # (T, N) — per-bar DOLLAR volume (shares×price); slippage participation denominator (F1)
        timestamps: np.ndarray,          # (T,) — UTC epoch seconds (int64)
        initial_capital: float = 100_000.0,
        taker_fee_pct: float = 0.0002,           # 2 bps (liquid futures/ETF)
        slippage_base_bps: float = 1.0,
        slippage_impact_bps: float = 5.0,
        target_vol_asset: float = 0.10,          # per-asset annualized vol target
        lev_cap: float = 2.0,                    # per-asset leverage cap after vol-scaling
        max_gross_exposure: float = 3.0,         # vol-targeted book may lever > 1
        vol_floor: float = 1e-6,                 # guard against div-by-zero in vol-scaling
        allow_short: bool = True,
        reward_type: str = "dsr",                # {"dsr", "sortino", "simple"}
        dsr_eta: float = 0.01,                   # DSR EMA adaptation rate (Moody-Saffell)
        sortino_window: int = 63,                # downside-deviation window (bars)
        reward_scaling: float = 1.0,
        turnover_penalty: float = 0.001,
        reward_clip_range: tuple = (-5.0, 5.0),
        min_trade_pct: float = 0.005,            # skip dust trades < 0.5% weight
        no_trade_band: float = 0.0,              # v1.1: trade only to the band edge
        rebalance_interval: int = 1,             # v1.1: decision cadence in bars (1 = every bar)
        cost_penalty_scale: float = 0.0,         # v1.1: realized (fee+slip)/PV reward penalty
        circuit_breaker_threshold: float = 0.1,
        random_start: bool = False,
        random_start_pct: float = 0.1,
        enable_trade_log: bool = False,
    ) -> None:
        super().__init__()

        # --- Validate inputs ---
        assert price_ary.ndim == 2, "price_ary must be (T, N)"
        assert tech_ary.ndim == 2, "tech_ary must be (T, N * tech_dim)"
        T, n_assets = price_ary.shape
        assert tech_ary.shape[0] == T, "tech_ary rows must match price_ary"
        assert tech_ary.shape[1] % n_assets == 0, "tech_ary cols must be a multiple of N"
        for name, arr in (("vol_ary", vol_ary), ("carry_ary", carry_ary),
                          ("volume_ary", volume_ary)):
            assert arr.shape == (T, n_assets), f"{name} must be (T, N) == {(T, n_assets)}"
        assert timestamps.shape == (T,), "timestamps must be (T,)"

        # --- Store data arrays ---
        self.price_ary = price_ary.astype(np.float64)
        self.tech_ary = tech_ary.astype(np.float32)
        self.vol_ary = vol_ary.astype(np.float64)
        self.carry_ary = carry_ary.astype(np.float64)
        self.volume_ary = volume_ary.astype(np.float64)
        self.timestamps = timestamps.astype(np.int64)

        # --- Dimensions ---
        self.n_assets = n_assets
        self.tech_dim = tech_ary.shape[1] // n_assets
        self.max_step = T - 1

        # --- Environment parameters ---
        self.initial_capital = float(initial_capital)
        self.taker_fee_pct = float(taker_fee_pct)
        self.slippage_base_bps = float(slippage_base_bps)
        self.slippage_impact_bps = float(slippage_impact_bps)
        self.target_vol_asset = float(target_vol_asset)
        self.lev_cap = float(lev_cap)
        self.max_gross_exposure = float(max_gross_exposure)
        self.vol_floor = float(vol_floor)
        self.allow_short = bool(allow_short)
        self.reward_type = reward_type
        self.dsr_eta = float(dsr_eta)
        self.sortino_window = int(sortino_window)
        self.reward_scaling = float(reward_scaling)
        self.turnover_penalty = float(turnover_penalty)
        self.reward_clip_range = reward_clip_range
        self.min_trade_pct = float(min_trade_pct)
        # --- v1.1 cost levers (S553-cont-34: gross alpha real, net killed by
        # turnover cost — median cost_gap 0.41 vs the 0.15 gate). Defaults keep
        # v1 behavior byte-identical (keystone baseline-parity unaffected).
        self.no_trade_band = float(no_trade_band)
        self.rebalance_interval = max(int(rebalance_interval), 1)
        self.cost_penalty_scale = float(cost_penalty_scale)
        self.circuit_breaker_threshold = float(circuit_breaker_threshold)
        self.random_start = bool(random_start)
        self.random_start_pct = float(random_start_pct)
        self.enable_trade_log = enable_trade_log

        # --- Reward: shared Moody-Saffell DSR calculator (reused from V7) ---
        self._dsr = DSRCalculator(eta=self.dsr_eta, scale=self.reward_scaling)

        # --- Observation & action spaces ---
        # obs = pv_pct(1) + tech(N*td) + pos(N) + upnl(N) + carry(N) + cost(N) + enb(1)
        self.obs_dim = 1 + (self.n_assets * self.tech_dim) + 4 * self.n_assets + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
        )
        action_low = -1.0 if self.allow_short else 0.0
        self.action_space = gym.spaces.Box(
            low=action_low, high=1.0, shape=(self.n_assets,), dtype=np.float32,
        )

        # --- Pre-compute asset availability mask (zero-price protection) ---
        self._asset_available = self.price_ary > 1e-10  # (T, N)

        # --- Pre-allocated working arrays (OPT: avoid per-step allocations) ---
        self._obs_buffer = np.empty(self.obs_dim, dtype=np.float32)
        self._closed_frac_buffer = np.zeros(n_assets, dtype=np.float64)
        self._cost_buffer = np.zeros(n_assets, dtype=np.float32)

        # --- Cached step results (OPT: avoid recomputation in _get_obs) ---
        self._cached_unrealized: np.ndarray | None = None
        self._cached_portfolio_value: float = 0.0
        self._cached_abs_positions: np.ndarray | None = None

        # --- Runtime state (initialized in reset) ---
        self.step_idx = 0
        self.margin_balance = 0.0
        self.positions = np.zeros(n_assets, dtype=np.float64)        # signed weights
        self.entry_prices = np.zeros(n_assets, dtype=np.float64)
        self.entry_notionals = np.zeros(n_assets, dtype=np.float64)  # fixed notional at entry
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.cumulative_carry = 0.0
        self.portfolio_values: list[float] = []
        self.returns_history: list[float] = []
        self.trade_log: list[dict] | None = [] if enable_trade_log else None

    # -----------------------------------------------------------------------
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
        self.cumulative_carry = 0.0
        self.portfolio_values = [self.initial_capital]
        self.returns_history = []
        self._dsr.reset()
        if self.enable_trade_log:
            self.trade_log = []

        self._cached_unrealized = None
        self._cached_portfolio_value = 0.0
        self._cached_abs_positions = None

        return self._get_obs(), {}

    # -----------------------------------------------------------------------
    def step(self, action: np.ndarray):
        if hasattr(action, "cpu"):
            action = action.cpu()
        lo = -1.0 if self.allow_short else 0.0
        action = np.asarray(action, dtype=np.float64).ravel()
        # Defensive: a non-finite conviction (e.g. warmup signal before history
        # exists, or a NaN leaking from a policy) maps to flat — never a NaN weight.
        np.nan_to_num(action, copy=False, nan=0.0, posinf=1.0, neginf=lo)
        action = action.clip(lo, 1.0)

        # --- Conviction -> target weights (vol-scaling + leverage caps) ---
        # Uses vol at the CURRENT (decision) bar, which is causal (<= t-1), BEFORE
        # advancing the clock — mirrors how obs is built from tech_ary[step_idx].
        # v1.1 rebalance cadence: off-cadence bars HOLD current weights (the action
        # is a no-op); only every `rebalance_interval`-th decision bar may trade.
        if self.rebalance_interval > 1 and (self.step_idx % self.rebalance_interval) != 0:
            target_weights = self.positions.copy()
        else:
            target_weights = self._action_to_weights(action)

        # --- Advance to next bar ---
        self.step_idx += 1
        price = self.price_ary[self.step_idx]

        # Zero out actions for assets with invalid/zero prices
        target_weights[~self._asset_available[self.step_idx]] = 0.0
        prev_price = self.price_ary[self.step_idx - 1]

        # --- Portfolio value BEFORE this bar's price move (on held positions) ---
        unrealized_at_prev = self._calc_unrealized_pnl(prev_price)
        portfolio_value_before = max(
            self.margin_balance + unrealized_at_prev.sum(),
            self.initial_capital * 0.001,
        )

        # --- Apply carry on OLD positions BEFORE rebalance (matches exchange order) ---
        carry_pnl = self._apply_carry(price)
        self.cumulative_carry += carry_pnl
        self.margin_balance += carry_pnl

        # --- Execute rebalance: compute deltas and apply costs ---
        old_positions = self.positions.copy()
        delta_weights = target_weights - old_positions

        # v1.1 no-trade band: deltas inside the band are held; larger deltas trade
        # only to the nearest band EDGE (target -/+ band), the classic partial-
        # rebalance turnover saver — each trade is `band` smaller than full-to-target.
        # Availability-forced closes (invalid-price asset, target zeroed above) BYPASS
        # the band: shrinking them would strand a band-sized residual in a dead asset
        # forever (every later delta sits "inside band" while the price stays invalid).
        if self.no_trade_band > 0.0:
            inside = np.abs(delta_weights) <= self.no_trade_band
            banded = np.where(
                inside, 0.0,
                delta_weights - np.sign(delta_weights) * self.no_trade_band,
            )
            delta_weights = np.where(
                self._asset_available[self.step_idx], banded, delta_weights,
            )
        abs_delta = np.abs(delta_weights)

        # Filter dust trades
        dust_mask = abs_delta < self.min_trade_pct
        delta_weights[dust_mask] = 0.0
        abs_delta[dust_mask] = 0.0

        total_fees, total_slippage = self._calc_transaction_costs_fast(
            delta_weights, abs_delta, price, portfolio_value_before,
        )
        realized_this_step = self._realize_pnl(old_positions, delta_weights, price)

        self.positions = old_positions + delta_weights
        self._update_entry_prices(old_positions, delta_weights, price, portfolio_value_before)

        self.cumulative_fees += total_fees + total_slippage
        self.realized_pnl += realized_this_step
        self.margin_balance -= (total_fees + total_slippage)
        self.margin_balance += realized_this_step

        # --- Liquidation guard: force-close and realize remaining PnL ---
        if self.margin_balance < 0:
            forced_pnl = float(self._calc_unrealized_pnl(price).sum())
            self.realized_pnl += forced_pnl
            self.margin_balance += forced_pnl
            self.margin_balance = max(self.margin_balance, 0.0)
            self.positions = np.zeros(self.n_assets, dtype=np.float64)
            self.entry_prices = np.zeros(self.n_assets, dtype=np.float64)
            self.entry_notionals = np.zeros(self.n_assets, dtype=np.float64)

        # --- New portfolio value ---
        new_unrealized = self._calc_unrealized_pnl(price)
        unrealized_sum = float(new_unrealized.sum())
        portfolio_value = self.margin_balance + unrealized_sum
        self.portfolio_values.append(portfolio_value)
        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-500:]

        # --- Reward ---
        if portfolio_value_before > 1e-6:
            step_return = (portfolio_value - portfolio_value_before) / portfolio_value_before
        else:
            step_return = 0.0
        self.returns_history.append(step_return)
        if len(self.returns_history) > self.sortino_window * 2:
            self.returns_history = self.returns_history[-self.sortino_window:]

        reward = self._calc_reward(
            step_return, abs_delta, portfolio_value_before,
            realized_cost=total_fees + total_slippage,
        )

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
                        "carry": float(carry_pnl * abs(old_positions[i]) / (abs_old + 1e-10)) if abs_old > 0 else 0.0,
                    })

        # --- OPT: cache results for _get_obs() reuse ---
        abs_pos = np.abs(self.positions)
        self._cached_unrealized = new_unrealized
        self._cached_portfolio_value = portfolio_value
        self._cached_abs_positions = abs_pos

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
            "cumulative_carry": self.cumulative_carry,
            "step_return": step_return,
            "turnover": float(abs_delta.sum()),
            "gross_exposure": float(abs_pos.sum()),
            "net_exposure": float(self.positions.sum()),
            "n_long": int((self.positions > self.min_trade_pct).sum()),
            "n_short": int((self.positions < -self.min_trade_pct).sum()),
            "circuit_triggered": circuit_triggered,
            "position": self.positions.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Conviction -> weights (the NEW allocator logic vs CryptoPerpEnv)
    # -----------------------------------------------------------------------
    def _action_to_weights(self, conviction: np.ndarray) -> np.ndarray:
        """Vol-scale signed conviction into target weights, then cap gross exposure.

        ``w_i = clip(conviction_i * clip(target_vol/vol_i[t], <= lev_cap),
        -lev_cap, +lev_cap)`` — byte-identical to the validated
        ``xsec_momentum_falsification.vol_scaled_weights``. ``vol_ary[step_idx]`` is the
        CAUSAL realized vol (uses returns ``<= t-1``); read at the decision bar before
        the clock advances. Assets with non-finite / non-positive vol get weight 0.
        """
        vol = self.vol_ary[self.step_idx]
        valid = np.isfinite(vol) & (vol > self.vol_floor)
        scale = np.zeros(self.n_assets, dtype=np.float64)
        np.divide(self.target_vol_asset, vol, out=scale, where=valid)
        np.minimum(scale, self.lev_cap, out=scale)         # cap upper (scale >= 0)
        scale[~valid] = 0.0
        weights = np.clip(conviction * scale, -self.lev_cap, self.lev_cap)
        weights[~valid] = 0.0
        return self._enforce_gross_exposure(weights)

    def _enforce_gross_exposure(self, weights: np.ndarray) -> np.ndarray:
        """Cap ``sum(|w|) <= max_gross_exposure`` by PROPORTIONAL scaling (preserves
        relative sizes and signs — leverage-invariant for a fixed gross)."""
        result = weights.copy()
        gross = np.abs(result).sum()
        if gross > self.max_gross_exposure and gross > 1e-12:
            result *= self.max_gross_exposure / gross
        return result

    # -----------------------------------------------------------------------
    # PnL calculations (copied verbatim from CryptoPerpEnv — SHORT-ACCT safe)
    # -----------------------------------------------------------------------
    def _calc_unrealized_pnl(self, current_price: np.ndarray) -> np.ndarray:
        """Per-asset unrealized PnL on fixed entry notionals:
        ``sign(pos) * entry_notional * (price/entry_price - 1)``."""
        pnl = np.zeros(self.n_assets, dtype=np.float64)
        active = (np.abs(self.positions) > 1e-8) & (self.entry_notionals > 1e-8)
        if active.any():
            price_ratio = current_price[active] / (self.entry_prices[active] + 1e-10) - 1.0
            pnl[active] = np.sign(self.positions[active]) * self.entry_notionals[active] * price_ratio
        return pnl

    def _realize_pnl(
        self, old_positions: np.ndarray, delta_weights: np.ndarray, current_price: np.ndarray,
    ) -> float:
        """Realize PnL for positions being reduced or closed (fixed entry notionals)."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        has_position = abs_old >= 1e-8
        has_entry = np.abs(self.entry_prices) >= 1e-10
        has_notional = self.entry_notionals >= 1e-8
        active = has_position & has_entry & has_notional
        if not active.any():
            return 0.0

        closed_fraction = self._closed_frac_buffer
        closed_fraction[:] = 0.0

        flipped = active & (np.sign(old_positions) != np.sign(new_positions)) & (abs_new > 1e-8)
        closed_fraction[flipped] = 1.0
        reduced = active & ~flipped & (abs_new < abs_old)
        if reduced.any():
            closed_fraction[reduced] = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]

        price_change = current_price / (self.entry_prices + 1e-10) - 1.0
        pnl = np.sign(old_positions) * closed_fraction * self.entry_notionals * price_change
        return float(pnl.sum())

    def _update_entry_prices(
        self, old_positions: np.ndarray, delta_weights: np.ndarray,
        current_price: np.ndarray, portfolio_value: float,
    ) -> None:
        """Update entry prices/notionals for position changes (fixed-notional bookkeeping)."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        closed = abs_new < 1e-8
        self.entry_prices[closed] = 0.0
        self.entry_notionals[closed] = 0.0

        from_flat = ~closed & (abs_old < 1e-8)
        self.entry_prices[from_flat] = current_price[from_flat]
        self.entry_notionals[from_flat] = abs_new[from_flat] * portfolio_value

        flipped = ~closed & ~from_flat & (np.sign(old_positions) != np.sign(new_positions))
        self.entry_prices[flipped] = current_price[flipped]
        self.entry_notionals[flipped] = abs_new[flipped] * portfolio_value

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
    # Transaction costs (copied from CryptoPerpEnv)
    # -----------------------------------------------------------------------
    def _calc_transaction_costs_fast(
        self, delta_weights: np.ndarray, abs_delta: np.ndarray,
        price: np.ndarray, portfolio_value: float,
    ) -> tuple[float, float]:
        """Fees (taker_fee * traded notional) + volume-dependent slippage."""
        active = abs_delta >= 1e-8
        if not active.any():
            return 0.0, 0.0

        notionals = abs_delta[active] * portfolio_value
        total_fees = float(np.sum(notionals) * self.taker_fee_pct)

        # Previous bar's DOLLAR volume (current-bar volume is unknowable at execution).
        # volume_ary is shares×price (F1), so notionals / bar_vols is a dimensionless
        # participation fraction. Missing/zero volume (halt, holiday, data gap) ⇒ assume
        # MAX impact (ratio 1.0), the conservative CryptoPerpEnv default — never silently free.
        vol_idx = max(self.step_idx - 1, 0)
        bar_vols = self.volume_ary[vol_idx, active]
        volume_ratios = np.where(bar_vols > 1e-6, notionals / bar_vols, 1.0)
        slippage_bps = self.slippage_base_bps + self.slippage_impact_bps * volume_ratios
        total_slippage = float(np.sum(notionals * slippage_bps * 1e-4))
        return total_fees, total_slippage

    # -----------------------------------------------------------------------
    # Carry (generalizes funding; v1 ships carry = 0 — ADR-6)
    # -----------------------------------------------------------------------
    def _apply_carry(self, price: np.ndarray) -> float:
        """Accrue per-bar carry on open positions.

        ``carry_ary[t, i]`` is the per-bar carry return earned by a LONG unit of
        notional: a long position EARNS positive carry, a short position pays it.
        Charged on the current notional value (entry_notional × price_ratio), matching
        the funding mechanics it generalizes. v1 ships ``carry_ary = 0`` (TSMOM-only),
        so this is a no-op until carry data is wired in v1.1.
        """
        carry_rates = self.carry_ary[self.step_idx]
        active = (np.abs(self.positions) >= 1e-8) & (np.abs(self.entry_prices) >= 1e-10)
        if not active.any():
            return 0.0
        current_notional = self.entry_notionals[active] * (price[active] / self.entry_prices[active])
        carry_pnl = np.sign(self.positions[active]) * current_notional * carry_rates[active]
        return float(carry_pnl.sum())

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------
    def _calc_reward(
        self, step_return: float, abs_delta: np.ndarray, portfolio_value: float,
        realized_cost: float = 0.0,
    ) -> float:
        """``reward = base_signal(step_return) - turnover_penalty * sum|Δw|
        - cost_penalty_scale * realized_cost / portfolio_value``.

        ``dsr``: Moody-Saffell Differential Sharpe Ratio (shared ``DSRCalculator``).
        ``sortino``: return / trailing downside deviation. ``simple``: scaled return.

        The v1.1 cost term re-charges the bar's ACTUAL fee+slippage (already inside
        ``step_return``) as a dense penalty, so the cost signal scales with the real
        fee structure rather than raw turnover (which is fee-blind).
        """
        if self.reward_type == "dsr":
            reward = self._dsr.compute(step_return)
        elif self.reward_type == "sortino":
            reward = self._sortino_reward(step_return)
        else:
            reward = step_return * self.reward_scaling

        if portfolio_value > 1e-6:
            reward -= float(abs_delta.sum()) * self.turnover_penalty
            if self.cost_penalty_scale > 0.0:
                reward -= self.cost_penalty_scale * realized_cost / portfolio_value

        clip_lo, clip_hi = self.reward_clip_range
        return max(clip_lo, min(clip_hi, reward))

    def _sortino_reward(self, step_return: float) -> float:
        """Return / trailing downside deviation ``sqrt(sum(min(r,0)^2)/(n-1))``."""
        if len(self.returns_history) < 2:
            return step_return * self.reward_scaling
        window = min(self.sortino_window, len(self.returns_history))
        recent = np.array(self.returns_history[-window:])
        downside_sq = np.minimum(recent, 0.0) ** 2
        n = len(downside_sq)
        dd = float(np.sqrt(np.sum(downside_sq) / max(n - 1, 1)))
        if dd < 1e-8 or not np.isfinite(dd):
            return step_return * self.reward_scaling
        return (step_return / dd) * self.reward_scaling

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the flat float32 observation (raw-numpy path preserved)."""
        cached_unrealized = self._cached_unrealized
        cached_abs_pos = self._cached_abs_positions
        if cached_unrealized is not None and cached_abs_pos is not None:
            unrealized = cached_unrealized
            portfolio_value = self._cached_portfolio_value
            abs_pos = cached_abs_pos
            self._cached_unrealized = None
            self._cached_abs_positions = None
        else:
            price = self.price_ary[self.step_idx]
            unrealized = self._calc_unrealized_pnl(price)
            portfolio_value = self.margin_balance + float(unrealized.sum())
            abs_pos = np.abs(self.positions)

        inv_cap = 1.0 / self.initial_capital
        buf = self._obs_buffer
        n = self.n_assets
        td = self.tech_dim

        # 1. Portfolio value as % of initial capital
        buf[0] = portfolio_value * inv_cap

        # 2. Per-asset signal features (already flattened float32)
        off = 1
        buf[off:off + n * td] = self.tech_ary[self.step_idx]
        off += n * td

        # 3. Current positions (signed weights)
        buf[off:off + n] = self.positions
        off += n

        # 4. Unrealized PnL per asset (÷ initial capital)
        buf[off:off + n] = unrealized * inv_cap
        off += n

        # 5. Carry per asset
        buf[off:off + n] = self.carry_ary[self.step_idx]
        off += n

        # 6. Cost to fully unwind each position (fee + slippage estimate)
        cost_buf = self._cost_buffer
        cost_buf[:] = 0.0
        active_pos = abs_pos > 1e-8
        if active_pos.any():
            pv_for_cost = max(portfolio_value, self.initial_capital * 0.01)
            obs_vol_idx = max(self.step_idx - 1, 0)
            notionals = abs_pos[active_pos] * pv_for_cost
            bar_vols = self.volume_ary[obs_vol_idx, active_pos]  # DOLLAR volume (F1)
            # NAN-01: the `> 1e-6` guard only catches a volume of EXACTLY ~zero. A bar with
            # a tiny-but-nonzero dollar volume (halt, holiday stub, data gap) divides an
            # O(capital) notional by ~0 and emits a participation ~1e10 — finite in float64,
            # therefore invisible to `sanitize_obs`'s isfinite scan, but `inf` under fp16 AMP.
            # Participation beyond MAX_OBS_PARTICIPATION is saturated nonsense either way
            # (trading 1e4x the bar's whole volume), so bound it at the source.
            vol_ratios = np.where(bar_vols > 1e-6, notionals / bar_vols, 1.0)
            np.minimum(vol_ratios, MAX_OBS_PARTICIPATION, out=vol_ratios)
            slip_bps = self.slippage_base_bps + self.slippage_impact_bps * vol_ratios
            costs = notionals * (self.taker_fee_pct + slip_bps * 1e-4)
            cost_buf[active_pos] = costs / (self.initial_capital + 1e-6)
        buf[off:off + n] = cost_buf
        off += n

        # 7. Portfolio concentration (Effective Number of Bets = 1/HHI, normalized)
        total_w = float(abs_pos.sum())
        if total_w > 1e-8:
            w_norm = abs_pos / total_w
            hhi = float((w_norm ** 2).sum())
            enb = (1.0 / hhi if hhi > 1e-8 else float(n)) / n
        else:
            enb = 0.0
        buf[off] = enb

        return sanitize_obs(buf)

    # -----------------------------------------------------------------------
    def render(self):
        s = self.get_portfolio_summary()
        logger.info(
            f"Step {s['step']:>5d} | PV {s['portfolio_value']:>12,.2f} | "
            f"Ret {s['total_return']:>+8.2%} | Gross {s['gross_exposure']:.2f} | "
            f"Net {s['net_exposure']:>+.2f} | Pos {s['n_positions']}",
        )

    def get_portfolio_summary(self) -> dict:
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
            "cumulative_carry": self.cumulative_carry,
            "gross_exposure": float(abs_w.sum()),
            "net_exposure": float(self.positions.sum()),
            "n_positions": int((abs_w > self.min_trade_pct).sum()),
            "n_long": int((self.positions > self.min_trade_pct).sum()),
            "n_short": int((self.positions < -self.min_trade_pct).sum()),
            "total_return": portfolio_value / self.initial_capital - 1.0,
        }

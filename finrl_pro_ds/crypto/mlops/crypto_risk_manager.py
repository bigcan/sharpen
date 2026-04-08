"""Crypto Perpetual Futures Risk Manager for Synapse Crypto 1H.

Full risk management module that is:
- DISABLED during backtesting (pass-through mode)
- ENABLED for paper trading and live trading

The no-leverage constraint (gross exposure ≤ 1.0) is always enforced
by the environment, even during backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CryptoRiskConfig:
    """Risk management configuration."""
    # Master switch
    enabled: bool = False

    # Drawdown
    max_drawdown_pct: float = 0.20
    circuit_breaker_cooldown_bars: int = 48  # 2 days

    # Exposure
    max_gross_exposure: float = 1.0       # No leverage (always enforced by env)
    max_position_pct: float = 0.15        # Max 15% per asset
    max_net_short_exposure: float = -0.50  # Max 50% net short

    # Turnover
    daily_turnover_limit: float = 1.50    # 150% per day
    daily_cost_budget_bps: float = 20.0

    # Crypto events
    flash_crash_threshold: float = 0.15   # 15% drop in 1h
    flash_crash_min_assets: int = 5
    funding_rate_alert: float = 0.001     # |rate| > 0.1% per 8h

    # Concentration
    max_cluster_exposure: float = 0.30
    min_effective_bets: float = 4.0
    correlation_window: int = 168
    correlation_cluster_threshold: float = 0.7

    # Margin
    min_margin_reserve_pct: float = 0.05  # 5% of initial capital

    # Prop firm: EOD trailing drawdown mode
    eod_trailing_drawdown: bool = False   # If true, floor updates at EOD only
    eod_hour_utc: int = 0                 # UTC hour for EOD floor update


@dataclass
class RiskState:
    """Tracks risk-related state across steps."""
    peak_portfolio_value: float = 0.0
    current_drawdown: float = 0.0
    circuit_breaker_active: bool = False
    circuit_breaker_cooldown_remaining: int = 0
    daily_turnover_accumulated: float = 0.0
    daily_cost_accumulated: float = 0.0
    bars_since_day_start: int = 0
    violations: list = field(default_factory=list)
    # EOD trailing drawdown (prop firm mode)
    eod_peak_value: float = 0.0
    last_eod_date: str = ""
    # FIX RSK-01: Declare field properly — was dynamically added via setattr,
    # which breaks frozen dataclasses and confuses static analysis.
    last_day_close_value: float = 0.0


class CryptoRiskManager:
    """Risk manager for crypto perpetual futures trading.

    When disabled (backtest mode), all checks pass through unchanged.
    When enabled (paper/live), full risk enforcement is applied.
    """

    def __init__(self, config: CryptoRiskConfig | None = None):
        self.config = config or CryptoRiskConfig()
        self.state = RiskState()
        self._last_accumulated_delta: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self, initial_capital: float) -> None:
        """Reset risk state for a new episode."""
        self.state = RiskState(
            peak_portfolio_value=initial_capital,
            eod_peak_value=initial_capital,
        )
        self._last_accumulated_delta = 0.0

    def check(
        self,
        action: np.ndarray,
        portfolio_value: float,
        margin_balance: float,
        positions: np.ndarray,
        funding_rates: np.ndarray,
        recent_returns: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, list[str]]:
        """Check and potentially modify action based on risk controls.

        Args:
            action: Target weights from arbitrator [-1, 1] per asset.
            portfolio_value: Current portfolio value.
            margin_balance: Current margin balance.
            positions: Current positions (signed weights).
            funding_rates: Current funding rates per asset.
            recent_returns: Last 1h per-asset returns for flash crash detection.
            **kwargs: Optional ``bar_time`` (epoch int or datetime) for EOD
                trailing drawdown mode.

        Returns:
            (modified_action, violations) — action may be unchanged if no risks.
        """
        if not self.enabled:
            return action.copy(), []

        violations = []
        modified = action.copy()

        # --- Flash crash check (integrated) ---
        if recent_returns is not None:
            crash_triggered, crash_msg = self.check_flash_crash(recent_returns)
            if crash_triggered:
                violations.append(crash_msg)
                self.state.circuit_breaker_active = True
                self.state.circuit_breaker_cooldown_remaining = self.config.circuit_breaker_cooldown_bars
                logger.warning("Circuit breaker ACTIVATED: flash crash")
                return np.zeros_like(action), violations

        # --- Update drawdown tracking ---
        if self.config.eod_trailing_drawdown:
            # Prop firm EOD mode: peak only updates at end-of-day boundary.
            # Requires bar_time kwarg or falls back to tick-by-tick.
            from datetime import datetime, timezone
            bar_time = kwargs.get("bar_time")
            if bar_time is not None:
                if isinstance(bar_time, (int, float)):
                    dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)
                else:
                    dt = bar_time
                today = dt.strftime("%Y%m%d")
                if today != self.state.last_eod_date:
                    # FIX CRM-01: Day boundary — update EOD peak using the
                    # PREVIOUS bar's PV (last_day_close_value), not the current
                    # bar's PV. Current bar is the first bar of the new day,
                    # which may have gapped. Prop firm EOD trailing DD uses
                    # the previous day's closing equity.
                    last_day_pv = getattr(self.state, "last_day_close_value", portfolio_value)
                    self.state.eod_peak_value = max(
                        self.state.eod_peak_value, last_day_pv,
                    )
                    self.state.last_eod_date = today
                # Always track latest PV as candidate for next day's close
                self.state.last_day_close_value = portfolio_value
                # Drawdown measured against EOD peak (not tick-by-tick peak)
                if self.state.eod_peak_value > 0:
                    self.state.current_drawdown = (
                        1.0 - portfolio_value / self.state.eod_peak_value
                    )
            else:
                # No bar_time: fall back to standard tick-by-tick tracking
                self.state.peak_portfolio_value = max(
                    self.state.peak_portfolio_value, portfolio_value,
                )
                if self.state.peak_portfolio_value > 0:
                    self.state.current_drawdown = (
                        1.0 - portfolio_value / self.state.peak_portfolio_value
                    )
        else:
            self.state.peak_portfolio_value = max(
                self.state.peak_portfolio_value, portfolio_value,
            )
            if self.state.peak_portfolio_value > 0:
                self.state.current_drawdown = (
                    1.0 - portfolio_value / self.state.peak_portfolio_value
                )

        # --- Circuit breaker check ---
        if self.state.circuit_breaker_active:
            self.state.circuit_breaker_cooldown_remaining -= 1
            if self.state.circuit_breaker_cooldown_remaining <= 0:
                self.state.circuit_breaker_active = False
                logger.info("Circuit breaker cooldown expired — resuming trading")
            else:
                violations.append(
                    f"CIRCUIT_BREAKER: Active, {self.state.circuit_breaker_cooldown_remaining} bars remaining",
                )
                return np.zeros_like(action), violations

        # --- Check 1: Max drawdown ---
        if self.state.current_drawdown > self.config.max_drawdown_pct:
            violations.append(
                f"MAX_DRAWDOWN: {self.state.current_drawdown:.2%} > "
                f"{self.config.max_drawdown_pct:.0%} — circuit breaker activated",
            )
            self.state.circuit_breaker_active = True
            self.state.circuit_breaker_cooldown_remaining = self.config.circuit_breaker_cooldown_bars
            logger.warning(f"Circuit breaker ACTIVATED: drawdown {self.state.current_drawdown:.2%}")
            return np.zeros_like(action), violations

        # --- Check 2: Per-asset position limit ---
        # C13 fix: Only restrict growth beyond the limit, not reduction.
        # This prevents a ratchet effect where oversized positions can only shrink.
        for i in range(len(modified)):
            if abs(modified[i]) > self.config.max_position_pct:
                if abs(modified[i]) > abs(positions[i]):
                    # Growing beyond limit — cap it
                    sign = np.sign(modified[i])
                    violations.append(
                        f"POSITION_LIMIT: Asset {i} weight {modified[i]:.3f} "
                        f"exceeds {self.config.max_position_pct:.0%}",
                    )
                    modified[i] = sign * self.config.max_position_pct

        # --- Check 3: Max net short exposure ---
        net = modified.sum()
        if net < self.config.max_net_short_exposure:
            violations.append(
                f"NET_SHORT: {net:.3f} below limit {self.config.max_net_short_exposure}",
            )
            # FIX CRM-02: Scale short positions to meet constraint. If scaling
            # shorts alone can't fix it (scale >= 1.0 because violation is driven
            # by large longs), proportionally reduce ALL positions.
            short_mask = modified < 0
            if short_mask.any():
                short_sum = modified[short_mask].sum()
                long_sum = modified[~short_mask].sum()
                target_short = self.config.max_net_short_exposure - long_sum
                if short_sum < -1e-8:
                    scale = target_short / short_sum
                    if scale <= 1.0:
                        # Can fix by reducing shorts
                        modified[short_mask] *= scale
                    else:
                        # Can't fix by shorts alone — scale everything down
                        net_after = modified.sum()
                        if abs(net_after) > 1e-8:
                            target_net = self.config.max_net_short_exposure
                            overall_scale = target_net / net_after
                            modified *= min(overall_scale, 1.0)

        # --- Check 4: Daily turnover limit ---
        # Note: "daily" = 24 consecutive bars (~1 day for 1H bars).
        # Does not align with UTC midnight; acceptable for risk budgeting.
        delta = np.abs(modified - positions).sum()

        # M13 fix: Reset BEFORE incrementing to get exactly 24 bars per cycle
        if self.state.bars_since_day_start >= 24:
            self.state.daily_turnover_accumulated = 0.0
            self.state.daily_cost_accumulated = 0.0
            self.state.bars_since_day_start = 0
        self.state.bars_since_day_start += 1

        pre_delta_accumulated = self.state.daily_turnover_accumulated

        if pre_delta_accumulated + delta > self.config.daily_turnover_limit:
            violations.append(
                f"DAILY_TURNOVER: {pre_delta_accumulated + delta:.2f} > "
                f"{self.config.daily_turnover_limit:.2f}",
            )
            # Scale action to use exactly the remaining turnover budget
            remaining = max(0, self.config.daily_turnover_limit - pre_delta_accumulated)
            if delta > 1e-8:
                scale = min(remaining / delta, 1.0)
                modified = positions + (modified - positions) * scale

        # Turnover accumulation deferred to after all checks (R6 fix below).

        # --- Check 5: Funding rate alert ---
        # M15 fix: Only reduce if position and funding have same sign
        # (i.e., you're PAYING funding, not earning it).
        # Long pays when rate > 0; short pays when rate < 0.
        high_funding = np.abs(funding_rates) > self.config.funding_rate_alert
        if high_funding.any():
            for i in np.where(high_funding)[0]:
                if abs(modified[i]) > 0.01 and np.sign(modified[i]) == np.sign(funding_rates[i]):
                    violations.append(
                        f"FUNDING_ALERT: Asset {i} rate={funding_rates[i]:.5f} "
                        f"(paying funding)",
                    )
                    modified[i] *= 0.5  # Halve exposure

        # --- Check 6: Margin reserve ---
        if portfolio_value > 0 and margin_balance / portfolio_value < self.config.min_margin_reserve_pct:
            violations.append(
                f"MARGIN_LOW: {margin_balance/portfolio_value:.2%} < "
                f"{self.config.min_margin_reserve_pct:.0%}",
            )
            # Reduce gross exposure by 20%
            modified *= 0.8

        # --- Check 7: Concentration (ENB) ---
        abs_w = np.abs(modified)
        total_w = abs_w.sum()
        if total_w > 0.1:
            w_norm = abs_w / total_w
            hhi = float((w_norm ** 2).sum())
            enb = 1.0 / hhi if hhi > 1e-8 else len(modified)

            if enb < self.config.min_effective_bets:
                violations.append(
                    f"CONCENTRATION: ENB={enb:.1f} < {self.config.min_effective_bets}",
                )
                # M1: Blend toward equal-sized positions (preserve signs)
                n_active = max(1, int((np.abs(modified) > 0.01).sum()))
                avg_size = total_w / n_active
                blend = 0.5  # Blend 50% toward equal size
                for j in range(len(modified)):
                    if abs(modified[j]) > 0.01:
                        target_size = np.sign(modified[j]) * avg_size
                        modified[j] = modified[j] * (1 - blend) + target_size * blend

        # --- Re-enforce gross exposure after modifications ---
        gross = np.abs(modified).sum()
        if gross > self.config.max_gross_exposure:
            modified *= self.config.max_gross_exposure / gross

        # R6 fix: Accumulate the FINAL actual delta after all checks (5-7)
        # have potentially modified the action. Previous code accumulated
        # after Check 4 only, understating/overstating actual turnover.
        final_delta = np.abs(modified - positions).sum()
        self.state.daily_turnover_accumulated = pre_delta_accumulated + final_delta
        self._last_accumulated_delta = float(final_delta)

        self.state.violations = violations
        if violations:
            logger.debug(f"Risk violations: {violations}")

        return modified, violations

    def rollback_last_turnover(self) -> None:
        """Undo turnover accumulated by the last check() call.

        Call this when the broker skips/rejects a trade so that
        the turnover budget is not consumed by orders that never executed.
        """
        self.state.daily_turnover_accumulated = max(
            0.0,
            self.state.daily_turnover_accumulated - self._last_accumulated_delta,
        )
        self._last_accumulated_delta = 0.0

    def check_flash_crash(
        self,
        returns_1h: np.ndarray,
    ) -> tuple[bool, str]:
        """Check for market-wide flash crash.

        Args:
            returns_1h: Last 1h returns for all assets.

        Returns:
            (triggered, message)
        """
        if not self.enabled:
            return False, ""

        crash_assets = np.sum(returns_1h < -self.config.flash_crash_threshold)
        if crash_assets >= self.config.flash_crash_min_assets:
            msg = (
                f"FLASH_CRASH: {crash_assets} assets dropped > "
                f"{self.config.flash_crash_threshold:.0%} in 1h"
            )
            logger.warning(msg)
            return True, msg

        return False, ""

    def get_risk_report(self) -> dict:
        """Get current risk state summary."""
        return {
            "enabled": self.enabled,
            "current_drawdown": self.state.current_drawdown,
            "peak_value": self.state.peak_portfolio_value,
            "circuit_breaker_active": self.state.circuit_breaker_active,
            "daily_turnover": self.state.daily_turnover_accumulated,
            "n_violations": len(self.state.violations),
            "violations": self.state.violations.copy(),
        }

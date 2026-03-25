"""AlphaSeek risk manager — HFT-tuned safety layer for discrete BTC trading.

Adapts the existing CryptoRiskManager patterns for single-asset, discrete
position trading at 2-second frequency. Adds HFT-specific checks:
spread health, latency monitoring, and tighter drawdown limits.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AlphaSeekRiskConfig:
    """Risk parameters tuned for HFT single-asset discrete trading."""
    enabled: bool = True
    max_drawdown_pct: float = 0.05          # 5% — tighter than GMGP1's 10%
    max_daily_loss_pct: float = 0.02        # 2% daily hard stop
    circuit_breaker_cooldown_ticks: int = 300  # 10 minutes at 2s ticks
    daily_turnover_limit: int = 500         # max trades per day
    flash_crash_pct: float = 0.02           # 2% drop in 60 seconds
    flash_crash_window_s: int = 60
    max_spread_multiplier: float = 5.0      # halt if spread > 5x historical mean
    max_latency_ms: float = 500.0           # skip tick if processing > 500ms


class AlphaSeekRiskManager:
    """Risk enforcement for AlphaSeek live trading.

    Checks (in order):
        1. Circuit breaker (paused after drawdown breach)
        2. Max drawdown
        3. Daily loss limit
        4. Daily turnover limit
        5. Flash crash detection
        6. Spread health
        7. Latency check

    Usage:
        risk = AlphaSeekRiskManager(config)
        risk.reset(initial_capital=10000.0)
        action, violations = risk.check(
            action_int=1, position=0, mid_price=100000.0,
            spread=0.5, tick_latency_ms=50.0,
        )
    """

    def __init__(self, config: AlphaSeekRiskConfig | None = None):
        self.config = config or AlphaSeekRiskConfig()
        self._initial_capital: float = 0.0
        self._peak_value: float = 0.0
        self._portfolio_value: float = 0.0
        self._daily_start_value: float = 0.0
        self._daily_trade_count: int = 0
        self._last_daily_reset: datetime | None = None
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_ticks_remaining: int = 0
        self._price_history: list[tuple[float, float]] = []  # (timestamp, price)
        self._spread_ema: float = 0.0
        self._spread_ema_alpha: float = 0.01  # slow EMA for spread baseline

    def reset(self, initial_capital: float):
        """Initialize with starting capital."""
        self._initial_capital = initial_capital
        self._peak_value = initial_capital
        self._portfolio_value = initial_capital
        self._daily_start_value = initial_capital
        self._daily_trade_count = 0
        self._last_daily_reset = datetime.now(timezone.utc)
        self._circuit_breaker_active = False
        self._circuit_breaker_ticks_remaining = 0
        self._price_history.clear()
        self._spread_ema = 0.0

    def update_portfolio_value(self, value: float):
        """Update current portfolio value (call every tick)."""
        self._portfolio_value = value
        self._peak_value = max(self._peak_value, value)

    def check(
        self,
        action_int: int,
        position: int,
        mid_price: float,
        spread: float = 0.0,
        tick_latency_ms: float = 0.0,
    ) -> tuple[int, list[str]]:
        """Check all risk conditions and return (possibly modified) action.

        Args:
            action_int: proposed action delta {-1, 0, +1}
            position: current position {-1, 0, +1}
            mid_price: current midpoint price
            spread: current bid-ask spread
            tick_latency_ms: time spent processing this tick so far

        Returns:
            (modified_action_int, violations) where violations is a list
            of strings describing any triggered risk checks.
        """
        if not self.config.enabled:
            return action_int, []

        violations: list[str] = []
        now = datetime.now(timezone.utc)

        # --- Daily reset ---
        self._check_daily_reset(now)

        # --- 1. Circuit breaker ---
        if self._circuit_breaker_active:
            self._circuit_breaker_ticks_remaining -= 1
            if self._circuit_breaker_ticks_remaining <= 0:
                self._circuit_breaker_active = False
                logger.info("Circuit breaker deactivated")
            else:
                violations.append(
                    f"circuit_breaker_active ({self._circuit_breaker_ticks_remaining} ticks remaining)"
                )
                # Force close if in position, otherwise hold
                if position != 0:
                    return -position, violations
                return 0, violations

        # --- 2. Max drawdown ---
        if self._peak_value > 0:
            drawdown = (self._peak_value - self._portfolio_value) / self._peak_value
            if drawdown >= self.config.max_drawdown_pct:
                self._circuit_breaker_active = True
                self._circuit_breaker_ticks_remaining = self.config.circuit_breaker_cooldown_ticks
                violations.append(
                    f"max_drawdown_breach ({drawdown:.4f} >= {self.config.max_drawdown_pct})"
                )
                logger.warning("MAX DRAWDOWN BREACH: %.4f. Circuit breaker activated.", drawdown)
                if position != 0:
                    return -position, violations
                return 0, violations

        # --- 3. Daily loss limit ---
        if self._daily_start_value > 0:
            daily_loss = (self._daily_start_value - self._portfolio_value) / self._daily_start_value
            if daily_loss >= self.config.max_daily_loss_pct:
                violations.append(
                    f"daily_loss_breach ({daily_loss:.4f} >= {self.config.max_daily_loss_pct})"
                )
                if position != 0:
                    return -position, violations
                return 0, violations

        # --- 4. Daily turnover limit ---
        if action_int != 0 and self._daily_trade_count >= self.config.daily_turnover_limit:
            violations.append(
                f"daily_turnover_limit ({self._daily_trade_count} >= {self.config.daily_turnover_limit})"
            )
            return 0, violations

        # --- 5. Flash crash detection ---
        now_ts = time.time()
        self._price_history.append((now_ts, mid_price))
        # Trim old entries
        cutoff = now_ts - self.config.flash_crash_window_s
        self._price_history = [
            (t, p) for t, p in self._price_history if t >= cutoff
        ]
        if len(self._price_history) >= 2:
            oldest_price = self._price_history[0][1]
            if oldest_price > 0:
                price_change = (mid_price - oldest_price) / oldest_price
                if price_change < -self.config.flash_crash_pct:
                    violations.append(
                        f"flash_crash ({price_change:.4f} in {self.config.flash_crash_window_s}s)"
                    )
                    if position != 0:
                        return -position, violations
                    return 0, violations

        # --- 6. Spread health ---
        if spread > 0:
            if self._spread_ema <= 0:
                self._spread_ema = spread
            else:
                self._spread_ema = (
                    self._spread_ema * (1 - self._spread_ema_alpha)
                    + spread * self._spread_ema_alpha
                )
            if self._spread_ema > 0 and spread > self._spread_ema * self.config.max_spread_multiplier:
                violations.append(
                    f"spread_unhealthy ({spread:.6f} > {self.config.max_spread_multiplier}x "
                    f"EMA {self._spread_ema:.6f})"
                )
                return 0, violations

        # --- 7. Latency check ---
        if tick_latency_ms > self.config.max_latency_ms:
            violations.append(
                f"high_latency ({tick_latency_ms:.1f}ms > {self.config.max_latency_ms}ms)"
            )
            return 0, violations

        # Track trades
        if action_int != 0:
            self._daily_trade_count += 1

        return action_int, violations

    def _check_daily_reset(self, now: datetime):
        """Reset daily counters at UTC midnight."""
        if self._last_daily_reset is None:
            self._last_daily_reset = now
            return

        if now.date() > self._last_daily_reset.date():
            logger.info(
                "Daily risk reset: trades=%d, start_value=%.2f -> %.2f",
                self._daily_trade_count, self._daily_start_value, self._portfolio_value,
            )
            self._daily_trade_count = 0
            self._daily_start_value = self._portfolio_value
            self._last_daily_reset = now

    @property
    def drawdown(self) -> float:
        """Current drawdown as a fraction."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._portfolio_value) / self._peak_value

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def get_state(self) -> dict:
        """Return state for monitoring."""
        return {
            "drawdown": round(self.drawdown, 6),
            "daily_trade_count": self._daily_trade_count,
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_ticks_remaining": self._circuit_breaker_ticks_remaining,
            "spread_ema": round(self._spread_ema, 8),
            "portfolio_value": round(self._portfolio_value, 2),
            "peak_value": round(self._peak_value, 2),
        }

"""TickClock — sub-second heartbeat scheduler for HFT trading.

Unlike BarClock (which aligns to exchange candle close boundaries),
TickClock is a self-clocked heartbeat at configurable interval (default 2s).
It tracks drift and late detection for monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TickClock:
    """Produces periodic tick events at a fixed interval.

    Usage:
        clock = TickClock(interval_seconds=2.0)
        while True:
            tick_time = await clock.wait_for_next_tick()
            # ... process tick ...
    """

    def __init__(
        self,
        interval_seconds: float = 2.0,
        max_late_seconds: float = 1.0,
        execution_delay_seconds: float = 0.0,
    ):
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")

        self.interval_seconds = interval_seconds
        self.max_late_seconds = max_late_seconds
        self.execution_delay_seconds = execution_delay_seconds

        self._ticks_processed: int = 0
        self._ticks_skipped: int = 0
        self._total_drift_ms: float = 0.0
        self._max_drift_ms: float = 0.0
        self._next_tick_time: float | None = None
        self._started = False

    async def wait_for_next_tick(self) -> datetime:
        """Sleep until the next tick and return the tick timestamp (UTC).

        Returns:
            UTC datetime of this tick.

        If the tick is more than max_late_seconds late (e.g., after system
        suspend), the tick is skipped and the clock realigns to the next
        future boundary.
        """
        now = time.monotonic()

        if not self._started:
            # First tick: align to next interval boundary
            self._next_tick_time = now + self.interval_seconds
            self._started = True
        else:
            self._next_tick_time += self.interval_seconds

        # If we're behind (e.g., processing took too long), skip ahead
        while self._next_tick_time < now - self.max_late_seconds:
            self._ticks_skipped += 1
            self._next_tick_time += self.interval_seconds

        # Sleep until target time
        sleep_duration = self._next_tick_time - now + self.execution_delay_seconds
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

        # Track drift
        actual_time = time.monotonic()
        drift_ms = (actual_time - self._next_tick_time) * 1000
        self._total_drift_ms += abs(drift_ms)
        self._max_drift_ms = max(self._max_drift_ms, abs(drift_ms))
        self._ticks_processed += 1

        return datetime.now(timezone.utc)

    @property
    def ticks_processed(self) -> int:
        return self._ticks_processed

    @property
    def ticks_skipped(self) -> int:
        return self._ticks_skipped

    @property
    def avg_drift_ms(self) -> float:
        if self._ticks_processed == 0:
            return 0.0
        return self._total_drift_ms / self._ticks_processed

    @property
    def max_drift_ms(self) -> float:
        return self._max_drift_ms

    def get_stats(self) -> dict:
        """Return clock statistics for monitoring."""
        return {
            "ticks_processed": self._ticks_processed,
            "ticks_skipped": self._ticks_skipped,
            "avg_drift_ms": round(self.avg_drift_ms, 2),
            "max_drift_ms": round(self._max_drift_ms, 2),
            "interval_seconds": self.interval_seconds,
        }

    def __repr__(self) -> str:
        return (
            f"TickClock(interval={self.interval_seconds}s, "
            f"ticks={self._ticks_processed}, skipped={self._ticks_skipped})"
        )

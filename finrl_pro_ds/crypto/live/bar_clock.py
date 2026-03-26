"""Bar Clock — Schedules actions at UTC bar boundaries.

Aligns to exchange candle close times (UTC standard for crypto).
E.g., 15-min bars close at :00, :15, :30, :45.

Includes configurable execution delay after bar close to ensure
the exchange has finalized the candle before we fetch it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class BarClock:
    """Schedules trading actions at bar boundaries.

    Usage:
        clock = BarClock(bar_interval_minutes=15, execution_delay_seconds=5.0)
        while True:
            bar_time = await clock.wait_for_next_bar()
            # bar_time is the bar close timestamp (UTC)
            # ... fetch bar, run agent, execute trade ...
    """

    def __init__(
        self,
        bar_interval_minutes: int = 15,
        execution_delay_seconds: float = 5.0,
        max_late_seconds: float = 30.0,
    ):
        """
        Args:
            bar_interval_minutes: Bar interval in minutes. Must evenly divide 60
                                  or be a multiple of 60 (e.g., 1, 5, 15, 60, 240).
            execution_delay_seconds: Seconds to wait after bar close before acting.
                                     Ensures exchange has finalized the candle.
            max_late_seconds: If we're more than this many seconds late for a bar,
                              skip it and wait for the next one.
        """
        self.interval = bar_interval_minutes
        self.delay = execution_delay_seconds
        self.max_late = max_late_seconds
        self._bar_count = 0

    def _next_bar_close(self, now: datetime) -> datetime:
        """Calculate the next bar close time aligned to UTC boundaries.

        For sub-hourly intervals (1, 5, 15, 30): align to minute boundaries.
        For hourly+ intervals (60, 240): align to hour boundaries.

        FIX AUD-H01: If now is exactly on a boundary, treat it as "just closed"
        and return the NEXT boundary (not the current one). This is correct
        because if we're exactly at :15:00.000, the :15 bar has just closed
        and we should wait for :30.
        """
        now_utc = now.astimezone(timezone.utc)

        if self.interval < 60:
            # Sub-hourly: align to minute boundaries within each hour
            # FIX: Use total minutes from midnight for correct cross-hour alignment
            total_minutes = now_utc.hour * 60 + now_utc.minute
            # If exactly on a boundary with 0 seconds, the bar just closed
            (total_minutes % self.interval == 0
                           and now_utc.second == 0 and now_utc.microsecond == 0)

            current_minute = now_utc.minute
            bars_elapsed = current_minute // self.interval
            next_bar_minute = (bars_elapsed + 1) * self.interval

            # If on exact boundary, we already computed the correct next bar
            # (bars_elapsed includes current, so +1 is correct)

            if next_bar_minute >= 60:
                # Roll over to next hour
                next_bar = now_utc.replace(
                    minute=0, second=0, microsecond=0,
                ) + timedelta(hours=1, minutes=next_bar_minute - 60)
            else:
                next_bar = now_utc.replace(
                    minute=next_bar_minute, second=0, microsecond=0,
                )
        elif self.interval == 60:
            # Hourly: next hour boundary
            next_bar = now_utc.replace(
                minute=0, second=0, microsecond=0,
            ) + timedelta(hours=1)
        else:
            # Multi-hour (e.g., 240 = 4h): align to interval boundaries from midnight
            hours = self.interval // 60
            current_hour = now_utc.hour
            bars_elapsed = current_hour // hours
            next_bar_hour = (bars_elapsed + 1) * hours

            if next_bar_hour >= 24:
                # Roll over to next day
                next_day = now_utc.replace(
                    hour=0, minute=0, second=0, microsecond=0,
                ) + timedelta(days=1)
                next_bar = next_day + timedelta(hours=next_bar_hour - 24)
            else:
                next_bar = now_utc.replace(
                    hour=next_bar_hour, minute=0, second=0, microsecond=0,
                )

        return next_bar

    async def wait_for_next_bar(self) -> datetime:
        """Sleep until the next bar close + execution delay.

        FIX AUD-H02: Uses iterative loop instead of recursion to handle
        late wakeups (e.g., system suspend) without stack overflow risk.

        Returns:
            The bar close timestamp (UTC). This is the timestamp of the
            completed candle that should be fetched.
        """
        while True:
            now = datetime.now(timezone.utc)
            next_bar = self._next_bar_close(now)

            # Target wake time = bar close + execution delay
            target = next_bar + timedelta(seconds=self.delay)
            sleep_seconds = (target - now).total_seconds()

            if sleep_seconds > 0:
                if self._bar_count == 0:
                    logger.info(
                        f"BarClock: waiting {sleep_seconds:.1f}s for next "
                        f"{self.interval}-min bar close at {next_bar.strftime('%H:%M:%S')} UTC",
                    )
                await asyncio.sleep(sleep_seconds)

            # Check if we're too late (e.g., system was suspended)
            actual_time = datetime.now(timezone.utc)
            lateness = (actual_time - target).total_seconds()
            if lateness <= self.max_late:
                self._bar_count += 1
                return next_bar

            logger.warning(
                f"BarClock: {lateness:.1f}s late for bar at "
                f"{next_bar.strftime('%H:%M:%S')} UTC (max_late={self.max_late}s). "
                f"Skipping to next bar.",
            )

    def get_bar_interval_timedelta(self) -> timedelta:
        """Return the bar interval as a timedelta."""
        return timedelta(minutes=self.interval)

    @property
    def bars_processed(self) -> int:
        return self._bar_count

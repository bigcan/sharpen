"""CME-aware Bar Clock — skips maintenance windows, weekends, and holidays.

Wraps the existing BarClock via composition. The LiveTradingEngine
calls wait_for_next_bar() and gets back only valid trading bars.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from finrl_pro_ds.crypto.live.bar_clock import BarClock
from finrl_pro_ds.futures.live.cme_calendar import CMEGlobexCalendar

logger = logging.getLogger(__name__)


class CMEBarClock:
    """Market-hours-aware bar clock for CME Globex futures.

    Composes BarClock and adds a market-hours gate:
        1. If market is closed, sleep until next open.
        2. Delegate to inner BarClock for precise bar alignment.
        3. Verify the returned bar time is in trading hours.
        4. If not (edge case at maintenance boundary), retry.

    Usage:
        calendar = CMEGlobexCalendar()
        clock = CMEBarClock(bar_interval_minutes=15, calendar=calendar)
        while True:
            bar_time = await clock.wait_for_next_bar()
            # bar_time is guaranteed to be during CME trading hours
    """

    def __init__(
        self,
        bar_interval_minutes: int = 15,
        execution_delay_seconds: float = 10.0,
        max_late_seconds: float = 60.0,
        calendar: CMEGlobexCalendar | None = None,
    ):
        self._inner = BarClock(
            bar_interval_minutes=bar_interval_minutes,
            execution_delay_seconds=execution_delay_seconds,
            max_late_seconds=max_late_seconds,
        )
        self._calendar = calendar or CMEGlobexCalendar()
        self._skipped_bars = 0

    @property
    def interval(self) -> int:
        return self._inner.interval

    @property
    def bars_processed(self) -> int:
        return self._inner.bars_processed

    @property
    def skipped_bars(self) -> int:
        """Number of bars skipped due to market closure."""
        return self._skipped_bars

    def get_bar_interval_timedelta(self) -> timedelta:
        return self._inner.get_bar_interval_timedelta()

    async def wait_for_next_bar(self) -> datetime:
        """Wait for the next bar that falls within CME trading hours.

        Sleeps through maintenance windows, weekends, and holidays
        automatically. Returns a UTC datetime for the bar close.
        """
        max_retries = 100  # Safety valve

        for _ in range(max_retries):
            now = datetime.now(timezone.utc)

            # Gate: if market is closed, sleep until it opens
            if not self._calendar.is_market_open(now):
                next_open = self._calendar.next_open(now)
                sleep_secs = (next_open - now).total_seconds()

                if sleep_secs > 60:
                    reason = self._closure_reason(now)
                    logger.info(
                        f"CMEBarClock: Market closed ({reason}). "
                        f"Sleeping {sleep_secs / 3600:.1f}h until "
                        f"{next_open.strftime('%Y-%m-%d %H:%M')} UTC",
                    )

                # Sleep until market opens, plus a small buffer
                await asyncio.sleep(max(sleep_secs + 2.0, 0))
                continue

            # Market is open — delegate to inner BarClock
            bar_time = await self._inner.wait_for_next_bar()

            # Verify the bar time is valid (edge case: bar close at 5:00 PM ET)
            if self._calendar.is_market_open(bar_time):
                return bar_time

            # Bar landed on maintenance boundary — skip
            self._skipped_bars += 1
            logger.debug(
                f"CMEBarClock: Skipping bar at {bar_time.strftime('%H:%M')} UTC "
                f"(maintenance boundary)",
            )

        raise RuntimeError("CMEBarClock: exceeded max retries waiting for valid bar")

    def _closure_reason(self, dt_utc: datetime) -> str:
        """Human-readable reason why the market is closed."""
        if self._calendar.is_holiday(dt_utc):
            return "holiday"
        if self._calendar.is_weekend(dt_utc):
            return "weekend"
        if self._calendar.is_maintenance_window(dt_utc):
            return "daily maintenance"
        return "closed"

"""CFD Bar Clock — Market-hours-aware bar clock for XAUUSD CFD.

Wraps the existing BarClock via composition (same pattern as CMEBarClock).
Adds a market-hours gate for the XAUUSD CFD schedule:

    Open:  Sunday 22:00 UTC
    Close: Friday 22:00 UTC
    Daily break: 21:00-22:00 UTC (Mon-Fri, server rollover)

No holidays — forex/CFD market doesn't close for CME or bank holidays.
No DST handling — schedule is fixed in UTC (unlike CME which uses ET).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from finrl_pro_ds.crypto.live.bar_clock import BarClock

logger = logging.getLogger(__name__)

# XAUUSD CFD schedule (UTC)
_MARKET_OPEN_HOUR = 22  # Sunday 22:00 UTC
_MARKET_CLOSE_HOUR = 22  # Friday 22:00 UTC
_DAILY_BREAK_START = 21  # 21:00 UTC daily
_DAILY_BREAK_END = 22  # 22:00 UTC daily


class CFDBarClock:
    """Market-hours-aware bar clock for XAUUSD CFD trading.

    Composes BarClock and adds a market-hours gate:
        1. If market is closed, sleep until next open.
        2. Delegate to inner BarClock for precise bar alignment.
        3. Verify the returned bar time is in trading hours.
        4. If not (edge case at daily break boundary), retry.

    Usage:
        clock = CFDBarClock(bar_interval_minutes=15, execution_delay_seconds=5.0)
        while True:
            bar_time = await clock.wait_for_next_bar()
            # bar_time is guaranteed to be during XAUUSD trading hours
    """

    def __init__(
        self,
        bar_interval_minutes: int = 15,
        execution_delay_seconds: float = 5.0,
        max_late_seconds: float = 30.0,
    ):
        self._inner = BarClock(
            bar_interval_minutes=bar_interval_minutes,
            execution_delay_seconds=execution_delay_seconds,
            max_late_seconds=max_late_seconds,
        )
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
        """Wait for the next bar that falls within XAUUSD trading hours.

        Sleeps through daily breaks and weekends automatically.
        Returns a UTC datetime for the bar close.
        """
        max_retries = 100  # Safety valve

        for _ in range(max_retries):
            now = datetime.now(timezone.utc)

            # Gate: if market is closed, sleep until it opens
            if not self.is_market_open(now):
                next_open = self.next_market_open(now)
                sleep_secs = (next_open - now).total_seconds()

                if sleep_secs > 60:
                    reason = self._closure_reason(now)
                    logger.info(
                        f"CFDBarClock: Market closed ({reason}). "
                        f"Sleeping {sleep_secs / 3600:.1f}h until "
                        f"{next_open.strftime('%Y-%m-%d %H:%M')} UTC"
                    )

                # Sleep until market opens, plus a small buffer
                await asyncio.sleep(max(sleep_secs + 2.0, 0))
                continue

            # Market is open — delegate to inner BarClock
            bar_time = await self._inner.wait_for_next_bar()

            # Verify the bar time is valid
            if self.is_market_open(bar_time):
                return bar_time

            # Bar landed on daily break boundary — skip
            self._skipped_bars += 1
            logger.debug(
                f"CFDBarClock: Skipping bar at {bar_time.strftime('%H:%M')} UTC "
                f"(daily break boundary)"
            )

        raise RuntimeError("CFDBarClock: exceeded max retries waiting for valid bar")

    @staticmethod
    def is_market_open(dt_utc: datetime) -> bool:
        """Check if XAUUSD CFD market is open at the given UTC time.

        Schedule:
            Sunday 22:00 UTC → Friday 22:00 UTC (continuous)
            Daily break: 21:00-22:00 UTC (Mon-Fri)
            Saturday: closed all day
        """
        dt = dt_utc.astimezone(timezone.utc)
        weekday = dt.weekday()  # 0=Mon, 6=Sun
        hour = dt.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only from 22:00
        if weekday == 6:
            return hour >= _MARKET_OPEN_HOUR

        # Friday: closed from 22:00
        if weekday == 4 and hour >= _MARKET_CLOSE_HOUR:
            return False

        # Mon-Fri: daily break 21:00-22:00
        if _DAILY_BREAK_START <= hour < _DAILY_BREAK_END:
            return False

        return True

    @staticmethod
    def next_market_open(dt_utc: datetime) -> datetime:
        """Calculate the next market open time from a closed period.

        Returns:
            UTC datetime of the next market open.
        """
        dt = dt_utc.astimezone(timezone.utc)
        weekday = dt.weekday()
        hour = dt.hour

        # In daily break (21:00-22:00): opens at 22:00 same day
        if weekday not in (5, 6) and _DAILY_BREAK_START <= hour < _DAILY_BREAK_END:
            # But not Friday (Friday 21-22 → weekend → Sunday 22:00)
            if weekday == 4:
                # Friday close → Sunday 22:00
                days_until_sunday = 2
                return dt.replace(
                    hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_until_sunday)
            return dt.replace(
                hour=_DAILY_BREAK_END, minute=0, second=0, microsecond=0
            )

        # Weekend: Friday 22:00+ → Sunday 22:00
        if weekday == 4 and hour >= _MARKET_CLOSE_HOUR:
            # Friday after close → Sunday 22:00
            return dt.replace(
                hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0
            ) + timedelta(days=2)

        if weekday == 5:
            # Saturday → Sunday 22:00
            return dt.replace(
                hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

        if weekday == 6 and hour < _MARKET_OPEN_HOUR:
            # Sunday before 22:00 → Sunday 22:00
            return dt.replace(
                hour=_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0
            )

        # Shouldn't reach here if is_market_open returned False
        # Fallback: advance 1 hour and retry
        return dt + timedelta(hours=1)

    def _closure_reason(self, dt_utc: datetime) -> str:
        """Human-readable reason why the market is closed."""
        dt = dt_utc.astimezone(timezone.utc)
        weekday = dt.weekday()
        hour = dt.hour

        if weekday == 5:
            return "weekend (Saturday)"
        if weekday == 6 and hour < _MARKET_OPEN_HOUR:
            return "weekend (Sunday pre-open)"
        if weekday == 4 and hour >= _MARKET_CLOSE_HOUR:
            return "weekend (Friday close)"
        if _DAILY_BREAK_START <= hour < _DAILY_BREAK_END:
            return "daily rollover break"
        return "closed"

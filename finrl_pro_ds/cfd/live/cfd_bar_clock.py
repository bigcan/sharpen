"""CFD Bar Clock — Market-hours-aware bar clock for XAUUSD CFD.

Wraps the existing BarClock via composition (same pattern as CMEBarClock).
Adds a market-hours gate for the XAUUSD CFD schedule:

    IC Markets server time: EET (UTC+2 winter / UTC+3 summer DST)
    Daily break: server 00:00-01:00 → 22:00-23:00 UTC (winter) / 21:00-22:00 UTC (summer)
    Open:  Sunday 22:00/23:00 UTC (DST dependent)
    Close: Friday 22:00/23:00 UTC (DST dependent)

No holidays — forex/CFD market doesn't close for CME or bank holidays.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from finrl_pro_ds.crypto.live.bar_clock import (
    BarClock,
    _JUMP_ERROR_THRESHOLD,
    _JUMP_WARN_THRESHOLD,
)

logger = logging.getLogger(__name__)

# IC Markets server timezone (EET/EEST) — daily break is at server midnight.
_SERVER_TZ = ZoneInfo("Europe/Athens")


def _get_schedule_hours(dt_utc: datetime) -> tuple[int, int, int]:
    """Return (market_open_hour, break_start_hour, break_end_hour) in UTC.

    IC Markets server follows EET (UTC+2 winter, UTC+3 summer DST).
    Daily break is server 00:00-01:00 → shifts in UTC with DST.
    """
    server_dt = dt_utc.astimezone(_SERVER_TZ)
    offset_hours = int(server_dt.utcoffset().total_seconds()) // 3600
    # Server midnight (00:00) in UTC
    break_start = (24 - offset_hours) % 24  # 22 winter, 21 summer
    break_end = (break_start + 1) % 24
    # Market open/close at same hour as break end
    market_hour = break_start
    return market_hour, break_start, break_end


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

    @property
    def clock_jump_detected(self) -> bool:
        """True if a large clock jump was detected (delegates to inner BarClock)."""
        return self._inner.clock_jump_detected

    def clear_clock_jump(self) -> None:
        """Reset the clock jump flag after the engine has handled it."""
        self._inner.clear_clock_jump()

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
                # Record clocks for jump detection around long market-closed sleeps
                actual_sleep = max(sleep_secs + 2.0, 0)
                mono_before = time.monotonic()
                wall_before = time.time()
                await asyncio.sleep(actual_sleep)
                self._check_gate_clock_jump(wall_before, mono_before)
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

    def _check_gate_clock_jump(
        self, wall_before: float, mono_before: float,
    ) -> None:
        """Detect clock jumps during market-closed gate sleeps."""
        wall_elapsed = time.time() - wall_before
        mono_elapsed = time.monotonic() - mono_before
        delta = abs(wall_elapsed - mono_elapsed)

        if delta > _JUMP_ERROR_THRESHOLD:
            # Propagate to inner clock so the flag is visible via property
            self._inner._clock_jump_detected = True
            logger.error(
                f"CFDBarClock: Large clock jump detected during market-closed sleep: "
                f"wall={wall_elapsed:.1f}s, monotonic={mono_elapsed:.1f}s, "
                f"delta={delta:.1f}s (>{_JUMP_ERROR_THRESHOLD}s). "
                f"Trading on stale/future bars is possible.",
            )
        elif delta > _JUMP_WARN_THRESHOLD:
            logger.warning(
                f"CFDBarClock: Clock jump detected during market-closed sleep: "
                f"wall={wall_elapsed:.1f}s, monotonic={mono_elapsed:.1f}s, "
                f"delta={delta:.1f}s",
            )

    @staticmethod
    def is_market_open(dt_utc: datetime) -> bool:
        """Check if XAUUSD CFD market is open at the given UTC time.

        Schedule adapts to DST (IC Markets EET/EEST server time):
            Sunday open: 22:00 UTC (winter) / 21:00 UTC (summer)
            Friday close: same hour
            Daily break: 1 hour starting at server midnight
            Saturday: closed all day
        """
        dt = dt_utc.astimezone(timezone.utc)
        weekday = dt.weekday()  # 0=Mon, 6=Sun
        hour = dt.hour

        market_hour, break_start, break_end = _get_schedule_hours(dt)

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only from market_hour
        if weekday == 6:
            return hour >= market_hour

        # Friday: closed from market_hour
        if weekday == 4 and hour >= market_hour:
            return False

        # Mon-Fri: daily break
        if break_start <= hour < break_end:
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

        market_hour, break_start, break_end = _get_schedule_hours(dt)

        # In daily break: opens at break_end same day
        if weekday not in (5, 6) and break_start <= hour < break_end:
            # But not Friday (Friday break → weekend → Sunday)
            if weekday == 4:
                # Friday close → Sunday at market_hour
                # Recompute for Sunday's DST state
                sunday = dt.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=2)
                sun_hour, _, _ = _get_schedule_hours(sunday)
                return sunday.replace(hour=sun_hour)
            return dt.replace(
                hour=break_end, minute=0, second=0, microsecond=0
            )

        # Weekend: Friday close+ → Sunday at market_hour
        if weekday == 4 and hour >= market_hour:
            sunday = dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=2)
            sun_hour, _, _ = _get_schedule_hours(sunday)
            return sunday.replace(hour=sun_hour)

        if weekday == 5:
            sunday = dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            sun_hour, _, _ = _get_schedule_hours(sunday)
            return sunday.replace(hour=sun_hour)

        if weekday == 6 and hour < market_hour:
            return dt.replace(
                hour=market_hour, minute=0, second=0, microsecond=0
            )

        # Fallback: advance 1 hour and retry
        return dt + timedelta(hours=1)

    def _closure_reason(self, dt_utc: datetime) -> str:
        """Human-readable reason why the market is closed."""
        dt = dt_utc.astimezone(timezone.utc)
        weekday = dt.weekday()
        hour = dt.hour
        market_hour, break_start, break_end = _get_schedule_hours(dt)

        if weekday == 5:
            return "weekend (Saturday)"
        if weekday == 6 and hour < market_hour:
            return "weekend (Sunday pre-open)"
        if weekday == 4 and hour >= market_hour:
            return "weekend (Friday close)"
        if break_start <= hour < break_end:
            return "daily rollover break"
        return "closed"

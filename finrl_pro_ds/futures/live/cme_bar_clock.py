"""CME-aware Bar Clock — skips maintenance windows, weekends, and holidays.

Wraps the existing BarClock via composition. The LiveTradingEngine
calls wait_for_next_bar() and gets back only valid trading bars.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from typing import Callable, Optional

from finrl_pro_ds.crypto.live.bar_clock import (
    BarClock,
    _JUMP_ERROR_THRESHOLD,
    _JUMP_WARN_THRESHOLD,
)
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

    # Max seconds between health heartbeats during market-closed sleeps.
    # Keeps the health file fresh so Docker doesn't mark us UNHEALTHY.
    _HEARTBEAT_INTERVAL = 120

    def __init__(
        self,
        bar_interval_minutes: int = 15,
        execution_delay_seconds: float = 10.0,
        max_late_seconds: float = 60.0,
        calendar: CMEGlobexCalendar | None = None,
        heartbeat_callback: Optional[Callable[[str], None]] = None,
    ):
        self._inner = BarClock(
            bar_interval_minutes=bar_interval_minutes,
            execution_delay_seconds=execution_delay_seconds,
            max_late_seconds=max_late_seconds,
        )
        self._calendar = calendar or CMEGlobexCalendar()
        self._skipped_bars = 0
        self._heartbeat_callback = heartbeat_callback

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

                # Sleep until market opens in chunks, writing health
                # heartbeats so Docker doesn't mark us UNHEALTHY.
                actual_sleep = max(sleep_secs + 2.0, 0)
                mono_before = time.monotonic()
                wall_before = time.time()
                remaining = actual_sleep
                while remaining > 0:
                    chunk = min(remaining, self._HEARTBEAT_INTERVAL)
                    await asyncio.sleep(chunk)
                    remaining -= chunk
                    if remaining > 0 and self._heartbeat_callback is not None:
                        try:
                            self._heartbeat_callback("market_closed")
                        except Exception:
                            pass  # Non-critical
                self._check_gate_clock_jump(wall_before, mono_before)
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
                f"CMEBarClock: Large clock jump detected during market-closed sleep: "
                f"wall={wall_elapsed:.1f}s, monotonic={mono_elapsed:.1f}s, "
                f"delta={delta:.1f}s (>{_JUMP_ERROR_THRESHOLD}s). "
                f"Trading on stale/future bars is possible.",
            )
        elif delta > _JUMP_WARN_THRESHOLD:
            logger.warning(
                f"CMEBarClock: Clock jump detected during market-closed sleep: "
                f"wall={wall_elapsed:.1f}s, monotonic={mono_elapsed:.1f}s, "
                f"delta={delta:.1f}s",
            )

    def minutes_to_friday_close(self, dt_utc: datetime) -> float | None:
        """Return minutes until Friday market close, or None if not Friday.

        CME Gold closes Friday at 5:00 PM ET (or 1:00 PM ET on early close).
        Used by the live engine to trigger pre-weekend position flattening.
        """
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        et = dt_utc.astimezone(_ET)
        if et.weekday() != 4:  # Not Friday
            return None
        # Use calendar's next_close which handles early close days
        try:
            close_utc = self._calendar.next_close(dt_utc)
        except RuntimeError:
            return None
        remaining = (close_utc - dt_utc.astimezone(timezone.utc)).total_seconds() / 60.0
        if remaining <= 0:
            return 0.0
        return remaining

    def _closure_reason(self, dt_utc: datetime) -> str:
        """Human-readable reason why the market is closed."""
        if self._calendar.is_holiday(dt_utc):
            return "holiday"
        if self._calendar.is_weekend(dt_utc):
            return "weekend"
        if self._calendar.is_maintenance_window(dt_utc):
            return "daily maintenance"
        return "closed"

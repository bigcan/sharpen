"""CME Globex trading calendar for Gold futures (GC/MGC).

GC/MGC trade Sunday-Friday on CME Globex:
    Open:  Sunday 6:00 PM ET
    Close: Friday 5:00 PM ET
    Daily maintenance: 5:00 PM - 6:00 PM ET (1 hour gap each weekday)

All public methods accept/return UTC datetimes.
DST handled via zoneinfo (stdlib Python 3.9+).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Eastern Time zone (handles EST/EDT automatically)
_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc

# CME Globex GC maintenance window in ET
_MAINTENANCE_START = time(17, 0)  # 5:00 PM ET
_MAINTENANCE_END = time(18, 0)    # 6:00 PM ET

# Weekly schedule: opens Sunday 6 PM ET, closes Friday 5 PM ET
_WEEK_OPEN_DAY = 6   # Sunday
_WEEK_OPEN_TIME = time(18, 0)  # 6:00 PM ET
_WEEK_CLOSE_DAY = 4  # Friday
_WEEK_CLOSE_TIME = time(17, 0)  # 5:00 PM ET

# CME observed holidays for 2026 (full-day closures, no Globex session).
# Source: CME Group holiday calendar. Update annually.
CME_HOLIDAYS_2026: set[date] = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth National Independence Day  # FIX IB-12
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
}

# Early close days (close at 1:00 PM ET instead of 5:00 PM ET)
# FIX IB-14: Added July 2 and Dec 31 (CME metals early close days)
CME_EARLY_CLOSE_2026: set[date] = {
    date(2026, 7, 2),    # Day before Independence Day (observed)
    date(2026, 11, 27),  # Black Friday
    date(2026, 12, 24),  # Christmas Eve
    date(2026, 12, 31),  # New Year's Eve
}


class CMEGlobexCalendar:
    """CME Globex Gold futures trading hours with DST awareness.

    Thread-safe, stateless (no mutable instance state).
    """

    def __init__(
        self,
        holidays: set[date] | None = None,
        early_closes: set[date] | None = None,
    ):
        self._holidays = holidays if holidays is not None else CME_HOLIDAYS_2026
        self._early_closes = early_closes if early_closes is not None else CME_EARLY_CLOSE_2026

    def is_market_open(self, dt_utc: datetime) -> bool:
        """Check if CME Globex GC is open at this UTC time."""
        et = dt_utc.astimezone(_ET)

        # Holiday check: if the current ET date is a holiday, market is closed
        if et.date() in self._holidays:
            return False

        weekday = et.weekday()  # 0=Mon, 6=Sun
        et_time = et.time()

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only after 6 PM ET
        if weekday == 6:
            return et_time >= _WEEK_OPEN_TIME

        # Friday: open until 5 PM ET (then closed for weekend)
        if weekday == 4:
            if et_time >= _WEEK_CLOSE_TIME:
                return False
            # Check early close on the previous day affecting today's maintenance
            return not self._in_maintenance(et)

        # Monday-Thursday: open except during maintenance (5-6 PM ET)
        if self._in_maintenance(et):
            return False

        # FIX IB-13: After maintenance (6 PM+ ET), check if the NEXT trading
        # date is a holiday. If so, no Globex session opens — the entire
        # evening session is cancelled (e.g., Dec 24 evening before Dec 25).
        if et_time >= _MAINTENANCE_END:
            next_date = (et + timedelta(days=1)).date()
            if next_date in self._holidays:
                return False

        return True

    def _in_maintenance(self, et: datetime) -> bool:
        """Check if we're in the daily 5-6 PM ET maintenance window."""
        et_time = et.time()

        # Check if this is an early close day
        if et.date() in self._early_closes:
            early_close = time(13, 0)  # 1:00 PM ET
            # Early close: market closed from 1 PM to 6 PM ET
            if early_close <= et_time < _MAINTENANCE_END:
                return True
            return False

        return _MAINTENANCE_START <= et_time < _MAINTENANCE_END

    def next_open(self, dt_utc: datetime) -> datetime:
        """Return the next market open time after dt_utc (UTC).

        If market is currently open, returns the next open after
        the current session closes (i.e., after maintenance or weekend).
        If market is closed, returns when it next opens.
        """
        et = dt_utc.astimezone(_ET)

        # Try up to 10 days ahead (handles long holiday weekends)
        for offset_hours in range(0, 240, 1):
            candidate_et = et + timedelta(hours=offset_hours)
            candidate_utc = candidate_et.astimezone(_UTC)

            if not self.is_market_open(candidate_utc) and offset_hours == 0:
                continue

            if offset_hours == 0:
                # Market is currently open — find the next close, then next open
                close_utc = self.next_close(dt_utc)
                return self.next_open(close_utc + timedelta(seconds=1))

            if self.is_market_open(candidate_utc):
                # Found an open time — now back up to the exact open boundary
                return self._find_exact_open(candidate_utc)

        raise RuntimeError(f"Could not find next market open within 10 days of {dt_utc}")

    def _find_exact_open(self, approx_utc: datetime) -> datetime:
        """Binary search backward to find the exact open boundary."""
        et = approx_utc.astimezone(_ET)
        weekday = et.weekday()

        # Sunday: opens at 6 PM ET
        if weekday == 6:
            open_et = et.replace(hour=18, minute=0, second=0, microsecond=0)
            return open_et.astimezone(_UTC)

        # Weekday after maintenance: opens at 6 PM ET previous day
        # (maintenance is 5-6 PM, so open is 6 PM)
        open_et = et.replace(hour=18, minute=0, second=0, microsecond=0)

        # If current time is before 6 PM, the open was yesterday at 6 PM
        if et.time() < _MAINTENANCE_END:
            open_et = (et - timedelta(days=1)).replace(
                hour=18, minute=0, second=0, microsecond=0,
            )

        # Check for holidays: if the computed open day is a holiday, go back further
        while open_et.date() in self._holidays:
            open_et -= timedelta(days=1)
            open_et = open_et.replace(hour=18, minute=0, second=0, microsecond=0)

        return open_et.astimezone(_UTC)

    def next_close(self, dt_utc: datetime) -> datetime:
        """Return the next market close (maintenance start) after dt_utc (UTC).

        Returns the next 5:00 PM ET (or 1:00 PM ET on early close days).
        """
        et = dt_utc.astimezone(_ET)

        # Try up to 7 days ahead
        for day_offset in range(8):
            candidate_date = et.date() + timedelta(days=day_offset)
            candidate_weekday = candidate_date.weekday()

            # Skip weekends for close times
            if candidate_weekday >= 5:
                continue

            # Holiday: no close (market doesn't open)
            if candidate_date in self._holidays:
                continue

            # Determine close time for this day
            if candidate_date in self._early_closes:
                close_time = time(13, 0)  # 1 PM ET
            elif candidate_weekday == 4:
                close_time = _WEEK_CLOSE_TIME  # Friday 5 PM ET
            else:
                close_time = _MAINTENANCE_START  # Weekday 5 PM ET

            close_et = datetime.combine(candidate_date, close_time, tzinfo=_ET)
            close_utc = close_et.astimezone(_UTC)

            if close_utc > dt_utc:
                return close_utc

        raise RuntimeError(f"Could not find next market close within 7 days of {dt_utc}")

    def is_maintenance_window(self, dt_utc: datetime) -> bool:
        """Check if we're in the 5-6 PM ET daily maintenance."""
        et = dt_utc.astimezone(_ET)
        return self._in_maintenance(et)

    def is_weekend(self, dt_utc: datetime) -> bool:
        """Check if we're in the weekend gap (Fri 5 PM - Sun 6 PM ET)."""
        et = dt_utc.astimezone(_ET)
        weekday = et.weekday()

        if weekday == 5:  # Saturday
            return True
        if weekday == 6 and et.time() < _WEEK_OPEN_TIME:  # Sunday before 6 PM
            return True
        if weekday == 4 and et.time() >= _WEEK_CLOSE_TIME:  # Friday after 5 PM
            return True
        return False

    def is_holiday(self, dt_utc: datetime) -> bool:
        """Check if the given UTC time falls on a CME holiday."""
        et = dt_utc.astimezone(_ET)
        return et.date() in self._holidays

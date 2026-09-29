"""Ex-ante NYSE trading calendar for the forward runner (TAILWIND Tier-2 roadmap X2).

The linear-core book rebalances at month-end sessions, and under ``execution.decision_lead_bars:
1`` the rebalance fills at the month-end close from data ending the session before. A forward
runner therefore has to know, at close d-1, whether session d is the last session of its month.
The price data cannot tell it (the next bar does not exist yet); an exchange calendar can,
because NYSE publishes its schedule in advance. This module is that calendar: rule-based full
closures, so it never needs the network and never reads data stamped after the day in question.

Coverage and rules (full-day closures only; early closes do not change a daily close-to-close
book):
- New Year's Day: Jan 1, or Mon Jan 2 when Jan 1 is a Sunday. When Jan 1 is a Saturday there is
  NO weekday holiday (NYSE stays open on Fri Dec 31; e.g. 2010-12-31, 2021-12-31).
- Martin Luther King Jr. Day: 3rd Monday of January (NYSE since 1998).
- Washington's Birthday: 3rd Monday of February.
- Good Friday: the Friday before Western Easter.
- Memorial Day: last Monday of May.
- Juneteenth: June 19 (Sat -> Fri, Sun -> Mon), from 2022.
- Independence Day: July 4 (Sat -> Fri, Sun -> Mon).
- Labor Day: 1st Monday of September.
- Thanksgiving: 4th Thursday of November.
- Christmas: Dec 25 (Sat -> Fri, Sun -> Mon).
- Unscheduled closures, dated: 9/11 (2001-09-11..14), national days of mourning (Reagan
  2004-06-11, Ford 2007-01-02, G.H.W. Bush 2018-12-05, Carter 2025-01-09) and Hurricane Sandy
  (2012-10-29..30).

A closure announced after this table was written is not known here. The runner's freshness gate
catches it: the data will have no bar for a session the calendar expects, so the run fails
closed until the date is added to ``SPECIAL_CLOSURES``. Years before ``MIN_YEAR`` raise, as the
rules above were not all in force.

``tests/data/test_trading_calendar.py`` pins the rules on hand-checked dates and reproduces the
cached ETF panel's 2006-2026 session calendar exactly.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

MIN_YEAR = 1998                       # MLK Day joined the NYSE calendar in 1998
NYSE_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = dt.time(16, 0)        # early closes (13:00) only make this conservative

SPECIAL_CLOSURES: frozenset[dt.date] = frozenset(dt.date.fromisoformat(d) for d in (
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",   # September 11 attacks
    "2004-06-11",                                             # President Reagan, day of mourning
    "2007-01-02",                                             # President Ford, day of mourning
    "2012-10-29", "2012-10-30",                               # Hurricane Sandy
    "2018-12-05",                                             # President G.H.W. Bush, day of mourning
    "2025-01-09",                                             # President Carter, day of mourning
))


def _easter(year: int) -> dt.date:
    """Western (Gregorian) Easter Sunday: the anonymous Meeus/Jones/Butcher algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th ``weekday`` (Mon=0) of the month; n = -1 for the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[dt.date]:
    """NYSE full-day closures (rules plus dated special closures) falling in ``year``."""
    if year < MIN_YEAR:
        raise ValueError(f"NYSE calendar rules are only encoded from {MIN_YEAR}, got {year}")
    days = set()
    new_year = dt.date(year, 1, 1)
    if new_year.weekday() == 6:
        days.add(new_year + dt.timedelta(days=1))
    elif new_year.weekday() != 5:
        days.add(new_year)
    days.add(_nth_weekday(year, 1, 0, 3))                   # MLK Day
    days.add(_nth_weekday(year, 2, 0, 3))                   # Washington's Birthday
    days.add(_easter(year) - dt.timedelta(days=2))          # Good Friday
    days.add(_nth_weekday(year, 5, 0, -1))                  # Memorial Day
    if year >= 2022:
        days.add(_observed(dt.date(year, 6, 19)))           # Juneteenth
    days.add(_observed(dt.date(year, 7, 4)))                # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))                   # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))                  # Thanksgiving
    days.add(_observed(dt.date(year, 12, 25)))              # Christmas
    days |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return frozenset(days)


def _as_date(day) -> dt.date:
    return pd.Timestamp(day).date()


def is_session(day) -> bool:
    """True if NYSE holds a regular trading session on ``day``."""
    d = _as_date(day)
    return d.weekday() < 5 and d not in holidays(d.year)


def sessions(start, end) -> pd.DatetimeIndex:
    """Every session in ``[start, end]`` as tz-naive midnight timestamps (the panel's index
    convention)."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return pd.DatetimeIndex([d for d in days if is_session(d)])


def next_sessions(day, n: int) -> pd.DatetimeIndex:
    """The ``n`` sessions strictly after ``day``."""
    out, d = [], pd.Timestamp(day).normalize()
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if is_session(d):
            out.append(d)
    return pd.DatetimeIndex(out)


def previous_session(day) -> pd.Timestamp:
    """The last session strictly before ``day``."""
    d = pd.Timestamp(day).normalize()
    while True:
        d -= pd.Timedelta(days=1)
        if is_session(d):
            return d


def is_month_end(day) -> bool:
    """True if session ``day`` is the last session of its calendar month: the rebalance
    session of the monthly book, known ex ante."""
    if not is_session(day):
        raise ValueError(f"{pd.Timestamp(day).date()} is not an NYSE session")
    return next_sessions(day, 1)[0].month != pd.Timestamp(day).month


def last_complete_session(now: dt.datetime, *, settle_minutes: int) -> pd.Timestamp:
    """The latest session whose regular close plus ``settle_minutes`` is at or before ``now``
    (timezone-aware). This is the as-of session a daily run may act on; any bar the vendor
    returns for a later date is an in-progress bar and must be dropped. Using the 16:00
    close on early-close days only delays the cutoff, never advances it."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(NYSE_TZ)
    today = pd.Timestamp(local.date())
    cutoff = dt.datetime.combine(local.date(), REGULAR_CLOSE, NYSE_TZ) + dt.timedelta(
        minutes=int(settle_minutes))
    if is_session(today) and local >= cutoff:
        return today
    return previous_session(today)

"""The ex-ante NYSE calendar the forward runner schedules month-end fills from (Tier-2 X2).

A wrong session breaks the runner in one of two ways. A missed closure makes it expect a bar
that never prints, so the freshness gate fails the run. A missed month-end fills the rebalance
on the wrong close. So the rules are pinned on hand-checked dates (each rule and each dated
special closure), and on the cached ETF panel: every 2006-2026 session is reproduced exactly.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from sharpen.data import trading_calendar as tc

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("day", [
    "2024-03-29",                                  # Good Friday
    "2022-06-20",                                  # Juneteenth (Sun 19th -> Mon)
    "2023-01-02",                                  # New Year's (Sun -> Mon)
    "2026-07-03",                                  # Independence Day (Sat -> Fri)
    "2027-12-24",                                  # Christmas (Sat -> Fri)
    "2026-11-26",                                  # Thanksgiving, 4th Thursday
    "2026-01-19", "2026-02-16", "2026-05-25", "2026-09-07",   # MLK, Presidents, Memorial, Labor
    "2001-09-11", "2001-09-14", "2004-06-11", "2007-01-02",   # 9/11; Reagan; Ford
    "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",   # Sandy; G.H.W. Bush; Carter
])
def test_closures(day):
    assert not tc.is_session(day)


@pytest.mark.parametrize("day", [
    "2021-12-31",       # New Year's on a Saturday: NYSE does NOT close the Friday before
    "2010-12-31",
    "2021-06-18",       # Juneteenth only from 2022
    "2026-07-02",
    "2024-03-28",
])
def test_open_days(day):
    assert tc.is_session(day)


def test_weekends_are_not_sessions():
    assert not tc.is_session("2026-08-01") and not tc.is_session("2026-08-02")


@pytest.mark.parametrize("year, easter", [
    (2008, "2008-03-23"), (2011, "2011-04-24"), (2024, "2024-03-31"), (2025, "2025-04-20"),
    (2026, "2026-04-05"), (2038, "2038-04-25"),
])
def test_easter(year, easter):
    assert tc._easter(year) == dt.date.fromisoformat(easter)


def test_month_end_follows_the_calendar_not_the_weekday():
    assert tc.is_month_end("2026-07-31")
    assert not tc.is_month_end("2026-07-30")
    assert tc.is_month_end("2016-02-29")
    # March 2024 ends on Thursday the 28th: Good Friday closes the 29th. A weekday calendar
    # would schedule the rebalance for a session that never trades.
    assert tc.is_month_end("2024-03-28")
    with pytest.raises(ValueError, match="not an NYSE session"):
        tc.is_month_end("2024-03-29")


def test_next_and_previous_sessions_skip_closures():
    assert list(tc.next_sessions("2024-03-28", 2).date) == [dt.date(2024, 4, 1), dt.date(2024, 4, 2)]
    assert tc.previous_session("2024-04-01") == pd.Timestamp("2024-03-28")
    assert list(tc.sessions("2026-07-02", "2026-07-07").date) == [
        dt.date(2026, 7, 2), dt.date(2026, 7, 6), dt.date(2026, 7, 7)]


def test_last_complete_session_waits_for_the_close_plus_settle():
    ny = tc.NYSE_TZ
    at = lambda y, m, d, h, mi: dt.datetime(y, m, d, h, mi, tzinfo=ny)  # noqa: E731
    assert tc.last_complete_session(at(2026, 7, 31, 15, 59), settle_minutes=30) == pd.Timestamp("2026-07-30")
    assert tc.last_complete_session(at(2026, 7, 31, 16, 29), settle_minutes=30) == pd.Timestamp("2026-07-30")
    assert tc.last_complete_session(at(2026, 7, 31, 16, 30), settle_minutes=30) == pd.Timestamp("2026-07-31")
    assert tc.last_complete_session(at(2026, 8, 1, 12, 0), settle_minutes=30) == pd.Timestamp("2026-07-31")
    assert tc.last_complete_session(at(2026, 8, 3, 9, 0), settle_minutes=30) == pd.Timestamp("2026-07-31")
    # the same instant expressed in UTC resolves identically
    utc = at(2026, 7, 31, 16, 45).astimezone(dt.timezone.utc)
    assert tc.last_complete_session(utc, settle_minutes=30) == pd.Timestamp("2026-07-31")


def test_fails_closed_outside_its_rules():
    with pytest.raises(ValueError, match="only encoded"):
        tc.is_session("1997-07-03")
    with pytest.raises(ValueError, match="timezone-aware"):
        tc.last_complete_session(dt.datetime(2026, 7, 31, 17, 0), settle_minutes=30)



"""Unit tests for CFDBarClock — XAUUSD market hours schedule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from finrl_pro_ds.cfd.live.cfd_bar_clock import CFDBarClock


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def utc(year, month, day, hour, minute=0) -> datetime:
    """Create a UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------
# is_market_open tests
# ---------------------------------------------------------------

def test_weekday_during_session():
    """Monday 14:00 UTC → market open."""
    # 2026-03-30 is a Monday
    dt = utc(2026, 3, 30, 14)
    assert CFDBarClock.is_market_open(dt) is True


def test_weekday_morning():
    """Tuesday 08:00 UTC → market open."""
    dt = utc(2026, 3, 31, 8)
    assert CFDBarClock.is_market_open(dt) is True


def test_daily_break_start():
    """Monday 21:00 UTC → closed (daily break starts)."""
    dt = utc(2026, 3, 30, 21)
    assert CFDBarClock.is_market_open(dt) is False


def test_daily_break_mid():
    """Monday 21:30 UTC → closed."""
    dt = utc(2026, 3, 30, 21, 30)
    assert CFDBarClock.is_market_open(dt) is False


def test_daily_break_end():
    """Monday 22:00 UTC → market open (break ended)."""
    dt = utc(2026, 3, 30, 22)
    assert CFDBarClock.is_market_open(dt) is True


def test_friday_before_close():
    """Friday 20:00 UTC → market open."""
    # 2026-04-03 is a Friday
    dt = utc(2026, 4, 3, 20)
    assert CFDBarClock.is_market_open(dt) is True


def test_friday_close():
    """Friday 22:00 UTC → closed."""
    dt = utc(2026, 4, 3, 22)
    assert CFDBarClock.is_market_open(dt) is False


def test_friday_after_close():
    """Friday 23:00 UTC → closed."""
    dt = utc(2026, 4, 3, 23)
    assert CFDBarClock.is_market_open(dt) is False


def test_saturday():
    """Saturday any time → closed."""
    # 2026-04-04 is a Saturday
    for hour in [0, 6, 12, 18, 23]:
        dt = utc(2026, 4, 4, hour)
        assert CFDBarClock.is_market_open(dt) is False, f"Saturday {hour}:00 should be closed"


def test_sunday_before_open():
    """Sunday 21:00 UTC → closed (pre-open)."""
    # 2026-04-05 is a Sunday
    dt = utc(2026, 4, 5, 21)
    assert CFDBarClock.is_market_open(dt) is False


def test_sunday_at_open():
    """Sunday 22:00 UTC → market open."""
    dt = utc(2026, 4, 5, 22)
    assert CFDBarClock.is_market_open(dt) is True


def test_sunday_late():
    """Sunday 23:00 UTC → market open."""
    dt = utc(2026, 4, 5, 23)
    assert CFDBarClock.is_market_open(dt) is True


# ---------------------------------------------------------------
# next_market_open tests
# ---------------------------------------------------------------

def test_next_open_from_daily_break():
    """21:30 Mon → 22:00 Mon."""
    dt = utc(2026, 3, 30, 21, 30)
    expected = utc(2026, 3, 30, 22)
    assert CFDBarClock.next_market_open(dt) == expected


def test_next_open_from_friday_break():
    """21:30 Fri → Sunday 22:00 (Friday break = weekend start)."""
    dt = utc(2026, 4, 3, 21, 30)
    expected = utc(2026, 4, 5, 22)
    assert CFDBarClock.next_market_open(dt) == expected


def test_next_open_from_friday_close():
    """Friday 22:30 → Sunday 22:00."""
    dt = utc(2026, 4, 3, 22, 30)
    expected = utc(2026, 4, 5, 22)
    assert CFDBarClock.next_market_open(dt) == expected


def test_next_open_from_saturday():
    """Saturday 12:00 → Sunday 22:00."""
    dt = utc(2026, 4, 4, 12)
    expected = utc(2026, 4, 5, 22)
    assert CFDBarClock.next_market_open(dt) == expected


def test_next_open_from_sunday_morning():
    """Sunday 08:00 → Sunday 22:00."""
    dt = utc(2026, 4, 5, 8)
    expected = utc(2026, 4, 5, 22)
    assert CFDBarClock.next_market_open(dt) == expected


# ---------------------------------------------------------------
# Property delegation
# ---------------------------------------------------------------

def test_interval_property():
    clock = CFDBarClock(bar_interval_minutes=15)
    assert clock.interval == 15


def test_interval_property_1h():
    clock = CFDBarClock(bar_interval_minutes=60)
    assert clock.interval == 60


def test_bar_interval_timedelta():
    clock = CFDBarClock(bar_interval_minutes=15)
    assert clock.get_bar_interval_timedelta() == timedelta(minutes=15)


def test_initial_bars_processed():
    clock = CFDBarClock()
    assert clock.bars_processed == 0


def test_initial_skipped_bars():
    clock = CFDBarClock()
    assert clock.skipped_bars == 0

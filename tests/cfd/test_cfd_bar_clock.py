"""Unit tests for CFDBarClock — XAUUSD market hours schedule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sharpen.cfd.live.cfd_bar_clock import CFDBarClock


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


# ---------------------------------------------------------------
# Heartbeat callback (S490) — mirror CMEBarClock so cTrader XAUUSD
# stays Docker-healthy through daily rollover + weekend gates.
# ---------------------------------------------------------------

def test_heartbeat_callback_default_none():
    """Default construction leaves the callback slot empty."""
    clock = CFDBarClock()
    assert clock._heartbeat_callback is None


def test_heartbeat_callback_assignable():
    """Engine wires the callback post-construction via hasattr check."""
    clock = CFDBarClock()
    calls = []
    clock._heartbeat_callback = lambda phase: calls.append(phase)
    clock._heartbeat_callback("market_closed")
    assert calls == ["market_closed"]


def test_heartbeat_interval_constant():
    """Heartbeat cadence matches CMEBarClock (120s)."""
    assert CFDBarClock._HEARTBEAT_INTERVAL == 120


def test_heartbeat_fires_during_market_closed_sleep(monkeypatch):
    """S490 F3: chunked sleep fires heartbeat once per chunk (incl. final).

    Regression guard: before S490 the CFDBarClock slept in a single
    `await asyncio.sleep()`, so the health file staled during the 21-22 UTC
    daily rollover and Docker marked the container UNHEALTHY → watchdog
    auto-restart. After S490 F4, callback fires after every chunk so the
    file stays fresh across the market-closed → market-open transition.
    """
    import asyncio

    clock = CFDBarClock()
    calls = []
    clock._heartbeat_callback = lambda phase: calls.append(phase)

    # Flip market from closed → open so wait_for_next_bar exits the sleep
    # branch and delegates to the inner clock after one pass.
    state = {"calls": 0}

    def mock_is_open(_dt):
        state["calls"] += 1
        return state["calls"] > 1  # first call = closed, rest = open

    clock.is_market_open = mock_is_open

    # Return a 500s sleep window regardless of real wall clock.
    clock.next_market_open = lambda now: now + timedelta(seconds=500)

    # Stub inner BarClock so it returns a bar_time immediately (no real sleep).
    bar_time = utc(2026, 4, 22, 10, 0)

    async def fake_inner_wait():
        return bar_time

    clock._inner.wait_for_next_bar = fake_inner_wait

    # Patch asyncio.sleep in the cfd_bar_clock module to a no-op coroutine
    # so the chunked loop completes instantly.
    import sharpen.cfd.live.cfd_bar_clock as mod

    async def fake_sleep(_secs):
        pass

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    async def run():
        return await clock.wait_for_next_bar()

    result = asyncio.run(run())
    assert result == bar_time

    # sleep_secs = 500, actual_sleep = 502. Chunks: 120,120,120,120,22 → 5.
    # F4 guarantees the callback fires once per chunk, including the final.
    assert len(calls) == 5, f"expected 5 heartbeat calls, got {len(calls)}"
    assert all(phase == "market_closed" for phase in calls)


def test_heartbeat_silent_when_callback_not_wired(monkeypatch):
    """When _heartbeat_callback is None, chunked sleep must not crash."""
    import asyncio

    clock = CFDBarClock()
    assert clock._heartbeat_callback is None

    state = {"calls": 0}
    clock.is_market_open = lambda _dt: (state.__setitem__("calls", state["calls"] + 1) or state["calls"] > 1)
    clock.next_market_open = lambda now: now + timedelta(seconds=500)

    bar_time = utc(2026, 4, 22, 10, 0)

    async def fake_inner_wait():
        return bar_time

    clock._inner.wait_for_next_bar = fake_inner_wait

    import sharpen.cfd.live.cfd_bar_clock as mod

    async def fake_sleep(_secs):
        pass

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    async def run():
        return await clock.wait_for_next_bar()

    # Should complete without raising even though callback is None.
    assert asyncio.run(run()) == bar_time

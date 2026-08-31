"""Daily-loss trigger tests for LiveTradingEngine._check_daily_loss.

Covers the S448 Defect 3 fix (ship with next ctrader image rebuild):

    Median-of-3 PV smoothing lags raw PV by up to 2 bars on fast adverse
    moves. On 2026-04-13 XAUUSD rallied 4707→4716 while the paper container
    was short; smoothed_return sat at -9.37% when raw was already past
    FTMO's 10% daily-loss line, final DD -11.74%.

Fix (Option A): trip when the raw reading has breached the limit on ≥2
consecutive bars, even if the median has not caught up. A single spike
(API hiccup pattern) still cannot trip the limit on its own.
"""

from __future__ import annotations

import asyncio
import collections
from datetime import datetime, timezone

from sharpen.crypto.live.live_engine import LiveTradingEngine


def _make_engine(limit: float = 0.04, start_value: float = 10_000.0) -> LiveTradingEngine:
    """Build a LiveTradingEngine bypassing __init__ — only the fields used
    by _check_daily_loss are set. Side-effect methods are stubbed so the
    test can observe whether a trip fired."""
    eng = object.__new__(LiveTradingEngine)
    eng._portfolio_value = start_value
    eng._daily_start_value = start_value
    eng._last_daily_reset_date = None
    eng._max_daily_loss_pct = limit
    eng._pv_buffer = collections.deque(maxlen=3)
    eng._consecutive_raw_breach_count = 0

    eng._trip_fired = False
    eng._halt_detail = None

    async def _fake_flatten():
        eng._trip_fired = True

    def _fake_write_halt_state(reason, detail, now_utc):
        eng._halt_detail = (reason, detail)

    def _fake_request_stop(reason):
        pass

    eng._emergency_flatten = _fake_flatten
    eng._write_halt_state = _fake_write_halt_state
    eng._request_stop = _fake_request_stop
    return eng


async def _step(eng: LiveTradingEngine, pv: float, bar_time: datetime) -> None:
    eng._portfolio_value = pv
    await eng._check_daily_loss(bar_time)


def test_single_raw_spike_does_not_trip():
    """API-hiccup pattern: one anomalous low reading then recovery. Must NOT trip."""
    async def run():
        eng = _make_engine(limit=0.04)
        t = datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc)

        await _step(eng, 10_000.0, t)                                # steady
        await _step(eng, 9_500.0, t.replace(minute=15))              # -5% spike
        await _step(eng, 10_000.0, t.replace(minute=30))             # recovered

        assert eng._trip_fired is False
        assert eng._consecutive_raw_breach_count == 0
    asyncio.run(run())


def test_two_consecutive_raw_breaches_trip():
    """S448 Defect 3 regression: two consecutive raw breaches trip."""
    async def run():
        eng = _make_engine(limit=0.04)
        t = datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc)

        await _step(eng, 10_000.0, t)
        assert eng._trip_fired is False

        await _step(eng, 9_500.0, t.replace(minute=15))             # raw -5%
        assert eng._trip_fired is False
        assert eng._consecutive_raw_breach_count == 1

        await _step(eng, 9_450.0, t.replace(minute=30))             # raw -5.5%
        assert eng._trip_fired is True
        assert eng._halt_detail is not None
        reason, _ = eng._halt_detail
        assert reason == "daily_loss_limit"
    asyncio.run(run())


def test_raw_persisted_trip_beats_median_lag():
    """Median lags: two consecutive raw breaches while median is still
    clean must trip via the raw_persisted branch (the XAUUSD lag case)."""
    async def run():
        eng = _make_engine(limit=0.04)
        t = datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc)

        # Warm buffer so median stays high when breach starts.
        await _step(eng, 10_000.0, t)
        await _step(eng, 10_000.0, t.replace(minute=15))

        # Breach #1: raw -5%, median of [10000, 10000, 9500] = 10000 → smoothed 0%.
        await _step(eng, 9_500.0, t.replace(minute=30))
        assert eng._trip_fired is False
        assert eng._consecutive_raw_breach_count == 1

        # Breach #2: raw -5%, median of [10000, 9500, 9500] = 9500 → smoothed -5%.
        # Must trip (either via smoothed or raw_persisted — both acceptable).
        await _step(eng, 9_500.0, t.replace(minute=45))
        assert eng._trip_fired is True
    asyncio.run(run())


def test_counter_resets_on_non_breach_bar():
    """Non-breach bar between breaches must reset the counter to 0, so a
    later single breach doesn't trip via raw_persisted."""
    async def run():
        eng = _make_engine(limit=0.04)
        t = datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc)

        await _step(eng, 10_000.0, t)
        await _step(eng, 9_500.0, t.replace(minute=15))     # breach, counter=1
        assert eng._consecutive_raw_breach_count == 1
        assert eng._trip_fired is False
        await _step(eng, 9_800.0, t.replace(minute=30))     # -2%, reset
        assert eng._consecutive_raw_breach_count == 0
        assert eng._trip_fired is False
    asyncio.run(run())


def test_counter_resets_on_new_utc_day():
    """Daily reset must clear the consecutive counter, not leave it armed."""
    async def run():
        eng = _make_engine(limit=0.04)
        day1 = datetime(2026, 4, 13, 23, 45, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)

        await _step(eng, 10_000.0, day1)
        await _step(eng, 9_500.0, day1.replace(minute=50))  # breach, counter=1
        assert eng._consecutive_raw_breach_count == 1

        # New UTC day — daily_start_value resets, counter resets.
        await _step(eng, 9_500.0, day2)
        assert eng._consecutive_raw_breach_count == 0
        assert eng._daily_start_value == 9_500.0
        assert eng._trip_fired is False
    asyncio.run(run())

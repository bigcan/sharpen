"""Unit tests for OandaDataLoader — no live connection required.

Covers the two non-trivial bits:
  - M1 -> M3 resample logic (ADR-3)
  - Incomplete-candle filtering
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pandas as pd
import pytest

from finrl_pro_ds.cfd.data.oanda_data_loader import (
    OandaDataLoader,
    _resample_ohlcv,
    _parse_datetime,
    _to_oanda_dt,
)


def _make_loader():
    broker = MagicMock()
    broker._account_id = "101-001-1234567-001"
    broker._session = MagicMock(spec=httpx.AsyncClient)
    broker._session.get = AsyncMock()
    return OandaDataLoader(broker=broker, instrument="EUR_USD")


def _mock_candles_response(candles: list[dict]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"candles": candles}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def test_to_oanda_dt_naive_assumes_utc():
    dt = datetime(2026, 5, 26, 6, 28, 30)
    out = _to_oanda_dt(dt)
    assert out.endswith("Z")
    assert "2026-05-26T06:28:30" in out


def test_to_oanda_dt_aware_passes_through():
    dt = datetime(2026, 5, 26, 6, 28, 30, tzinfo=timezone.utc)
    out = _to_oanda_dt(dt)
    assert out.endswith("Z")


def test_parse_datetime_z_suffix():
    dt = _parse_datetime("2026-05-26T06:28:30Z")
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 6


def test_parse_datetime_nanoseconds_trimmed():
    """OANDA emits 9-digit fractional seconds; fromisoformat handles 6."""
    dt = _parse_datetime("2026-05-26T06:28:30.030305889Z")
    assert dt.tzinfo == timezone.utc
    assert dt.minute == 28


def test_parse_datetime_datetime_passthrough():
    dt_in = datetime(2026, 5, 26, 6, 28, tzinfo=timezone.utc)
    dt_out = _parse_datetime(dt_in)
    assert dt_out == dt_in


# ---------------------------------------------------------------------------
# M1 -> M3 resample (ADR-3)
# ---------------------------------------------------------------------------


def _m1_df(n_bars: int, start: datetime, prices: list[float] = None) -> pd.DataFrame:
    """Construct synthetic M1 OHLCV with monotone-increasing OHLC if no prices given."""
    rows = []
    for i in range(n_bars):
        ts = start + timedelta(minutes=i)
        if prices is not None:
            p = prices[i]
        else:
            p = 1.16 + i * 0.0001
        rows.append({
            "timestamp": ts,
            "ticker": "EURUSD",
            "open": p,
            "high": p + 0.0001,
            "low": p - 0.0001,
            "close": p + 0.00005,
            "volume": 100,
        })
    return pd.DataFrame(rows)


def test_resample_m1_to_m3_basic():
    start = datetime(2026, 5, 26, 6, 0, tzinfo=timezone.utc)
    df = _m1_df(6, start)  # 6 M1 bars -> 2 M3 bars

    out = _resample_ohlcv(df, "EURUSD", "3min")
    assert len(out) == 2

    # Bar 0 covers M1 bars [0, 1, 2]:
    #   open = first M1 open  = 1.16
    #   high = max of M1 highs = 1.16 + 2*0.0001 + 0.0001 = 1.1603
    #   low  = min of M1 lows  = 1.16 - 0.0001 = 1.1599
    #   close = last M1 close = 1.16 + 2*0.0001 + 0.00005 = 1.16025
    bar0 = out.iloc[0]
    assert bar0["open"] == pytest.approx(1.16)
    assert bar0["high"] == pytest.approx(1.1603)
    assert bar0["low"] == pytest.approx(1.1599)
    assert bar0["close"] == pytest.approx(1.16025)
    assert bar0["volume"] == 300  # sum of 3 M1 volumes
    # Label-left convention: M3 bar 0 timestamp = M1 bar 0 timestamp
    assert pd.Timestamp(bar0["timestamp"]) == pd.Timestamp(start)


def test_resample_m1_to_m3_label_left_closed_left():
    """Verify that the bar at 06:00 contains M1 bars 06:00, 06:01, 06:02."""
    start = datetime(2026, 5, 26, 6, 0, tzinfo=timezone.utc)
    df = _m1_df(9, start)
    out = _resample_ohlcv(df, "EURUSD", "3min")
    expected_starts = [
        start,
        start + timedelta(minutes=3),
        start + timedelta(minutes=6),
    ]
    assert len(out) == 3
    for i, expected in enumerate(expected_starts):
        assert pd.Timestamp(out.iloc[i]["timestamp"]) == pd.Timestamp(expected)


def test_resample_m1_to_m3_drops_empty_bars():
    """Gap in M1 stream -> no M3 bar for that period (NaN open dropped)."""
    start = datetime(2026, 5, 26, 6, 0, tzinfo=timezone.utc)
    df = _m1_df(3, start)  # M1 06:00, 06:01, 06:02
    # Plus 3 M1 bars at 06:09, 06:10, 06:11 (gap at 06:03-06:08)
    later = start + timedelta(minutes=9)
    df2 = _m1_df(3, later)
    df = pd.concat([df, df2]).reset_index(drop=True)
    out = _resample_ohlcv(df, "EURUSD", "3min")
    # Should be 2 bars: one at 06:00, one at 06:09. The empty bars at
    # 06:03 and 06:06 should be dropped.
    assert len(out) == 2
    starts = [pd.Timestamp(out.iloc[i]["timestamp"]) for i in range(len(out))]
    assert pd.Timestamp(start) in starts
    assert pd.Timestamp(later) in starts


def test_resample_preserves_ticker_column():
    start = datetime(2026, 5, 26, 6, 0, tzinfo=timezone.utc)
    df = _m1_df(3, start)
    out = _resample_ohlcv(df, "EURUSD", "3min")
    assert all(out["ticker"] == "EURUSD")


def test_resample_empty_input_returns_empty():
    out = _resample_ohlcv(
        pd.DataFrame(columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"]),
        "EURUSD",
        "3min",
    )
    assert len(out) == 0


# ---------------------------------------------------------------------------
# fetch_ohlcv via mocked session
# ---------------------------------------------------------------------------


def _candle(time_str: str, o: float, h: float, lo: float, c: float, v: int = 100, complete: bool = True):
    return {
        "time": time_str,
        "volume": v,
        "complete": complete,
        "mid": {"o": str(o), "h": str(h), "l": str(lo), "c": str(c)},
    }


def test_fetch_ohlcv_filters_incomplete_candles():
    """A bar with `complete: false` must be excluded."""
    import asyncio

    loader = _make_loader()
    loader._broker._session.get.return_value = _mock_candles_response([
        _candle("2026-05-26T06:00:00Z", 1.160, 1.161, 1.159, 1.1605),
        _candle("2026-05-26T06:01:00Z", 1.1605, 1.162, 1.16, 1.161, complete=False),
    ])

    df = asyncio.run(loader.fetch_ohlcv(
        ["EURUSD"], "2026-05-26T06:00:00Z", "2026-05-26T06:02:00Z", "1m",
    ))
    assert len(df) == 1
    assert df.iloc[0]["close"] == pytest.approx(1.1605)


def test_fetch_ohlcv_m1_native():
    import asyncio

    loader = _make_loader()
    candles = [
        _candle("2026-05-26T06:00:00Z", 1.160, 1.161, 1.159, 1.1605),
        _candle("2026-05-26T06:01:00Z", 1.1605, 1.162, 1.16, 1.161),
        _candle("2026-05-26T06:02:00Z", 1.161, 1.1615, 1.1605, 1.1612),
    ]
    loader._broker._session.get.return_value = _mock_candles_response(candles)

    df = asyncio.run(loader.fetch_ohlcv(
        ["EURUSD"], "2026-05-26T06:00:00Z", "2026-05-26T06:03:00Z", "1m",
    ))
    assert len(df) == 3
    assert list(df.columns) == [
        "timestamp", "ticker", "open", "high", "low", "close", "volume"
    ]
    assert all(df["ticker"] == "EURUSD")

    # Verify granularity sent to OANDA
    call = loader._broker._session.get.call_args
    params = call.kwargs.get("params") or {}
    assert params.get("granularity") == "M1"


def test_fetch_ohlcv_m3_fetches_m1_and_resamples():
    """For timeframe='3m', loader requests M1 from OANDA and resamples to M3."""
    import asyncio

    loader = _make_loader()
    # 6 M1 candles -> 2 M3 bars
    candles = [
        _candle("2026-05-26T06:00:00Z", 1.160, 1.161, 1.159, 1.1605),
        _candle("2026-05-26T06:01:00Z", 1.1605, 1.162, 1.16, 1.161),
        _candle("2026-05-26T06:02:00Z", 1.161, 1.1615, 1.1605, 1.1612),
        _candle("2026-05-26T06:03:00Z", 1.1612, 1.1620, 1.1610, 1.1618),
        _candle("2026-05-26T06:04:00Z", 1.1618, 1.1625, 1.1615, 1.1622),
        _candle("2026-05-26T06:05:00Z", 1.1622, 1.1630, 1.1620, 1.1628),
    ]
    loader._broker._session.get.return_value = _mock_candles_response(candles)

    df = asyncio.run(loader.fetch_ohlcv(
        ["EURUSD"], "2026-05-26T06:00:00Z", "2026-05-26T06:06:00Z", "3m",
    ))

    # Loader should have requested M1 from OANDA (not M3)
    call = loader._broker._session.get.call_args
    params = call.kwargs.get("params") or {}
    assert params.get("granularity") == "M1"

    # Result is 2 M3 bars
    assert len(df) == 2
    # First M3 bar: open=1.160, close of last M1 in window=1.1612, high=max, low=min
    bar0 = df.iloc[0]
    assert bar0["open"] == pytest.approx(1.160)
    assert bar0["high"] == pytest.approx(1.1620)
    assert bar0["low"] == pytest.approx(1.159)
    assert bar0["close"] == pytest.approx(1.1612)


def test_fetch_ohlcv_unsupported_timeframe_raises():
    import asyncio
    loader = _make_loader()
    with pytest.raises(ValueError, match="Unsupported"):
        asyncio.run(loader.fetch_ohlcv(["EURUSD"], "x", "y", "11s"))


def test_fetch_ohlcv_empty_window_returns_empty():
    import asyncio
    loader = _make_loader()
    df = asyncio.run(loader.fetch_ohlcv(
        ["EURUSD"], "2026-05-26T06:00:00Z", "2026-05-26T05:00:00Z", "1m",
    ))
    assert len(df) == 0


def test_fetch_ohlcv_clips_future_end_dt():
    """OANDA rejects `to > now` with 'Time is in the future' HTTP 400.

    Live engine's bar loop routinely passes `end = next_bar_close` which is a
    few seconds in the future when the engine fires (e.g. for 3-min bar
    closing 07:30:00 UTC, engine fires at 07:30:10 with end=07:31:00).
    Loader must clip `end` to `now - small_buffer` so the request is valid.

    Regression: deployed sg1-eurusd-oanda 2026-05-26 hit HTTP 400 on every
    bar fetch because of this; loader returned empty df + engine logged
    'No new bars, skipping' indefinitely.
    """
    import asyncio
    from datetime import datetime, timezone, timedelta

    loader = _make_loader()
    # Capture the params actually sent to OANDA
    sent_params: list[dict] = []

    async def fake_get(url, params=None, timeout=None):
        sent_params.append(params or {})
        return _mock_candles_response([])

    loader._broker._session.get = fake_get

    # Caller asks for a window ending 5 minutes in the future
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=10)).isoformat()
    end = (now + timedelta(minutes=5)).isoformat()

    asyncio.run(loader.fetch_ohlcv(["EURUSD"], start, end, "1m"))

    # Assert the params sent had `to` clipped to <= now (with a 10s slack)
    assert sent_params, "loader made no request"
    to_str = sent_params[-1].get("to", "")
    # to_str format: "2026-05-26T07:30:00.000000Z"
    to_dt = _parse_datetime(to_str)
    # Should be at most 10s after `now` (we allow 10s to give the
    # _parse_datetime + clock-skew slack room). The clip target is `now - 5s`,
    # so a strict assertion is `to_dt <= now`.
    assert to_dt <= now + timedelta(seconds=2), (
        f"OandaDataLoader did NOT clip future end_dt: to={to_dt} > now={now}"
    )

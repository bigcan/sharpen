"""OANDA v20 historical OHLCV loader — REST /candles + M1->M3 resample.

Implements the same async fetch_ohlcv() interface as
``CTraderDataLoader``, ``IBDataLoader``, and ``CryptoLoader`` so
``LiveObsBuilder.bootstrap()`` and ``LiveTradingEngine._fetch_new_bars()``
work without modification.

OANDA API characteristics:
    - ~5000 candles per request
    - Practice rate limit ~30 req/s (we pace at 0.2s = 5 req/s)
    - No native M3 granularity (only M1/M2/M4/M5/M10/M15/M30/H1/...);
      ADR-3 resamples from M1 on the client side for SG-1's 3-min bars
    - `candle.complete: false` means the bar is still forming — drop it

See `.agent/artifacts/oanda_broker_architecture.md` for the full design.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# Native OANDA granularities (a subset; add as needed).
_GRANULARITY_MAP = {
    "5s": "S5",
    "10s": "S10",
    "15s": "S15",
    "30s": "S30",
    "1m": "M1",
    "2m": "M2",
    "4m": "M4",
    "5m": "M5",
    "10m": "M10",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "2h": "H2",
    "3h": "H3",
    "4h": "H4",
    "6h": "H6",
    "8h": "H8",
    "12h": "H12",
    "1d": "D",
    "1w": "W",
    "1mo": "M",
}

# Timeframes that we synthesize from M1 because OANDA has no native bin.
# Values are the pandas resample rule (label-left, closed-left). For 3-min
# bars the M1 fetch is 3× the wire volume but stays well within rate limits
# and matches how the cTrader loader currently produces SG-1 M3 bars.
_RESAMPLE_FROM_M1 = {
    "3m": "3min",
    "7m": "7min",  # speculative — not used today; safe to leave
}

# Max candles per request (OANDA documented cap is 5000).
_MAX_CANDLES_PER_REQUEST = 5000

# Minimum interval between requests (s). 0.2s = 5 req/s; practice cap is 30/s.
_MIN_REQUEST_INTERVAL = 0.2


class OandaDataLoader:
    """Fetches OHLCV bars from OANDA v20 REST API.

    Compatible with the loader interface expected by ``LiveObsBuilder``
    and ``LiveTradingEngine``. Single-asset by default (the engine's
    bar loop passes a single-element list).
    """

    def __init__(
        self,
        broker,
        account_id: str | None = None,
        instrument: str = "EUR_USD",
        price: str = "M",
    ):
        """
        Args:
            broker: OandaBroker instance (provides ``_session`` + token).
            account_id: OANDA account ID. Falls back to broker._account_id.
            instrument: OANDA symbol (e.g. "EUR_USD"). Used when the engine
                passes its config-side symbol (e.g. "EURUSD") — mapped via
                broker._oanda_symbol.
            price: Pricing component to fetch — "M" (mid), "B" (bid),
                "A" (ask). Mid is the standard for training compatibility.
        """
        self._broker = broker
        self._account_id = account_id or broker._account_id
        self._instrument = instrument
        self._price = price
        self._last_request_time: float = 0.0
        # Compatibility with cTrader's loader attribute names — referenced
        # by some callers (defensive; LiveTradingEngine does not currently
        # read these but the cTrader loader provided them).
        self._exchange = None

    @property
    def _session(self):
        """Dynamically fetch active httpx session from broker."""
        return self._broker._session

    async def fetch_ohlcv(
        self,
        assets: list[str],
        start: str,
        end: str,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single asset.

        Args:
            assets: List of asset symbols (only first used).
            start: ISO datetime string (inclusive).
            end: ISO datetime string (exclusive).
            timeframe: Bar size — see _GRANULARITY_MAP / _RESAMPLE_FROM_M1.

        Returns:
            DataFrame with columns:
                [timestamp, ticker, open, high, low, close, volume]
            Sorted by timestamp ascending. Incomplete candles dropped.
        """
        asset = assets[0] if assets else "EURUSD"

        # Map config-side symbol to OANDA convention (mirrors broker).
        from sharpen.cfd.execution.oanda_broker import _SYMBOL_MAP
        oanda_symbol = _SYMBOL_MAP.get(asset, self._instrument)

        # Pick native granularity if available, else resample-from-M1.
        if timeframe in _GRANULARITY_MAP:
            granularity = _GRANULARITY_MAP[timeframe]
            resample_rule = None
        elif timeframe in _RESAMPLE_FROM_M1:
            granularity = "M1"
            resample_rule = _RESAMPLE_FROM_M1[timeframe]
            logger.debug(
                "OANDA: synthesizing %s bars by resampling M1 (rule=%s)",
                timeframe, resample_rule,
            )
        else:
            raise ValueError(
                f"Unsupported OANDA timeframe: {timeframe}. "
                f"Native: {sorted(_GRANULARITY_MAP)}. "
                f"Synthesized: {sorted(_RESAMPLE_FROM_M1)}."
            )

        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)

        # OANDA strictly rejects `to` in the future ("Invalid value specified
        # for 'to'. Time is in the future"). The live engine's bar loop
        # routinely passes `end = next_bar_close` which is a few seconds in
        # the future right after the engine fires. Clip to `now - 5s` so the
        # current bar has time to land in OANDA's candles endpoint as complete.
        # The data loader still respects the caller's `start`; only `end` is
        # trimmed. The incomplete-candle filter (`complete: true` check) is
        # the secondary guard for the very-recent edge.
        now_minus_buffer = datetime.now(timezone.utc) - timedelta(seconds=5)
        if end_dt > now_minus_buffer:
            end_dt = now_minus_buffer

        if start_dt >= end_dt:
            return _empty_ohlcv_df()

        # Chunk window so each request stays under _MAX_CANDLES_PER_REQUEST.
        chunk_delta = _chunk_delta_for(granularity)
        rows: list[dict] = []
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + chunk_delta, end_dt)
            try:
                chunk = await self._fetch_chunk(
                    oanda_symbol, granularity, cursor, chunk_end,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "OANDA fetch_ohlcv chunk failed [%s -> %s]: %s",
                    cursor.isoformat(), chunk_end.isoformat(), exc,
                )
                # Don't fast-fail the whole bootstrap — gap will surface
                # downstream as warmup-quality < 1.0 and trigger the
                # engine's bar-skip path.
                cursor = chunk_end
                continue
            rows.extend(chunk)
            cursor = chunk_end

        if not rows:
            return _empty_ohlcv_df()

        df = pd.DataFrame(rows)
        # Strip out the OANDA fixed asset label; engine wants the
        # config-side `asset` value as ticker.
        df["ticker"] = asset
        df = df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.reset_index(drop=True)

        if resample_rule is not None:
            df = _resample_ohlcv(df, asset, resample_rule)

        return df

    async def _fetch_chunk(
        self,
        oanda_symbol: str,
        granularity: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[dict]:
        """One /candles GET. Returns parsed rows (excludes incomplete bars)."""
        await self._enforce_pacing()

        if self._session is None:
            raise RuntimeError(
                "OandaDataLoader: broker session is None — broker not connected"
            )

        params = {
            "granularity": granularity,
            "from": _to_oanda_dt(start_dt),
            "to": _to_oanda_dt(end_dt),
            "price": self._price,
            "smooth": "false",
            "includeFirst": "true",
        }
        resp = await self._session.get(
            f"/v3/instruments/{oanda_symbol}/candles",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        candles = payload.get("candles", [])

        rows: list[dict] = []
        for c in candles:
            if not c.get("complete", False):
                continue
            try:
                mid = c.get(self._price.lower()) or c.get("mid") or {}
                ts = _parse_datetime(c["time"])
                rows.append({
                    "timestamp": ts,
                    "open": float(mid.get("o")),
                    "high": float(mid.get("h")),
                    "low": float(mid.get("l")),
                    "close": float(mid.get("c")),
                    "volume": float(c.get("volume", 0)),
                })
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("OANDA candle parse error: %s c=%r", exc, c)
        return rows

    async def _enforce_pacing(self) -> None:
        """Sleep to respect _MIN_REQUEST_INTERVAL between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    )


def _chunk_delta_for(granularity: str) -> timedelta:
    """Time-delta per chunk that fits ~_MAX_CANDLES_PER_REQUEST bars."""
    # ~4500 bars per chunk to leave headroom.
    n = 4500
    if granularity == "M1":
        return timedelta(minutes=n)
    if granularity == "M2":
        return timedelta(minutes=2 * n)
    if granularity == "M4":
        return timedelta(minutes=4 * n)
    if granularity == "M5":
        return timedelta(minutes=5 * n)
    if granularity == "M10":
        return timedelta(minutes=10 * n)
    if granularity == "M15":
        return timedelta(minutes=15 * n)
    if granularity == "M30":
        return timedelta(minutes=30 * n)
    if granularity == "H1":
        return timedelta(hours=n)
    if granularity == "H4":
        return timedelta(hours=4 * n)
    if granularity == "D":
        return timedelta(days=n)
    # Conservative default
    return timedelta(hours=12)


def _to_oanda_dt(dt: datetime) -> str:
    """Format a datetime as the RFC3339 string OANDA accepts.

    OANDA accepts both 'Z' suffix and explicit +00:00; we use Z for
    compactness. Datetime must be timezone-aware UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.utcoffset() != timedelta(0):
        dt = dt.astimezone(timezone.utc)
    # OANDA accepts nanosecond precision; we provide microsecond which
    # is automatically padded.
    return dt.isoformat().replace("+00:00", "Z")


def _parse_datetime(dt_str) -> datetime:
    """Robust ISO parser accepting str or datetime."""
    if isinstance(dt_str, datetime):
        if dt_str.tzinfo is None:
            return dt_str.replace(tzinfo=timezone.utc)
        return dt_str.astimezone(timezone.utc)
    if isinstance(dt_str, pd.Timestamp):
        ts = dt_str.to_pydatetime()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    s = str(dt_str)
    # OANDA RFC3339 with nanosecond precision: 2026-05-26T06:28:30.030305889Z
    # Python's fromisoformat (3.11+) parses 'Z' suffix natively.
    if s.endswith("Z"):
        # Trim sub-microsecond digits (Python's fromisoformat handles 6 frac
        # digits; OANDA emits 9). Keep 6 if present.
        if "." in s:
            head, frac = s[:-1].split(".", 1)
            frac = frac[:6]
            s = f"{head}.{frac}+00:00"
        else:
            s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resample_ohlcv(df: pd.DataFrame, ticker: str, rule: str) -> pd.DataFrame:
    """Resample 1-min OHLCV to a wider rule (M3, etc.).

    Convention: label-left, closed-left — matches how the cTrader loader
    builds M3 from M1, and matches how training data was binned.
    """
    if df.empty:
        return df
    work = df.set_index("timestamp")
    # Drop the ticker column for the aggregate; reinstate at the end.
    cols = ["open", "high", "low", "close", "volume"]
    agg = work[cols].resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    # Drop bars that contained no M1 data (NaN open).
    agg = agg.dropna(subset=["open"])
    agg["ticker"] = ticker
    agg = agg.reset_index()
    agg = agg[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
    return agg

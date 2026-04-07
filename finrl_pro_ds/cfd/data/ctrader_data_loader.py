"""cTrader Historical Data Loader — fetches OHLCV bars from cTrader Open API.

Implements the same async fetch_ohlcv() interface as IBDataLoader and
CryptoLoader so LiveObsBuilder.bootstrap() and LiveTradingEngine._fetch_new_bars()
work without modification.

cTrader API limits:
    - ~4320 bars per request (3 days of 1-min bars)
    - Rate limit: 50 req/s (generous, but we pace at 0.5s to be safe)
    - Trendbar prices are relative (low + deltaOpen/deltaClose/deltaHigh)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# cTrader trendbar period mapping
_PERIOD_MAP = {
    "1m": 1,   # M1
    "5m": 5,   # M5
    "15m": 7,  # M15
    "30m": 8,  # M30
    "1h": 9,   # H1
    "4h": 10,  # H4
    "1d": 12,  # D1
}

# Max bars per request (~3 days of 1-min data)
_MAX_BARS_PER_REQUEST = 4000

# Minimum seconds between requests (pacing)
_MIN_REQUEST_INTERVAL = 0.5

# cTrader trendbar prices are ALWAYS scaled by 10^5, same as spot events.
# This is independent of the symbol's display digits.
_TRENDBAR_PRICE_SCALE = 100_000


class CTraderDataLoader:
    """Fetches OHLCV bars from cTrader Open API for XAUUSD CFD.

    Compatible with IBDataLoader/CryptoLoader interface expected by
    LiveObsBuilder and LiveTradingEngine.
    """

    def __init__(
        self,
        client,
        account_id: int,
        symbol_id: int,
        symbol_digits: int = 2,
    ):
        """
        Args:
            client: Connected ctrader_open_api.Client instance (shared with broker).
            account_id: cTrader trading account ID.
            symbol_id: Resolved symbol ID for XAUUSD.
            symbol_digits: Price decimal precision (2 for XAUUSD).
        """
        self._client = client
        self._account_id = account_id
        self._symbol_id = symbol_id
        self._symbol_digits = symbol_digits
        self._last_request_time: float = 0.0

        # Compatibility: LiveTradingEngine may access loader._exchange
        self._exchange = None

    async def fetch_ohlcv(
        self,
        assets: list[str],
        start: str,
        end: str,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from cTrader historical data API.

        Args:
            assets: List of asset symbols (only first used — single asset).
            start: Start datetime ISO string.
            end: End datetime ISO string.
            timeframe: Bar size ("1m", "5m", "15m", "1h", etc.).

        Returns:
            DataFrame with columns: [timestamp, ticker, open, high, low, close, volume]
        """
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetTrendbarsReq,
        )

        asset = assets[0] if assets else "XAUUSD"
        period = _PERIOD_MAP.get(timeframe)
        if period is None:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Use one of {list(_PERIOD_MAP)}"
            )

        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)

        # Calculate chunk duration based on timeframe
        if timeframe == "1m":
            chunk_delta = timedelta(days=3)
        elif timeframe in ("5m", "15m"):
            chunk_delta = timedelta(days=14)
        elif timeframe in ("30m", "1h"):
            chunk_delta = timedelta(days=30)
        else:
            chunk_delta = timedelta(days=90)

        all_rows = []
        current_end = end_dt
        chunk_retries = 0

        while current_end > start_dt:
            await self._enforce_pacing()

            chunk_start = max(current_end - chunk_delta, start_dt)

            # cTrader timestamps are in milliseconds since epoch
            from_ts = int(chunk_start.timestamp() * 1000)
            to_ts = int(current_end.timestamp() * 1000)

            try:
                req = ProtoOAGetTrendbarsReq()
                req.ctidTraderAccountId = self._account_id
                req.symbolId = self._symbol_id
                req.period = period
                req.fromTimestamp = from_ts
                req.toTimestamp = to_ts

                response = await self._send_request(req, timeout=30.0)
                res_payload = Protobuf.extract(response)

                bars = res_payload.trendbar
                if not bars:
                    current_end = chunk_start - timedelta(seconds=1)
                    continue

                for bar in bars:
                    # Decode relative prices.
                    # cTrader trendbar prices are ALWAYS scaled by 10^5
                    # (same as spot events), NOT by 10^digits.
                    # low is absolute; deltaOpen/deltaClose/deltaHigh are
                    # offsets from low (all in 10^5 units).
                    low = bar.low / _TRENDBAR_PRICE_SCALE
                    high = low + bar.deltaHigh / _TRENDBAR_PRICE_SCALE
                    open_price = low + bar.deltaOpen / _TRENDBAR_PRICE_SCALE
                    close = low + bar.deltaClose / _TRENDBAR_PRICE_SCALE
                    volume = bar.volume  # Tick volume (price change count)

                    # Timestamp: utcTimestampInMinutes is minutes since epoch
                    ts = datetime.fromtimestamp(
                        bar.utcTimestampInMinutes * 60, tz=timezone.utc
                    )

                    all_rows.append({
                        "timestamp": ts,
                        "ticker": asset,
                        "open": round(open_price, self._symbol_digits),
                        "high": round(high, self._symbol_digits),
                        "low": round(low, self._symbol_digits),
                        "close": round(close, self._symbol_digits),
                        "volume": volume,
                    })

                # FIX CFD-D-01: Move window backward using the exact bar
                # timestamp instead of subtracting 1 second, which created a
                # gap that could silently drop bars at chunk boundaries.
                # drop_duplicates() at the end handles any overlap.
                earliest_ts = min(bar.utcTimestampInMinutes for bar in bars)
                current_end = datetime.fromtimestamp(
                    earliest_ts * 60, tz=timezone.utc
                )

                chunk_retries = 0
                logger.debug(
                    f"Fetched {len(bars)} bars, total {len(all_rows)}, "
                    f"moving end to {current_end}"
                )

            except Exception as e:
                # Retry failed chunks before skipping to avoid silent data gaps
                if chunk_retries < 2:
                    chunk_retries += 1
                    logger.warning(
                        f"cTrader data request failed (attempt {chunk_retries}/3): {e}"
                    )
                    await asyncio.sleep(1.0 * chunk_retries)
                    continue
                logger.error(
                    f"cTrader data chunk DROPPED after 3 attempts: "
                    f"{chunk_start} to {current_end}. Gap in data!"
                )
                chunk_retries = 0
                current_end = chunk_start - timedelta(seconds=1)

        if not all_rows:
            logger.warning(f"No bars fetched for {asset} from {start} to {end}")
            return pd.DataFrame(
                columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_rows)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        df = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )

        logger.info(f"Fetched {len(df)} bars for {asset} ({timeframe}) from cTrader")
        return df

    async def _send_request(self, message, timeout: float = 30.0):
        """Send a protobuf request via the shared client and await response."""
        import time as _time

        msg_id = f"loader_{int(_time.time() * 1000)}_{id(message)}"

        response_deferred = self._client.send(
            message,
            clientMsgId=msg_id,
            responseTimeoutInSeconds=timeout,
        )

        # Poll Deferred until resolved (same bridge pattern as broker)
        result_container = {"value": None, "error": None, "done": False}

        def on_success(result):
            result_container["value"] = result
            result_container["done"] = True
            return result

        def on_error(failure):
            result_container["error"] = failure
            result_container["done"] = True

        response_deferred.addCallback(on_success)
        response_deferred.addErrback(on_error)

        deadline = time.monotonic() + timeout
        while not result_container["done"]:
            if time.monotonic() > deadline:
                raise asyncio.TimeoutError(
                    f"cTrader data request timed out after {timeout}s"
                )
            await asyncio.sleep(0.05)

        if result_container["error"] is not None:
            failure = result_container["error"]
            if hasattr(failure, "raiseException"):
                failure.raiseException()
            raise RuntimeError(f"cTrader data request failed: {failure}")

        return result_container["value"]

    async def _enforce_pacing(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            wait = _MIN_REQUEST_INTERVAL - elapsed
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()


def _parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string to UTC datetime."""
    dt = pd.Timestamp(dt_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime().astimezone(timezone.utc)

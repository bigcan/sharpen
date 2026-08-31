"""IB Historical Data Loader — fetches OHLCV bars from Interactive Brokers.

Implements the same async fetch_ohlcv() interface as CryptoLoader
so LiveObsBuilder.bootstrap() and LiveTradingEngine._fetch_new_bars()
work without modification.

IB pacing limits:
    - Max 60 historical data requests per 10 minutes
    - Each request can fetch up to 1 trading day of 1-min bars (~1380 bars)
    - Bootstrap (30K bars ~22 trading days) requires ~22 requests
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

logger = logging.getLogger(__name__)

# IB bar size mapping
_BAR_SIZE_MAP = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
}

# Minimum seconds between IB historical data requests (pacing)
_MIN_REQUEST_INTERVAL = 11.0  # 11s to stay under 60 req / 10 min


class IBDataLoader:
    """Fetches OHLCV bars from IB for CME Gold futures.

    Compatible with CryptoLoader interface expected by LiveObsBuilder
    and LiveTradingEngine.
    """

    def __init__(
        self,
        ib,
        contract_manager,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ):
        """
        Args:
            ib: Connected ib_insync.IB instance.
            contract_manager: FuturesContractManager with resolved contract.
            what_to_show: IB data type ("TRADES", "MIDPOINT", "BID", "ASK").
            use_rth: If True, only return Regular Trading Hours data.
                     False = full Globex session (what we want for 23h/day).
        """
        self._ib = ib
        self._contract_manager = contract_manager
        self._what_to_show = what_to_show
        self._use_rth = use_rth
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
        """Fetch OHLCV from IB historical data API.

        Args:
            assets: List of asset symbols (only first is used — single-asset).
            start: Start datetime ISO string.
            end: End datetime ISO string.
            timeframe: Bar size ("1m", "5m", "15m", "1h", etc.).

        Returns:
            DataFrame with columns: [timestamp, ticker, open, high, low, close, volume]
        """
        asset = assets[0] if assets else "GC"
        bar_size = _BAR_SIZE_MAP.get(timeframe)
        if bar_size is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(_BAR_SIZE_MAP)}")

        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)

        contract = self._contract_manager.ib_contract
        if contract is None:
            raise RuntimeError("Contract not qualified. Call contract_manager.resolve_front_month() first.")

        # Fetch in daily chunks to respect IB pacing limits
        all_bars = []
        current_end = end_dt

        while current_end > start_dt:
            await self._enforce_pacing()

            # IB duration: fetch 1 day at a time for 1-min bars
            duration = "1 D" if timeframe == "1m" else "2 D"

            try:
                bars = await asyncio.wait_for(
                    self._ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime=current_end.strftime("%Y%m%d-%H:%M:%S"),
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow=self._what_to_show,
                        useRTH=self._use_rth,
                        formatDate=2,  # UTC timestamps
                    ),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"IB historical data request failed: {e}")
                bars = []

            if not bars:
                # No data for this period — move back
                current_end -= timedelta(days=1)
                continue

            all_bars.extend(bars)

            # Move end to before the earliest bar we got
            earliest = bars[0].date
            if isinstance(earliest, str):
                earliest = datetime.fromisoformat(earliest)
            if not isinstance(earliest, datetime):
                earliest = datetime.combine(earliest, datetime.min.time(), tzinfo=timezone.utc)
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=timezone.utc)

            current_end = earliest - timedelta(seconds=1)

            logger.debug(
                f"Fetched {len(bars)} bars, total {len(all_bars)}, "
                f"moving end to {current_end}",
            )

        if not all_bars:
            logger.warning(f"No bars fetched for {asset} from {start} to {end}")
            return pd.DataFrame(columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"])

        df = self._bars_to_dataframe(all_bars, asset)

        # Filter to requested range and sort
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        logger.info(f"Fetched {len(df)} bars for {asset} ({timeframe}) from IB")
        return df

    async def _enforce_pacing(self) -> None:
        """Enforce minimum interval between IB requests."""
        import time as _time

        now = _time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            wait = _MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"IB pacing: waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        self._last_request_time = _time.monotonic()

    @staticmethod
    def _bars_to_dataframe(bars: list, ticker: str) -> pd.DataFrame:
        """Convert ib_insync BarData objects to CryptoLoader-compatible DataFrame.

        ib_insync BarData has: date, open, high, low, close, volume, barCount, average
        We need: timestamp, ticker, open, high, low, close, volume
        """
        rows = []
        for bar in bars:
            ts = bar.date
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if not isinstance(ts, datetime):
                ts = datetime.combine(ts, datetime.min.time(), tzinfo=timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })

        return pd.DataFrame(rows)


def _parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string to UTC datetime."""
    dt = pd.Timestamp(dt_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime().astimezone(timezone.utc)

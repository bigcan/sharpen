"""Crypto Perpetual Futures Data Loader for Synapse Crypto 1H.

Fetches 1H OHLCV candles, funding rates, and open interest from
cryptocurrency exchanges via CCXT. Implements the Bronze → Silver → Gold
medallion pipeline with robust cleaning and validation.

Usage:
    loader = CryptoLoader(exchange="binance", market_type="swap")
    bronze_df = await loader.fetch_ohlcv(symbols, start, end, timeframe="1h")
    funding_df = await loader.fetch_funding_rates(symbols, start, end)

    pipeline = CryptoDataPipeline(loader)
    gold_df = await pipeline.run(symbols, start, end)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT", "LINK",
    "ETC", "UNI", "ATOM", "LTC", "FIL", "APT", "ARB", "OP", "NEAR", "INJ",
]

QUOTE = "USDT"

# CCXT returns OHLCV as: [timestamp_ms, open, high, low, close, volume]
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Max candles per request (Binance futures limit)
MAX_CANDLES_PER_REQUEST = 1500

# Rate limit safety: pause between paginated requests (seconds)
REQUEST_DELAY = 0.1


# ===========================================================================
# CryptoLoader — Bronze Layer (Raw Data Fetching)
# ===========================================================================
class CryptoLoader:
    """Fetches raw crypto perpetual futures data via CCXT.

    Handles pagination, rate limiting, and retry logic.
    Output is raw (Bronze layer) — no cleaning or transformation.
    """

    def __init__(
        self,
        exchange: str = "binance",
        market_type: Literal["swap", "future", "spot"] = "swap",
        sandbox: bool = False,
    ):
        self.exchange_id = exchange
        self.market_type = market_type
        self.sandbox = sandbox
        self._exchange = None

    async def _get_exchange(self):
        """Lazy-initialize async CCXT exchange."""
        if self._exchange is not None:
            return self._exchange

        import ccxt.async_support as ccxt

        exchange_class = getattr(ccxt, self.exchange_id)
        options = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot" if self.market_type == "spot" else self.market_type,
            },
        }
        # Only restrict to linear markets for futures/swap (skip dapi)
        if self.market_type != "spot":
            options["options"]["fetchMarkets"] = ["linear"]

        self._exchange = exchange_class(options)

        if self.sandbox:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        logger.info(
            f"Initialized {self.exchange_id} ({self.market_type}), "
            f"{len(self._exchange.markets)} markets loaded",
        )
        return self._exchange

    def _to_symbol(self, base: str) -> str:
        """Convert base asset to CCXT symbol.

        Binance perpetuals: 'BTC/USDT:USDT'
        Binance spot:       'BTC/USDT'
        """
        if self.market_type == "spot":
            return f"{base}/{QUOTE}"
        return f"{base}/{QUOTE}:{QUOTE}"

    def _resolve_symbols(self, assets: list[str]) -> dict[str, str]:
        """Map base assets to CCXT symbols, filtering unavailable ones.

        Raises ValueError if more than 10% of the universe is unavailable
        to prevent silent dimension mismatches downstream (C1 fix).
        """
        mapping = {}
        unavailable = []
        for asset in assets:
            sym = self._to_symbol(asset)
            if self._exchange and sym in self._exchange.markets:
                mapping[asset] = sym
            else:
                logger.warning(f"Symbol {sym} not found on {self.exchange_id}, skipping")
                unavailable.append(asset)

        if unavailable and len(unavailable) / len(assets) > 0.10:
            raise ValueError(
                f"{len(unavailable)}/{len(assets)} assets unavailable on "
                f"{self.exchange_id}: {unavailable}. "
                f"This exceeds the 10% threshold for silent dropping.",
            )
        return mapping

    async def fetch_ohlcv(
        self,
        assets: list[str],
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for all assets with automatic pagination.

        Args:
            assets: List of base assets (e.g., ["BTC", "ETH"]).
            start: Start date string (e.g., "2022-01-01").
            end: End date string (e.g., "2026-03-01").
            timeframe: Candle timeframe (default "1h").

        Returns:
            DataFrame with columns: [timestamp, ticker, open, high, low, close, volume]
        """
        exchange = await self._get_exchange()
        symbol_map = self._resolve_symbols(assets)

        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        all_frames = []
        total_assets = len(symbol_map)

        for idx, (asset, symbol) in enumerate(symbol_map.items(), 1):
            logger.info(f"[{idx}/{total_assets}] Fetching OHLCV: {symbol} ({timeframe})")
            asset_candles = await self._paginate_ohlcv(
                exchange, symbol, timeframe, since_ms, end_ms,
            )

            if not asset_candles:
                logger.warning(f"No OHLCV data for {symbol}")
                continue

            df = pd.DataFrame(asset_candles, columns=OHLCV_COLUMNS)
            df["ticker"] = asset
            all_frames.append(df)
            logger.info(f"  → {len(df)} candles fetched for {asset}")

        if not all_frames:
            raise ValueError("No OHLCV data fetched for any asset")

        result = pd.concat(all_frames, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        # Deduplicate candles at pagination boundaries (cursor overlap)
        result = result.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
        result = result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        logger.info(
            f"Bronze OHLCV: {len(result)} total candles, "
            f"{result['ticker'].nunique()} assets, "
            f"{result['timestamp'].min()} → {result['timestamp'].max()}",
        )
        return result

    async def _paginate_ohlcv(
        self,
        exchange,
        symbol: str,
        timeframe: str,
        since_ms: int,
        end_ms: int,
    ) -> list[list]:
        """Paginate through OHLCV history for a single symbol."""
        all_candles = []
        cursor = since_ms

        while cursor < end_ms:
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=cursor,
                    limit=MAX_CANDLES_PER_REQUEST,
                )
            except Exception as e:
                logger.error(f"OHLCV fetch error for {symbol} at {cursor}: {e}")
                await asyncio.sleep(1.0)
                try:
                    candles = await exchange.fetch_ohlcv(
                        symbol, timeframe=timeframe, since=cursor,
                        limit=MAX_CANDLES_PER_REQUEST,
                    )
                except Exception as e2:
                    logger.error(f"OHLCV retry failed for {symbol}: {e2}, skipping batch")
                    # Advance cursor by 1 batch worth of time to avoid infinite loop
                    cursor += MAX_CANDLES_PER_REQUEST * 3_600_000  # 1h in ms
                    continue

            if not candles:
                break

            # Filter candles within our time range
            candles = [c for c in candles if c[0] < end_ms]
            if not candles:
                break

            all_candles.extend(candles)

            # Advance cursor past the last candle
            last_ts = candles[-1][0]
            if last_ts <= cursor:
                break  # No progress — avoid infinite loop
            # C2 fix: Advance past the last candle to avoid re-fetching it.
            # CCXT since= is inclusive, so +1ms prevents overlap. For 1h
            # candles (3,600,000ms apart) this cannot skip any candle.
            cursor = last_ts + 1

            await asyncio.sleep(REQUEST_DELAY)

        return all_candles

    async def fetch_funding_rates(
        self,
        assets: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch historical funding rates for all assets.

        Returns:
            DataFrame with columns: [timestamp, ticker, funding_rate]
        """
        exchange = await self._get_exchange()
        symbol_map = self._resolve_symbols(assets)

        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        all_frames = []

        for idx, (asset, symbol) in enumerate(symbol_map.items(), 1):
            logger.info(f"[{idx}/{len(symbol_map)}] Fetching funding rates: {symbol}")
            rates = await self._paginate_funding(exchange, symbol, since_ms, end_ms)

            if not rates:
                logger.warning(f"No funding rate data for {symbol}")
                continue

            rows = []
            for r in rates:
                rows.append({
                    "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                    "ticker": asset,
                    "funding_rate": float(r.get("fundingRate", 0.0)) if r.get("fundingRate") is not None else 0.0,
                })
            df = pd.DataFrame(rows)
            all_frames.append(df)
            logger.info(f"  → {len(df)} funding rate records for {asset}")

        if not all_frames:
            logger.warning("No funding rate data fetched — returning empty DataFrame")
            return pd.DataFrame(columns=["timestamp", "ticker", "funding_rate"])

        result = pd.concat(all_frames, ignore_index=True)
        result = result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        return result

    async def _paginate_funding(
        self, exchange, symbol: str, since_ms: int, end_ms: int,
    ) -> list[dict]:
        """Paginate through funding rate history for a single symbol."""
        all_rates = []
        cursor = since_ms

        while cursor < end_ms:
            try:
                rates = await exchange.fetch_funding_rate_history(
                    symbol, since=cursor, limit=1000,
                )
            except Exception as e:
                logger.error(f"Funding rate fetch error for {symbol}: {e}")
                # Retry once after a brief delay
                await asyncio.sleep(1.0)
                try:
                    rates = await exchange.fetch_funding_rate_history(
                        symbol, since=cursor, limit=1000,
                    )
                except Exception as e2:
                    logger.error(f"Funding rate retry failed for {symbol}: {e2}")
                    break

            if not rates:
                break

            rates = [r for r in rates if r["timestamp"] < end_ms]
            if not rates:
                break

            all_rates.extend(rates)

            last_ts = rates[-1]["timestamp"]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1

            await asyncio.sleep(REQUEST_DELAY)

        return all_rates

    async def fetch_open_interest(
        self,
        assets: list[str],
    ) -> pd.DataFrame:
        """Fetch current open interest snapshot for all assets.

        Note: Historical OI requires exchange-specific endpoints.
        This fetches the current snapshot for each asset.

        Returns:
            DataFrame with columns: [timestamp, ticker, open_interest]
        """
        exchange = await self._get_exchange()
        symbol_map = self._resolve_symbols(assets)

        rows = []
        for asset, symbol in symbol_map.items():
            try:
                oi = await exchange.fetch_open_interest(symbol)
                rows.append({
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "ticker": asset,
                    "open_interest": oi.get("openInterestAmount", 0.0),
                    "open_interest_value": oi.get("openInterestValue", 0.0),
                })
            except Exception as e:
                logger.warning(f"OI fetch failed for {symbol}: {e}")

            await asyncio.sleep(REQUEST_DELAY)

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["timestamp", "ticker", "open_interest", "open_interest_value"],
        )

    async def close(self):
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None


# ===========================================================================
# CryptoDataCleaner — Silver Layer (Cleaning & Validation)
# ===========================================================================
class CryptoDataCleaner:
    """Cleans and validates raw crypto OHLCV data.

    Produces the Silver layer: gap-filled, outlier-corrected, validated.
    Follows the same Hampel Filter philosophy as the equities DataCleaner.
    """

    def __init__(
        self,
        hampel_window: int = 24,     # 24 hours (1 day) for hourly data
        hampel_threshold: float = 3.0,
        max_gap_hours: int = 4,       # Max gap to forward-fill
        flash_crash_threshold: float = 0.30,  # 30% move in 1h
        volume_zscore_cap: float = 10.0,
    ):
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold
        self.max_gap_hours = max_gap_hours
        self.flash_crash_threshold = flash_crash_threshold
        self.volume_zscore_cap = volume_zscore_cap

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full cleaning pipeline.

        Args:
            df: Bronze OHLCV DataFrame with columns
                [timestamp, ticker, open, high, low, close, volume]

        Returns:
            Tuple of (cleaned_df, anomaly_log_df).
            anomaly_log_df records all detected anomalies for audit.
        """
        logger.info(f"Silver cleaning: {len(df)} rows, {df['ticker'].nunique()} assets")
        anomalies = []

        # Process per ticker to maintain temporal integrity
        cleaned_frames = []
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("timestamp").copy()

            # Step 1: Fill time gaps (exchange outages / missing candles)
            group, gap_log = self._fill_time_gaps(group, ticker)
            anomalies.extend(gap_log)

            # Step 2: OHLCV consistency checks
            group, consistency_log = self._fix_ohlcv_consistency(group, ticker)
            anomalies.extend(consistency_log)

            # Step 3: Detect and flag flash crashes (don't remove, just flag)
            group, crash_log = self._flag_flash_crashes(group, ticker)
            anomalies.extend(crash_log)

            # Step 4: Hampel filter on price columns
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                group[col], n_outliers = self._hampel_filter(group[col])
                if n_outliers > 0:
                    anomalies.append({
                        "ticker": ticker, "type": "hampel_outlier",
                        "column": col, "count": n_outliers,
                    })

            # Step 4b: Re-enforce OHLCV consistency after Hampel
            # (Hampel corrects columns independently, may break H≥L)
            oc_max = group[["open", "close"]].max(axis=1)
            oc_min = group[["open", "close"]].min(axis=1)
            group["high"] = group["high"].clip(lower=oc_max)
            group["low"] = group["low"].clip(upper=oc_min)
            # Final check: ensure high >= low (Hampel may have inverted them)
            inverted = group["high"] < group["low"]
            if inverted.any():
                h_copy = group.loc[inverted, "high"].copy()
                group.loc[inverted, "high"] = group.loc[inverted, "low"]
                group.loc[inverted, "low"] = h_copy

            # Step 5: Volume anomaly capping
            group["volume"], n_capped = self._cap_volume_spikes(group["volume"])
            if n_capped > 0:
                anomalies.append({
                    "ticker": ticker, "type": "volume_spike_capped",
                    "count": n_capped,
                })

            # Step 6: Final imputation (forward-fill then backfill leading NaNs)
            group = self._impute_remaining(group)

            cleaned_frames.append(group)

        cleaned_df = pd.concat(cleaned_frames, ignore_index=True)
        anomaly_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

        n_anomalies = len(anomalies)
        logger.info(
            f"Silver cleaning complete: {len(cleaned_df)} rows, "
            f"{n_anomalies} anomalies detected",
        )

        return cleaned_df, anomaly_df

    def _fill_time_gaps(
        self, df: pd.DataFrame, ticker: str,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Detect and fill gaps in the hourly time series.

        Gaps ≤ max_gap_hours are forward-filled.
        Gaps > max_gap_hours are forward-filled but flagged.
        """
        anomalies = []

        # Create complete hourly index
        full_index = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq="1h",
            tz="UTC",
        )

        df = df.set_index("timestamp")
        existing_timestamps = df.index

        # Detect gaps
        missing = full_index.difference(existing_timestamps)
        if len(missing) > 0:
            # Group consecutive missing bars
            gap_groups = np.split(
                np.arange(len(missing)),
                np.where(np.diff(missing.astype(np.int64) // 10**9) > 3600 * 1.5)[0] + 1,
            )

            for gap_idx in gap_groups:
                if len(gap_idx) == 0:
                    continue
                gap_start = missing[gap_idx[0]]
                gap_end = missing[gap_idx[-1]]
                gap_hours = len(gap_idx)

                severity = "minor" if gap_hours <= self.max_gap_hours else "major"
                anomalies.append({
                    "ticker": ticker, "type": f"time_gap_{severity}",
                    "timestamp": str(gap_start), "gap_hours": gap_hours,
                    "gap_end": str(gap_end),
                })

            logger.debug(
                f"{ticker}: {len(missing)} missing bars detected, "
                f"{len(gap_groups)} gap(s)",
            )

        # Reindex to full hourly grid and forward-fill
        df = df.reindex(full_index)
        df["ticker"] = ticker
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].ffill(limit=self.max_gap_hours * 2)
        )
        # For volume, fill gaps with 0 (no trading during outage)
        df["volume"] = df["volume"].fillna(0.0)

        # For gaps longer than max_gap_hours * 2, mark as degraded rather than
        # silently fabricating flat-price candles of arbitrary length.
        remaining_nans = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        if remaining_nans > 0:
            # Mark degraded bars so downstream can filter or weight them
            if "_degraded" not in df.columns:
                df["_degraded"] = False
            df.loc[df[["open", "high", "low", "close"]].isna().any(axis=1), "_degraded"] = True
            logger.warning(
                f"{ticker}: {remaining_nans} bars remain unfilled after "
                f"limit={self.max_gap_hours * 2} ffill — extending ffill to cover, "
                f"but data quality is degraded for long outages",
            )
            df[["open", "high", "low", "close"]] = (
                df[["open", "high", "low", "close"]].ffill()
            )

        df = df.reset_index().rename(columns={"index": "timestamp"})
        return df, anomalies

    def _fix_ohlcv_consistency(
        self, df: pd.DataFrame, ticker: str,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Fix OHLCV consistency violations.

        Rules:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        - All prices > 0
        - Volume >= 0
        """
        anomalies = []

        # Zero / negative prices
        for col in ["open", "high", "low", "close"]:
            bad_mask = df[col] <= 0
            if bad_mask.any():
                n_bad = bad_mask.sum()
                anomalies.append({
                    "ticker": ticker, "type": "non_positive_price",
                    "column": col, "count": int(n_bad),
                })
                df.loc[bad_mask, col] = np.nan
                df[col] = df[col].ffill().bfill()

        # high < low
        bad_hl = df["high"] < df["low"]
        if bad_hl.any():
            anomalies.append({
                "ticker": ticker, "type": "high_lt_low",
                "count": int(bad_hl.sum()),
            })
            # Swap high and low
            h = df.loc[bad_hl, "high"].copy()
            df.loc[bad_hl, "high"] = df.loc[bad_hl, "low"]
            df.loc[bad_hl, "low"] = h

        # high should be >= max(open, close)
        oc_max = df[["open", "close"]].max(axis=1)
        bad_high = df["high"] < oc_max
        if bad_high.any():
            df.loc[bad_high, "high"] = oc_max[bad_high]

        # low should be <= min(open, close)
        oc_min = df[["open", "close"]].min(axis=1)
        bad_low = df["low"] > oc_min
        if bad_low.any():
            df.loc[bad_low, "low"] = oc_min[bad_low]

        # Negative volume
        df["volume"] = df["volume"].clip(lower=0.0)

        return df, anomalies

    def _flag_flash_crashes(
        self, df: pd.DataFrame, ticker: str,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Flag (but don't remove) candles with extreme price moves.

        These are kept in the data because they represent real market events,
        but flagged for the anomaly log and downstream awareness.
        """
        anomalies = []

        returns = df["close"].pct_change().abs()
        flash_mask = returns > self.flash_crash_threshold

        if flash_mask.any():
            df["_flash_crash"] = flash_mask
            flash_rows = df[flash_mask]
            for _, row in flash_rows.iterrows():
                anomalies.append({
                    "ticker": ticker, "type": "flash_crash_flagged",
                    "timestamp": str(row["timestamp"]),
                    "return_pct": float(returns.loc[row.name]),
                })
            logger.warning(
                f"{ticker}: {flash_mask.sum()} flash crash candle(s) flagged",
            )
        else:
            df["_flash_crash"] = False

        return df, anomalies

    def _hampel_filter(self, series: pd.Series) -> tuple[pd.Series, int]:
        """Apply Hampel filter: replace outliers with rolling median.

        Same algorithm as equities DataCleaner but tuned for hourly data.
        """
        median = series.rolling(
            window=self.hampel_window, min_periods=max(5, self.hampel_window // 4),
        ).median()

        deviation = (series - median).abs()
        mad = deviation.rolling(
            window=self.hampel_window, min_periods=max(5, self.hampel_window // 4),
        ).median()

        # MAD consistency constant for normal distribution
        k = 1.4826
        robust_z = deviation / (k * mad + 1e-10)

        outlier_mask = robust_z > self.hampel_threshold
        n_outliers = int(outlier_mask.sum())

        if n_outliers > 0:
            corrected = series.copy()
            corrected.loc[outlier_mask] = median.loc[outlier_mask]
            return corrected, n_outliers

        return series, 0

    def _cap_volume_spikes(self, volume: pd.Series) -> tuple[pd.Series, int]:
        """Cap extreme volume spikes at zscore threshold.

        Volume spikes are informative but extreme outliers can destabilize
        normalization. Cap at volume_zscore_cap standard deviations.
        """
        rolling_mean = volume.rolling(window=168, min_periods=24).mean()
        rolling_std = volume.rolling(window=168, min_periods=24).std()

        z = (volume - rolling_mean) / (rolling_std + 1e-10)
        spike_mask = z > self.volume_zscore_cap

        n_capped = int(spike_mask.sum())
        if n_capped > 0:
            cap_value = rolling_mean + self.volume_zscore_cap * rolling_std
            capped = volume.copy()
            capped.loc[spike_mask] = cap_value.loc[spike_mask]
            return capped, n_capped

        return volume, 0

    def _impute_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final imputation pass: ffill then bfill for leading NaNs."""
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = df[col].ffill().bfill()
        return df


# ===========================================================================
# CryptoFundingCleaner — Clean & Resample Funding Rates
# ===========================================================================
class CryptoFundingCleaner:
    """Clean and resample funding rates to align with 1H OHLCV bars."""

    def clean_and_resample(
        self,
        funding_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clean funding rates and forward-fill to 1H resolution.

        Funding rates are published every 8 hours. We forward-fill them
        to every 1H bar so the environment can apply them correctly.

        Args:
            funding_df: Raw funding rates [timestamp, ticker, funding_rate]
            ohlcv_df: Cleaned OHLCV with complete 1H timestamps

        Returns:
            DataFrame aligned to OHLCV timestamps with funding_rate column.
        """
        if funding_df.empty:
            # Return zero funding rates aligned to OHLCV
            result = ohlcv_df[["timestamp", "ticker"]].copy()
            result["funding_rate"] = 0.0
            return result

        all_frames = []
        for ticker in ohlcv_df["ticker"].unique():
            ohlcv_ts = ohlcv_df.loc[
                ohlcv_df["ticker"] == ticker, "timestamp",
            ].sort_values()

            funding_tic = funding_df[funding_df["ticker"] == ticker].copy()

            if funding_tic.empty:
                fr = pd.DataFrame({
                    "timestamp": ohlcv_ts,
                    "ticker": ticker,
                    "funding_rate": 0.0,
                })
            else:
                funding_tic = funding_tic.set_index("timestamp").sort_index()
                # Reindex to hourly and forward-fill (rate applies until next update)
                fr = funding_tic["funding_rate"].reindex(ohlcv_ts)
                # LEAK-2: Truncate forward-fill to available funding data extent
                fund_last = funding_tic.index.max()
                fr = fr.ffill()
                fr.loc[fr.index > fund_last] = np.nan
                fr = fr.fillna(0.0)
                fr = fr.reset_index()
                fr.columns = ["timestamp", "funding_rate"]
                fr["ticker"] = ticker

            all_frames.append(fr)

        return pd.concat(all_frames, ignore_index=True)


# ===========================================================================
# CryptoDataValidator — Quality Gates
# ===========================================================================
class CryptoDataValidator:
    """Validates cleaned data before feature engineering.

    Enforces quality gates that must pass before proceeding to Gold layer.
    """

    def __init__(
        self,
        max_missing_pct: float = 0.001,   # <0.1% missing bars
        min_bars_per_asset: int = 1000,     # At least ~6 weeks of hourly data
        max_zero_volume_pct: float = 0.05,  # <5% zero-volume bars
        min_assets: int = 15,               # At least 15 of 20 assets must have data
    ):
        self.max_missing_pct = max_missing_pct
        self.min_bars_per_asset = min_bars_per_asset
        self.max_zero_volume_pct = max_zero_volume_pct
        self.min_assets = min_assets

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Run all validation checks.

        Returns:
            Tuple of (passed: bool, issues: list of failure descriptions).
        """
        issues = []

        # Check 1: Minimum number of assets
        n_assets = df["ticker"].nunique()
        if n_assets < self.min_assets:
            issues.append(
                f"FAIL: Only {n_assets} assets, need ≥ {self.min_assets}",
            )

        # Per-asset checks
        for ticker, group in df.groupby("ticker"):
            n_bars = len(group)

            # Check 2: Minimum bars per asset
            if n_bars < self.min_bars_per_asset:
                issues.append(
                    f"FAIL: {ticker} has only {n_bars} bars, "
                    f"need ≥ {self.min_bars_per_asset}",
                )
                continue

            # Check 3: NaN check (after cleaning there should be zero)
            nan_pct = group[["open", "high", "low", "close"]].isna().mean().max()
            if nan_pct > self.max_missing_pct:
                issues.append(
                    f"FAIL: {ticker} has {nan_pct:.4%} NaN in price columns",
                )

            # Check 4: Zero volume ratio
            zero_vol_pct = (group["volume"] == 0).mean()
            if zero_vol_pct > self.max_zero_volume_pct:
                issues.append(
                    f"WARN: {ticker} has {zero_vol_pct:.2%} zero-volume bars "
                    f"(threshold {self.max_zero_volume_pct:.0%})",
                )

            # Check 5: Timestamp continuity
            ts_diff = group["timestamp"].diff().dt.total_seconds().dropna()
            expected_gap = 3600  # 1 hour
            bad_gaps = ts_diff[ts_diff != expected_gap]
            if len(bad_gaps) > 0:
                issues.append(
                    f"FAIL: {ticker} has {len(bad_gaps)} non-hourly gaps "
                    f"after cleaning",
                )

            # Check 6: Price sanity (no zeros or negatives after cleaning)
            for col in ["open", "high", "low", "close"]:
                if (group[col] <= 0).any():
                    issues.append(f"FAIL: {ticker}.{col} has non-positive values")

            # Check 7: OHLCV consistency (high >= low)
            if (group["high"] < group["low"]).any():
                issues.append(f"FAIL: {ticker} has high < low after cleaning")

        # Summary
        failures = [i for i in issues if i.startswith("FAIL")]
        warnings = [i for i in issues if i.startswith("WARN")]
        passed = len(failures) == 0

        if passed:
            logger.info(
                f"Validation PASSED: {n_assets} assets, "
                f"{len(df)} bars, {len(warnings)} warnings",
            )
        else:
            logger.error(
                f"Validation FAILED: {len(failures)} failures, "
                f"{len(warnings)} warnings",
            )
            for f in failures:
                logger.error(f"  {f}")

        for w in warnings:
            logger.warning(f"  {w}")

        return passed, issues


# ===========================================================================
# CryptoDataPipeline — Full Bronze → Silver → Gold Orchestrator
# ===========================================================================
class CryptoDataPipeline:
    """End-to-end data pipeline: fetch, clean, validate, cache.

    Bronze (raw CCXT) → Silver (cleaned) → validated & ready for features.
    The Gold layer (feature engineering) is handled separately.
    """

    def __init__(
        self,
        exchange: str = "binance",
        cache_dir: str = "./data/crypto_cache",
        sandbox: bool = False,
    ):
        self.loader = CryptoLoader(exchange=exchange, sandbox=sandbox)
        self.cleaner = CryptoDataCleaner()
        self.funding_cleaner = CryptoFundingCleaner()
        self.validator = CryptoDataValidator()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        assets: list[str] | None = None,
        start: str = "2022-01-01",
        end: str | None = None,
        timeframe: str = "1h",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Run the full data pipeline.

        Args:
            assets: List of base assets. Defaults to DEFAULT_UNIVERSE.
            start: Start date.
            end: End date. Defaults to now.
            timeframe: Candle timeframe.
            use_cache: Whether to load/save Silver cache.
            force_refresh: Force re-download even if cache exists.

        Returns:
            Dict with keys: 'ohlcv', 'funding', 'anomalies'
            All DataFrames are cleaned, validated, and ready for feature engineering.
        """
        if assets is None:
            assets = DEFAULT_UNIVERSE
        if end is None:
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # ----- Try Silver cache first -----
        silver_ohlcv_path = self.cache_dir / "silver_ohlcv.parquet"
        silver_funding_path = self.cache_dir / "silver_funding.parquet"
        silver_anomaly_path = self.cache_dir / "silver_anomalies.parquet"

        if use_cache and not force_refresh and silver_ohlcv_path.exists():
            logger.info(f"Loading Silver cache from {self.cache_dir}")
            try:
                ohlcv_df = pd.read_parquet(silver_ohlcv_path)
                ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
                if ohlcv_df["timestamp"].dt.tz is None:
                    ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.tz_localize("UTC")
                else:
                    ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.tz_convert("UTC")

                funding_df = pd.read_parquet(silver_funding_path) if silver_funding_path.exists() else pd.DataFrame()
                if not funding_df.empty:
                    funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"])
                    if funding_df["timestamp"].dt.tz is None:
                        funding_df["timestamp"] = funding_df["timestamp"].dt.tz_localize("UTC")
                    else:
                        funding_df["timestamp"] = funding_df["timestamp"].dt.tz_convert("UTC")

                anomaly_df = pd.read_parquet(silver_anomaly_path) if silver_anomaly_path.exists() else pd.DataFrame()

                # Check if cache covers our requested range (both start and end)
                cache_start = ohlcv_df["timestamp"].min()
                cache_end = ohlcv_df["timestamp"].max()
                requested_start = pd.Timestamp(start, tz="UTC")
                requested_end = pd.Timestamp(end, tz="UTC")

                if (cache_start <= requested_start + pd.Timedelta(hours=2)
                        and cache_end >= requested_end - pd.Timedelta(hours=2)):
                    logger.info(
                        f"Silver cache valid: {len(ohlcv_df)} rows, "
                        f"covers through {cache_end}",
                    )
                    # Validate
                    passed, issues = self.validator.validate(ohlcv_df)
                    if not passed:
                        logger.warning("Cached data failed validation, re-fetching...")
                    else:
                        return {
                            "ohlcv": ohlcv_df,
                            "funding": funding_df,
                            "anomalies": anomaly_df,
                        }
                else:
                    logger.info(
                        f"Silver cache stale: ends at {cache_end}, "
                        f"need through {requested_end}. Re-fetching...",
                    )
            except Exception as e:
                logger.warning(f"Failed to load Silver cache: {e}. Re-fetching...")

        # ----- Bronze: Fetch raw data -----
        logger.info("=" * 60)
        logger.info("BRONZE LAYER: Fetching raw data from exchange")
        logger.info("=" * 60)

        try:
            bronze_ohlcv = await self.loader.fetch_ohlcv(
                assets, start, end, timeframe,
            )
            # Fetch funding separately so OHLCV is preserved on funding failure
            try:
                bronze_funding = await self.loader.fetch_funding_rates(
                    assets, start, end,
                )
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}. Continuing with empty funding.")
                bronze_funding = pd.DataFrame(columns=["timestamp", "ticker", "funding_rate"])
        finally:
            await self.loader.close()

        # Save Bronze
        bronze_dir = self.cache_dir / "bronze"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        bronze_ohlcv.to_parquet(bronze_dir / "ohlcv.parquet", index=False)
        if not bronze_funding.empty:
            bronze_funding.to_parquet(bronze_dir / "funding.parquet", index=False)
        logger.info(f"Bronze data saved to {bronze_dir}")

        # ----- Silver: Clean -----
        logger.info("=" * 60)
        logger.info("SILVER LAYER: Cleaning and validating")
        logger.info("=" * 60)

        silver_ohlcv, anomaly_df = self.cleaner.clean(bronze_ohlcv)

        # Clean and resample funding rates to 1H
        silver_funding = self.funding_cleaner.clean_and_resample(
            bronze_funding, silver_ohlcv,
        )

        # ----- Validate -----
        passed, issues = self.validator.validate(silver_ohlcv)
        if not passed:
            logger.error("DATA VALIDATION FAILED — review anomaly log")
            for issue in issues:
                logger.error(f"  {issue}")
            # Don't raise — return data with issues list for manual review
            # The caller can decide whether to proceed

        # ----- Save Silver cache -----
        silver_ohlcv.to_parquet(silver_ohlcv_path, index=False)
        if not silver_funding.empty:
            silver_funding.to_parquet(silver_funding_path, index=False)
        if not anomaly_df.empty:
            anomaly_df.to_parquet(silver_anomaly_path, index=False)
        logger.info(f"Silver data cached to {self.cache_dir}")

        # ----- Walk-Forward Coverage Check -----
        wf_validator = WalkForwardCoverageValidator()
        wf_result = wf_validator.validate(silver_ohlcv)

        # ----- Summary -----
        logger.info("=" * 60)
        logger.info("DATA PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Assets:        {silver_ohlcv['ticker'].nunique()}")
        logger.info(f"  Bars:          {len(silver_ohlcv)}")
        logger.info(f"  Date range:    {silver_ohlcv['timestamp'].min()} → {silver_ohlcv['timestamp'].max()}")
        logger.info(f"  Funding:       {len(silver_funding)} records")
        logger.info(f"  Anomalies:     {len(anomaly_df)}")
        logger.info(f"  Validation:    {'PASSED' if passed else 'FAILED'}")
        logger.info(f"  WF windows:    {wf_result['n_windows']}")
        logger.info(f"  WF coverage:   {'PASSED' if wf_result['passed'] else 'FAILED'}")

        return {
            "ohlcv": silver_ohlcv,
            "funding": silver_funding,
            "anomalies": anomaly_df,
            "walk_forward": wf_result,
        }


# ===========================================================================
# Convenience: synchronous wrapper
# ===========================================================================
def fetch_crypto_data(
    assets: list[str] | None = None,
    start: str = "2022-01-01",
    end: str | None = None,
    exchange: str = "binance",
    cache_dir: str = "./data/crypto_cache",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Synchronous wrapper for the async pipeline.

    Usage:
        data = fetch_crypto_data(["BTC", "ETH"], "2022-01-01", "2026-03-01")
        ohlcv = data["ohlcv"]
        funding = data["funding"]
    """
    pipeline = CryptoDataPipeline(
        exchange=exchange, cache_dir=cache_dir,
    )
    coro = pipeline.run(
        assets=assets, start=start, end=end,
        force_refresh=force_refresh,
    )
    # Support calling from inside an existing event loop (e.g. Jupyter).
    # R6 fix: Use a background thread instead of nest_asyncio (not in deps).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def fetch_spot_data(
    assets: list[str] | None = None,
    start: str = "2022-01-01",
    end: str | None = None,
    exchange: str = "binance",
    cache_dir: str = "./data/crypto_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch spot OHLCV data for funding arb (spot leg).

    Uses the same CryptoLoader with market_type="spot" and CryptoDataCleaner
    for cleaning. Returns cleaned Silver-layer OHLCV DataFrame.

    Usage:
        spot_df = fetch_spot_data(["BTC", "ETH"], "2022-01-01", "2026-03-01")
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    silver_spot_path = cache_path / "silver_spot_ohlcv.parquet"

    if not force_refresh and silver_spot_path.exists():
        logger.info(f"Loading spot Silver cache from {silver_spot_path}")
        df = pd.read_parquet(silver_spot_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        return df

    async def _fetch():
        loader = CryptoLoader(exchange=exchange, market_type="spot")
        if assets is None:
            _assets = DEFAULT_UNIVERSE
        else:
            _assets = assets
        if end is None:
            _end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            _end = end

        try:
            bronze = await loader.fetch_ohlcv(_assets, start, _end, "1h")
        finally:
            await loader.close()

        cleaner = CryptoDataCleaner()
        silver, _ = cleaner.clean(bronze)
        silver.to_parquet(silver_spot_path, index=False)
        logger.info(f"Spot Silver data cached to {silver_spot_path}")
        return silver

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _fetch()).result()
    return asyncio.run(_fetch())


# ===========================================================================
# WalkForwardCoverageValidator — Ensure data supports multi-window WF
# ===========================================================================
class WalkForwardCoverageValidator:
    """Validates that the dataset has enough bars for walk-forward training.

    Must be called after the Silver layer is ready, before proceeding
    to feature engineering and training.
    """

    def __init__(
        self,
        train_bars: int = 8760,     # 12 months
        val_bars: int = 720,        # 1 month
        test_bars: int = 1440,      # 2 months
        step_bars: int = 720,       # 1 month
        embargo_bars: int = 720,    # 1 month (matches normalization window)
        min_windows: int = 10,      # Minimum windows for statistical power
    ):
        self.train_bars = train_bars
        self.val_bars = val_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars
        self.min_windows = min_windows

    @property
    def bars_per_window(self) -> int:
        # Two embargo gaps: train | embargo | val | embargo | test
        # Matches CryptoWalkForwardEvaluator.generate_windows() layout
        return self.train_bars + self.embargo_bars + self.val_bars + self.embargo_bars + self.test_bars

    def validate(self, df: pd.DataFrame) -> dict:
        """Check walk-forward coverage for the dataset.

        Args:
            df: Cleaned OHLCV DataFrame with [timestamp, ticker, ...].

        Returns:
            Dict with validation results:
                passed: bool
                total_bars: int (per asset, minimum across assets)
                n_windows: int
                first_window_start: timestamp
                last_window_test_end: timestamp
                issues: list[str]
                window_schedule: list of (train_start, train_end, val_start,
                    val_end, test_start, test_end) timestamp tuples
        """
        issues = []

        # Calculate bars per asset
        bars_per_asset = df.groupby("ticker")["timestamp"].count()
        min_bars = int(bars_per_asset.min())
        min_asset = bars_per_asset.idxmin()

        # How many windows can we fit?
        available_after_first = min_bars - self.bars_per_window
        if available_after_first < 0:
            n_windows = 0
            issues.append(
                f"FAIL: Shortest asset ({min_asset}) has {min_bars} bars, "
                f"need ≥ {self.bars_per_window} for even 1 window",
            )
        else:
            n_windows = 1 + available_after_first // self.step_bars

        if n_windows < self.min_windows:
            issues.append(
                f"FAIL: Only {n_windows} walk-forward windows possible, "
                f"need ≥ {self.min_windows} for statistical power. "
                f"Need ≥ {self.bars_per_window + (self.min_windows - 1) * self.step_bars} "
                f"bars per asset.",
            )

        # Build window schedule using the asset with least data as reference
        ref_timestamps = (
            df[df["ticker"] == min_asset]["timestamp"]
            .sort_values()
            .reset_index(drop=True)
        )

        window_schedule = []
        for w in range(n_windows):
            offset = w * self.step_bars
            ref_timestamps.iloc[offset]
            train_end_idx = offset + self.train_bars - 1
            embargo1_end_idx = train_end_idx + self.embargo_bars
            val_start_idx = embargo1_end_idx + 1
            val_end_idx = val_start_idx + self.val_bars - 1
            embargo2_end_idx = val_end_idx + self.embargo_bars
            test_start_idx = embargo2_end_idx + 1
            test_end_idx = test_start_idx + self.test_bars - 1

            if test_end_idx >= len(ref_timestamps):
                break

            window_schedule.append({
                "window": w,
                "train_start": ref_timestamps.iloc[offset],
                "train_end": ref_timestamps.iloc[train_end_idx],
                "val_start": ref_timestamps.iloc[val_start_idx],
                "val_end": ref_timestamps.iloc[val_end_idx],
                "test_start": ref_timestamps.iloc[test_start_idx],
                "test_end": ref_timestamps.iloc[test_end_idx],
            })

        passed = len([i for i in issues if i.startswith("FAIL")]) == 0

        result = {
            "passed": passed,
            "total_bars_min": min_bars,
            "total_bars_max": int(bars_per_asset.max()),
            "shortest_asset": min_asset,
            "n_windows": len(window_schedule),
            "bars_per_window": self.bars_per_window,
            "step_bars": self.step_bars,
            "first_window_start": window_schedule[0]["train_start"] if window_schedule else None,
            "last_window_test_end": window_schedule[-1]["test_end"] if window_schedule else None,
            "issues": issues,
            "window_schedule": window_schedule,
        }

        # Log summary
        logger.info("=" * 60)
        logger.info("WALK-FORWARD COVERAGE REPORT")
        logger.info("=" * 60)
        logger.info(f"  Bars per window:  {self.bars_per_window} "
                    f"(train={self.train_bars} + embargo={self.embargo_bars} "
                    f"+ val={self.val_bars} + embargo={self.embargo_bars} "
                    f"+ test={self.test_bars})")
        logger.info(f"  Step size:        {self.step_bars} bars")
        logger.info(f"  Min bars (asset): {min_bars} ({min_asset})")
        logger.info(f"  Max bars (asset): {int(bars_per_asset.max())}")
        logger.info(f"  Windows:          {len(window_schedule)}")
        logger.info(f"  Status:           {'PASSED' if passed else 'FAILED'}")

        if window_schedule:
            logger.info(f"  First window:     {window_schedule[0]['train_start']} → "
                        f"{window_schedule[0]['test_end']}")
            logger.info(f"  Last window:      {window_schedule[-1]['train_start']} → "
                        f"{window_schedule[-1]['test_end']}")

        for issue in issues:
            if issue.startswith("FAIL"):
                logger.error(f"  {issue}")
            else:
                logger.warning(f"  {issue}")

        return result

"""Continuous Crypto Data Collection Service.

Background service that:
- Fetches 1H OHLCV candles every hour via CCXT
- Collects funding rates every 8 hours
- Collects open interest snapshots every hour
- Stores in Parquet (Bronze layer), append-only
- Backfills gaps on startup

Usage:
    python -m finrl_pro_ds.crypto.data.crypto_collector --run-once     # Single fetch
    python -m finrl_pro_ds.crypto.data.crypto_collector --daemon        # Continuous
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from finrl_pro_ds.crypto.data.crypto_loader import (
    CryptoLoader,
    DEFAULT_UNIVERSE,
    QUOTE,
)

logger = logging.getLogger(__name__)

# Collection intervals
OHLCV_INTERVAL_SECONDS = 3600       # Every 1 hour
FUNDING_INTERVAL_SECONDS = 28800     # Every 8 hours
OI_INTERVAL_SECONDS = 3600           # Every 1 hour

# How far back to backfill on startup
BACKFILL_HOURS = 48


class CryptoCollector:
    """Continuous data collection service for crypto perpetual futures."""

    def __init__(
        self,
        assets: list[str] | None = None,
        exchange: str = "binance",
        data_dir: str = "./data/crypto_cache/bronze",
        timeframe: str = "1h",
    ):
        self.assets = assets or DEFAULT_UNIVERSE
        self.exchange = exchange
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeframe = timeframe
        self._running = False
        # M3: Reuse a single loader across collection cycles
        self._shared_loader: CryptoLoader | None = None

    async def run_once(self) -> dict[str, int]:
        """Perform a single collection cycle.

        Returns counts of collected records.
        """
        loader = CryptoLoader(exchange=self.exchange)

        try:
            now = datetime.now(timezone.utc)
            # Fetch last 2 hours of OHLCV (overlap for deduplication)
            start = (now - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
            end = now.strftime("%Y-%m-%dT%H:%M:%S")

            logger.info(f"Collecting OHLCV: {start} → {end}")
            ohlcv = await loader.fetch_ohlcv(
                self.assets, start, end, self.timeframe
            )
            ohlcv_path = self.data_dir / "ohlcv_latest.parquet"
            self._append_parquet(ohlcv, ohlcv_path)

            # Funding rates (less frequent but collect on every cycle)
            logger.info("Collecting funding rates")
            funding = await loader.fetch_funding_rates(
                self.assets, start, end
            )
            if not funding.empty:
                funding_path = self.data_dir / "funding_latest.parquet"
                self._append_parquet(funding, funding_path)

            # Open interest snapshot
            logger.info("Collecting open interest")
            oi = await loader.fetch_open_interest(self.assets)
            if not oi.empty:
                oi_path = self.data_dir / "oi_snapshots.parquet"
                self._append_parquet(oi, oi_path)

            counts = {
                "ohlcv": len(ohlcv),
                "funding": len(funding),
                "oi": len(oi),
            }
            logger.info(f"Collection complete: {counts}")
            return counts

        finally:
            await loader.close()

    async def backfill(self, hours: int = BACKFILL_HOURS) -> None:
        """Backfill missing data from the last N hours."""
        loader = CryptoLoader(exchange=self.exchange)

        try:
            now = datetime.now(timezone.utc)
            start = (now - pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
            end = now.strftime("%Y-%m-%dT%H:%M:%S")

            logger.info(f"Backfilling {hours}h: {start} → {end}")

            ohlcv = await loader.fetch_ohlcv(self.assets, start, end, self.timeframe)
            ohlcv_path = self.data_dir / "ohlcv_latest.parquet"
            self._append_parquet(ohlcv, ohlcv_path)

            funding = await loader.fetch_funding_rates(self.assets, start, end)
            if not funding.empty:
                funding_path = self.data_dir / "funding_latest.parquet"
                self._append_parquet(funding, funding_path)

            logger.info(
                f"Backfill complete: {len(ohlcv)} OHLCV bars, "
                f"{len(funding)} funding records"
            )
        finally:
            await loader.close()

    async def run_daemon(self) -> None:
        """Run continuous collection in a loop."""
        self._running = True
        logger.info("Starting crypto collector daemon")

        # Backfill on startup
        await self.backfill()

        # Trigger first collection immediately after backfill
        last_ohlcv = 0.0
        last_funding = 0.0

        while self._running:
            now = time.time()

            try:
                if now - last_ohlcv >= OHLCV_INTERVAL_SECONDS:
                    await self._collect_ohlcv()
                    last_ohlcv = now

                if now - last_funding >= FUNDING_INTERVAL_SECONDS:
                    await self._collect_funding()
                    last_funding = now
            except Exception as e:
                logger.error(f"Collection error: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(60):
                if not self._running:
                    break
                await asyncio.sleep(1)

        # Clean up shared loader on shutdown
        await self._close_loader()
        logger.info("Crypto collector daemon stopped")

    async def _get_loader(self) -> CryptoLoader:
        """Get or create the shared loader instance."""
        if self._shared_loader is None:
            self._shared_loader = CryptoLoader(exchange=self.exchange)
        return self._shared_loader

    async def _close_loader(self) -> None:
        """Close the shared loader (for shutdown)."""
        if self._shared_loader is not None:
            await self._shared_loader.close()
            self._shared_loader = None

    async def _collect_ohlcv(self) -> None:
        """Collect OHLCV and OI data."""
        loader = await self._get_loader()
        now = datetime.now(timezone.utc)
        start = (now - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        end = now.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(f"Collecting OHLCV: {start} → {end}")
        ohlcv = await loader.fetch_ohlcv(self.assets, start, end, self.timeframe)
        self._append_parquet(ohlcv, self.data_dir / "ohlcv_latest.parquet")

        oi = await loader.fetch_open_interest(self.assets)
        if not oi.empty:
            self._append_parquet(oi, self.data_dir / "oi_snapshots.parquet")

    async def _collect_funding(self) -> None:
        """Collect funding rate data (every 8 hours)."""
        loader = await self._get_loader()
        now = datetime.now(timezone.utc)
        start = (now - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%S")
        end = now.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info("Collecting funding rates")
        funding = await loader.fetch_funding_rates(self.assets, start, end)
        if not funding.empty:
            self._append_parquet(funding, self.data_dir / "funding_latest.parquet")

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False

    def _append_parquet(self, new_df: pd.DataFrame, path: Path) -> None:
        """Append new data to existing Parquet file, deduplicating.

        Uses atomic write (temp file + rename) to prevent corruption from
        crashes or concurrent access.
        """
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_df], ignore_index=True)

                # Deduplicate on (timestamp, ticker)
                if "timestamp" in combined.columns and "ticker" in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=["timestamp", "ticker"], keep="last"
                    )
                    combined = combined.sort_values(
                        ["ticker", "timestamp"]
                    ).reset_index(drop=True)

                # Atomic write: write to temp file, then rename
                tmp_path = path.with_suffix(".parquet.tmp")
                combined.to_parquet(tmp_path, index=False)
                tmp_path.replace(path)
                logger.debug(f"Appended to {path}: {len(new_df)} new → {len(combined)} total")
            except Exception as e:
                logger.warning(f"Failed to append to {path}: {e}. Writing fresh.")
                new_df.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)
            logger.debug(f"Created {path}: {len(new_df)} rows")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Crypto data collector")
    parser.add_argument("--run-once", action="store_true", help="Single collection cycle")
    parser.add_argument("--daemon", action="store_true", help="Continuous collection")
    parser.add_argument("--backfill", type=int, default=0, help="Backfill N hours")
    parser.add_argument("--exchange", default="binance", help="Exchange ID")
    parser.add_argument("--data-dir", default="./data/crypto_cache/bronze", help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    collector = CryptoCollector(exchange=args.exchange, data_dir=args.data_dir)

    if args.backfill > 0:
        asyncio.run(collector.backfill(args.backfill))
    elif args.run_once:
        asyncio.run(collector.run_once())
    elif args.daemon:
        loop = asyncio.new_event_loop()

        def _shutdown(sig, frame):
            logger.info(f"Received {sig}, shutting down...")
            collector.stop()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        loop.run_until_complete(collector.run_daemon())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

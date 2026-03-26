"""
Market Making Data Pipeline — Download, Process, Validate BTC/USDT klines.

Usage:
    python scripts/mm_data_pipeline.py --symbol BTCUSDT --start 2025-01 --end 2025-12
    python scripts/mm_data_pipeline.py --process-only  # Skip download, just process existing files

Output: data/processed/btc_usdt_mm_1min.parquet
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "binance_raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def download_klines(symbol: str, start_date: str, end_date: str):
    """Download monthly 1-min klines from Binance Vision."""
    sys.path.insert(0, PROJECT_ROOT)
    from finrl_pro_ds.data.binance_loader.downloader import (
        download_file,
        generate_monthly_urls,
    )

    os.makedirs(RAW_DIR, exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m")
    end_dt = datetime.strptime(end_date, "%Y-%m")

    urls = generate_monthly_urls(symbol, start_dt, end_dt)
    # Only download klines (skip depth for Level 1)
    kline_urls = [u for u in urls if u["type"] == "klines"]

    logger.info(f"Downloading {len(kline_urls)} kline files for {symbol}")
    success = 0
    for task in kline_urls:
        path = os.path.join(RAW_DIR, task["filename"])
        if download_file(task["url"], path):
            success += 1

    logger.info(f"Downloaded {success}/{len(kline_urls)} kline files")
    return success > 0


def process_klines(symbol: str, start_date: str, end_date: str, output_path: str):
    """Process downloaded kline zips into a single parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m")
    end_dt = datetime.strptime(end_date, "%Y-%m")

    all_dfs = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        ym = current.strftime("%Y-%m")
        kline_file = os.path.join(RAW_DIR, f"{symbol}-1m-{ym}.zip")

        if not os.path.exists(kline_file):
            logger.warning(f"Missing kline file: {kline_file}, skipping")
            current += relativedelta(months=1)
            continue

        logger.info(f"Processing {kline_file}")
        df = pd.read_csv(kline_file, header=None, compression="zip")
        # Binance kline columns: open_time, O, H, L, C, V, close_time, ...
        df.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tb_base_vol", "tb_quote_vol", "ignore",
        ]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        all_dfs.append(df)
        current += relativedelta(months=1)

    if not all_dfs:
        logger.error("No kline data found")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Validate
    n_bars = len(combined)
    n_nan = combined[["open", "high", "low", "close"]].isna().sum().sum()
    n_invalid = ((combined["high"] < combined["low"]) | (combined["close"] <= 0)).sum()

    logger.info(f"Combined: {n_bars} bars, {n_nan} NaN values, {n_invalid} invalid bars")

    # Drop NaN/invalid
    combined = combined.dropna(subset=["open", "high", "low", "close"])
    combined = combined[combined["close"] > 0]
    combined = combined[combined["high"] >= combined["low"]]

    # Forward-fill gaps
    combined[["open", "high", "low", "close", "volume"]] = combined[
        ["open", "high", "low", "close", "volume"]
    ].ffill()

    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} bars to {output_path}")

    # Summary
    date_range = f"{combined['timestamp'].min()} → {combined['timestamp'].max()}"
    logger.info(f"Date range: {date_range}")
    logger.info(f"Price range: {combined['close'].min():.2f} → {combined['close'].max():.2f}")
    logger.info(f"Mean volume: {combined['volume'].mean():.2f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="MM Data Pipeline — BTC/USDT klines")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--start", default="2025-01", help="Start month (YYYY-MM)")
    parser.add_argument("--end", default="2025-12", help="End month (YYYY-MM)")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--process-only", action="store_true", help="Skip download")
    args = parser.parse_args()

    output = args.output or os.path.join(
        PROCESSED_DIR, f"{args.symbol.lower()}_mm_1min.parquet",
    )

    if not args.process_only:
        download_klines(args.symbol, args.start, args.end)

    process_klines(args.symbol, args.start, args.end, output)


if __name__ == "__main__":
    main()

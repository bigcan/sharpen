"""
Fetch CoinAPI LOB Data
Downloads historical order-book snapshots from CoinAPI REST API,
resamples to 1-minute intervals, and saves as Parquet files compatible
with the DeepScalper pipeline.

Usage:
    python scripts/data/fetch_coinapi_lob.py \
        --api-key YOUR_KEY \
        --start-date 2025-08-01 \
        --end-date 2026-02-01 \
        --output data/raw/coinapi_lob/ \
        --dry-run
"""
import os
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Constants ───────────────────────────────────────────────────────
BASE_URL = "https://rest.coinapi.io/v1"
DEFAULT_SYMBOL = "BINANCE_SPOT_BTC_USDT"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


def fetch_day(api_key: str, symbol: str, date_str: str, limit_levels: int = 5) -> list:
    """
    Fetch all order-book snapshots for a single day from CoinAPI.

    Returns raw JSON list of snapshots.
    """
    url = f"{BASE_URL}/orderbooks/{symbol}/history"
    headers = {"X-CoinAPI-Key": api_key, "Accept": "application/json"}
    params = {
        "date": date_str,
        "limit_levels": limit_levels,
        "limit": 100000,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=300)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited — back off
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  ⚠ Rate limited (429). Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            elif resp.status_code in (500, 502, 503, 504):
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  ⚠ Server error ({resp.status_code}). Retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  ✗ HTTP {resp.status_code}: {resp.text[:200]}")
                return []
        except requests.exceptions.Timeout:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  ⚠ Timeout. Retrying in {wait:.0f}s...")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}")
            return []

    print(f"  ✗ Failed after {MAX_RETRIES} retries for {date_str}")
    return []


def flatten_snapshots(snapshots: list, num_levels: int = 5) -> pd.DataFrame:
    """
    Flatten CoinAPI order-book snapshots into a tabular DataFrame.

    CoinAPI response format per snapshot:
    {
        "symbol_id": "BINANCE_SPOT_BTC_USDT",
        "time_exchange": "2025-08-01T00:00:00.123Z",
        "time_coinapi": "2025-08-01T00:00:00.456Z",
        "asks": [{"price": 65000.1, "size": 0.5}, ...],
        "bids": [{"price": 64999.9, "size": 0.3}, ...]
    }

    Output columns: timestamp, bid_price_1..N, bid_vol_1..N, ask_price_1..N, ask_vol_1..N
    """
    if not snapshots:
        return pd.DataFrame()

    rows = []
    for snap in snapshots:
        ts = snap.get("time_exchange") or snap.get("time_coinapi")
        if not ts:
            continue

        row = {"timestamp": ts}

        # Bids (sorted descending by price — CoinAPI default)
        bids = snap.get("bids", [])
        for i in range(num_levels):
            if i < len(bids):
                row[f"bid_price_{i+1}"] = float(bids[i]["price"])
                row[f"bid_vol_{i+1}"] = float(bids[i]["size"])
            else:
                row[f"bid_price_{i+1}"] = np.nan
                row[f"bid_vol_{i+1}"] = np.nan

        # Asks (sorted ascending by price — CoinAPI default)
        asks = snap.get("asks", [])
        for i in range(num_levels):
            if i < len(asks):
                row[f"ask_price_{i+1}"] = float(asks[i]["price"])
                row[f"ask_vol_{i+1}"] = float(asks[i]["size"])
            else:
                row[f"ask_price_{i+1}"] = np.nan
                row[f"ask_vol_{i+1}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def resample_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick-level LOB snapshots to 1-minute bars.

    Strategy: Take the LAST snapshot in each 1-minute window.
    This gives us the most recent order-book state at each minute boundary.
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Group by 1-minute floor, take last snapshot per group
    resampled = df.resample("1min").last()

    # Drop minutes with no snapshots
    resampled = resampled.dropna(subset=["bid_price_1"])

    resampled = resampled.reset_index()
    return resampled


def load_progress(progress_file: str) -> set:
    """Load set of completed dates from progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: str, completed: set):
    """Save completed dates to progress file."""
    with open(progress_file, "w") as f:
        json.dump(sorted(completed), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CoinAPI LOB Data for DeepScalper"
    )
    parser.add_argument("--api-key", required=False, help="CoinAPI API key (or set COINAPI_KEY env var)")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="data/raw/coinapi_lob/",
        help="Output directory for daily parquets",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"CoinAPI symbol ID (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument("--levels", type=int, default=5, help="LOB depth levels")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Delay between daily fetches (seconds)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch 1 day only for testing"
    )
    parser.add_argument(
        "--concat-only",
        action="store_true",
        help="Skip fetching, just concatenate existing daily files",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("COINAPI_KEY")
    if not api_key:
        parser.error("API Key must be provided via --api-key or COINAPI_KEY env var")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = str(output_dir / "_progress.json")
    final_output = str(output_dir / "coinapi_lob_final.parquet")

    # ── Concat-only mode ──
    if args.concat_only:
        print("Concatenating daily parquet files...")
        _concat_daily_files(output_dir, final_output)
        return

    # ── Date range ──
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if args.dry_run:
        all_dates = all_dates[:1]
        print(f"🧪 Dry run: fetching only {all_dates[0]}")

    # ── Load progress ──
    completed = load_progress(progress_file)
    remaining = [d for d in all_dates if d not in completed]

    print(f"📊 CoinAPI LOB Fetcher")
    print(f"   Symbol:     {args.symbol}")
    print(f"   Date range: {args.start_date} → {args.end_date}")
    print(f"   Total days: {len(all_dates)}")
    print(f"   Completed:  {len(completed)}")
    print(f"   Remaining:  {len(remaining)}")
    print(f"   Levels:     {args.levels}")
    print()

    if not remaining:
        print("✓ All dates already fetched. Use --concat-only to merge.")
        return

    # ── Fetch loop ──
    total_rows = 0
    for idx, date_str in enumerate(remaining):
        print(f"[{idx+1}/{len(remaining)}] Fetching {date_str}...")

        daily_file = output_dir / f"lob_{date_str}.parquet"

        # Fetch raw snapshots
        snapshots = fetch_day(api_key, args.symbol, date_str, args.levels)

        if not snapshots:
            print(f"  ⚠ No data for {date_str}, skipping.")
            completed.add(date_str)
            save_progress(progress_file, completed)
            continue

        print(f"  → {len(snapshots)} raw snapshots")

        # Flatten to DataFrame
        df = flatten_snapshots(snapshots, args.levels)
        if df.empty:
            print(f"  ⚠ Flattening produced empty DataFrame, skipping.")
            completed.add(date_str)
            save_progress(progress_file, completed)
            continue

        # Resample to 1-minute
        df_1min = resample_to_1min(df)
        print(f"  → {len(df_1min)} 1-min snapshots")

        if not df_1min.empty:
            # Sanity checks
            prices = df_1min["bid_price_1"].dropna()
            if len(prices) > 0:
                print(
                    f"  → Price range: ${prices.min():,.2f} – ${prices.max():,.2f}"
                )

            df_1min.to_parquet(daily_file, index=False)
            total_rows += len(df_1min)

        completed.add(date_str)
        save_progress(progress_file, completed)

        # Rate limit delay
        if idx < len(remaining) - 1:
            time.sleep(args.delay)

    print(f"\n✓ Fetching complete. Total 1-min rows: {total_rows:,}")

    # ── Concatenate ──
    print("\nConcatenating daily files...")
    _concat_daily_files(output_dir, final_output)


def _concat_daily_files(output_dir: Path, final_output: str):
    """Concatenate all daily parquet files into a single final file."""
    daily_files = sorted(output_dir.glob("lob_*.parquet"))

    if not daily_files:
        print("✗ No daily parquet files found!")
        return

    print(f"  Found {len(daily_files)} daily files")
    dfs = []
    for f in daily_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Error reading {f.name}: {e}")

    if not dfs:
        print("✗ No valid data files!")
        return

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )

    final_df.to_parquet(final_output, index=False)
    print(f"\n✓ Saved {len(final_df):,} rows to {final_output}")
    print(f"  Date range: {final_df['timestamp'].min()} → {final_df['timestamp'].max()}")
    print(f"  Columns: {final_df.columns.tolist()}")

    # Quick validation
    expected_cols = []
    for i in range(1, 6):
        expected_cols.extend(
            [f"bid_price_{i}", f"bid_vol_{i}", f"ask_price_{i}", f"ask_vol_{i}"]
        )
    missing = [c for c in expected_cols if c not in final_df.columns]
    if missing:
        print(f"  ⚠ WARNING: Missing expected columns: {missing}")
    else:
        print(f"  ✓ All 20 LOB columns present")

    nan_pct = final_df[expected_cols].isna().mean().mean() * 100
    print(f"  NaN coverage: {nan_pct:.1f}%")


if __name__ == "__main__":
    main()

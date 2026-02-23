"""
Fetch GC.c.2 (active Gold contract) MBP-10 data from Databento.
Streams to DBN file, then resamples to 1-min LOB snapshots.

Cost: ~$54.26 for full year 2025.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db
import pandas as pd
import numpy as np

API_KEY = os.environ.get('DATABENTO_API_KEY')
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set")
    sys.exit(1)

client = db.Historical(API_KEY)

DATASET = "GLBX.MDP3"
DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "GC.c.2"
START = "2025-01-01"
END = "2025-12-31"


def fetch_mbp10():
    """Fetch full year MBP-10 tick data and save as DBN file."""
    out_path = RAW_DIR / "gc_2025_mbp10.dbn.zst"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Raw file already exists: {out_path} ({size_mb:.1f} MB)")
        response = input("  Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("  Skipping download, using existing file.")
            return out_path

    print(f"  Fetching MBP-10 for {SYMBOL} ({START} to {END})...")
    print(f"  This may take several minutes for a full year of tick data...")
    t0 = time.time()

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        stype_in="continuous",
        schema="mbp-10",
        start=START,
        end=END,
    )

    # Save raw file first
    data.to_file(str(out_path))
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved raw: {out_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")

    return out_path


def resample_mbp10_to_1min(raw_path: Path) -> pd.DataFrame:
    """
    Load MBP-10 data and resample to 1-minute LOB snapshots.

    Takes the LAST book snapshot per minute (end-of-minute state).
    Extracts 5 levels of depth from the 10 available.
    """
    print(f"\n  Loading MBP-10 from {raw_path}...")
    t0 = time.time()

    store = db.DBNStore.from_file(str(raw_path))
    df = store.to_df()
    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} tick records in {elapsed:.0f}s")
    print(f"  Columns: {list(df.columns)}")

    # Print first few rows to understand structure
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    print(f"\n  Column dtypes:")
    for col in df.columns:
        print(f"    {col}: {df[col].dtype}")

    return df


def build_lob_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MBP-10 tick data to 1-minute LOB snapshots.

    Databento MBP-10 columns include:
    - bid_px_00 through bid_px_09 (10 bid levels)
    - ask_px_00 through ask_px_09 (10 ask levels)
    - bid_sz_00 through bid_sz_09 (bid sizes)
    - ask_sz_00 through ask_sz_09 (ask sizes)
    - price, size, action, side (trade info)
    """
    print(f"\n  Building 1-minute LOB snapshots...")

    # Identify bid/ask columns
    bid_px_cols = [f'bid_px_{i:02d}' for i in range(5)]
    ask_px_cols = [f'ask_px_{i:02d}' for i in range(5)]
    bid_sz_cols = [f'bid_sz_{i:02d}' for i in range(5)]
    ask_sz_cols = [f'ask_sz_{i:02d}' for i in range(5)]

    # Check which columns exist
    available = set(df.columns)
    for col_list, name in [(bid_px_cols, "bid_px"), (ask_px_cols, "ask_px"),
                            (bid_sz_cols, "bid_sz"), (ask_sz_cols, "ask_sz")]:
        missing = [c for c in col_list if c not in available]
        if missing:
            print(f"  WARNING: Missing {name} columns: {missing}")
            # Try alternative naming
            alt_names = [c for c in available if name.replace('_', '') in c.replace('_', '')]
            print(f"  Available alternatives: {alt_names[:10]}")

    # Resample: take last snapshot per minute
    # First, ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'ts_event' in df.columns:
            df = df.set_index('ts_event')
        elif 'timestamp' in df.columns:
            df = df.set_index('timestamp')

    # Remove timezone for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Select LOB columns
    lob_cols = bid_px_cols + ask_px_cols + bid_sz_cols + ask_sz_cols
    existing_lob_cols = [c for c in lob_cols if c in df.columns]

    if not existing_lob_cols:
        print("  ERROR: No LOB columns found. Checking all columns...")
        print(f"  All columns: {sorted(df.columns.tolist())}")
        return pd.DataFrame()

    # Also include price columns for OHLCV construction
    price_col = 'price' if 'price' in df.columns else None
    size_col = 'size' if 'size' in df.columns else None

    # Resample to 1-minute: take LAST snapshot per minute
    lob_1min = df[existing_lob_cols].resample('1min').last()

    # Build OHLCV from trade prices if available
    if price_col and size_col:
        # Filter to actual trades (non-zero price)
        trades = df[df[price_col] > 0][[price_col, size_col]]
        ohlcv = trades[price_col].resample('1min').agg(
            open='first', high='max', low='min', close='last'
        )
        vol = trades[size_col].resample('1min').sum()
        ohlcv['volume'] = vol
    else:
        # Construct from mid prices
        if bid_px_cols[0] in df.columns and ask_px_cols[0] in df.columns:
            mid = (df[bid_px_cols[0]] + df[ask_px_cols[0]]) / 2
            ohlcv = mid.resample('1min').agg(
                open='first', high='max', low='min', close='last'
            )
            ohlcv['volume'] = 0

    # Merge LOB + OHLCV
    result = pd.concat([lob_1min, ohlcv], axis=1)

    # Drop rows where close is NaN (no trading)
    result = result.dropna(subset=['close'])

    # Rename columns to match DeepScalper format
    rename_map = {}
    for i in range(5):
        rename_map[f'bid_px_{i:02d}'] = f'bid_price_{i+1}'
        rename_map[f'ask_px_{i:02d}'] = f'ask_price_{i+1}'
        rename_map[f'bid_sz_{i:02d}'] = f'bid_vol_{i+1}'
        rename_map[f'ask_sz_{i:02d}'] = f'ask_vol_{i+1}'

    result = result.rename(columns=rename_map)

    # Reset index
    result = result.reset_index()
    result = result.rename(columns={result.columns[0]: 'timestamp'})

    # Compute mid_price
    if 'bid_price_1' in result.columns and 'ask_price_1' in result.columns:
        result['mid_price'] = (result['bid_price_1'] + result['ask_price_1']) / 2

    print(f"  Built {len(result):,} 1-minute LOB snapshots")
    print(f"  Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"  Columns: {list(result.columns)}")

    # Validate LOB data quality
    if 'bid_price_1' in result.columns:
        valid_lob = result['bid_price_1'].notna() & (result['bid_price_1'] > 0)
        print(f"  Valid LOB rows: {valid_lob.sum():,} / {len(result):,} "
              f"({valid_lob.mean()*100:.1f}%)")

    if 'bid_price_1' in result.columns and 'ask_price_1' in result.columns:
        spread = result['ask_price_1'] - result['bid_price_1']
        print(f"  Spread stats: mean={spread.mean():.4f}, "
              f"median={spread.median():.4f}, "
              f"min={spread.min():.4f}, max={spread.max():.4f}")

    return result


def main():
    print("=" * 60)
    print("FETCH GC MBP-10 ORDER BOOK DATA")
    print("=" * 60)

    # Step 1: Fetch raw data
    raw_path = fetch_mbp10()

    # Step 2: Load and inspect
    df = resample_mbp10_to_1min(raw_path)

    if df is None or len(df) == 0:
        print("  ERROR: Failed to load data")
        return

    # Step 3: Build LOB snapshots
    result = build_lob_snapshots(df)

    if len(result) == 0:
        print("  ERROR: Failed to build LOB snapshots")
        return

    # Step 4: Save
    out_path = DATA_DIR / "gc_2025_lob_1min.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(result):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Asset: Gold (GC.c.2, active month)")
    print(f"  Period: {START} to {END}")
    print(f"  1-min bars: {len(result):,}")
    print(f"  Trading days: {result['timestamp'].dt.date.nunique()}")
    print(f"  Avg bars/day: {len(result) / max(result['timestamp'].dt.date.nunique(), 1):.0f}")

    if 'close' in result.columns:
        print(f"  Price range: ${result['close'].min():.2f} - ${result['close'].max():.2f}")

    lob_cols = [c for c in result.columns if 'bid_price' in c or 'ask_price' in c]
    print(f"  LOB depth levels: {len(lob_cols) // 2}")

    print(f"\n  Ready for feature engineering and RF signal test.")


if __name__ == "__main__":
    main()

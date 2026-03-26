"""
Fetch Gold (GC) front-month LOB data using individual delivery contracts.
==========================================================================

Databento's continuous contracts (GC.c.0, GC.c.2) produce broken LOB data
at roll boundaries — 127x fewer ticks and 20x worse spreads vs the actual
contract. This script fetches each delivery contract individually during
its front-month period and stitches them into a single clean parquet.

Gold delivery months: Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)
Each contract is front-month for ~2 months before its delivery month.

Roll schedule (approximate — actual roll is last biz day before FND):
  Jan        → GCG5 (Feb delivery)
  Feb-Mar    → GCJ5 (Apr delivery)
  Apr-May    → GCM5 (Jun delivery)
  Jun-Jul    → GCQ5 (Aug delivery)
  Aug-Sep    → GCV5 (Oct delivery)
  Oct-Nov    → GCZ5 (Dec delivery)
  Dec        → GCG6 (Feb 2026 delivery)

Output: data/cme/gc_2025_lob1_1min_stitched.parquet (~200K+ 1-min bars)

Usage:
    python scripts/fetch_gc_front_month.py
    python scripts/fetch_gc_front_month.py --force   # Re-fetch even if exists
"""

import argparse
import gc as garbage_collect
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db  # noqa: E402
import pandas as pd  # noqa: E402

API_KEY = os.environ.get('DATABENTO_API_KEY')
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set in .env")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
OUT_PATH = DATA_DIR / "gc_2025_lob1_1min_stitched.parquet"

# Roll schedule: (symbol, period_start, period_end)
# Each contract is fetched during its front-month window.
# Fetched month-by-month within each window to avoid OOM (~16M ticks/month).
ROLL_SCHEDULE = [
    ("GCG5", "2025-01-01", "2025-01-31"),   # Feb delivery — front-month Jan
    ("GCJ5", "2025-02-01", "2025-03-31"),   # Apr delivery — front-month Feb-Mar
    ("GCM5", "2025-04-01", "2025-05-31"),   # Jun delivery — front-month Apr-May
    ("GCQ5", "2025-06-01", "2025-07-31"),   # Aug delivery — front-month Jun-Jul
    ("GCV5", "2025-08-01", "2025-09-30"),   # Oct delivery — front-month Aug-Sep
    ("GCZ5", "2025-10-01", "2025-11-30"),   # Dec delivery — front-month Oct-Nov
    ("GCG6", "2025-12-01", "2025-12-31"),   # Feb 2026 delivery — front-month Dec
]

LOB_COLS = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']


def fetch_and_resample(client, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch MBP-1 for one month of a specific contract, resample to 1-min."""
    t0 = time.time()
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        stype_in="raw_symbol",
        schema="mbp-1",
        start=start,
        end=end,
    )
    df = data.to_df()
    elapsed = time.time() - t0

    if len(df) == 0:
        return pd.DataFrame()

    print(f"    {symbol} {start}: {len(df):,} ticks ({elapsed:.0f}s)", end="")

    # Remove timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter to only columns that exist
    lob_cols = [c for c in LOB_COLS if c in df.columns]
    if not lob_cols:
        print(" -> NO LOB COLUMNS")
        return pd.DataFrame()

    # Take last LOB snapshot per minute
    df_1min = df[lob_cols].resample('1min').last()

    # OHLCV from trade prices
    if 'price' in df.columns:
        trade_mask = df['price'] > 0
        if trade_mask.any():
            trades = df.loc[trade_mask]
            ohlcv = trades['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last',
            )
            if 'size' in trades.columns:
                ohlcv['volume'] = trades['size'].resample('1min').sum()
            else:
                ohlcv['volume'] = 0
            df_1min = df_1min.join(ohlcv, how='left')

    # Drop empty minutes (no LOB update in that minute)
    df_1min = df_1min.dropna(subset=lob_cols[:1])

    # Rename to standard column names
    rename = {
        'bid_px_00': 'bid_price_1', 'ask_px_00': 'ask_price_1',
        'bid_sz_00': 'bid_vol_1', 'ask_sz_00': 'ask_vol_1',
    }
    df_1min = df_1min.rename(columns=rename)
    df_1min['mid_price'] = (df_1min['bid_price_1'] + df_1min['ask_price_1']) / 2

    # Fill OHLCV gaps with mid_price
    for col in ['close', 'open', 'high', 'low']:
        if col not in df_1min.columns:
            df_1min[col] = df_1min['mid_price']
        else:
            df_1min[col] = df_1min[col].fillna(df_1min['mid_price'])
    if 'volume' not in df_1min.columns:
        df_1min['volume'] = 0
    df_1min['volume'] = df_1min['volume'].fillna(0)

    df_1min = df_1min.reset_index()
    ts_col = 'ts_event' if 'ts_event' in df_1min.columns else df_1min.columns[0]
    df_1min = df_1min.rename(columns={ts_col: 'timestamp'})

    # Tag source contract
    df_1min['contract'] = symbol

    print(f" -> {len(df_1min):,} 1-min bars")

    # Free memory
    del df, data
    garbage_collect.collect()

    return df_1min


def fetch_contract_period(client, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a contract's full front-month period, month by month."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    all_chunks = []
    current = datetime.strptime(start, "%Y-%m-%d")
    period_end = datetime.strptime(end, "%Y-%m-%d")

    while current <= period_end:
        # Fetch one month at a time
        month_end = min(current + relativedelta(months=1), period_end + relativedelta(days=1))
        try:
            chunk = fetch_and_resample(
                client, symbol,
                current.strftime("%Y-%m-%d"),
                month_end.strftime("%Y-%m-%d"),
            )
            if len(chunk) > 0:
                all_chunks.append(chunk)
        except Exception as e:
            print(f"    {symbol} {current.strftime('%Y-%m')}: ERROR - {e}")

        current = month_end

    if not all_chunks:
        return pd.DataFrame()

    return pd.concat(all_chunks, ignore_index=True)


def validate_stitched(df: pd.DataFrame) -> None:
    """Print data quality report for stitched output."""
    print(f"\n{'='*70}")
    print("DATA QUALITY REPORT")
    print(f"{'='*70}")

    print(f"  Total 1-min bars: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Monthly bar counts
    df_ts = pd.to_datetime(df['timestamp'])
    monthly = df_ts.dt.to_period('M').value_counts().sort_index()
    print("\n  Monthly bar counts:")
    for period, count in monthly.items():
        print(f"    {period}: {count:,} bars")

    # Spread analysis
    mid = (df['bid_price_1'] + df['ask_price_1']) / 2
    spread_bps = ((df['ask_price_1'] - df['bid_price_1']) / mid) * 10000
    print("\n  Spread (bps):")
    print(f"    median={spread_bps.median():.2f}, mean={spread_bps.mean():.2f}")
    print(f"    p5={spread_bps.quantile(0.05):.2f}, p95={spread_bps.quantile(0.95):.2f}")
    tight = (spread_bps < 5).mean() * 100
    print(f"    Bars < 5 bps: {tight:.1f}%")

    # Monthly spread
    print("\n  Monthly median spread (bps):")
    for month in sorted(df_ts.dt.month.unique()):
        mask = df_ts.dt.month == month
        sp = spread_bps[mask]
        print(f"    Month {month:2d}: median={sp.median():.2f} bps ({mask.sum():,} bars)")

    # Contract source breakdown
    if 'contract' in df.columns:
        print("\n  Contract sources:")
        for contract, count in df['contract'].value_counts().sort_index().items():
            print(f"    {contract}: {count:,} bars")

    # NaN check
    nan_count = df.isnull().sum().sum()
    print(f"\n  NaN cells: {nan_count}")

    # Crossed quotes
    crossed = (df['ask_price_1'] <= df['bid_price_1']).sum()
    print(f"  Crossed quotes: {crossed}")

    # Price range
    print(f"  Gold price range: ${df['bid_price_1'].min():.2f} - ${df['bid_price_1'].max():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GC front-month LOB data from individual delivery contracts")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if output file exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not args.force:
        print(f"Already exists: {OUT_PATH}")
        print("Use --force to re-fetch.")
        return

    client = db.Historical(API_KEY)
    all_segments = []

    print("="*70)
    print("GOLD FRONT-MONTH LOB FETCH — Individual Delivery Contracts")
    print("="*70)

    for symbol, start, end in ROLL_SCHEDULE:
        print(f"\n  [{symbol}] Fetching {start} to {end}...")
        try:
            segment = fetch_contract_period(client, symbol, start, end)
            if len(segment) > 0:
                all_segments.append(segment)
                print(f"  [{symbol}] Total: {len(segment):,} 1-min bars")
            else:
                print(f"  [{symbol}] WARNING: No data returned")
        except Exception as e:
            print(f"  [{symbol}] ERROR: {e}")

    if not all_segments:
        print("\nERROR: No data fetched from any contract")
        sys.exit(1)

    # Concatenate all segments
    print(f"\nStitching {len(all_segments)} contract segments...")
    result = pd.concat(all_segments, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'])

    # Sort by timestamp, deduplicate (shouldn't have overlaps with clean schedule)
    result = result.sort_values('timestamp').reset_index(drop=True)

    # Check for duplicate timestamps (overlap at roll boundaries)
    dupes = result.duplicated(subset=['timestamp'], keep='first')
    n_dupes = dupes.sum()
    if n_dupes > 0:
        print(f"  Removing {n_dupes} duplicate timestamps at roll boundaries")
        result = result[~dupes].reset_index(drop=True)

    # Save
    result.to_parquet(OUT_PATH, index=False, engine='pyarrow')
    print(f"\nSaved: {OUT_PATH} ({len(result):,} rows, {OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # Validate
    validate_stitched(result)


if __name__ == "__main__":
    main()

"""
Process GC MBP-10 data in chunks to extract 1-min LOB snapshots.

The full year file is 9.1 GB compressed, too large for single-load.
Strategy: fetch fresh quarterly data via Databento API (already paid for),
process each quarter to 1-min snapshots, concatenate.

Actually: Databento charges per byte, so re-fetching costs money.
Instead: use the replay() iterator on the existing DBN file to stream
through records without loading everything into memory.

Output: gc_2025_lob_1min.parquet with 5-level LOB + OHLCV
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db  # noqa: E402
import pandas as pd  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "cme"
RAW_PATH = DATA_DIR / "raw" / "gc_2025_mbp10.dbn.zst"


def process_with_replay():
    """
    Stream through MBP-10 records using replay() iterator.
    Accumulate 1-minute LOB snapshots without loading full dataset.
    """
    print(f"  Loading DBN store from {RAW_PATH}...")
    print(f"  File size: {RAW_PATH.stat().st_size / 1e9:.2f} GB")

    store = db.DBNStore.from_file(str(RAW_PATH))

    # Get metadata
    print(f"  Schema: {store.schema}")
    print(f"  Symbol map: {store.symbology_resolution}")

    # State for accumulating 1-min snapshots
    current_minute = None
    last_snapshot = {}
    snapshots = []
    record_count = 0
    t0 = time.time()

    print("  Streaming through records...")

    for record in store.replay():
        record_count += 1

        # Progress every 10M records
        if record_count % 10_000_000 == 0:
            elapsed = time.time() - t0
            rate = record_count / elapsed
            print(f"    {record_count/1e6:.0f}M records processed "
                  f"({elapsed:.0f}s, {rate/1e6:.1f}M/s, "
                  f"{len(snapshots)} minutes accumulated)")

        # Extract timestamp and truncate to minute
        ts = pd.Timestamp(record.ts_event, unit='ns', tz='UTC')
        minute = ts.floor('min')

        # When minute changes, save the last snapshot
        if current_minute is not None and minute != current_minute:
            if last_snapshot:
                snapshots.append(last_snapshot.copy())
            last_snapshot = {}

        current_minute = minute

        # Extract LOB levels from record
        # MBP-10 record has .levels attribute with 10 BidAskPair entries
        try:
            snapshot = {'timestamp': current_minute.tz_localize(None)}

            # Extract 5 levels
            for i in range(min(5, len(record.levels))):
                level = record.levels[i]
                snapshot[f'bid_price_{i+1}'] = level.bid_px / 1e9  # Fixed-point to float
                snapshot[f'ask_price_{i+1}'] = level.ask_px / 1e9
                snapshot[f'bid_vol_{i+1}'] = float(level.bid_sz)
                snapshot[f'ask_vol_{i+1}'] = float(level.ask_sz)

            # Extract trade price for OHLCV construction
            price = record.price / 1e9 if hasattr(record, 'price') and record.price > 0 else None
            size = record.size if hasattr(record, 'size') else 0

            if price and price > 0:
                if 'open' not in last_snapshot or last_snapshot['open'] is None:
                    snapshot['open'] = price
                snapshot['close'] = price
                snapshot['high'] = max(last_snapshot.get('high', 0), price)
                snapshot['low'] = min(last_snapshot.get('low', float('inf')), price)
                snapshot['volume'] = last_snapshot.get('volume', 0) + size
            else:
                # Keep OHLCV from previous records in this minute
                snapshot['open'] = last_snapshot.get('open')
                snapshot['close'] = last_snapshot.get('close')
                snapshot['high'] = last_snapshot.get('high', 0)
                snapshot['low'] = last_snapshot.get('low', float('inf'))
                snapshot['volume'] = last_snapshot.get('volume', 0)

            last_snapshot = snapshot

        except Exception as e:
            if record_count < 10:
                print(f"    Record {record_count} error: {e}")
                print(f"    Record type: {type(record)}")
                print(f"    Record attrs: {[a for a in dir(record) if not a.startswith('_')]}")
            continue

    # Don't forget the last minute
    if last_snapshot:
        snapshots.append(last_snapshot.copy())

    elapsed = time.time() - t0
    print(f"  Processed {record_count:,} records in {elapsed:.0f}s")
    print(f"  Generated {len(snapshots):,} 1-minute snapshots")

    # Build DataFrame
    df = pd.DataFrame(snapshots)

    return df


def process_with_ndarray_chunks():
    """
    Alternative: use to_ndarray with chunked iteration.
    Falls back to this if replay() is too slow.
    """
    print(f"  Loading DBN store from {RAW_PATH}...")
    store = db.DBNStore.from_file(str(RAW_PATH))

    print(f"  Schema: {store.schema}")

    # Try chunked ndarray iteration
    chunk_size = 5_000_000  # 5M records per chunk
    all_snapshots = []
    chunk_num = 0
    t0 = time.time()

    print(f"  Processing in chunks of {chunk_size:,} records...")

    try:
        # Use the internal iterator
        for chunk_arr in store.to_ndarray_iter(chunk_size):
            chunk_num += 1
            chunk_df = pd.DataFrame(chunk_arr)

            # Convert ts_event to datetime
            if 'ts_event' in chunk_df.columns:
                chunk_df['ts_event'] = pd.to_datetime(chunk_df['ts_event'], unit='ns', utc=True)
                chunk_df['minute'] = chunk_df['ts_event'].dt.floor('min')
            elif chunk_df.index.name == 'ts_event':
                chunk_df['minute'] = chunk_df.index.floor('min')

            # Take last record per minute
            lob_cols = []
            for i in range(5):
                for prefix in ['bid_px', 'ask_px', 'bid_sz', 'ask_sz']:
                    col = f'{prefix}_{i:02d}'
                    if col in chunk_df.columns:
                        lob_cols.append(col)

            if lob_cols:
                minute_snapshots = chunk_df.groupby('minute')[lob_cols].last()
                all_snapshots.append(minute_snapshots)

            elapsed = time.time() - t0
            print(f"    Chunk {chunk_num}: {len(chunk_df):,} records → "
                  f"{len(minute_snapshots):,} minutes ({elapsed:.0f}s)")

    except AttributeError:
        print("  to_ndarray_iter not available, trying alternative...")
        # Try to_df with limited rows by date
        return process_by_quarter()

    # Concatenate and deduplicate
    if all_snapshots:
        df = pd.concat(all_snapshots)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        # Rename columns
        rename = {}
        for i in range(5):
            rename[f'bid_px_{i:02d}'] = f'bid_price_{i+1}'
            rename[f'ask_px_{i:02d}'] = f'ask_price_{i+1}'
            rename[f'bid_sz_{i:02d}'] = f'bid_vol_{i+1}'
            rename[f'ask_sz_{i:02d}'] = f'ask_vol_{i+1}'
        df = df.rename(columns=rename)
        df = df.reset_index()
        df = df.rename(columns={'minute': 'timestamp'})
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        return df

    return pd.DataFrame()


def process_by_quarter():
    """
    Fetch quarterly data via API (uses existing cost credit).
    Most reliable approach for large datasets.
    """
    API_KEY = os.environ.get('DATABENTO_API_KEY')
    if not API_KEY:
        print("  ERROR: No API key for quarterly fetch")
        return pd.DataFrame()

    client = db.Historical(API_KEY)

    quarters = [
        ("2025-01-01", "2025-04-01"),
        ("2025-04-01", "2025-07-01"),
        ("2025-07-01", "2025-10-01"),
        ("2025-10-01", "2026-01-01"),
    ]

    all_dfs = []
    for i, (start, end) in enumerate(quarters):
        print(f"\n  Quarter {i+1}/4: {start} to {end}")

        try:
            # Check cost first
            cost = client.metadata.get_cost(
                dataset="GLBX.MDP3",
                symbols=["GC.c.2"],
                stype_in="continuous",
                schema="mbp-10",
                start=start,
                end=end,
            )
            print(f"    Estimated cost: ${cost:.2f}")

            # Fetch
            t0 = time.time()
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["GC.c.2"],
                stype_in="continuous",
                schema="mbp-10",
                start=start,
                end=end,
            )

            df = data.to_df()
            elapsed = time.time() - t0
            print(f"    Fetched {len(df):,} records in {elapsed:.0f}s")

            # Resample to 1-minute
            df_1min = resample_quarter(df)
            all_dfs.append(df_1min)

            # Free memory
            del df
            import gc
            gc.collect()

        except MemoryError:
            print(f"    MemoryError on Q{i+1} — trying monthly...")
            # Fall back to monthly
            monthly_dfs = process_quarter_monthly(client, start, end)
            all_dfs.extend(monthly_dfs)
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return result

    return pd.DataFrame()


def process_quarter_monthly(client, q_start, q_end):
    """Process a quarter month by month if quarterly is too large."""
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    start_dt = datetime.strptime(q_start, "%Y-%m-%d")
    end_dt = datetime.strptime(q_end, "%Y-%m-%d")
    results = []

    current = start_dt
    while current < end_dt:
        next_month = current + relativedelta(months=1)
        if next_month > end_dt:
            next_month = end_dt

        start_str = current.strftime("%Y-%m-%d")
        end_str = next_month.strftime("%Y-%m-%d")
        print(f"      Month: {start_str} to {end_str}")

        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["GC.c.2"],
                stype_in="continuous",
                schema="mbp-10",
                start=start_str,
                end=end_str,
            )
            df = data.to_df()
            print(f"      {len(df):,} records")
            df_1min = resample_quarter(df)
            results.append(df_1min)
            del df
            import gc
            gc.collect()
        except Exception as e:
            print(f"      Error: {e}")

        current = next_month

    return results


def resample_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Resample tick-level MBP-10 to 1-minute LOB snapshots."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'ts_event' in df.columns:
            df = df.set_index('ts_event')

    # Remove timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Identify LOB columns (first 5 levels)
    lob_cols = []
    for i in range(5):
        for prefix in ['bid_px', 'ask_px', 'bid_sz', 'ask_sz']:
            col = f'{prefix}_{i:02d}'
            if col in df.columns:
                lob_cols.append(col)

    # Take last snapshot per minute
    df_1min = df[lob_cols].resample('1min').last().dropna(how='all')

    # Build OHLCV from trade prices
    if 'price' in df.columns:
        trades = df[df['price'] > 0]
        if len(trades) > 0:
            ohlcv = trades['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last'
            )
            vol = trades['size'].resample('1min').sum() if 'size' in trades.columns else 0
            df_1min = df_1min.join(ohlcv, how='left')
            if isinstance(vol, pd.Series):
                df_1min['volume'] = vol

    # Rename columns to DeepScalper format
    rename = {}
    for i in range(5):
        rename[f'bid_px_{i:02d}'] = f'bid_price_{i+1}'
        rename[f'ask_px_{i:02d}'] = f'ask_price_{i+1}'
        rename[f'bid_sz_{i:02d}'] = f'bid_vol_{i+1}'
        rename[f'ask_sz_{i:02d}'] = f'ask_vol_{i+1}'
    df_1min = df_1min.rename(columns=rename)

    # Clean up
    df_1min = df_1min.reset_index()
    ts_col = 'ts_event' if 'ts_event' in df_1min.columns else df_1min.columns[0]
    df_1min = df_1min.rename(columns={ts_col: 'timestamp'})

    # Add mid_price
    if 'bid_price_1' in df_1min.columns and 'ask_price_1' in df_1min.columns:
        df_1min['mid_price'] = (df_1min['bid_price_1'] + df_1min['ask_price_1']) / 2

    # Drop rows with no LOB data
    df_1min = df_1min.dropna(subset=['bid_price_1'])

    print(f"    Resampled to {len(df_1min):,} 1-minute snapshots")
    return df_1min


def main():
    print("=" * 70)
    print("PROCESS GC MBP-10 → 1-MIN LOB SNAPSHOTS")
    print("=" * 70)

    if not RAW_PATH.exists():
        print(f"  Raw file not found: {RAW_PATH}")
        print("  Falling back to quarterly API fetch...")
        df = process_by_quarter()
    else:
        # Try ndarray chunks first (fastest), fall back to quarterly
        try:
            df = process_with_ndarray_chunks()
        except Exception as e:
            print(f"  ndarray approach failed: {e}")
            print("  Falling back to quarterly API fetch...")
            df = process_by_quarter()

    if len(df) == 0:
        print("  ERROR: No data processed")
        return

    # Save
    out_path = DATA_DIR / "gc_2025_lob_1min.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Columns: {list(df.columns)}")

    # Validate
    if 'bid_price_1' in df.columns:
        valid = df['bid_price_1'].notna() & (df['bid_price_1'] > 0)
        print(f"  Valid LOB rows: {valid.sum():,} / {len(df):,} ({valid.mean()*100:.1f}%)")

    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        spread = df['ask_price_1'] - df['bid_price_1']
        valid_spread = spread[spread > 0]
        if len(valid_spread) > 0:
            print(f"  Spread: mean={valid_spread.mean():.4f}, "
                  f"median={valid_spread.median():.4f}")

    print(f"\n  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Trading days: {pd.to_datetime(df['timestamp']).dt.date.nunique()}")


if __name__ == "__main__":
    main()

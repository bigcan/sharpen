"""
G0: Gold (GC) Data Preparation — 1-min LOB → 5-min OHLCV+LOB
================================================================

Input:  data/cme/gc_2025_lob1_1min.parquet  (237K rows, 11 cols)
Output: data/processed/gc_2025_full_year_5min.parquet (~47K bars)

Steps:
  1. Load raw 1-min Gold LOB1 data
  2. Clean: drop/ffill NaN rows, filter crossed quotes (ask <= bid)
  3. Filter CME maintenance window (22:00-23:00 UTC daily) + weekends
  4. Resample 1-min → 5-min: OHLCV agg + LOB last snapshot
  5. Save processed parquet

Splits (configured in YAML, not embedded here):
  Train: Jan-Oct 2025
  Val:   Nov 2025
  Test:  Dec 2025
  norm_cutoff_date: 2025-11-01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw GC LOB1 1-min parquet."""
    print(f"[G0] Loading {path}")
    try:
        df = pd.read_parquet(path, engine='fastparquet')
    except Exception:
        df = pd.read_parquet(path, engine='pyarrow')
    print(f"[G0] Raw shape: {df.shape}, columns: {df.columns.tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NaN rows and crossed quotes."""
    n_before = len(df)

    # Ensure timestamp column
    if 'timestamp' not in df.columns and df.index.name == 'timestamp':
        df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Count NaN rows
    nan_mask = df.isnull().any(axis=1)
    n_nan = nan_mask.sum()
    print(f"[G0] NaN rows: {n_nan}")

    # Forward-fill NaN (LOB snapshots can have brief gaps)
    df = df.ffill()

    # Drop any remaining NaN (e.g. at start)
    df = df.dropna().reset_index(drop=True)

    # Filter crossed quotes (ask <= bid)
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        crossed = df['ask_price_1'] <= df['bid_price_1']
        n_crossed = crossed.sum()
        print(f"[G0] Crossed quotes: {n_crossed}")
        df = df[~crossed].reset_index(drop=True)

    print(f"[G0] After cleaning: {n_before} → {len(df)} rows")
    return df


def filter_cme_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter CME maintenance window (22:00-23:00 UTC daily) and weekends."""
    n_before = len(df)

    # CME Gold futures: Sunday 18:00 ET - Friday 17:00 ET
    # Maintenance window: 17:00-18:00 ET daily = 22:00-23:00 UTC
    hour = df['timestamp'].dt.hour
    maintenance_mask = (hour == 22)
    n_maintenance = maintenance_mask.sum()
    df = df[~maintenance_mask].reset_index(drop=True)

    # Filter weekends (Saturday=5, Sunday=6)
    # Note: CME opens Sunday evening, so some Sunday data is valid.
    # We filter Saturday entirely and Sunday before 23:00 UTC (18:00 ET open).
    dow = df['timestamp'].dt.dayofweek
    hour_vals = df['timestamp'].dt.hour  # re-fetch after maintenance filter
    saturday_mask = (dow == 5)
    sunday_before_open = (dow == 6) & (hour_vals < 23)
    weekend_mask = saturday_mask | sunday_before_open
    n_weekend = weekend_mask.sum()
    df = df[~weekend_mask.values].reset_index(drop=True)

    print(f"[G0] Filtered: {n_maintenance} maintenance + {n_weekend} weekend rows")
    print(f"[G0] After CME filter: {n_before} → {len(df)} rows")
    return df


def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min → 5-min with OHLCV aggregation + LOB last snapshot."""
    df = df.set_index('timestamp')

    # Define aggregation rules
    agg_rules = {}

    # OHLCV columns
    ohlcv_map = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    for col, func in ohlcv_map.items():
        if col in df.columns:
            agg_rules[col] = func

    # LOB columns — take last snapshot in 5-min bar
    lob_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ('bid_price_', 'ask_price_', 'bid_vol_', 'ask_vol_')
    )]
    for col in lob_cols:
        agg_rules[col] = 'last'

    # Any other numeric columns — take last
    for col in df.columns:
        if col not in agg_rules:
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                agg_rules[col] = 'last'

    resampled = df.resample('5min').agg(agg_rules)

    # Drop bars with no LOB data (empty 5-min windows from non-trading hours).
    # resample('5min') creates bins for the ENTIRE time range including overnight
    # gaps — these produce NaN in LOB columns. Drop them.
    if 'bid_price_1' in resampled.columns:
        resampled = resampled.dropna(subset=['bid_price_1'])
    else:
        resampled = resampled.dropna(how='all')

    # Forward-fill any remaining OHLCV gaps (rare — should be near-zero after LOB filter)
    for col in ['open', 'high', 'low', 'close']:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()

    # Forward-fill LOB volumes (very rare 1-2 bar gaps within session)
    for col in lob_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()

    # Drop any remaining NaN rows (e.g. very start of data)
    resampled = resampled.dropna()

    resampled = resampled.reset_index()
    print(f"[G0] After 5-min resample: {len(resampled)} bars")
    return resampled


def clamp_spread(df: pd.DataFrame, max_spread_bps: float) -> pd.DataFrame:
    """Clamp wide spreads to max_spread_bps around mid-price.

    GC.c.2 data has extreme spreads in odd months due to contract roll.
    This clamps bid/ask symmetrically around mid to simulate front-month
    liquidity. Preserves mid_price and OHLCV unchanged.
    """
    if 'bid_price_1' not in df.columns or 'ask_price_1' not in df.columns:
        return df

    mid = (df['bid_price_1'] + df['ask_price_1']) / 2
    spread_bps = (df['ask_price_1'] - df['bid_price_1']) / df['bid_price_1'] * 10000
    wide_mask = spread_bps > max_spread_bps
    n_wide = wide_mask.sum()

    if n_wide > 0:
        # Clamp: set bid/ask to mid ± half of max spread
        half_spread = mid * (max_spread_bps / 10000) / 2
        df.loc[wide_mask, 'bid_price_1'] = (mid - half_spread)[wide_mask]
        df.loc[wide_mask, 'ask_price_1'] = (mid + half_spread)[wide_mask]
        # Update mid_price if present
        if 'mid_price' in df.columns:
            df.loc[wide_mask, 'mid_price'] = mid[wide_mask]
        pct = n_wide / len(df) * 100
        print(f"[G0] Spread clamp: {n_wide}/{len(df)} bars ({pct:.1f}%) clamped to {max_spread_bps:.1f} bps")

    return df


def validate_output(df: pd.DataFrame) -> bool:
    """Validate output has no NaN, no crossed quotes, reasonable bar count."""
    issues = []

    # Check NaN
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isnull().any()].tolist()
        issues.append(f"NaN found in {nan_count} cells, columns: {nan_cols}")

    # Check crossed quotes
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        crossed = (df['ask_price_1'] <= df['bid_price_1']).sum()
        if crossed > 0:
            issues.append(f"{crossed} crossed quotes remain")

    # Check bar count (expect ~45-50K for full year 5-min)
    n = len(df)
    if n < 30000:
        issues.append(f"Only {n} bars (expected ~45-50K)")
    elif n > 80000:
        issues.append(f"{n} bars seems too high (expected ~45-50K)")

    if issues:
        print("[G0] VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print(f"[G0] VALIDATION PASSED: {n} bars, no NaN, no crossed quotes")
    return True


def main():
    parser = argparse.ArgumentParser(description="G0: Prepare Gold (GC) 5-min data")
    parser.add_argument("--input", default="data/cme/gc_2025_lob1_1min.parquet",
                        help="Path to raw 1-min LOB1 parquet")
    parser.add_argument("--output", default="data/processed/gc_2025_full_year_5min.parquet",
                        help="Output path for processed 5-min parquet")
    parser.add_argument("--no-resample", action="store_true",
                        help="Skip resampling (output 1-min cleaned data)")
    parser.add_argument("--max-spread-bps", type=float, default=0,
                        help="Clamp spreads wider than this (bps). 0=disabled. "
                             "Recommended: 5.0 for GC.c.2 data (fixes contract-roll spread noise)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.exists(input_path):
        print(f"[G0] ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Pipeline
    df = load_raw_data(input_path)
    df = clean_data(df)
    df = filter_cme_hours(df)

    if not args.no_resample:
        df = resample_5min(df)

    # Spread clamp (fix GC.c.2 contract-roll spread noise)
    if args.max_spread_bps > 0:
        df = clamp_spread(df, args.max_spread_bps)

    # Validate
    ok = validate_output(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"[G0] Saved to {output_path}")

    # Summary
    print("\n[G0] Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    if 'bid_price_1' in df.columns:
        print(f"  Gold price range: ${df['bid_price_1'].min():.2f} - ${df['bid_price_1'].max():.2f}")

    if not ok:
        print("\n[G0] WARNING: Validation failed — check output before using in pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()

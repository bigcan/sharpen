"""
Prepare Bitfinex BTC Perp 5-min Data for DeepScalper Pipeline.
===============================================================

Input:  data/bitfinex/btc_usdt_perp_2025_1min.parquet
Output: data/processed/btc_bitfinex_2025_full_year_5min.parquet

Steps:
  1. Load raw 1-min OHLCV from Bitfinex
  2. Resample 1-min → 5-min: OHLCV aggregation
  3. Synthesize LOB-like columns from OHLCV (bid/ask from high-low spread)
  4. Validate output
  5. Save processed parquet

Note: Bitfinex OHLCV-only data has NO real LOB (bid/ask/volume at each level).
We synthesize bid_price_1/ask_price_1 from the bar range as a proxy:
  - mid_price = (open + close) / 2
  - half_spread = estimated from high-low range (scaled)
  - bid_price_1 = mid - half_spread
  - ask_price_1 = mid + half_spread

This is a reasonable proxy for backtesting but NOT equivalent to true MBP-1 data.
For real LOB features, we need Tardis.dev or self-collected WebSocket snapshots.

Splits (same as BTC full-year):
  Train: Jan-Oct 2025
  Val:   Nov 2025
  Test:  Dec 2025
  norm_cutoff_date: 2025-11-01

Usage:
  python scripts/prepare_bitfinex_5min.py
  python scripts/prepare_bitfinex_5min.py --spread-method fixed --fixed-spread-bps 1.0
"""

import argparse
import os
import sys

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw Bitfinex 1-min OHLCV parquet."""
    print(f"[PREP] Loading {path}")
    df = pd.read_parquet(path, engine='pyarrow')
    print(f"[PREP] Raw shape: {df.shape}, columns: {df.columns.tolist()}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min → 5-min with standard OHLCV aggregation."""
    df = df.set_index('timestamp')

    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    # Include mid_price if present
    if 'mid_price' in df.columns:
        agg_rules['mid_price'] = 'last'

    resampled = df.resample('5min').agg(agg_rules)

    # Drop empty bars (no trading in that 5-min window)
    resampled = resampled.dropna(subset=['close'])
    resampled = resampled[resampled['volume'] > 0]

    resampled = resampled.reset_index()
    print(f"[PREP] After 5-min resample: {len(resampled):,} bars")
    return resampled


def synthesize_lob(df: pd.DataFrame, method: str = 'hl_range',
                   fixed_spread_bps: float = 1.0) -> pd.DataFrame:
    """Synthesize bid/ask columns from OHLCV data.

    Methods:
      - 'hl_range': Estimate spread from high-low range (Parkinson-inspired).
        Typical BTC perp spread ≈ 0.5-2 bps on liquid venues.
        We use: spread ≈ (high - low) * scale_factor, clamped to reasonable range.
      - 'fixed': Use a fixed spread in bps around mid-price.
    """
    mid = (df['open'] + df['close']) / 2

    if method == 'hl_range':
        # High-low range as spread proxy
        # For 5-min bars, hl_range >> actual spread, so we scale down
        hl_range = df['high'] - df['low']
        # Typical BTC 5-min bar range ≈ 20-50 bps, actual spread ≈ 1-3 bps
        # Scale factor: spread ≈ range * 0.05 (empirically reasonable)
        estimated_spread = hl_range * 0.05
        # Clamp to [0.5, 5.0] bps
        min_spread = mid * 0.5 / 10000
        max_spread = mid * 5.0 / 10000
        estimated_spread = estimated_spread.clip(lower=min_spread, upper=max_spread)
    elif method == 'fixed':
        estimated_spread = mid * fixed_spread_bps / 10000
    else:
        raise ValueError(f"Unknown spread method: {method}")

    half_spread = estimated_spread / 2
    df['bid_price_1'] = mid - half_spread
    df['ask_price_1'] = mid + half_spread
    df['mid_price'] = mid

    # Synthesize volume at L1 (rough proxy: split total volume)
    df['bid_vol_1'] = df['volume'] * 0.5
    df['ask_vol_1'] = df['volume'] * 0.5

    # Report spread stats
    spread_bps = (df['ask_price_1'] - df['bid_price_1']) / df['bid_price_1'] * 10000
    print(f"[PREP] Synthesized LOB ({method}):")
    print(f"  Spread: mean={spread_bps.mean():.2f} bps, "
          f"median={spread_bps.median():.2f} bps, "
          f"min={spread_bps.min():.2f}, max={spread_bps.max():.2f}")

    return df


def validate_output(df: pd.DataFrame) -> bool:
    """Validate output has no NaN, reasonable bar count."""
    issues = []

    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isnull().any()].tolist()
        issues.append(f"NaN found in {nan_count} cells, columns: {nan_cols}")

    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        crossed = (df['ask_price_1'] <= df['bid_price_1']).sum()
        if crossed > 0:
            issues.append(f"{crossed} crossed quotes remain")

    n = len(df)
    # BTC trades 24/7 = 365 days * 288 bars/day = ~105K bars
    if n < 50000:
        issues.append(f"Only {n} bars (expected ~100K for full year 5-min)")
    elif n > 150000:
        issues.append(f"{n} bars seems too high (expected ~100K)")

    if issues:
        print(f"[PREP] VALIDATION WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print(f"[PREP] VALIDATION PASSED: {n:,} bars, no NaN, no crossed quotes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Bitfinex BTC 5-min data")
    parser.add_argument("--input", default="data/bitfinex/btc_usdt_perp_2025_1min.parquet",
                        help="Path to raw 1-min Bitfinex parquet")
    parser.add_argument("--output", default="data/processed/btc_bitfinex_2025_full_year_5min.parquet",
                        help="Output path for processed 5-min parquet")
    parser.add_argument("--spread-method", default="hl_range",
                        choices=["hl_range", "fixed"],
                        help="Spread estimation method")
    parser.add_argument("--fixed-spread-bps", type=float, default=1.0,
                        help="Fixed spread in bps (only used with --spread-method fixed)")
    parser.add_argument("--no-resample", action="store_true",
                        help="Skip resampling (keep 1-min)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.exists(input_path):
        print(f"[PREP] ERROR: Input file not found: {input_path}")
        sys.exit(1)

    df = load_raw_data(input_path)

    # Resample
    if not args.no_resample:
        df = resample_5min(df)

    # Synthesize LOB columns
    df = synthesize_lob(df, method=args.spread_method,
                        fixed_spread_bps=args.fixed_spread_bps)

    # Validate
    ok = validate_output(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\n[PREP] Saved to {output_path}")

    # Summary
    print(f"\n[PREP] Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  BTC price range: ${df['close'].min():,.2f} — ${df['close'].max():,.2f}")
    print(f"  Trading days: {df['timestamp'].dt.date.nunique()}")

    # Monthly bar counts
    monthly = df.groupby(df['timestamp'].dt.month).size()
    print(f"\n  Monthly bar counts:")
    for month, count in monthly.items():
        print(f"    Month {month:>2}: {count:>6,} bars")

    if not ok:
        print("\n[PREP] WARNING: Validation had warnings — check output before pipeline use")


if __name__ == "__main__":
    main()

"""
Resample 1-Min Parquet Data to 3-Min Bars
==========================================

Resamples OHLCV + LOB data from 1-minute to 3-minute resolution.

Resampling rules (consistent with dp_oracle_1min.py):
  - open: first
  - high: max
  - low: min
  - close: last
  - volume: sum
  - mid_price: last
  - LOB columns (bid_price_1, ask_price_1, bid_vol_1, ask_vol_1, etc.): last
  - contract (if present): last
  - timestamp: first of each 3-min group

Usage:
    # Single file
    python scripts/resample_3min.py --input data/bitfinex/btc_usdt_perp_2025_1min.parquet \
                                     --output data/processed/btc_bitfinex_2025_3min.parquet

    # Batch mode: process both BTC and Gold
    python scripts/resample_3min.py --batch
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# --- Default batch paths (relative to repo root) ---
BATCH_JOBS = [
    {
        "name": "BTC",
        "input": "data/bitfinex/btc_usdt_perp_2025_1min.parquet",
        "output": "data/processed/btc_bitfinex_2025_3min.parquet",
        "expected_bars": 145_000,
    },
    {
        "name": "Gold",
        "input": "data/cme/gc_2025_lob1_1min_stitched.parquet",
        "output": "data/processed/gc_2025_3min_front.parquet",
        "expected_bars": 113_000,
    },
]

# Columns with fixed aggregation rules
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "mid_price": "last",
}

# LOB / metadata columns: take the last value in each window
LOB_COLUMNS = [
    "bid_price_1", "ask_price_1",
    "bid_vol_1", "ask_vol_1",
    "contract",
]


def resample_to_3min(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Read a 1-min parquet file, resample to 3-min bars, write output.

    Returns the resampled DataFrame for summary reporting.
    """
    df = pd.read_parquet(input_path)
    n_input = len(df)
    print(f"  Input:  {input_path}")
    print(f"  Rows:   {n_input:,}")
    print(f"  Cols:   {list(df.columns)}")

    # --- Ensure datetime index on timestamp ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("No 'timestamp' column and index is not DatetimeIndex")

    # --- Build aggregation dict ---
    agg = {}
    for col, func in OHLCV_AGG.items():
        if col in df.columns:
            agg[col] = func

    for col in LOB_COLUMNS:
        if col in df.columns:
            agg[col] = "last"

    # Catch any extra numeric columns not in our known sets
    known_cols = set(OHLCV_AGG.keys()) | set(LOB_COLUMNS)
    extra_cols = [c for c in df.columns if c not in known_cols]
    if extra_cols:
        print(f"  WARNING: Unknown columns (ignored): {extra_cols}")

    # --- Resample ---
    resampled = df.resample("3min").agg(agg)

    # --- Drop NaN in critical OHLC columns ---
    critical = [c for c in ["open", "high", "low", "close"] if c in resampled.columns]
    n_before = len(resampled)
    resampled = resampled.dropna(subset=critical)
    n_dropped = n_before - len(resampled)

    # --- Reset index: timestamp becomes a column (first ts of each group) ---
    resampled = resampled.reset_index()

    # --- Write output ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(str(out_path), index=False, engine="pyarrow")

    # --- Summary stats ---
    n_output = len(resampled)
    ts = resampled["timestamp"]
    date_min = ts.iloc[0]
    date_max = ts.iloc[-1]

    print(f"\n  Output: {output_path}")
    print(f"  Bars:   {n_output:,}  (from {n_input:,} 1-min bars)")
    print(f"  Ratio:  {n_input / n_output:.2f}:1")
    print(f"  Range:  {date_min} -> {date_max}")
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with NaN in OHLC columns")

    # --- Gap detection ---
    diffs = ts.diff().dropna()
    expected_delta = pd.Timedelta(minutes=3)
    gaps = diffs[diffs > expected_delta]
    if len(gaps) > 0:
        print(f"\n  Gaps detected: {len(gaps):,} intervals > 3 min")
        # Show top 10 largest gaps
        top_gaps = gaps.sort_values(ascending=False).head(10)
        for idx, gap in top_gaps.items():
            gap_start = ts.iloc[idx - 1] if idx > 0 else "?"
            print(f"    {gap_start} -> gap of {gap}")
    else:
        print("  No gaps detected (continuous 3-min bars)")

    return resampled


def main():
    parser = argparse.ArgumentParser(
        description="Resample 1-min parquet data to 3-min bars",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input 1-min parquet file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output 3-min parquet file")
    parser.add_argument("--batch", action="store_true",
                        help="Process both BTC and Gold with default paths")
    args = parser.parse_args()

    if not args.batch and (args.input is None or args.output is None):
        print("ERROR: Provide --input and --output, or use --batch mode.")
        parser.print_help()
        sys.exit(1)

    if args.batch:
        print("=" * 70)
        print("BATCH MODE: Resampling 1-min -> 3-min for all datasets")
        print("=" * 70)

        for job in BATCH_JOBS:
            print(f"\n{'—' * 60}")
            print(f"  Dataset: {job['name']}")
            print(f"{'—' * 60}")

            inp = job["input"]
            if not Path(inp).exists():
                print(f"  SKIPPED: Input not found at {inp}")
                continue

            result = resample_to_3min(inp, job["output"])
            n = len(result)
            expected = job["expected_bars"]
            pct = n / expected * 100 if expected > 0 else 0
            print(f"\n  Expected ~{expected:,} bars, got {n:,} ({pct:.0f}%)")

        print(f"\n{'=' * 70}")
        print("BATCH COMPLETE")
        print(f"{'=' * 70}")

    else:
        print("=" * 70)
        print("Resampling 1-min -> 3-min")
        print("=" * 70)
        resample_to_3min(args.input, args.output)
        print("\nDone.")


if __name__ == "__main__":
    main()

"""
Preprocess Jan–Jun 2025 CoinAPI LOB Data for DeepScalper Pipeline
=================================================================
Transforms raw 1-minute LOB snapshots (5-level) into a model-ready Parquet file
with both Micro and Macro features pre-computed.

Input:  data/raw/coinapi_lob/coinapi_lob_final.parquet  (237k rows, 21 cols)
Output: data/processed/btc_2025_jan_jun.parquet

Key Design Decision — Synthetic OHLCV:
  Since the raw data contains only LOB snapshots (no true OHLCV bars), we
  approximate OHLCV from the LOB data:
    - Open / High / Low / Close  ≈  Mid Price  = (bid_price_1 + ask_price_1) / 2
    - Volume  ≈  Total Liquidity Depth  = sum(bid_vol_1..5 + ask_vol_1..5)
  This is an approximation. Mid-price is used uniformly for O/H/L/C because
  each row is a single snapshot — there is no intra-bar price variation.
  Volume is proxied by total visible depth, NOT actual trade volume.
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from finrl_pro_ds.data.feature_engineering import (  # noqa: E402
    DeepScalperFeatureEngineer,
)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "coinapi_lob" / "coinapi_lob_final.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "btc_2025_jan_jun.parquet"

# Number of initial rows to drop for SMA warm-up (longest SMA window = 30)
WARMUP_ROWS = 30


def main():
    print("=" * 70)
    print("DeepScalper 2025 Data Preprocessor")
    print("=" * 70)

    # ── Step 0: Load Raw Data ────────────────────────────────────────────
    print(f"\n[1/6] Loading raw LOB data from {RAW_PATH}...")
    df = pd.read_parquet(RAW_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Strip timezone to produce naive datetime64[ns] (consistent with existing data)
    if hasattr(df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None:
        print(f"  Stripping timezone ({df['timestamp'].dt.tz}) → naive UTC...")
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Sort by timestamp (safety)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── Step 1: Synthetic OHLCV Generation ───────────────────────────────
    print("\n[2/6] Generating synthetic OHLCV from LOB snapshots...")
    # Mid Price = (best_bid + best_ask) / 2
    # NOTE: Since each row is a single 1-min snapshot, O=H=L=C=Mid.
    # This is an approximation — true OHLCV would require tick-level data.
    mid = (df['bid_price_1'].values + df['ask_price_1'].values) / 2.0
    df['open'] = mid
    df['high'] = mid
    df['low'] = mid
    df['close'] = mid

    # Volume ≈ Total visible depth across all 5 levels (bid + ask)
    # NOTE: This is NOT trade volume. It's a liquidity depth proxy.
    vol_cols = [f'{side}_vol_{i}' for i in range(1, 6) for side in ('bid', 'ask')]
    df['volume'] = df[vol_cols].sum(axis=1)

    print(f"  Mid price range: ${mid.min():,.2f} – ${mid.max():,.2f}")
    print(f"  Synthetic volume range: {df['volume'].min():.4f} – {df['volume'].max():.4f}")

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    fe = DeepScalperFeatureEngineer()

    # 2a. Micro Features
    print("\n[3/6] Processing Micro features (OFI, spread, log_ret, normalization)...")
    df = fe.process_micro(df)

    # Collect micro columns added by process_micro
    micro_added = [c for c in df.columns if c.startswith(('mid_price', 'spread', 'vol_imbalance', 'log_ret', 'n_'))]
    print(f"  Micro features added: {len(micro_added)} columns")

    # 2b. Macro Features
    print("\n[4/6] Processing Macro features (z-scores, SMAs)...")
    macro_df = fe.process_macro(df)

    # Merge macro columns back into the main DataFrame
    # Same row count (1:1 LOB-to-OHLCV), so direct column assignment is safe.
    assert len(macro_df) == len(df), (
        f"Row count mismatch: micro={len(df)}, macro={len(macro_df)}"
    )
    for col in macro_df.columns:
        df[col] = macro_df[col].values

    macro_cols = list(macro_df.columns)
    print(f"  Macro features added: {len(macro_cols)} columns → {macro_cols}")

    # ── Step 3: Data Cleaning ────────────────────────────────────────────
    print(f"\n[5/6] Cleaning data (dropping first {WARMUP_ROWS} warm-up rows)...")

    # Drop SMA warm-up rows
    df = df.iloc[WARMUP_ROWS:].reset_index(drop=True)
    print(f"  Rows after warm-up drop: {df.shape[0]:,}")

    # Assert 0 NaNs
    nan_counts = df.isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print("  ⚠️  NaN columns found:")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {cnt} NaNs")
        # Fill remaining NaNs with 0 as safety net
        df = df.fillna(0.0)
        print("  → Filled remaining NaNs with 0.0")
    else:
        print("  ✓ 0 NaN values — clean dataset")

    # Final sort
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── Step 4: Save ─────────────────────────────────────────────────────
    print(f"\n[6/6] Saving processed dataset to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.1f} MB")

    # ── Quality Report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUALITY REPORT")
    print("=" * 70)
    print(f"  Rows:        {df.shape[0]:,}")
    print(f"  Columns:     {df.shape[1]}")
    print(f"  Date range:  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  NaN total:   {df.isnull().sum().sum()}")
    print(f"  File size:   {file_size_mb:.1f} MB")
    print("\n  Column groups:")

    # Group columns for readability
    lob_cols = [c for c in df.columns if any(c.startswith(p) for p in ('bid_', 'ask_'))]
    norm_cols = [c for c in df.columns if c.startswith('n_')]
    ofi_cols = [c for c in df.columns if c.startswith('vol_imbalance')]
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    macro_z = [c for c in df.columns if c.startswith(('z_', 'zd_'))]
    other_cols = [c for c in df.columns if c not in lob_cols + norm_cols + ofi_cols + ohlcv_cols + macro_z + ['timestamp', 'mid_price', 'spread_1', 'log_ret']]

    print(f"    LOB raw:        {len(lob_cols)} cols")
    print(f"    Normalized:     {len(norm_cols)} cols")
    print(f"    OFI:            {len(ofi_cols)} cols")
    print(f"    Synthetic OHLCV:{len([c for c in ohlcv_cols if c in df.columns])} cols")
    print(f"    Macro z-scores: {len(macro_z)} cols")
    print("    Other:          mid_price, spread_1, log_ret" + (f", {other_cols}" if other_cols else ""))

    # Spot-check macro feature ranges
    print("\n  Macro feature ranges (should be within ±100 bps):")
    for col in macro_z[:4]:
        vals = df[col].values
        print(f"    {col:12s}: [{vals.min():+8.2f}, {vals.max():+8.2f}]  mean={vals.mean():+.4f}")

    print("\n  ✓ Dataset ready for DeepScalper training")
    print(f"  → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

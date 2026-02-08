"""
Data Quality Audit for DeepScalper Pipeline.
Inspects the parquet file used for training and validates:
1. Column presence & types
2. Value ranges (prices, volumes, features)
3. NaN / Inf counts
4. Feature distributions (mean, std, min, max)
5. Temporal coverage & gaps
6. Normalization sanity (are n_bid_price_1 etc. generated?)
"""
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "data/btc_lob_jan2023.parquet"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def audit_raw_data(df):
    """Audit the raw parquet before feature engineering."""
    section("1. RAW PARQUET AUDIT")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")

    # Check for critical columns
    critical = [
        'timestamp', 'bid_price_1', 'ask_price_1',
        'bid_vol_1', 'ask_vol_1'
    ]
    for col in critical:
        present = col in df.columns
        print(f"  {'✓' if present else '✗'} {col}: {'FOUND' if present else 'MISSING'}")

    # NaN / Inf check
    section("2. NaN / Inf AUDIT")
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"Columns with NaNs ({len(nan_cols)}):")
        for col, count in nan_cols.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("No NaN values found. ✓")

    # Check for Inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col].values).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf
    if inf_counts:
        print(f"\nColumns with Inf ({len(inf_counts)}):")
        for col, count in inf_counts.items():
            print(f"  {col}: {count}")
    else:
        print("No Inf values found. ✓")

    # Price sanity
    section("3. PRICE SANITY")
    for col in ['bid_price_1', 'ask_price_1']:
        if col in df.columns:
            vals = df[col].values
            print(f"  {col}: min={np.nanmin(vals):.2f}, max={np.nanmax(vals):.2f}, "
                  f"mean={np.nanmean(vals):.2f}, std={np.nanstd(vals):.2f}")
            zeros = (vals == 0).sum()
            negatives = (vals < 0).sum()
            print(f"    Zeros: {zeros}, Negatives: {negatives}")

    # Spread sanity
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        spread = df['ask_price_1'].values - df['bid_price_1'].values
        mid = (df['ask_price_1'].values + df['bid_price_1'].values) / 2.0
        spread_bps = (spread / mid) * 10000
        print(f"\n  Spread (raw): min={spread.min():.4f}, max={spread.max():.4f}, mean={spread.mean():.4f}")
        print(f"  Spread (bps): min={spread_bps.min():.2f}, max={spread_bps.max():.2f}, mean={spread_bps.mean():.2f}")
        crossed = (spread < 0).sum()
        print(f"  Crossed spreads: {crossed}")

    # Volume sanity
    section("4. VOLUME SANITY")
    for i in range(1, 6):
        for side in ['bid', 'ask']:
            col = f'{side}_vol_{i}'
            if col in df.columns:
                vals = df[col].values
                print(f"  {col}: min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}, "
                      f"mean={np.nanmean(vals):.4f}, zeros={int((vals == 0).sum())}")

    # Temporal coverage
    section("5. TEMPORAL COVERAGE")
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        print(f"  Start: {ts.min()}")
        print(f"  End:   {ts.max()}")
        print(f"  Duration: {ts.max() - ts.min()}")
        # Check for gaps
        diffs = ts.diff().dropna()
        print(f"  Median interval: {diffs.median()}")
        print(f"  Max interval:    {diffs.max()}")
        print(f"  Min interval:    {diffs.min()}")

    # Check if pre-computed features exist
    section("6. PRE-COMPUTED FEATURES CHECK")
    norm_cols = [f'n_bid_price_{i}' for i in range(1, 6)] + [f'n_ask_price_{i}' for i in range(1, 6)]
    macro_cols = ['z_open', 'z_high', 'z_low', 'z_close', 'z_adj_close']
    derived_cols = ['mid_price', 'spread_1', 'log_ret', 'vol_imbalance_1']

    for label, cols in [("Normalized LOB", norm_cols), ("Macro Z-features", macro_cols), ("Derived Micro", derived_cols)]:
        found = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        print(f"  {label}: {len(found)}/{len(cols)} present")
        if missing:
            print(f"    Missing: {missing}")

    # OHLCV check (needed for macro generation)
    section("7. OHLCV CHECK (for Macro Feature Generation)")
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv:
        if col in df.columns:
            vals = df[col].dropna().values
            print(f"  {col}: min={np.nanmin(vals):.2f}, max={np.nanmax(vals):.2f}, "
                  f"mean={np.nanmean(vals):.2f}, unique={len(np.unique(vals))}")
        else:
            print(f"  {col}: MISSING")


def audit_processed_data():
    """Run feature engineering and audit the processed output."""
    section("8. FEATURE ENGINEERING OUTPUT AUDIT")
    try:
        from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

        handler = ParquetDataHandler(
            file_path=DATA_PATH,
            ticker="BTCUSDT",
            feature_config={"volatility_horizon": 100},
            start_date="2023-01-10 00:00:00",
            end_date="2023-01-18 23:59:59"
        )

        print(f"  Handler loaded: {handler._len} rows")
        print(f"  Feature columns ({len(handler._feature_cols)}): {handler._feature_cols[:20]}...")

        # Check if normalized columns exist after processing
        norm_present = [c for c in handler._feature_cols if c.startswith('n_')]
        print(f"  Normalized columns: {len(norm_present)}")
        if norm_present:
            print(f"    Examples: {norm_present[:5]}")

        # Sample first 5 rows
        handler.reset()
        print(f"\n  First 5 rows sample:")
        for i in range(min(5, handler._len)):
            row = handler.step()
            if row is None:
                break
            # Print key values
            bid1 = row.get('bid_price_1', 'N/A')
            ask1 = row.get('ask_price_1', 'N/A')
            n_bid1 = row.get('n_bid_price_1', 'N/A')
            n_ask1 = row.get('n_ask_price_1', 'N/A')
            mid = row.get('mid_price', 'N/A')
            spread = row.get('spread_1', 'N/A')
            log_ret = row.get('log_ret', 'N/A')
            vi1 = row.get('vol_imbalance_1', 'N/A')
            print(f"    Row {i}: bid1={bid1}, ask1={ask1}, n_bid1={n_bid1}, n_ask1={n_ask1}, "
                  f"mid={mid}, spread={spread}, log_ret={log_ret}, vi1={vi1}")

        # Statistics on normalized features
        if norm_present:
            print(f"\n  Normalized Feature Statistics:")
            for col in norm_present[:10]:
                if col in handler._data_arrays:
                    arr = handler._data_arrays[col]
                    print(f"    {col}: min={arr.min():.4f}, max={arr.max():.4f}, "
                          f"mean={arr.mean():.4f}, std={arr.std():.4f}")

        # Check macro features
        macro_cols = ['z_close', 'z_open', 'zd_5']
        print(f"\n  Macro Feature Statistics:")
        for col in macro_cols:
            if col in handler._data_arrays:
                arr = handler._data_arrays[col]
                print(f"    {col}: min={arr.min():.4f}, max={arr.max():.4f}, "
                      f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                      f"zeros={int((arr == 0).sum())}/{len(arr)}")
            else:
                print(f"    {col}: NOT IN DATA ARRAYS")

        handler.close()

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("DeepScalper Data Quality Audit")
    print(f"File: {DATA_PATH}")

    try:
        df = pd.read_parquet(DATA_PATH, engine='fastparquet')
    except Exception:
        df = pd.read_parquet(DATA_PATH, engine='pyarrow')

    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    audit_raw_data(df)
    audit_processed_data()

    section("AUDIT COMPLETE")

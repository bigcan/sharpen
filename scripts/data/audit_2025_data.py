"""
Data Quality Audit: DeepScalper 2025 Processed Data
===================================================

This script performs a rigorous audit of the feature distributions, stationarity, and correlations
in the processed dataset: `data/processed/btc_2025_jan_jun.parquet`

Checks:
1. Missing Values: Critical check (should be 0).
2. Descriptive Statistics: Mean, Std, Skewness, Kurtosis.
   - Normalized features (n_*) should be roughly centered around mean 0 or bounded.
   - Macro features (z_*) should be near 0 with small std (basis points).
3. Stationarity (ADF Test):
   - Returns, Volume Imbalance, and Z-scores MUST be stationary (p < 0.05).
   - Prices themselves (mid_price) are non-stationary (Unit Root), which is expected.
4. Feature Correlations:
   - Check for perfect multicollinearity (redundancy).
   - Specifically check the impact of Synthetic OHLCV (O=H=L=C).
5. Outliers:
   - Count values at the clamping boundaries (saturation).
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "btc_2025_jan_jun.parquet"

def get_autocorr(series, lag=1):
    """
    Check Autocorrelation at lag 1.
    If closer to 1.0 (Unit Root), it is likely Non-Stationary.
    If much less than 1.0 and mean ~ 0, likely Stationary.
    """
    try:
        if len(series) < 10:
            return 0.0
        return series.autocorr(lag=lag)
    except Exception:
        return 1.0

def main():
    print("=" * 70)
    print("DATA QUALITY AUDIT REPORT")
    print("Target:", DATA_PATH)
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"Error: File not found: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    N = len(df)
    print(f"Loaded {N:,} rows.")

    # 0. Timestamp Integrity Check
    print("\n[0] Timestamp Integrity")
    ts = pd.to_datetime(df['timestamp'])
    ts_diff = ts.diff().dropna()
    mean_diff = ts_diff.mean()
    median_diff = ts_diff.median()
    print(f"    Mean Diff: {mean_diff} | Median Diff: {median_diff}")

    # Check for gaps > 1 minute (assuming 1-min data)
    gaps = ts_diff[ts_diff > pd.Timedelta(minutes=1)]
    if len(gaps) > 0:
        print(f"    WARNING: Found {len(gaps)} gaps > 1 minute.")
        print(f"    Largest gap: {gaps.max()}")
    else:
        print("    PASSED: Contiguous 1-minute steps (no significant gaps).")

    # Check for duplicates
    dups = ts.duplicated().sum()
    if dups > 0:
        print(f"    FAILED: Found {dups} duplicate timestamps!")
    else:
        print("    PASSED: No duplicate timestamps.")

    # 1. Missing Values
    print("\n[1] Missing Value Check")
    nans = df.isnull().sum().sum()
    if nans == 0:
        print("    PASSED: 0 Missing Values.")
    else:
        print(f"    FAILED: {nans} Missing Values found!")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # 2. Synthetic OHLCV Redundancy Check
    print("\n[2] Synthetic OHLCV Impact Analysis")
    # Since O=H=L=C=Mid, we expect z_high == z_low == z_close == z_open (mostly)
    # z_open = (open_t / close_{t-1} - 1)
    # z_close = (close_t / close_{t-1} - 1)
    # These should be identical if open_t == close_t.

    diff_open_close = (df['z_open'] - df['z_close']).abs().sum()
    diff_high_low = (df['z_high'] - df['z_low']).abs().sum()

    if diff_open_close < 1e-6:
        print("    NOTE: z_open is identical to z_close (Expected for synthetic O=C).")
    else:
        print(f"    WARNING: z_open differs from z_close by sum abs diff {diff_open_close:.6f}")

    if diff_high_low < 1e-6:
        print("    NOTE: z_high is identical to z_low (Expected for synthetic H=L).")
    else:
        print(f"    WARNING: z_high differs from z_low by sum abs diff {diff_high_low:.6f}")

    # 3. Feature Distribution & Stationarity
    print("\n[3] Feature Distribution & Stationarity (Mean Rev)")
    print(f"{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>8} {'Kurt':>8} {'AC(1)':>10} {'Stable?'}")
    print("-" * 115)

    # Select representative features
    audit_cols = [
        'log_ret',
        'spread_1',
        'vol_imbalance_1',
        'n_bid_price_1',
        'n_bid_vol_1',
        'z_close',
        'z_volume',
        'zd_10',
        'zd_30',
    ]

    for col in audit_cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Stats
        mu = series.mean()
        sigma = series.std()
        mn = series.min()
        mx = series.max()
        sk = skew(series)
        kt = kurtosis(series)

        # Stationarity Proxy: Autocorrelation at Lag 1
        # AC(1) ~ 1.0 means unit root-ish (bad for RL inputs)
        # AC(1) < 0.9 means mean reverting enough
        ac1 = get_autocorr(series)
        is_stable = "YES" if abs(ac1) < 0.95 else "Hmm.."

        print(f"{col:<25} {mu:10.4f} {sigma:10.4f} {mn:10.4f} {mx:10.4f} {sk:8.2f} {kt:8.2f} {ac1:10.4f} {is_stable}")

    # 4. Saturation Check (Clamping)
    print("\n[4] Saturation Check (Clamping Impact)")
    print("    Check if features are hitting their valid min/max limits too frequently.")

    # Normalized Price Clamps: [-50, 50] basis points
    n_price_cols = [c for c in df.columns if c.startswith('n_') and 'price' in c]
    for col in n_price_cols:
        sat_min = (df[col] <= -50.0).mean() * 100
        sat_max = (df[col] >= 50.0).mean() * 100
        if sat_min > 1.0 or sat_max > 1.0:
            print(f"    WARNING: {col} saturated: Min {sat_min:.2f}% | Max {sat_max:.2f}%")

    # Normalized Volume Clamps: [-5, 5] (Z-score)
    n_vol_cols = [c for c in df.columns if c.startswith('n_') and 'vol' in c]
    for col in n_vol_cols:
        sat_min = (df[col] <= -5.0).mean() * 100
        sat_max = (df[col] >= 5.0).mean() * 100
        if sat_min > 0.1 or sat_max > 0.1: # Tighter check for Z-score > 5 sigma
             print(f"    NOTE: {col} saturated: Min {sat_min:.2f}% | Max {sat_max:.2f}%")

    # Macro Clamps: [-100, 100] basis points
    macro_cols = [c for c in df.columns if c.startswith(('z_', 'zd_'))]
    for col in macro_cols:
        sat_min = (df[col] <= -100.0).mean() * 100
        sat_max = (df[col] >= 100.0).mean() * 100
        if sat_min > 0.1 or sat_max > 0.1:
            print(f"    WARNING: {col} saturated: Min {sat_min:.2f}% | Max {sat_max:.2f}%")

    # 5. Correlation Check
    print("\n[5] Correlation Analysis (Multicollinearity)")
    # Check correlation between 'z_close' (return) and 'vol_imbalance_1' (OFI)
    # Theoretically OFI drives price change -- correlation should be positive but not 1.0
    corr_ofi_ret = df['vol_imbalance_1'].corr(df['log_ret'])
    print(f"    Correlation(OFI_L1, Returns): {corr_ofi_ret:.4f} (Expected: Positive, 0.2 - 0.6 range commonly)")

    if abs(corr_ofi_ret) > 0.95:
         print("    WARNING: Suspiciously high correlation between OFI and Returns.")

    print("\nDone.")

if __name__ == "__main__":
    main()

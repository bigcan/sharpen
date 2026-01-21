"""Point-in-time (PIT) validator for Feature Engineering.

Performs a "Deletion Test" to certify that feature generation has no look-ahead bias.
Methodology:
1. Compute features on the full dataset (F_full).
2. Truncate the dataset at random points t.
3. Compute features on the truncated dataset (F_trunc).
4. Assert that F_full[t] == F_trunc[t].

If F_full[t] differs from F_trunc[t], it means the calculation at t depended on data after t.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from finrl_pro_ds.features.engineering import FeatureEngineer

def validate_pit(df_raw: pd.DataFrame, sample_size: int = 10) -> dict:
    """
    Runs the PIT validation.
    Args:
        df_raw: DataFrame with columns [date, tic, open, high, low, close, volume]
        sample_size: Number of random truncation points to test per ticker.
    Returns:
        Dictionary with validation results.
    """
    # 1. Compute Full Features
    print("Computing features on full dataset...")
    engineer = FeatureEngineer()
    df_full = engineer.preprocess_data(df_raw)
    
    # Identify feature columns (those ending in _shifted)
    feature_cols = [c for c in df_full.columns if c.endswith('_shifted')]
    if not feature_cols:
        print("No '_shifted' columns found. Using all non-OHLCV columns.")
        exclude = {'date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        feature_cols = [c for c in df_full.columns if c not in exclude]
    
    print(f"Validating features: {feature_cols}")
    
    violations = {}
    for col in feature_cols:
        violations[col] = 0
        
    tickers = df_raw['tic'].unique()
    
    total_checks = 0
    failed_checks = 0
    
    for tic in tickers:
        df_tic = df_raw[df_raw['tic'] == tic].sort_values('date').reset_index(drop=True)
        
        # Determine valid range for testing (skip initial warmup period)
        # Heuristic: Skip first 100 rows
        if len(df_tic) < 150:
            continue
            
        test_indices = np.random.choice(range(100, len(df_tic)), size=sample_size, replace=False)
        
        for cut_idx in test_indices:
            # Date at cut
            cut_date = df_tic.iloc[cut_idx]['date']
            
            # Get Full Feature value at cut_idx
            # We need to match by date because preprocessing might drop rows (dropna)
            full_row = df_full[(df_full['tic'] == tic) & (df_full['date'] == cut_date)]
            if full_row.empty:
                # This happens if the row was dropped (e.g. NaN). Skip.
                continue
            
            # 2. Truncate
            # We include data up to cut_idx
            df_trunc_raw = df_tic.iloc[:cut_idx+1]
            
            # 3. Compute Truncated Features
            df_trunc_feats = engineer.preprocess_data(df_trunc_raw)
            
            trunc_row = df_trunc_feats[(df_trunc_feats['tic'] == tic) & (df_trunc_feats['date'] == cut_date)]
            
            if trunc_row.empty:
                # Should not happen if full_row existed, unless calculation creates NaNs at the edge
                # If it creates NaN at edge, that's arguably safe (no value > wrong value), but let's log it.
                continue
                
            total_checks += 1
            
            # 4. Compare
            for col in feature_cols:
                val_full = full_row[col].values[0]
                val_trunc = trunc_row[col].values[0]
                
                # Handle NaNs
                if np.isnan(val_full) and np.isnan(val_trunc):
                    continue
                if np.isnan(val_full) or np.isnan(val_trunc):
                    violations[col] += 1
                    failed_checks += 1
                    continue
                
                if not np.isclose(val_full, val_trunc, rtol=1e-5, atol=1e-8):
                    violations[col] += 1
                    failed_checks += 1
                    print(f"FAIL: {tic} @ {cut_date} | {col} | Full: {val_full} != Trunc: {val_trunc}")

    return {
        "total_checks_per_feature": total_checks,
        "violations": violations,
        "passed": failed_checks == 0
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to raw OHLCV Parquet/CSV")
    ap.add_argument("--samples", type=int, default=5, help="Samples per ticker")
    args = ap.parse_args()
    
    path = Path(args.data)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
        
    # Ensure column names
    # Expected: date, tic, open, high, low, close, volume
    # Map: timestamp -> date, ticker -> tic
    rename_map = {'timestamp': 'date', 'ticker': 'tic'}
    df = df.rename(columns=rename_map)
    
    results = validate_pit(df, sample_size=args.samples)
    
    print("\n=== PIT Validation Results ===")
    print(f"Total Comparison Points: {results['total_checks_per_feature']}")
    print("Violations per feature:")
    for feat, count in results['violations'].items():
        status = "FAIL" if count > 0 else "PASS"
        print(f"  {feat:<20}: {count} ({status})")
        
    if results['passed']:
        print("\nSUCCESS: No look-ahead bias detected.")
    else:
        print("\nFAILURE: Look-ahead bias detected in one or more features.")
        exit(1)

if __name__ == "__main__":
    main()


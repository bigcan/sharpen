import pandas as pd
import numpy as np
from finrl_pro_ds.features.custom_features import add_fracdiff_features, FracDiffConfig

def verify_fracdiff():
    csv_path = "tmp/features_to_validate.csv"
    df = pd.read_csv(csv_path)
    
    # Ensure we have 'tic' column or add a dummy one if missing (parquet might have dropped it or it's single ticker)
    if "tic" not in df.columns:
        df["tic"] = "SPY" # Assuming single asset for now based on file context
    
    # Ensure date is sorted (it should be)
    if "date" not in df.columns:
        # create dummy date index if missing
        df["date"] = pd.date_range(start="2016-01-01", periods=len(df))

    # Keep original values to compare
    original_fracdiff = df["fd_close_d0p5_w256"].copy()
    
    # Re-calculate
    # config from features.meta.json: d=0.5, window=256, min_weight=1e-5
    cfg = FracDiffConfig(d=0.5, window=256, min_weight=1e-5)
    
    # We need to pass a dataframe with 'close', 'date', 'tic'
    # and we expect the function to append 'fd_close_d0p5_w256'
    
    # Drop the existing feature column to avoid collision or confusion? 
    # The function overwrites or appends. Let's operate on a copy.
    df_recalc = df[["date", "tic", "close"]].copy()
    df_recalc = add_fracdiff_features(df_recalc, cfg)
    
    new_fracdiff = df_recalc["fd_close_d0p5_w256"]
    
    # Compare
    # Note: NaN handling.
    valid_mask = ~np.isnan(original_fracdiff) & ~np.isnan(new_fracdiff)
    diff = np.abs(original_fracdiff[valid_mask] - new_fracdiff[valid_mask])
    
    max_diff = diff.max() if len(diff) > 0 else 0.0
    
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-9:
        print("FracDiff verification PASS: Re-calculated values match CSV values.")
    else:
        print("FracDiff verification FAIL: values mismatch.")
        # Show some examples
        print(pd.concat([original_fracdiff, new_fracdiff], axis=1).dropna().head())

if __name__ == "__main__":
    verify_fracdiff()

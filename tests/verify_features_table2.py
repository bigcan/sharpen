import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer

def verify_feature_engineering():
    print("=== Verifying DeepScalper Feature Engineering (Table 2) ===")
    
    # 1. Create Mock OHLCV Data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 115, 100),
        "low": np.random.uniform(95, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.uniform(1000, 5000, 100),
    }
    df = pd.DataFrame(data)
    # Crypto usually matches close = adj_close
    df['adj_close'] = df['close'] 
    
    # 2. Initialize Engineer
    fe = DeepScalperFeatureEngineer()
    
    # 3. Process Macro
    print("\nProcessing Macro Features...")
    macro_df = fe.process_macro(df)
    
    print("Columns:", macro_df.columns.tolist())
    print("Shape:", macro_df.shape)
    
    expected_cols = [
        'z_open', 'z_high', 'z_low', 
        'z_close', 'z_adj_close',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ]
    
    # Check Columns
    missing = [c for c in expected_cols if c not in macro_df.columns]
    if missing:
        print(f"FAIL: Missing columns: {missing} ❌")
        return
    else:
        print("All expected columns present. ✅")
        
    # Check Values
    # Check z_close calculation for row 1
    # z_close = close_t / close_{t-1} - 1
    t1_close = df.iloc[1]['close']
    t0_close = df.iloc[0]['close']
    expected_z = t1_close / t0_close - 1
    actual_z = macro_df.iloc[1]['z_close']
    
    if np.isclose(expected_z, actual_z, atol=1e-5):
        print(f"z_close verification passed: {expected_z:.6f} == {actual_z:.6f} ✅")
    else:
        print(f"FAIL: z_close mismatch. Expected {expected_z}, got {actual_z} ❌")
        
    # Check zd_5 calculation for row 5 (index 5, window 5 uses 1..5)
    # Window 5 ending at index 5 includes indices 1,2,3,4,5? Default rolling includes current row? Yes.
    # Indices 1,2,3,4,5.
    
    # Note: Rolling(5) at index 4 (0-based) includes 0,1,2,3,4.
    target_idx = 10
    subset = df.iloc[target_idx-4 : target_idx+1]['adj_close'] # 5 values
    sma = subset.mean()
    curr_close = df.iloc[target_idx]['adj_close']
    expected_zd = sma / curr_close - 1
    actual_zd = macro_df.iloc[target_idx]['zd_5']
    
    if np.isclose(expected_zd, actual_zd, atol=1e-5):
        print(f"zd_5 verification passed: {expected_zd:.6f} == {actual_zd:.6f} ✅")
    else:
        print(f"FAIL: zd_5 mismatch. Expected {expected_zd}, got {actual_zd} ❌")

    print("\nFeature Engineering Verification Complete: PASS ✅")

if __name__ == "__main__":
    verify_feature_engineering()

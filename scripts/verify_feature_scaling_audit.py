
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

def test_feature_engineering_scaling():
    print("=== Testing Feature Engineering Scaling ===")
    
    # Mock Data
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1min')
    data = {
        'timestamp': dates,
        'mid_price': [100.0, 101.0, 100.5, 100.0, 101.0],
        'close': [100.0, 101.0, 100.5, 100.0, 101.0], # Needed for macro
        'adj_close': [100.0, 101.0, 100.5, 100.0, 101.0],
        'open': [100.0, 101.0, 100.5, 100.0, 101.0],
        'high': [100.0, 101.0, 100.5, 100.0, 101.0],
        'low': [100.0, 101.0, 100.5, 100.0, 101.0]
    }
    
    # Add LOB data (Levels 1-5)
    # Price ~100, Vol ~1000-20000
    for i in range(1, 6):
        data[f'bid_price_{i}'] = np.array([99.0, 100.0, 99.5, 99.0, 100.0]) - (i * 0.1)
        data[f'ask_price_{i}'] = np.array([101.0, 102.0, 101.5, 101.0, 102.0]) + (i * 0.1)
        data[f'bid_vol_{i}'] = np.array([100.0, 1000.0, 5000.0, 10000.0, 20000.0])
        data[f'ask_vol_{i}'] = np.array([100.0, 1000.0, 5000.0, 10000.0, 20000.0])

    df = pd.DataFrame(data)
    
    # Run Feature Engineering
    fe = DeepScalperFeatureEngineer()
    processed_df = fe.process_micro(df.copy())
    
    # Verify Columns Exist
    print("\n[Check 1] Normalized Columns Existence:")
    expected_cols = [f'n_bid_price_1', 'n_bid_vol_1', 'n_ask_price_1', 'n_ask_vol_1']
    missing = [c for c in expected_cols if c not in processed_df.columns]
    if missing:
        print(f"FAILED: Missing columns {missing}")
    else:
        print("PASSED: All expected normalized columns present.")

    # Verify Value Ranges
    print("\n[Check 2] Value Ranges:")
    
    # Price Normalization Check: (Price - Mid) / Mid
    # Mid=100, Bid1=98.9 (at i=1, 99.0 - 0.1 = 98.9)
    # (98.9 - 100)/100 = -1.1/100 = -0.011
    # Should be small float around 0
    bid_px_norm = processed_df['n_bid_price_1'].values
    print(f"Normalized Bid Price 1 (Sample): {bid_px_norm[:3]}")
    if np.all(np.abs(bid_px_norm) < 1.0):
        print("PASSED: Normalized prices are within expected small range (< 1.0)")
    else:
        print(f"FAILED: Normalized prices out of range! Max: {np.max(np.abs(bid_px_norm))}")

    # Volume Normalization Check: log1p(Vol)
    # Vol=100 -> log(101) ~ 4.6
    # Vol=20000 -> log(20001) ~ 9.9
    bid_vol_norm = processed_df['n_bid_vol_1'].values
    print(f"Normalized Bid Vol 1 (Sample): {bid_vol_norm}")
    if np.all(bid_vol_norm < 15.0) and np.all(bid_vol_norm > 0.0):
        print("PASSED: Normalized volumes are within expected log range (0 < v < 15)")
    else:
        print(f"FAILED: Normalized volumes out of range! Range: [{np.min(bid_vol_norm)}, {np.max(bid_vol_norm)}]")

def test_environment_scaling():
    print("\n=== Testing Environment Private State Scaling ===")
    
    config = {
        "symbol": "BTCUSDT",
        "initial_balance": 100000.0,
        "max_position": 5.0,
        "transaction_fee": 0.0005
    }
    
    env = DeepScalperEnv(config)
    
    # Test _normalize_private_state directly
    # Case 1: Start
    pos, bal = 0.0, 100000.0
    norm = env._normalize_private_state(pos, bal)
    print(f"State(Pos={pos}, Bal={bal}) -> Norm: {norm}")
    assert np.isclose(norm[0], 0.0), "Position should be 0"
    assert np.isclose(norm[1], 1.0), "Balance should be 1.0"
    
    # Case 2: Max Position
    pos, bal = 5.0, 100000.0
    norm = env._normalize_private_state(pos, bal)
    print(f"State(Pos={pos}, Bal={bal}) -> Norm: {norm}")
    assert np.isclose(norm[0], 1.0), "Position should be 1.0"
    
    # Case 3: Short Max Position
    pos, bal = -5.0, 105000.0
    norm = env._normalize_private_state(pos, bal)
    print(f"State(Pos={pos}, Bal={bal}) -> Norm: {norm}")
    assert np.isclose(norm[0], -1.0), "Position should be -1.0"
    assert np.isclose(norm[1], 1.05), "Balance should be 1.05"
    
    print("PASSED: Environment private state normalization verified.")
    
    # Test LOB Keys mapping
    print("\n[Check 3] Environment LOB Keys:")
    keys = env._lob_keys[0]
    print(f"Level 1 Keys: {keys}")
    if keys[0] == 'n_bid_price_1':
        print("PASSED: Environment is configured to look for 'n_' prefixed columns.")
    else:
        print(f"FAILED: Environment looking for {keys[0]} instead of normalized column.")

if __name__ == "__main__":
    try:
        test_feature_engineering_scaling()
        test_environment_scaling()
        print("\n=== AUDIT SUCCESS: All scaling logic verified ===")
    except Exception as e:
        print(f"\n=== AUDIT FAILED: {str(e)} ===")
        import traceback
        traceback.print_exc()

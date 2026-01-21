import pandas as pd
import numpy as np
import os

def generate_dummy_data(output_path="data/smoke_test_lob.parquet", n_rows=2000):
    # Ensure dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate timestamp
    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="1s")
    
    data = {
        "timestamp": dates,
        "symbol": ["BTCUSDT"] * n_rows
    }
    
    # Generate LOB columns (levels 1-5)
    base_price = 40000.0
    price_walk = np.random.randn(n_rows).cumsum() * 10
    mid_price = base_price + price_walk
    
    for i in range(1, 6):
        # Bid < Mid, Ask > Mid
        spread = 0.5 * i
        data[f"bid_price_{i}"] = mid_price - spread - np.random.rand(n_rows)*0.1
        data[f"ask_price_{i}"] = mid_price + spread + np.random.rand(n_rows)*0.1
        data[f"bid_vol_{i}"] = np.random.rand(n_rows) * 1.5
        data[f"ask_vol_{i}"] = np.random.rand(n_rows) * 1.5
        
    # Generate Macro columns
    macro_cols = [
        'rsi_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'atr_14', 'obv'
    ]
    
    for col in macro_cols:
        data[col] = np.random.randn(n_rows)
        
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Generated {n_rows} rows of dummy data at {output_path}")

if __name__ == "__main__":
    generate_dummy_data(output_path="c:/FinRL/FinRL-Pro_DS/data/smoke_data.parquet")


import pandas as pd
import numpy as np
import os

def debug():
    data_path = "debug_data.parquet"
    dates = pd.date_range("2023-01-01", periods=10, freq="1min")
    
    # Method 1
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(10),
        "bid_price_1": 100.0,
        "ask_price_1": 101.0
    })
    
    print("DF Columns before save:", df.columns.tolist())
    df.to_parquet(data_path)
    
    # Load back
    loaded = pd.read_parquet(data_path)
    print("Loaded Columns:", loaded.columns.tolist())
    print("Loaded Index:", loaded.index.name)
    
    if "timestamp" not in loaded.columns:
        print("FAIL: timestamp missing!")
    else:
        print("SUCCESS: timestamp found.")
        
    os.remove(data_path)

if __name__ == "__main__":
    debug()

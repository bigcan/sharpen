import pandas as pd
import numpy as np
import datetime

def generate_dummy_parquet(output_path="data/btc_lob_demo.parquet", rows=2000):
    print(f"Generating {rows} rows of dummy LOB data...")
    
    start_time = datetime.datetime.now()
    timestamps = [start_time + datetime.timedelta(seconds=i) for i in range(rows)]
    
    data = {"timestamp": timestamps}
    
    # LOB Cols: bid_price_1..5, bid_vol_1..5, ask_price_1..5, ask_vol_1..5
    base_price = 45000.0
    
    for level in range(1, 6):
        # Random walk for prices
        noise = np.random.randn(rows) * 10
        data[f"bid_price_{level}"] = base_price - level * 10 + noise
        data[f"ask_price_{level}"] = base_price + level * 10 + noise
        
        data[f"bid_vol_{level}"] = np.abs(np.random.randn(rows) * 1.5) + 0.1
        data[f"ask_vol_{level}"] = np.abs(np.random.randn(rows) * 1.5) + 0.1
        
    # Macro Cols (OHLCV)
    data["open"] = base_price + np.random.randn(rows)
    data["high"] = data["open"] + 50
    data["low"] = data["open"] - 50
    data["close"] = data["open"] + np.random.randn(rows) * 10
    data["volume"] = np.abs(np.random.randn(rows) * 100)
    
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_dummy_parquet()

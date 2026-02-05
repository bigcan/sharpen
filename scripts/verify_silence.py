
import os
import sys
# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
import logging

# Set up logging to verify silencing (though we are checking for stdout mainly)
logging.basicConfig(level=logging.INFO)

def test_silence():
    print("--- Starting Verification Run ---")
    
    # Path to sample data - assuming data/btc_lob_jan2023.parquet exists from previous context
    # or use a dummy path and expect error but cleanly
    file_path = 'data/btc_lob_jan2023.parquet'
    
    if not os.path.exists(file_path):
        print(f"Test file {file_path} not found. Using dummy check.")
        # Create dummy parquet
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1s'),
            'bid_price_1': np.random.rand(100),
            'ask_price_1': np.random.rand(100),
            'bid_vol_1': np.random.rand(100),
            'ask_vol_1': np.random.rand(100),
            'close': np.random.rand(100),
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'volume': np.random.rand(100)
        })
        file_path = 'data/dummy_test.parquet'
        df.to_parquet(file_path)
        
    config = {
        "volatility_horizon": 10
    }
    
    try:
        handler = ParquetDataHandler(file_path=file_path, ticker="BTCUSDT", feature_config=config)
        print("Handler instantiated.")
        handler.reset()
        print("Handler reset.")
        # Step a few times
        for _ in range(5):
            handler.step()
        print("Stepped 5 times.")
    except Exception as e:
        print(f"Error: {e}")
        
    print("--- End Verification Run ---")

if __name__ == "__main__":
    test_silence()

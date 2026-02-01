
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def inspect_data():
    file_path = "data/btc_lob_jan2023.parquet"
    if not os.path.exists(file_path):
        print(f"CRITICAL: File not found at {file_path}")
        # Try to find any parquet in data/ to test
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith(".parquet")]
            if files:
                file_path = f"data/{files[0]}"
                print(f"Using alternative file: {file_path}")
            else:
                print("No parquet files found in data/")
                return

    try:
        print(f"Loading {file_path}...")
        handler = ParquetDataHandler(file_path=file_path, ticker="BTCUSDT")
        
        print("\n--- Column Names ---")
        cols = handler._feature_cols
        print(cols)
        
        print("\n--- Feature Data Check ---")
        # Check for LOB cols
        expected_lob = []
        for i in range(1, 6):
            expected_lob.extend([
                f'bid_price_{i}', f'bid_vol_{i}',
                f'ask_price_{i}', f'ask_vol_{i}'
            ])
            
        missing = [c for c in expected_lob if c not in cols]
        if missing:
            print(f"CRITICAL: Missing LOB columns: {missing}")
        else:
            print("All LOB columns present.")

        print("\n--- Step Data Inspection ---")
        step_data = handler.step()
        if step_data:
            print("First Step Keys:", list(step_data.keys()))
            
            # Check values for Level 1
            print(f"Bid Px 1: {step_data.get('bid_price_1')}")
            print(f"Ask Px 1: {step_data.get('ask_price_1')}")
            print(f"Bid Vol 1: {step_data.get('bid_vol_1')}")
            print(f"Ask Vol 1: {step_data.get('ask_vol_1')}")
            
            # Check for Zeros
            if float(step_data.get('bid_vol_1', 0)) == 0:
                print("WARNING: Bid Vol 1 is 0.0!")
            if float(step_data.get('ask_vol_1', 0)) == 0:
                print("WARNING: Ask Vol 1 is 0.0!")
                
        else:
            print("Handler returned None for first step.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_data()

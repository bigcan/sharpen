import argparse
import sys
import os

# Minimal imports
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def main():
    print("Starting FE Debug...")
    
    # Argparse to satisfy deploy script interface
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()
    
    file_path = "/data/btc_lob_jan2023.parquet"
    if not os.path.exists(file_path):
        # Fallback to absolute if needed or error
        print(f"File {file_path} not found!") 
        # try checking local path if deployed differently?
        # But we know it's at /data
        return

    print(f"Initializing Handler with {file_path}...")
    try:
        handler = ParquetDataHandler(
            file_path=file_path,
            ticker="BTCUSDT",
            feature_config={}
        )
        print(f"Handler Initialized. Rows: {handler._len}")
        
        # Pull a few rows to verify
        print("Stepping 5 times...")
        for i in range(5):
             row = handler.step()
             print(f"Step {i}: {row.get('timestamp')}")
             
        print("FE Debug SUCCESS.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FE Debug CRASHED: {e}")

if __name__ == "__main__":
    main()

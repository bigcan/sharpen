import argparse
import sys
import os

print("STEP 0: Importing Torch First...", flush=True)
import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

print("STEP 1: Importing Handler...", flush=True)
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def main():
    file_path = "btc_lob_demo.parquet"
    print(f"STEP 2: Instantiating ParquetDataHandler: {file_path}", flush=True)
    
    try:
        # Check if file exists locally first (deployment case)
        if os.path.exists("btc_lob_demo.parquet"):
            file_path = "btc_lob_demo.parquet"
            print(f"Found local data file: {file_path}", flush=True)
        else:
            file_path = "/data/btc_lob_demo.parquet"
            print(f"Using default data path: {file_path}", flush=True)
        
        # Full Load
        handler = ParquetDataHandler(file_path=file_path, ticker="BTCUSDT")
        print("STEP 3: Handler Created Successfully.", flush=True)
        
        # Test Step
        print("STEP 4: Testing Step...", flush=True)
        handler.reset()
        row = handler.step()
        print(f"STEP 5: Got Row. Keys={list(row.keys())[:5]}", flush=True)
        
        print("FULL SUCCESS.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

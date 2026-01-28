import argparse
import sys
import os
import time

print("STEP 0: Importing Torch First...", flush=True)
import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

print("STEP 1: Importing Pandas...", flush=True)
import pandas as pd
import numpy as np

def main():
    file_path = "/data/btc_lob_jan2023.parquet"
    print(f"STEP 2: Loading Real Parquet: {file_path}", flush=True)
    
    try:
        t0 = time.time()
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"STEP 3: Loaded. Shape: {df.shape}. Time: {time.time()-t0:.2f}s", flush=True)
        
        df.columns = df.columns.astype(str).str.strip()
        print(f"STEP 4: Sanitized Columns.", flush=True)
        
        # Mimic ParquetDataHandler logic
        if 'timestamp' in df.columns:
            print("STEP 5: Converting Timestamp...", flush=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("STEP 6: Timestamp Converted.", flush=True)
        else:
            print("WARNING: No timestamp column.", flush=True)

        print("STEP 7: Sorting...", flush=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        print("STEP 8: Sorted.", flush=True)

        # Micro Features (Simulate basic FE)
        print("STEP 9: Accessing Columns (Micro FE Sim)...", flush=True)
        # Access a few columns to trigger lazy loading if any
        if 'bid_price_1' in df.columns:
            vals = df['bid_price_1'].values
            print(f"Sample: {vals[:5]}", flush=True)
            
        print("FULL SUCCESS.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

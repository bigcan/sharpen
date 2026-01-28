import argparse
import sys
import os
import time
import numpy as np

print("STEP 0: Importing Torch First...", flush=True)
import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

print("STEP 1: Importing Pandas...", flush=True)
import pandas as pd

def main():
    file_path = "/data/btc_lob_jan2023.parquet"
    print(f"STEP 2: Loading Real Parquet: {file_path}", flush=True)
    
    try:
        t0 = time.time()
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"STEP 3: Loaded. Shape: {df.shape}. Time: {time.time()-t0:.2f}s", flush=True)
        
        # Test 1: to_datetime & Sort
        if 'timestamp' in df.columns:
            print("STEP 4: Converting to Datetime...", flush=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print("STEP 5: Sorting...", flush=True)
            df = df.sort_values('timestamp')
            print("STEP 6: Sorted.", flush=True)
            
            # Test 2: Copy AFTER Sort
            print("STEP 7: Copying Sorted DF...", flush=True)
            df_copy = df.copy()
            print("STEP 8: Copy Done.", flush=True)
            
            # Test 3: Assign
            print("STEP 9: Assigning...", flush=True)
            df_copy['new_col'] = 1.0
            print("STEP 10: Assign Done.", flush=True)
        
        print("FULL SUCCESS.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

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
        
        # Test 1: Copy
        print("STEP 4: Testing Copy...", flush=True)
        df_copy = df.copy()
        print("STEP 5: Copy Done.", flush=True)
        
        # Test 2: Deep Copy
        print("STEP 6: Testing Deep Copy...", flush=True)
        df_deep = df.copy(deep=True)
        print("STEP 7: Deep Copy Done.", flush=True)
        
        print("FULL SUCCESS.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

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
        
        # Test Rolling Operation (mimic Volatility calc)
        print("STEP 4: Testing Rolling Operation (std)...", flush=True)
        if 'bid_price_1' in df.columns:
            series = df['bid_price_1']
            rolled = series.rolling(100).std()
            print(f"STEP 5: Rolling Std Done. Head: {rolled.head().tolist()}", flush=True)
        
        # Test Rolling Mean
        print("STEP 6: Testing Rolling Mean...", flush=True)
        rolled_mean = df.select_dtypes(include=[np.number]).iloc[:, 0].rolling(50).mean()
        print("STEP 7: Rolling Mean Done.", flush=True)
            
        print("FULL SUCCESS.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

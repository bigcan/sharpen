import pandas as pd
import os
import sys
import traceback
import torch
import numpy as np

def test_load():
    try:
        data_path = "/data/btc_lob_jan2023.parquet"
        print(f"Checking path: {data_path}")
        print(f"Torch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if not os.path.exists(data_path):
            print(f"ERROR: {data_path} does not exist!")
            return

        print(f"File size: {os.path.getsize(data_path)} bytes")
        
        print("Loading parquet with pyarrow...")
        df = pd.read_parquet(data_path, engine='pyarrow')
        print(f"Loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")
        print(f"First row timestamp: {df.iloc[0]['timestamp'] if 'timestamp' in df.columns else 'No timestamp'}")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_load()

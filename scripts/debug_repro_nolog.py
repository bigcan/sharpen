import argparse
import sys
import os
import numpy as np

print("STEP 0: Importing Torch First...", flush=True)
import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

print("STEP 1: Importing Pandas...", flush=True)
import pandas as pd

# INLINE DeepScalperFeatureEngineer (Simplified)
class DeepScalperFeatureEngineer:
    def __init__(self, config=None):
        self.config = config or {}
    
    def process_micro(self, lob_df):
        print("DEBUG: Entering process_micro", flush=True)
        df = lob_df.copy()
        if 'timestamp' in df.columns:
            sort_cols = ['timestamp']
            if 'level' in df.columns: sort_cols.append('level')
            df = df.sort_values(sort_cols)
        print("DEBUG: process_micro sorted", flush=True)
        
        # 1. Mid Prices
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        print("DEBUG: process_micro mid_price calc", flush=True)
        
        # 4. Log Returns - DISABLED
        # if 'mid_price' in df.columns:
        #    df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        print("DEBUG: process_micro log_ret SKIPPED", flush=True)

        return df

# INLINE ParquetDataHandler
class ParquetDataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fe = DeepScalperFeatureEngineer()
        self.load_data()

    def load_data(self):
        print("DEBUG: load_data start", flush=True)
        df = pd.read_parquet(self.file_path, engine='pyarrow')
        print("DEBUG: read_parquet done", flush=True)
        
        df.columns = df.columns.astype(str).str.strip()
        
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # FE
        print("DEBUG: calling process_micro", flush=True)
        micro = self.fe.process_micro(df)
        print("DEBUG: process_micro returned", flush=True)
        
        print("FULL SUCCESS inline.", flush=True)

def main():
    file_path = "/data/btc_lob_jan2023.parquet"
    print(f"STEP 2: Instantiating Inline Handler (No LogRet): {file_path}", flush=True)
    try:
        handler = ParquetDataHandler(file_path)
    except Exception as e:
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

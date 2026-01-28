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

# INLINE DeepScalperFeatureEngineer (Simplified to what we saw)
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
        
        # 4. Log Returns
        if 'mid_price' in df.columns:
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        print("DEBUG: process_micro log_ret calc", flush=True)

        return df

    def process_macro(self, df):
        print("DEBUG: Entering process_macro", flush=True)
        return df # No-op for now to test micro

    def align_multimodal(self, micro, macro):
        print("DEBUG: Entering align_multimodal", flush=True)
        return micro # No-op

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
        print("DEBUG: cols sanitized", flush=True)
        
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("DEBUG: timestamp converted", flush=True)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        print("DEBUG: sorted", flush=True)
        
        # FE
        print("DEBUG: calling process_micro", flush=True)
        micro = self.fe.process_micro(df)
        print("DEBUG: process_micro returned", flush=True)
        
        # Volatility Logic (Suspected)
        print("DEBUG: Volatility Logic Start", flush=True)
        if 'mid_price' in micro.columns:
            prices = micro['mid_price'].values.astype(np.float64)
            log_rets = np.zeros_like(prices)
            log_rets[1:] = np.log(prices[1:] / prices[:-1])
            print("DEBUG: Volatility log_rets done", flush=True)
            
            s = pd.Series(log_rets)
            rolling_std = s.rolling(100).std()
            print("DEBUG: Volatility rolling done", flush=True)
            
            vol_target = rolling_std.shift(-100).fillna(0.0).values
            print("DEBUG: Volatility shift done", flush=True)
        
        print("FULL SUCCESS inline.", flush=True)

def main():
    file_path = "/data/btc_lob_jan2023.parquet"
    print(f"STEP 2: Instantiating Inline Handler: {file_path}", flush=True)
    try:
        handler = ParquetDataHandler(file_path)
    except Exception as e:
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()

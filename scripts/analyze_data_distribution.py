import pandas as pd
import numpy as np
import os

FILE_PATH = "data/btc_lob_jan2023.parquet"

def analyze():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Loading {FILE_PATH}...")
    df = pd.read_parquet(FILE_PATH)
    
    print("-" * 40)
    print(f"**General Stats**")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory: {df.memory_usage().sum() / 1e6:.2f} MB")
    
    # Time Analysis
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start = df['timestamp'].min()
        end = df['timestamp'].max()
        duration = end - start
        print(f"\n**Time Range**")
        print(f"Start: {start}")
        print(f"End:   {end}")
        print(f"Duration: {duration}")
        
        # Gap Detection (assuming ~100ms or 1s frequency?)
        # Let's check median delta
        deltas = df['timestamp'].diff().dropna()
        median_delta = deltas.median()
        print(f"Median Time Delta: {median_delta}")
        
        # Large gaps (> 1 minute)
        gaps = deltas[deltas > pd.Timedelta(minutes=1)]
        print(f"Gaps > 1 min: {len(gaps)}")
        if len(gaps) > 0:
            print(f"Top 3 Gaps:\n{gaps.sort_values(ascending=False).head(3)}")
            
    # Price Analysis
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'close' in c.lower() or 'mid' in c.lower()]
    if price_cols:
        target_col = 'close' if 'close' in df.columns else price_cols[0]
        print(f"\n**Price Stats ({target_col})**")
        print(f"Mean: {df[target_col].mean():.2f}")
        print(f"Min:  {df[target_col].min():.2f}")
        print(f"Max:  {df[target_col].max():.2f}")
        print(f"Std:  {df[target_col].std():.2f}")
        
    # Volatility Check (Log Returns)
    if 'close' in df.columns:
         df['ret'] = np.log(df['close'] / df['close'].shift(1))
         print(f"\n**Return Stats**")
         print(f"Mean Ret: {df['ret'].mean():.8f}")
         print(f"Std Ret:  {df['ret'].std():.8f}")
         
    # Volume Analysis
    vol_cols = [c for c in df.columns if 'vol' in c.lower()]
    if vol_cols:
         target_col = 'volume' if 'volume' in df.columns else vol_cols[0]
         print(f"\n**Volume Stats ({target_col})**")
         print(f"Mean: {df[target_col].mean():.4f}")
         print(f"Sum:  {df[target_col].sum():.4f}")

    # Missing Values
    print(f"\n**Missing Values**")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("None")
    else:
        print(missing)

if __name__ == "__main__":
    analyze()

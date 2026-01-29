import pandas as pd
import os

files = [
    r"c:\data\btc_lob_jan2023.parquet"
]

for f in files:
    print(f"\n{'='*30}")
    print(f"FILE: {f}")
    print(f"{'='*30}")
    
    if not os.path.exists(f):
        print(f"File not found")
        continue
        
    try:
        df = pd.read_parquet(f)
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Time
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if time_cols:
            t_col = time_cols[0]
            print(f"\nTime Column: {t_col}")
            print(f"Start: {df[t_col].min()}")
            print(f"End:   {df[t_col].max()}")
            
        print(f"\nMemory Usage: {df.memory_usage().sum() / 1e6:.2f} MB")
            
    except Exception as e:
        print(f"Error: {e}")

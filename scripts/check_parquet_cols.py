import pandas as pd
import sys

file_path = "data/btc_lob_jan2023.parquet"
try:
    df = pd.read_parquet(file_path, engine='fastparquet')
    print(f"Columns in {file_path}:")
    cols = df.columns.tolist()
    print(cols)
    
    has_n_bid = 'n_bid_price_1' in cols
    print(f"\nHas 'n_bid_price_1': {has_n_bid}")
    
    has_macro = 'z_close' in cols
    print(f"Has 'z_close' (Macro): {has_macro}")
    
except Exception as e:
    print(f"Error: {e}")

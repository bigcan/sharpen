
import pandas as pd
import os
import sys

DATA_PATH = "data/btc_lob_jan2023.parquet"

def inspect_data():
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL: Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    print(f"Loading {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as e:
        print(f"CRITICAL: Failed to read parquet file. Error: {e}")
        sys.exit(1)

    print("\n--- Basic Info ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print("\n--- Columns (Sorted) ---")
    cols = sorted(df.columns.tolist())
    print(cols)
    
    print("\n--- Critical Feature Check ---")
    req_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    req_z = ['z_open', 'z_high', 'z_low', 'z_close']
    
    missing_ohlcv = [c for c in req_ohlcv if c not in df.columns]
    missing_z = [c for c in req_z if c not in df.columns]
    
    print(f"Missing OHLCV: {missing_ohlcv}")
    print(f"Missing Z-score strings: {missing_z}")
    time_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            time_col = col
            break
            
    if time_col:
        print(f"\n--- Time Coverage ({time_col}) ---")
        # Ensure regex doesn't break mixed types, but parquet should be typed
        try:
            # Check if it's numeric (unix) or string
            if pd.api.types.is_numeric_dtype(df[time_col]):
                # Assume nanoseconds or milliseconds - Standardize to datetime for display
                # If values are huge, likely ns. If ~1.6e9, likely seconds.
                # Just showing raw first/last is often safer if format unknown
                print(f"Min: {df[time_col].min()}")
                print(f"Max: {df[time_col].max()}")
                
                # Try conversion for readability
                try:
                    print(f"Min (Readable): {pd.to_datetime(df[time_col].min())}")
                    print(f"Max (Readable): {pd.to_datetime(df[time_col].max())}")
                except:
                    pass
            else:
                print(f"Min: {df[time_col].min()}")
                print(f"Max: {df[time_col].max()}")
        except Exception as e:
            print(f"Error checking time range: {e}")
    else:
        print("\nWARNING: No 'timestamp' or 'date' column identified.")

    print("\n--- Missing Values ---")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("No missing values found.")
    else:
        print(null_counts[null_counts > 0])

    # Check for Micro/Macro features presence
    print("\n--- Feature Groups ---")
    micro_cols = [c for c in df.columns if "micro" in c]
    macro_cols = [c for c in df.columns if "macro" in c]
    print(f"Micro features count: {len(micro_cols)}")
    print(f"Macro features count: {len(macro_cols)}")

if __name__ == "__main__":
    inspect_data()


import pandas as pd

# Load the parquet file
file_path = "data/btc_lob_jan2023.parquet"
try:
    df = pd.read_parquet(file_path)
    
    # Check for index or timestamp column
    if isinstance(df.index, pd.DatetimeIndex):
        start_date = df.index.min()
        end_date = df.index.max()
        print(f"Data Date Range (Index): {start_date} to {end_date}")
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Calculate frequency
        deltas = df['timestamp'].diff().dropna()
        median_delta = deltas.median()
        min_delta = deltas.min()
        
        print(f"Data Date Range (Timestamp Col): {start_date} to {end_date}")
        print(f"Median Time Step: {median_delta}")
        print(f"Min Time Step: {min_delta}")
        print(f"Total Rows: {len(df)}")
    else:
        print("No index or timestamp column found. Columns:", df.columns)
        print("Head:", df.head())
        
except Exception as e:
    print(f"Error reading file: {e}")

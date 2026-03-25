"""Debug Databento data format - check price scaling and data quality."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import databento as db  # noqa: E402

API_KEY = os.environ['DATABENTO_API_KEY']
client = db.Historical(API_KEY)

# Fetch just 1 day of CL to inspect format
print("=== CL (Crude Oil) - 1 day sample ===")
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["CL.c.0"],
    stype_in="continuous",
    schema="ohlcv-1m",
    start="2025-06-02",
    end="2025-06-03",
)

df = data.to_df()
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print("\nFirst 5 rows:")
print(df.head())
print("\nPrice stats (raw):")
for col in ['open', 'high', 'low', 'close']:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
print(f"Volume stats: min={df['volume'].min()}, max={df['volume'].max()}, mean={df['volume'].mean():.0f}")

# Check if prices need conversion
print(f"\nRaw open[0] = {df['open'].iloc[0]}")
print(f"Raw close[0] = {df['close'].iloc[0]}")

# Test different scalings
raw_price = df['close'].iloc[0]
print(f"\nScaling tests for close[0]={raw_price}:")
print(f"  / 1e9  = {raw_price / 1e9}")
print(f"  / 1e6  = {raw_price / 1e6}")
print(f"  / 1e3  = {raw_price / 1e3}")
print(f"  / 1e2  = {raw_price / 1e2}")
print(f"  / 1e1  = {raw_price / 1e1}")
print(f"  raw    = {raw_price}")

print("\n=== GC (Gold) - 1 day sample ===")
data_gc = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["GC.c.0"],
    stype_in="continuous",
    schema="ohlcv-1m",
    start="2025-06-02",
    end="2025-06-03",
)

df_gc = data_gc.to_df()
print(f"Shape: {df_gc.shape}")
print("\nFirst 5 rows:")
print(df_gc.head())
print("\nPrice stats (raw):")
for col in ['open', 'high', 'low', 'close']:
    print(f"  {col}: min={df_gc[col].min()}, max={df_gc[col].max()}, mean={df_gc[col].mean():.2f}")
print(f"Volume stats: min={df_gc['volume'].min()}, max={df_gc['volume'].max()}, mean={df_gc['volume'].mean():.0f}")

# Also check what symbol is being resolved
print(f"\nGC symbol column: {df_gc['symbol'].unique()}")
print(f"CL symbol column: {df['symbol'].unique()}")

# Try GC with parent symbol to see if we get more data
print("\n=== Try GCQ5 (specific August contract) ===")
try:
    data_gc2 = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["GC.c.0"],
        stype_in="continuous",
        schema="ohlcv-1m",
        start="2025-06-01",
        end="2025-06-07",
    )
    df_gc2 = data_gc2.to_df()
    print(f"Shape: {df_gc2.shape}")
    print(f"Bars per day: {len(df_gc2) / max(df_gc2.index.date.__len__(), 1):.0f}")
    print(df_gc2.head(10))
except Exception as e:
    print(f"Error: {e}")

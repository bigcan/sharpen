"""Sprint 5: Merge Jan-Jun + Jul-Dec 2025 parquets into full-year dataset."""
import pandas as pd

df1 = pd.read_parquet('data/processed/btc_2025_jan_jun.parquet')
df2 = pd.read_parquet('data/processed/btc_2025_jul_dec.parquet')

print(f"Jan-Jun: {df1.shape}, {df1['timestamp'].min()} to {df1['timestamp'].max()}")
print(f"Jul-Dec: {df2.shape}, {df2['timestamp'].min()} to {df2['timestamp'].max()}")

df = pd.concat([df1, df2]).sort_values('timestamp').reset_index(drop=True)

print(f"\nMerged: {df.shape}")
print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"NaN total: {df.isnull().sum().sum()}")
print(f"Duplicate timestamps: {df.duplicated(subset=['timestamp']).sum()}")
print(f"Columns: {len(df.columns)}")

# Verify boundary continuity
boundary_idx = len(df1) - 1
gap = df.iloc[boundary_idx + 1]['timestamp'] - df.iloc[boundary_idx]['timestamp']
print(f"Gap at boundary: {gap}")

df.to_parquet('data/processed/btc_2025_full_year.parquet', index=False)
print("\nSaved: data/processed/btc_2025_full_year.parquet")
print(f"File rows: {len(pd.read_parquet('data/processed/btc_2025_full_year.parquet'))}")

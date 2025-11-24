import pandas as pd
import sys

parquet_path = "finrl_pro/data/processed/294c694cf06b1882d09e811ca5e547f29b9153eafd39e72fc09e0d3ed6b8aa40/features.parquet"
csv_path = "tmp/features_to_validate.csv"

df = pd.read_parquet(parquet_path)
df.to_csv(csv_path, index=False) # Index (timestamp) should be a column or handled
print(f"Converted {parquet_path} to {csv_path}")

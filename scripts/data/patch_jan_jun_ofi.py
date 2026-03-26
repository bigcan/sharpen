"""Patch btc_2025_jan_jun.parquet: rename vol_imbalance_* → ofi_*"""
from pathlib import Path

import pandas as pd

path = Path("data/processed/btc_2025_jan_jun.parquet")
df = pd.read_parquet(path)

rename_map = {f"vol_imbalance_{i}": f"ofi_{i}" for i in range(1, 6)}
df = df.rename(columns=rename_map)
print(f"Renamed: {rename_map}")
print(f"OFI cols: {[c for c in df.columns if 'ofi' in c]}")

df.to_parquet(path, index=False)
print(f"Saved: {path}")

# Verify schema match
jd = pd.read_parquet("data/processed/btc_2025_jul_dec.parquet", columns=[])
print(f"Schemas match: {set(df.columns) == set(jd.columns)}")

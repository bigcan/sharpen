#!/bin/bash
cd /workspace/DeepScalper
export PATH=/root/miniconda3/bin:$PATH

python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_2025_full_year_5min.parquet')
micro_cols = [c for c in df.columns if c.startswith('micro_')]
macro_cols = [c for c in df.columns if c.startswith('macro_')]
print(f'Total columns: {len(df.columns)}')
print(f'Micro columns: {len(micro_cols)}')
print(f'Macro columns: {len(macro_cols)}')
print(f'Micro cols: {micro_cols}')
" 2>&1

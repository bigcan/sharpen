#!/bin/bash
cd /workspace/DeepScalper
export PATH=/root/miniconda3/bin:$PATH

python -u -c "
import sys
print(f'Python: {sys.version}')

print('Test 1: import pyarrow')
import pyarrow as pa
import pyarrow.parquet as pq
print(f'PyArrow: {pa.__version__}')

print('Test 2: read parquet')
table = pq.read_table('data/processed/btc_2025_full_year_5min.parquet')
print(f'Rows: {table.num_rows}, Cols: {table.num_columns}')

print('Test 3: pandas conversion')
import pandas as pd
df = table.to_pandas()
print(f'DataFrame: {df.shape}')
print(f'Columns: {list(df.columns[:10])}...')

print('Test 4: import numpy')
import numpy as np
print(f'NumPy: {np.__version__}')

print('Test 5: feature engineering import')
from sharpen.data.feature_engineering import FeatureEngineer
print('OK')

print('Test 6: ParquetDataHandler')
from sharpen.data.parquet_handler import ParquetDataHandler
handler = ParquetDataHandler(
    'data/processed/btc_2025_full_year_5min.parquet',
    ticker='BTCUSDT',
    start_date='2025-01-01',
    end_date='2025-10-31'
)
print(f'Handler OK: {len(handler.data)} rows')

print('Test 7: env creation')
from sharpen.envs.deep_scalper_env import DeepScalperEnv
cfg = {
    'margin_requirement': 0.05,
    'initial_balance': 100000,
    'window_size': 15,
    'reward': {'inventory_penalty_bps': 0.5},
    'action': {'discrete_dims': 3, 'max_position': 5.0}
}
env = DeepScalperEnv(cfg)
print(f'Env OK: inv_pen={env.inventory_penalty_bps}')

print('ALL PASSED')
" 2>&1

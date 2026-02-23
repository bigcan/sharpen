#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH

echo "=== Upgrading numpy to match pandas 2.3.3 ==="
pip install 'numpy>=2.0,<2.2' --quiet 2>&1 | tail -5

echo ""
echo "=== Versions after fix ==="
python -c "
import numpy; print(f'numpy: {numpy.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
import pyarrow; print(f'pyarrow: {pyarrow.__version__}')
import torch; print(f'torch: {torch.__version__}')
"

echo ""
echo "=== Verify parquet loading ==="
cd /workspace/DeepScalper
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_2025_full_year_5min.parquet')
print(f'DataFrame: {df.shape}')
print('PARQUET LOAD OK')
" 2>&1

echo ""
echo "=== Verify env creation ==="
python -c "
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
import yaml
cfg = yaml.safe_load(open('configs/phase_b13_inventory_penalty_5min.yaml'))
from scripts.run_full_pipeline import make_env
env = make_env(cfg)
obs, info = env.reset()
print(f'Env OK - micro: {obs[\"micro\"].shape}')
env.close()
print('ENV CREATION OK')
" 2>&1

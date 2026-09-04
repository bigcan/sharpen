#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
echo "=== Current versions ==="
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

echo ""
echo "=== Fixing PyArrow/Pandas compatibility ==="
pip install 'pyarrow>=17,<19' --quiet 2>&1 | tail -3

echo ""
echo "=== Updated versions ==="
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

echo ""
echo "=== Verify parquet loading ==="
cd /workspace/DeepScalper
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_2025_full_year_5min.parquet')
print(f'DataFrame: {df.shape}')
print(f'First cols: {list(df.columns[:5])}')
print('PARQUET LOAD OK')
" 2>&1

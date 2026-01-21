# DVC Setup — FinRL Pro Datasets

Use DVC to pin immutable dataset versions for reproducible experiments.

Supported dataset URIs in configs:
- `dvc://datasets/sp500_daily_2016_2025`
- `dvc://datasets/sp500_multi_2016_2025`

These are labels you control via your DVC remote structure (e.g., S3 folder names). They do not have to match local filenames, but keeping names aligned helps.

## One‑Time Setup

1) Initialize and configure remote
```
dvc init
dvc remote add -d s3 s3://<your-bucket>/finrl-pro
```

2) Add your datasets (examples)
```
# SPY daily
mkdir -p data
python scripts/make_spy_daily.py --start 2016-01-01 --end 2025-12-31 -o data/sp500_daily_2016_2025.parquet
dvc add data/sp500_daily_2016_2025.parquet

# Multi-asset universe (10–50 tickers)
python scripts/make_sp500_universe.py --start 2016-01-01 --end 2025-12-31 -o data/sp500_multi_2016_2025/
dvc add data/sp500_multi_2016_2025
```

3) Push data to remote
```
dvc push
```

4) Map URIs (convention)
- Create folders in your DVC remote:
  - `datasets/sp500_daily_2016_2025` → stores `data/sp500_daily_2016_2025.parquet.dvc`
  - `datasets/sp500_multi_2016_2025` → stores `data/sp500_multi_2016_2025.dvc`

Keep experiment YAMLs pointing to:
```
training.dataset_hash: dvc://datasets/sp500_daily_2016_2025
training.dataset_hash: dvc://datasets/sp500_multi_2016_2025
```

## Reproduce on another machine
```
git pull
dvc pull
python -m finrl_pro_ds.training.commands.run_matrix --experiments-dir finrl_pro_ds/configs/experiments --output-dir reports/matrix
```

## Notes
- We do not commit actual data; only small `.dvc` pointer files.
- The exact resolver for `dvc://` is a naming convention used in fingerprints; the pipeline treats it as a dataset ID for reproducibility and feature cache keys.
- If you later adopt a different remote or path, keep the label stable and remap behind the scenes to avoid breaking fingerprints.


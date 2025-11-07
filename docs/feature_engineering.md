# FinRL Pro Feature Engineering

This document describes the Pro-side feature engineering layer that augments
the upstream FinRLPodracer pipeline without modifying upstream code.

- PIT-safety: All features are computed per ticker and use only data available
  at or before t-1 for the value emitted at t. No backfill is used.

## Configuration

Add a `features:` block to your experiment YAML (e.g., `finrl_pro/configs/experiments/sp500_daily.yaml`):

```
features:
  enable_custom: true
  use_turbulence: true
  families:
    trend: true
    momentum: true
    vol: true
    volume: false
  stockstats_overrides: []
  advanced:
    fracdiff: { enable: true, cols: [close], d: 0.5, window: 256, min_weight: 1e-5 }
    wavelet:  { enable: false, cols: [close], wavelet: db4, level: 3, window: 256 }
  cache:
    enabled: true
    dir: finrl_pro/data/processed
```

The resolved feature configuration is hashed into a cache key and stored in
the experiment fingerprint as `features.cache_key`.

## Pro Environment (dynamic tech_dim)

- Build arrays with the Pro loader and create a dynamic env:

```
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env

asm = ProFeatureAssembler(dsn=os.getenv("FINRL_PRO_DB_DSN")).assemble_from_snapshot(
    snapshot_id="<uuid>", features_cfg=YOUR_FEATURES_CFG)
env = make_pro_env(asm)
```

- Use when you need more than 7 features per ticker or flexible feature sets.

## Ablation Quickstart

- Run a small study over feature families and advanced toggles using Optuna:

```
python -m finrl_pro.automl.feature_search \
  --experiment finrl_pro/configs/experiments/spy_snapshot.yaml \
  --trials 10 --study-name demo-ablation
```

- The study reuses features via the feature store (keyed by `features.cache_key`),
  logs `features.feature_set_id`, and persists fingerprints via `Trainer`.

## Advanced Features

- Fractional Differentiation: `finrl_pro.features.custom_features.add_fracdiff_features`
  - Parameters: `d`, `window`, `min_weight`, `cols`
  - Produces columns like `fd_close_d0p5_w256`

- Wavelets (SWT/MODWT): `finrl_pro.features.custom_features.add_wavelet_features`
  - Parameters: `wavelet`, `level`, `window`, `cols`
  - Produces band features like `wlt_close_D1_last`, `wlt_close_D1_energy`, and optional `wlt_close_trend`

## Indicator Families

`finrl_pro.features.families.resolve_indicator_list()` maps family toggles to
stockstats names (e.g., trend → `close_30_sma`, `close_60_sma`, `macd`).

## Caching

`finrl_pro.data.cache` provides a stable `feature_cache_key` and helpers to
load/save cached feature tables (Parquet + JSON meta) under
`finrl_pro/data/processed/<key>/`.

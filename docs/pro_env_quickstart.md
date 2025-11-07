# Pro Env Quickstart (Dynamic tech_dim)

This guide shows how to run with the dynamic Pro environment (Option B) and how to switch between Option A (fixed 7 indicators) and Option B.

- Config flag (per experiment YAML):
  - File: `finrl_pro/configs/experiments/<your_exp>.yaml`
  - Under `training:` set:
    - `use_pro_env: false` → Option A (fixed 7, upstream-compatible)
    - `use_pro_env: true`  → Option B (dynamic tech_dim via Pro env)

- Assemble arrays from a snapshot and create the Pro env:
  - `finrl_pro/data/loader_pro.py` produces arrays and the `feature_set_id` by reusing the feature store keyed by `features.cache_key`.
  - `finrl_pro/envs/factory.py` wraps the dynamic env creation.

Example:
```
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.envs.factory import make_pro_env

asm = ProFeatureAssembler(dsn=os.getenv("FINRL_PRO_DB_DSN")).assemble_from_snapshot(
    snapshot_id="<uuid>", features_cfg=YOUR_FEATURES_CFG)
env = make_pro_env(asm)
```

- Reproducibility:
  - `features.cache_key` + `features.feature_set_id` are logged in `module_versions` (see `finrl_pro/training/run_experiment.py`).
  - Keep seeds fixed across folds for stable comparisons.

- When to use Option B:
  - Need >7 features per ticker or varying feature sets in ablation.
  - Research on wavelet bands, fracdiff variants, or regime flags that exceed the upstream 7.

- When to use Option A:
  - Comparability with upstream env or small indicator sets; ablations that only swap which 7.


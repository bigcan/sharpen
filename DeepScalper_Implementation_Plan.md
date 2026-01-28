# Implementation Plan - HPO Production Readiness

## Goal Description
Prepare and verify the Hyperparameter Optimization (HPO) pipeline (`scripts/tune_deepscalper.py`) for the "Full Production Run". 

The current `tune` script is outdated and lacks the critical fixes verified in Phase 6 (WandB initialization, Config Mapping, Network Architecture alignment). We must align it with the stable `train_deepscalper_v3.py` to ensure it runs successfully on the GPUHub remote environment.

## User Review Required
> [!IMPORTANT]
> **WandB Policy**: For HPO (many trials), logging every step to WandB can be overwhelming. 
> *   **Proposal**: Initialize WandB in `disabled` mode or `offline` mode for trials, OR log only the final metric. The Trainer *expects* WandB to be active.
> *   **Decision**: I will set `wandb.init(mode="disabled")` inside the HPO loop to prevent crash, since Optuna tracks the metrics (Sharpe).

## Proposed Changes

### Scripts
#### [MODIFY] [tune_deepscalper.py](file:///c:/FinRL/FinRL-Pro_DS/scripts/tune_deepscalper.py)
*   **WandB Support**: Add `wandb.init(mode="disabled")` in `objective` function to satisfy Trainer requirement without flooding the dashboard.
*   **Network Config**: Replace hardcoded `net_conf` construction with the robust `key_map` and nested-config logic from `train_deepscalper_v3.py`.
*   **Agent Init**: Ensure `DeepScalperDQN` calculates `dqn_batch_size` correctly from config (mapping fix).
*   **Data Path**: Ensure `make_env` uses a relative path or `os.environ` hook for the Parquet file to work on remote.

### Configs
*   **Unified Config**: No changes needed; `tune` script will override values dynamically.

## Verification Plan

### Automated Tests
1.  **Local Smoke Test**: Run `python scripts/tune_deepscalper.py --trials 1 --steps 100 --debug` locally (if data exists) or check syntax.
2.  **Remote Smoke Deployment**: 
    *   Command: `python scripts/deploy_bare_metal.py --script scripts/tune_deepscalper.py --config configs/deepscalper_unified.yaml --run_name DS_HPO_Smoke_V1 --upload_data`
    *   Args: `--trials 2 --steps 1000` (Fast check).

### Success Criteria
*   Script runs 2 trials without crashing.
*   Optuna database (`hpo.db`) is created/updated.
*   Best parameters are logged.

# Implementation Plan - HPO Production Readiness

## Goal Description
Prepare and verify the Hyperparameter Optimization (HPO) pipeline (`scripts/tune_deepscalper.py`) for the "Full Production Run". 

The current `tune` script is outdated and lacks the critical fixes verified in Phase 6 (WandB initialization, Config Mapping, Network Architecture alignment). We must align it with the stable `train_deepscalper_v3.py` to ensure it runs successfully on the GPUHub remote environment.

## User Review Required
> [!IMPORTANT]
> **WandB Policy**: For HPO (many trials), logging every step to WandB can be overwhelming.
> *   **Proposal**: Initialize WandB in `disabled` mode or `offline` mode for trials, OR log only the final metric. The Trainer *expects* WandB to be active.
> *   **Decision**: I will set `wandb.init(mode="disabled")` inside the HPO loop to prevent crash, since Optuna tracks the metrics (Sharpe).

## Phase 2: Reporting Restoration

**Goal:** Restore the "New Reporting System" (WandB Dashboard + VectorBT) identified by the user.

### Dependencies (`requirements.txt`)
*   [ ] Uncomment/Add `vectorbt`
*   [ ] Add `plotly` (Required for VBT plotting)
*   [ ] Add `kaleido` (Required for static image export if used)

### Integration
*   [ ] **Backtest:** Verify `backtest_deepscalper.py` imports work after dependency install.
*   [ ] **HPO:** Patch `tune_deepscalper.py` to call `generate_wandb_report` on the best trial result.

## Phase 3: HPO Script Patching (`scripts/tune_deepscalper.py`)

**Goal:** Upgrade the outdated HPO script to match the Production-Verified Training/Backtest Verification.

### Fix List
1.  **WandB Initialization:** Add explicit `wandb.init(mode="disabled")` in `objective` function to satisfy Trainer requirement without flooding the dashboard.
2.  **Config Mapping:** Apply the same `dqn_batch_size` -> `batch_size` mapping logic used in `train_deepscalper_v3.py`.
3.  **Network Config:** Apply the nested dictionary restructuring logic for `DeepScalperNetwork`.
4.  **Reporting Hook:** Add the call to `generate_wandb_report` at the end of the script using the best model found.
5.  **Data Path:** Ensure `make_env` uses a relative path or `os.environ` hook for the Parquet file to work on remote.

### Configs
*   **Unified Config**: No changes needed; `tune` script will override values dynamically.

## Verification Plan

### Automated Tests
1.  **Tier 1: Smoke Test (Sanity)**: 
    *   Command: `python scripts/tune_deepscalper.py --trials 1 --steps 100 --debug`
    *   Goal: syntax check, import check.
2.  **Tier 2: Pilot Run**: 
    *   Command: `python scripts/deploy_bare_metal.py ... --config configs/deepscalper_unified.yaml ...`
    *   Args: `--trials 2 --steps 1000` (Fast check, but enough to trigger callbacks).
    *   Note: For full logic verification, use `deepscalper_smoke_test.yaml` (100k steps).

### Success Criteria
*   Script runs 2 trials without crashing.
*   Optuna database (`hpo.db`) is created/updated.
*   Best parameters are logged.

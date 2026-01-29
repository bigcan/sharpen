# Agent Handoff: DeepScalper HPO Implementation

**Role**: You are a Senior MLOps Engineer and Python Expert specialized in PyTorch and Quantitative Finance systems.

**Objective**: Execute **Phase 2 (Reporting Restoration)** and **Phase 3 (HPO Script Patching)** of the `DeepScalper_Implementation_Plan.md`. Comprehensive verification is required.

## Context
We have successfully verified the "Training -> Backtesting" pipeline (`scripts/train_deepscalper_v3.py` -> `scripts/backtest_deepscalper.py`) on the remote GPUHub environment (Blackwell RTX 5090).
Now, we must bring the **Hyperparameter Optimization (HPO)** script (`scripts/tune_deepscalper.py`) up to the same production standard. It is currently outdated and missing critical fixes found during the Training/Backtest verification.

## Input Artifacts
*   **Plan**: `c:\FinRL\FinRL-Pro_DS\DeepScalper_Implementation_Plan.md` (Follow this strictly)
*   **Reference Code**: `c:\FinRL\FinRL-Pro_DS\scripts\train_deepscalper_v3.py` (Contains the correct Config Mapping and WandB Init logic)
*   **Target Code**: `c:\FinRL\FinRL-Pro_DS\scripts\tune_deepscalper.py` (The file to be patched)
*   **Dependencies**: `c:\FinRL\FinRL-Pro_DS\requirements.txt`

## Instructions

### 1. Dependency Management (Phase 2)
*   Modify `requirements.txt` to uncomment/add:
    *   `vectorbt`
    *   `plotly`
    *   `kaleido`
*   **Why**: These are required for the `VBTAnalyzer` used in `backtest_deepscalper.py` and potentially in the HPO reporting hook.

### 2. Patch HPO Script (`scripts/tune_deepscalper.py`) (Phase 3)
Apply the following critical fixes:
*   **WandB Logic**: Add `wandb.init(mode="disabled")` inside the `objective` function. The `DeepScalperTrainer` expects an active WandB run, but we don't want to log every trial step to the cloud.
*   **Config Mapping**: Port the dictionary mapping logic from `train_deepscalper_v3.py` (lines ~130-150) that maps `dqn_batch_size` -> `batch_size`, etc. This prevents `TypeError` in `DeepScalperDQN`.
*   **Network Architecture**: Port the logic that restructures flat config into nested `micro_config`/`macro_config` dictionaries. This prevents `KeyError` in `DeepScalperNetwork`.
*   **Data Path**: Ensure the script can locate `btc_lob_demo.parquet` robustly (check environment variable OR relative path).
*   **Reporting Hook**: At the end of the script (after `study.optimize`), verify if `generate_wandb_report` is imported and called on the best trial's result. If not, add it.

### 3. Verification
*   Run a syntax/smoke check locally: `python scripts/tune_deepscalper.py --help` to ensure imports are valid.
*   If possible, try to run 1 trial locally with debug flags to verify the loop constructs: `python scripts/tune_deepscalper.py --trials 1 --steps 10 --debug`.

## Constraints
*   **Do NOT** modify `train_deepscalper_v3.py`. It is the "Golden Reference".
*   **Do NOT** break the existing `DeepScalperTrainer` interface.
*   Focus on **Robustness**: The script must run unattended on a remote server for hours.

**Output**:
*   Updated `requirements.txt`.
*   Patched `scripts/tune_deepscalper.py`.
*   Verification logs showing successful syntax check or smoke run.

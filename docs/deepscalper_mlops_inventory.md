# DeepScalper MLOps Pipeline Inventory

This document chronicles the inventory of scripts involved in the DeepScalper MLOps pipeline, categorized by function, as of Jan 2026.

## 1. Training & HPO (The Core)

*   `scripts/tune_deepscalper.py`
    *   **Function:** The primary entry point for Hyperparameter Optimization (HPO) and large-scale training (Mach 1/2/3).
    *   **Key Features:** Handles Optuna studies, trial orchestration, and Shared Memory initialization. Supports Independent and Joint HPO strategies.
*   `scripts/train_deepscalper_v3.py`
    *   **Function:** The latest standalone training script for single-run experiments (non-HPO).
*   `finrl_pro_ds/training/deepscalper_trainer.py`
    *   **Function:** The Core Trainer Class.
    *   **Logic:** Contains logic for the training loop, agent updates (DQN/PPO/A2C), AsyncVectorEnv handling, and WandB logging.

## 2. Deployment & Infrastructure

*   `scripts/deploy_bare_metal.py`
    *   **Function:** The main deployment utility.
    *   **Actions:** Zips the codebase, uploads it to the remote GPUHub server, installs dependencies, and launches the training command. Has "Gold Cache" for data optimization.
*   `scripts/remote_cmd.py`
    *   **Function:** Utility to execute arbitrary shell commands on the remote server (e.g., `ps aux`, `nvidia-smi`) without full deployment.
*   `scripts/monitor_status.py`
    *   **Function:** Dedicated script to query the status of running remote processes.

## 3. Data Pipeline

*   `finrl_pro_ds/data/parquet_handler.py`
    *   **Function:** Library module responsible for high-performance data loading, feature engineering, and Shared Memory management.
*   `scripts/generate_synthetic_data.py` / `scripts/generate_smoke_data.py`
    *   **Function:** Utilities to create dummy data for testing the pipeline without full datasets.

## 4. Analysis & Reporting

*   `scripts/analyze_hpo.py`
    *   **Function:** Query the local `hpo.db` (SQLite) to generate reports on trial results, best parameters, and failure rates.
*   `scripts/fetch_db.py`
    *   **Function:** Downloads the `hpo.db` from the remote server for local analysis.
*   `scripts/fetch_run_summary.py` / `scripts/fetch_remote_logs.py`
    *   **Function:** Utilities to retrieve specific logs or summaries from the remote workspace.

## 5. Debugging & Verification

*   `scripts/debug_full_loader.py`
    *   **Function:** Verifies the data loading and feature engineering pipeline integrity.
*   `scripts/debug_env.py`
    *   **Function:** Tests the DeepScalperEnv logic in isolation.
*   `scripts/verify_cuda.py`
    *   **Function:** Checks CUDA availability on the remote server.

## 6. Configuration

*   `configs/deepscalper_unified.yaml`
    *   **Function:** The master configuration file controlling environment parameters, model architecture (DeepScalper), and training settings.

---

## 7. Scheduled Refactors (Tech Debt)

| Item | Status | Description |
|---|---|---|
| `SynapseGatingNetwork` Alias | 🔴 Open | Remove backward-compat alias from `ensemble.py`. Rename all usages to `DeepScalperGatingNetwork`. |


# DeepScalper End-to-End MLOps Pipeline Implementation Plan

**Status:** Implemented (Jan 27, 2026)
**Context:** Mach 3 Optimization (RTX 5090 Shared Memory)
**Framework:** FinRL-Pro_DS (Extension Boundary Only)

## Overview
This document outlines the COMPLETED 5-phase plan that transitioned the DeepScalper "Mach 3" prototype into a unified, integrated data-to-deployment pipeline. The objective was to achieve a single-command workflow (`python -m finrl_pro_ds.cli deploy`) that handles configuration, training, auditing, and governance automatically.

---

## Phase 1: Unified Configuration & Data Hub (COMPLETED)

### 1.1 Objective
Consolidate fragmented configurations (`deepscalper_comprehensive.yaml`, `deepscalper_production.yaml`, code-level constants) into a rigid, hierarchal **Unified Schema**. Ensure data integrity across LOB (250ms) and OHLCV (1m) streams.

### 1.2 Implementation Steps
1.  **Define Unified Schema (`finrl_pro_ds/configs/schema.py`)**:
    *   [x] Use `dataclasses` (std lib) to define strict types for `DataConfig`, `ModelConfig`, `EnvConfig`, `TrainConfig`.
    *   [x] Implement a `ConfigLoader` that reads YAML, validates types, and enforces constraints (e.g., `batch_size % num_envs == 0`).
2.  **Create Master YAML (`configs/deepscalper_unified.yaml`)**:
    *   [x] Migrate all scattered params: `hidden_size`, `fusion_dim`, `reward_scaling`, `hindsight_horizon`, `grad_clip`.
    *   [x] Add `hardware` section: `use_amp`, `compile_mode`, `shm_size`.
3.  **Data Integrity Check (`finrl_pro_ds/data/integrity.py`)**:
    *   [x] Implement `FreshnessGuard`: Assert LOB timestamps increase monotonically and gap < 1s.
    *   [x] Implement `AlignmentGuard`: Assert every LOB tick maps to a valid OHLCV bar (Phase 13 requirement).

### 1.3 Validation
*   [x] Load config: `ConfigLoader.load('configs/deepscalper_unified.yaml')` -> Returns typed object.
*   [x] Data Load: `ParquetDataHandler` triggers `FreshnessGuard` on startup.

---

## Phase 2: High-Throughput Training Orchestration (COMPLETED)

### 2.1 Objective
Formalize the "Mach 3" optimizations (Shared Memory, AMP) into a robust `DeepScalperTrainer` that supports fractional gradient accumulation and automated health monitoring.

### 2.2 Implementation Steps
1.  **Fractional Gradient Accumulation (`finrl_pro_ds/training/accumulators.py`)**:
    *   [x] Create `GradientAccumulator` class to decouple `env_steps` from `optim_steps`.
    *   [x] Logic: `if (global_step % (batch_size // num_envs) == 0): optimize()`.
    *   **Goal**: Allow huge batch sizes (4096) on 24 envs without stalling on every step.
2.  **SPS Watchdog (`finrl_pro_ds/mlops/watchdog.py`)**:
    *   [x] Monitor `steps_per_second` rolling average.
    *   [x] Behavior: If SPS < threshold (e.g., 500) for N minutes, trigger `EmergencyDump` (save checkpoint + trace) and kill process to prevent zombie burns.
3.  **Independent Scalers**:
    *   [x] In `DeepScalperTrainer`: Instantiate separate `torch.cuda.amp.GradScaler` for `actor_loss`, `critic_loss`, and `aux_loss` to prevent value-head instability from collapsing policy gradients.

### 2.3 Validation
*   [x] Run `train_deepscalper.py` with `unified.yaml`.
*   [x] Verify SPS > 1000 on RTX 5090.
*   [x] Verify distinct scaler states in checkpoint dictionary.

---

## Phase 3: Institutional Financial Auditing (COMPLETED)

### 3.1 Objective
Standardize the "Smoke Test" vs. "Full Backtest" protocols and implement automated pass/fail auditing based on `VBTAnalyzer` metrics.

### 3.2 Implementation Steps
1.  **Enhance VBT Analyzer (`finrl_pro_ds/analytics/vbt_analyzer.py`)**:
    *   [x] Add granular getters: `get_sharpe()`, `get_sortino()`, `get_calmar()`, `get_max_drawdown()`.
    *   [x] Ensure all returns are floats (handling 1-element Arrays/Series).
2.  **Automated Audit Script (`scripts/audit_model.py`)**:
    *   [x] **Smoke Test Mode**: Run 1000 steps, assert `crash_count == 0` and `loss != NaN`.
    *   [x] **Financial Audit Mode**:
        *   Load Checkpoint -> Run Backtest on Holdout Data.
        *   Assert `Sharpe > 2.0` (or config threshold).
        *   Assert `MaxDD > -25%`.
        *   **Decay Check**: Compare Validation Score (Training) vs Backtest Score. If deviation > 20%, flag "Distribution Shift".

### 3.3 Validation
*   [x] `python scripts/audit_model.py --checkpoint ...` returns Exit Code 0 (Pass) or 1 (Fail).
*   [x] Markdown Report generated: `reports/audit_V{X}.md`.

---

## Phase 2.5: Automated Hyperparameter Tuning (HPO) (COMPLETED)

### 2.5.1 Objective
Integrate the Optuna-based `tune_deepscalper.py` as a first-class citizen in the pipeline, ensuring it consumes the **Unified Schema** and outputs a "Winning Config" that can be directly consumed by Phase 2 (Training).

### 2.5.2 Implementation Steps
1.  **Refactor `scripts/tune_deepscalper.py`**:
    *   [x] **Input**: Accept `deepscalper_unified.yaml` instead of hardcoded paths.
    *   [x] **Search Space**: Move search space definition (e.g., `learning_rate` range) to a new `hpo` section in the Unified Schema.
    *   [x] **Output**: Save the best trial parameters as `configs/generated/best_params_{timestamp}.yaml`.
2.  **CLI Integration**:
    *   [x] Add `python -m finrl_pro_ds.cli tune` command.
3.  **Distributed HPO (Future-Proofing)**:
    *   [x] Ensure the script is compatible with `optuna.storages.RDBStorage` (SQLite) for potential multi-node scaling.

### 2.5.3 Validation
*   [x] Run `python -m finrl_pro_ds.cli tune --trials 5`.
*   [x] Verify `best_params_*.yaml` is generated and valid.


---

## Phase 4: Remote MLOps & Deployment (CLI) (COMPLETED)

### 4.1 Objective
Create a unified CLI entry point (`python -m finrl_pro_ds.cli`) to simplify remote operations, replacing manual script calls.

### 4.2 Implementation Steps
1.  **CLI Entry Point (`finrl_pro_ds/cli.py`)**:
    *   [x] Use `argparse` with subcommands: `train`, `deploy`, `audit`, `tune`.
2.  **Deployment Logic (`finrl_pro_ds/ops/deploy.py`)**:
    *   [x] Wrap `scripts/deploy_bare_metal.py` logic.
    *   [x] **Provisioning**: Add flag `--provision` to auto-run `pip install ...` on remote.
    *   [x] **SFTP**: Cleanly separate code sync (`*.py`) from data sync (`*.parquet`).
    *   [x] **Heartbeat**: Launch background process with a generic heartbeat wrapper (`finrl_pro_ds/ops/heartbeat.py`) that touches a file every 10s.
3.  **Command Structure**:
    *   `python -m finrl_pro_ds.cli deploy --target gpuhub --config production`
    *   `python -m finrl_pro_ds.cli tune --trials 100`

### 4.3 Validation
*   [x] Execute `python -m finrl_pro_ds.cli deploy ...` from local Windows.
*   [x] Verify remote process starts and logs to W&B.

---

## Phase 5: Automated Governance (COMPLETED)

### 5.1 Objective
Automate the documentation of research findings and leaderboard updates, integrating with the `Research Logger` skill.

### 5.2 Implementation Steps
1.  **Governance Hook (`finrl_pro_ds/governance/hooks.py`)**:
    *   [x] Class `PipelineGovernor`.
    *   [x] Method `on_pipeline_success(run_id, metrics)`:
        *   Format a log entry (Markdown table row).
        *   Append to `randd_log.md`.
2.  **Leaderboard Sync (`finrl_pro_ds/governance/leaderboard.py`)**:
    *   [x] Maintain `leaderboard.json` database.
    *   [x] Sort by Sharpe Ratio.
    *   [x] Update on every audit.

### 5.3 Validation
*   [x] Complete a pipeline run.
*   [x] Check `randd_log.md` for auto-appended entry.
*   [x] Check `leaderboard.json` for updated rankings.

---

## Execution Dependencies
*   **Existing Scripts**: Refactor `deploy_bare_metal.py`, `train_deepscalper.py` into module-callable classes where possible.
*   **Libraries**: Maintain zero new dependencies (use `argparse`, `dataclasses`, `multiprocessing`).

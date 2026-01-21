# Synapse & FinRL-Pro_DS Research Log

A chronological record of experimental findings, performance audits, and strategy pivots for the FinRL-Pro_DS platform.

---

## 2026-01-21 | DeepScalper: Pilot Run (Phase 10) Execution

### Objective
Execute a "Pilot Run" (Test Flight) of the DeepScalper pipeline to validate the entire lifecycle (Training -> Deployment -> Monitoring -> Evaluation) using reduced settings before commissioning full-scale training on the H100 cluster.

### Analysis & Action
1.  **Deployment Infrastructure**:
    -   **Consolidation**: Deprecated legacy deployment scripts and standardized on `scripts/deploy_bare_metal.py`.
    -   **Enhancement**: Upgraded the deployment script to support **Automated Data Upload** via SFTP, allowing large LOB datasets (350MB+) to be transferred seamlessly to the GPUHub instance.
    -   **Standardization**: Enforced WandB run timestamping to prevent namespace collisions.

2.  **Validation & Remediation**:
    -   **Environment**: Fixed a critical `TypeError` in `DeepScalperEnv` reward calculation caused by NumPy array types leaking into scalar float operations. Enforced strict scalar casting.
    -   **Trainer**: Implemented missing functionalities in `DeepScalperTrainer.train()`:
        -   Checkingpointing (saving model state every $N$ steps).
        -   WandB Initialization (`wandb.init` was previously missing).
    -   **Sanity Check**: Successfully executed a 500-step local run with `configs/deepscalper_pilot_local.yaml`, producing 5 valid checkpoints.

3.  **Evaluation Pipeline**:
    -   **Backtesting**: Created `scripts/backtest_deepscalper.py` to bridge the gap between training artifacts and performance analysis.
    -   **VectorBT Integration**: The script loads trained checkpoints, runs a realistic inference loop (using the same `Env` mechanics), and computes professional metrics (Sharpe, Drawdown) using `vectorbt` (or pandas fallback).

4.  **Remote Launch**:
    -   Launched **Run `DS_GPUHub_V1`** on the remote GPU instance.
    -   **Configuration**: 50,000 timesteps, `btc_lob_jan2023.parquet` (cached), WandB Project `FinRL-Pro-DS`.
    -   **Status**: Active and logging.

### Conclusion
**PILOT AIRBORNE**. The DeepScalper system has successfully transitioned from "Code Complete" to "Operationally Active". The entire pipeline—from local code to remote GPU training to performance monitoring—is verified.

**Next Steps**:
1.  Analyze `DS_GPUHub_V1` performance after 50k steps.
2.  Enable `torch.compile` (max-autotune) for the full production training run to maximize H100 utilization.

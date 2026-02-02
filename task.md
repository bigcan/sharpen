# DeepScalper MLOps & Verification Tasks

## Status: Active

### MLOps Pipeline Verification (Jan 28, 2026)
- [x] **Validate Training Loop (Infinite Loop Fix)**
    - Verified `train_deepscalper_v3.py` progressing past Step 0.
    - Verified output logic and observation updates.
- [x] **Deploy Full Training Run (Remote)**
    - Run Name: `DS_Train_V3_Native_Full_Retry` (5000 Steps).
    - Status: **Success**.
    - Artifacts: `checkpoint_final_5000.pth`.
- [x] **Verify Checkpointing**
    - Fixed `DeepScalperTrainer` to enforce save at end of run.
    - Verified existence of `.pth` file on remote server.
- [x] **Deploy Backtesting Run (Remote)**
    - Run Name: `DS_Backtest_V3_Native_Full_Retry`.
    - Status: **Success**.
    - Metrics: PnL Verification Completed (WandB Report Generated).
- [x] **Fix Pipeline Glitches**
    - Fixed `wandb.init` missing in training script.
    - Fixed `DeepScalperNetwork` initialization args in backtest script.
    - Fixed Auto-Discovery logic to find checkpoints in `checkpoints/`.
    - Fixed `deploy_bare_metal.py` argument passing (`--run_name`).

### Next Steps (Phase 7)
- [ ] **Full Production Run:** Scale to 1M+ steps.
- [ ] **Hyperparameter Tuning:** Run Optuna optimization.
- [ ] **Model Analysis:** Investigate flat PnL/0% Return issue (likely initial balance fallback or data limitation).

### Audit & Verification (Feb 1, 2026)
### Audit & Verification (Feb 1, 2026)
- [ ] **Phase 3: DeepScalper Production Deployment & Audit**
    - [/] **Smoke Test Verification (Production Data)** `[Active]`
        - [x] Create `deepscalper_pilot_test.yaml` (Fast HPO/Training)
        - [x] Implement "Joint Fine-tuning" Logic (Phase 3)
        - [x] Deploy to GPUHub with Demo Data (Sanity Check)
        - [x] Resolve Deployment Bugs (Import Errors, Path Issues)
        - [x] Corrected Split Dates: Train (Jan 10-18), Val (Jan 19), Test (Jan 20).
            - [x] Specified precise timestamps (00:00:00 to 23:59:59).
            - [x] Identified LOB vs 1m discrepancy for Pilot duration.
            - [x] Enabled Performance Features: `torch_compile`, `use_shm`.
        - [ ] Run Smoke Test with **Production Data** (Jan 2023) `[Awaiting Clarification]`
        - [x] Fix HPO "Dirty State" (Clean `hpo.db` wipe)
        - [x] Implement WandB Tags for clean naming `[Completed]`
    - [ ] **Expert Audit Implementation**
        - [x] Configure Two-Phase Training (Specialists -> Gating)
        - [x] Implement "Evaluation Survival Check" (Doom Loop Fix) `[Completed]`
        - [x] Verify Shared Memory "Spawn" Context `[Completed]`
    - [ ] **Production Run (Full Scale)**
        - [ ] Deploy `deepscalper_production_fix.yaml`
        - [ ] Monitor Max Drawdown & Stability

### RTX 5090 Optimization (Feb 1, 2026)
- [x] **Precision Upgrade**: Enforce TF32 and FP16 (Mixed Precision) in `DeepScalperTrainer`.
    - [x] Fix Replay Ratio (`dqn_update_interval: 2048.0`) <!-- id: 11 -->
    - [x] Implement Micro-Gating (LOB Encoder) <!-- id: 12 -->
    - [/] Verify with Smoke Test (Production Data, Joint Phase) <!-- id: 13 -->
    - [ ] Deploy Production Config (Phase 3 Enabled) <!-- id: 14 -->

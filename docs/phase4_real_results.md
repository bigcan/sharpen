# Phase 4: Real Data Diagnostics & Stress Test Report

**Date:** 2025-11-20
**Status:** Phase 4 COMPLETE (Failed)

## Overview
Following the user request to "proceed to phase 4", a comprehensive diagnostic and stress testing suite was executed. This phase aimed to validate the robustness of the Phase 3 "Winner" (PPO Clip 0.30) using **Real Data** and **Real Training**, addressing concerns that previous results might have been simulated.

## 1. Root Cause Diagnostics

### A. Data Quality & PIT Audit
- **Objective:** Verify that feature engineering is Point-In-Time (PIT) safe.
- **Method:**
    - Analyzed `finrl_pro_ds.features.custom_features.add_fracdiff_features` code. Confirmed strict `t-1` shifting logic.
    - Ran `finrl_pro_ds.eval.pit_validator` on `ma20_lag` feature. Result: **PASS**.
    - Created `scripts/verify_fracdiff.py` to re-calculate fracdiff features from raw price. Result: **PASS** (matches CSV).
- **Conclusion:** The Feature Engineering pipeline is robust and PIT-safe.

### B. Environment & Cost Model Review
- **Objective:** Verify that `ProStockEnv` correctly applies costs and executes trades.
- **Method:**
    - Audited `finrl_pro_ds.envs.pro_stock_env.py`. Cost logic `price * delta * (1 + fee)` is correct.
    - Executed `scripts/trace_env.py` for a 50-step episode. Traced cash/asset flow manually.
- **Conclusion:** The Environment logic is correct. No double-charging or execution bugs found.

## 2. Real Training Validation (The "Reality Check")

A critical finding during diagnostics was that previous Phase 3 configurations lacked `real_training: true`, implying they relied on the built-in simulator (random noise generation). 

To rectify this, **Real Training** experiments were launched using the Phase 3 Winner configuration (`ppo_clip_0_30_fracdiff_d_0_5`) against the actual `sp500_daily_2016_2025` dataset.

### Experiment 1: Real Baseline (10 bps costs)
- **Config:** `phase4_real/baseline_real.yaml`
- **Result:** **FAILED** (Risk Breach)
- **Details:** `RuntimeError: Drawdown 0.3579 exceeds limit 0.2000.`
- **Analysis:** The agent, when exposed to real market dynamics, incurred a 35.8% drawdown, violating the 20% risk cap. This confirms the strategy is not robust on single-asset data.

### Experiment 2: Cost Stress Test (2x Costs)
- **Config:** `phase4_real/stress_2x.yaml`
- **Result:** Sharpe 0.20, MaxDD 3%.
- **Analysis:** The low MaxDD suggests the agent became extremely passive/timid under higher costs, effectively stopping trading to preserve capital.

### Experiment 3: Cost Stress Test (3x Costs)
- **Config:** `phase4_real/stress_3x.yaml`
- **Result:** Sharpe -0.40, MaxDD 2%.
- **Analysis:** Further degradation. The strategy holds mostly cash or makes losing trades.

## Conclusion
The "Phase 3 Winner" was likely an artifact of simulation or noise mining. On real data, the single-asset PPO strategy fails to meet basic risk gates (MaxDD > 35%) or fails to generate alpha under stress.

**Recommendation:** Proceed to Phase 5 (Multi-Asset Allocation) as planned, acknowledging that single-asset timing on this dataset is not viable with the current feature set.

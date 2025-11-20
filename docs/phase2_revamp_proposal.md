# Phase 2 Revamp: Expert Data Scientist Assessment & Proposal

**Date:** 2025-11-20
**Status:** Phase 2.8 Execution (Hybrid Baseline)
**Author:** Antigravity (AI Data Scientist)

## 1. Executive Summary & Diagnosis
... (Previous sections) ...

## 2. Proposed New Directions
... (Previous approaches) ...

### Approach E: The "FinRL Hybrid" Pivot
**Critique:** Previous attempts (Log HLOCV, Wavelets) failed because they were "jagged" and confusing to the PPO agent, which prefers smooth gradients.
**Philosophy:** Start with the FinRL recommended setup (smooth indicators), then upgrade the weak points with marketing/volume-aware features.

**Hybrid Feature Set:**
1.  **MACD:** Trend (Smooth).
2.  **RSI_14:** Momentum (Reactive, faster than 30).
3.  **VWAP_Ratio:** `VWMA_14 / Close` (Volume-weighted valuation).
4.  **ATR_Norm:** `ATR_14 / Close` (Normalized Volatility). **Fixes the non-stationary bug.**
5.  **Log_Volume:** `Log(Volume)` (Traffic Scale).

**Why this should work:**
-   **Smoothness:** MACD/RSI are friendly to Neural Networks.
-   **Stationarity:** ATR_Norm and VWAP_Ratio are relative (stationary), avoiding the OOD crash.
-   **Volume Context:** Log_Volume gives the agent confidence sizing.

## 3. Recommended Action Plan (Phase 2.8)

### 3.1 Implement Hybrid Features
**Action:** Implemented `add_hybrid_features` in `custom_features.py` and wired it into `loader_pro.py`.

### 3.2 Validation
Run `scripts/run_hybrid_baseline.py` on Seeds 41, 42, 43.
**Target:** Stability (Drawdown < 20%) and Positive Sharpe.

## 5. Results (2025-11-20)

### Real-Data Validation (Trend Baseline)
**Status:** FAILED. (Pending final logs, but expected failure given Log Baseline result).

### Real-Data Validation (Hybrid Baseline)
**Status:** In Progress (PID 9937).
- Features: MACD, RSI_14, VWAP_Ratio, ATR_Norm, Log_Volume.
- Goal: Prove that smooth + stationary + volume features provide the necessary signal for PPO.

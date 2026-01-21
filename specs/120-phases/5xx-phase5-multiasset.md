# Phase 5 Spec: Multi-Asset Allocation & Regime Robustness

**Owner:** Research Lead + Risk Lead  
**Status:** Draft  
**Gate:** Sharpe > 1.0 (2000-2025), MaxDD < 50%, Exposure Caps respected.

## 1. Motivation
Previous phases failed to produce robust agents because single-asset timing in low-signal-to-noise environments is fragile. Phase 5 shifts to **Multi-Asset Allocation** (Liquid 20) over a massive timeframe (25 years). This forces the agent to learn:
1.  **Diversification:** Using correlation structure to lower variance.
2.  **Regime Adaptation:** Surviving distinct market conditions (2008 crash, 2010-2019 Bull, 2020 Crash, 2022 Inflation).

## 2. Execution Plan

### Phase 5.1: Data & Regimes
*Goal: Ensure data covers all regimes and splits are "regime-aware" (not just random).*
*   **Dataset:** Liquid 20 (SPY, QQQ, IWM, Sector ETFs, Top Stocks, Gold, Treasuries).
*   **Period:** 2000-01-01 to 2025-01-01.
*   **Regime Labeling:** Use VIX levels and SPY Trend (SMA200) to label periods as:
    *   *Crisis* (VIX > 30)
    *   *Bull* (VIX < 20, Price > SMA200)
    *   *Correction* (VIX 20-30)
*   **Splits:** Construct Train/Val/Test sets that explicitly contain samples from *all* regimes, rather than simple chronological splitting which might hide a crisis in the test set.

### Phase 5.2: Features (Macro & Risk)
*Goal: Provide the agent with the state variables needed to identify regimes.*
*   **New Features:**
    *   **VIX:** Implied volatility (fear gauge).
    *   **Macro:** 10Y Treasury Yield (TNX), USD Index (DXY).
    *   **Correlation:** Rolling pairwise correlation (mean of matrix) to detect "risk-on/risk-off" synchronization.
    *   **Turbulence:** Mahalanobis distance (existing).

### Phase 5.3: Baselines
*Goal: Establish the "beat" bar.*
*   **Equal Weight:** 1/N rebalancing.
*   **Risk Parity:** Inverse volatility weighting.
*   **Momentum:** Long top N assets by 12-month return.

### Phase 5.4: Training (Staged)
*Goal: Curriculum learning to prevent "fear-learning" (never trading) or "greed-learning" (ignoring risk).*
*   **Stage 1 (Bull):** Train on low-volatility periods to learn mechanics and profit seeking.
*   **Stage 2 (Crisis):** Fine-tune on high-volatility periods to learn protection.
*   **Stage 3 (Full):** Full timeline training.

### Phase 5.5: Evaluation
*Goal: Detailed report card.*
*   **Per-Regime Performance:** Sharpe/Drawdown broken down by regime label.
*   **Turnover Analysis:** Ensure costs don't eat diversification benefits.

## 3. Requirements
*   **Environment:** `PortfolioAllocationEnv` (Action: Weights vector, sum=1).
*   **Constraint:** Long-only, no leverage.
*   **Wrapper:** `ActionSmoothingWrapper` (alpha=0.1) to prevent churn.

## 4. Artifacts
*   `specs/5xx-phase5-multiasset.md` (This file)
*   `data/liquid20_vix_macro.parquet`
*   `finrl_pro_ds/envs/portfolio_allocation.py`

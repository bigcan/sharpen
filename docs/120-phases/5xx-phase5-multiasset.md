# Phase 5: Multi-Asset Allocation & Constraints

## 1. Motivation
Single-asset trading (Phase 0-4) has proven fragile, with performance highly sensitive to market regimes and noise. To achieve robust, institutional-grade performance, we must transition to **Multi-Asset Portfolio Optimization**. By diversifying across a basket of assets (e.g., S&P 500 sector leaders), we can smooth the equity curve, reduce idiosyncratic risk, and leverage low pairwise correlations.

## 2. Goals
- **Portfolio Optimization:** Transition from independent asset timing to capital allocation (portfolio weights).
- **Risk Control:** Enforce strict exposure limits (e.g., Sector Caps, Max Position Size).
- **Long-Only Baseline:** Establish a stable long-only baseline before adding shorting or complex derivatives.

## 3. Requirements

### 3.1. Environment & Action Space
- **Environment:** `ProStockEnv` (or subclass) must support multi-asset inputs.
- **Action Space:** Continuous vector $a \in [0, 1]^N$ representing portfolio weights $w$.
- **Constraint:** $\sum w_i \le 1$ (Cash buffer allowed) or $\sum w_i = 1$ (Fully invested).
- **Wrapper:** `SoftmaxAllocationWrapper` to convert raw agent logits into valid portfolio weights.

### 3.2. Risk Constraints
- **Sector Exposure:** Max X% allocation to any single sector (if sector data available).
- **Position Limits:** Max Y% allocation to any single asset.
- **Turnover Control:** Penalize excessive rebalancing to account for transaction costs.

### 3.3. Data
- **Timeframe:** **2000-01-01 to 2025-01-01** (25 Years).
    - *Rationale:* Covers Dot-com crash, 2008 GFC, 2010s Bull Market, 2020 Covid, 2022 Inflation.
- **Universe:** "Liquid 20" Basket (High Capacity, Moderate Correlation).
    - **Indices:** SPY, QQQ, IWM
    - **Sectors:** XLK, XLF, XLE, XLV
    - **Stocks:** AAPL, MSFT, AMZN, NVDA, GOOGL, JPM, XOM, JNJ, PG, V
    - **Defensive:** GLD, TLT
- **Features:** Global features (VIX, Macro) + Per-Asset features (Price, Vol, Tech).

## 4. Acceptance Criteria
- **Performance:** Sharpe Ratio >= 1.0 (or >= 1.2x the best single-asset baseline).
- **Drawdown:** Max Drawdown < Buy & Hold (Equal Weight) and < Long/Flat baseline.
- **Constraints:** 100% adherence to position/sector limits in validation.
- **Stability:** Consistent performance across 3+ random seeds.

## 5. Artifacts
- `docs/120-phases/5xx-phase5-multiasset.md` (This file)
- `finrl_pro/configs/experiments/phase5/` (Experiment configs)
- `reports/phase5/` (Results and Analysis)

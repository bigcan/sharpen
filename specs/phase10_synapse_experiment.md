# Phase 10 Experiment: Synapse Arbitrator Validation

**Date:** 2025-12-04
**Status:** Preregistered
**Objective:** Validate Synapse Arbitrator (Profit-Based Scoring) against baselines using Walk-Forward Analysis on Colab.

## 1. Universe Selection
**Same as Phase 9 (20 Blue-Chip US Equities):**
- WMT, KO, PEP, MCD, NKE, COST, CL
- CSCO, INTC, ORCL, IBM, VZ, T, ADBE, TXN
- BA, CAT, DIS, PFE, HON

## 2. Data Protocol
- **Source:** Yahoo Finance (`yfinance`)
- **Range:** 2020-01-01 to 2024-12-31 (5 Years)
- **Features:** OHLCV + Technical Indicators (MACD, RSI, Bollinger, ADX)

## 3. Walk-Forward Design (Shortened for Speed)

| Phase | Train | Validation | Test |
|:------|:------|:-----------|:-----|
| W1 | 2020-01 to 2021-06 | 2021-07 to 2021-12 | 2022-01 to 2022-06 |
| W2 | 2020-07 to 2022-06 | 2022-07 to 2022-12 | 2023-01 to 2023-06 |
| W3 | 2021-01 to 2023-06 | 2023-07 to 2023-12 | 2024-01 to 2024-06 |

**Window Structure:**
- Train: 18 months (Rolling)
- Validation: 6 months
- Test (OOS): 6 months

## 4. Specialist Training (Per Window)
Train 3 specialist agents on regime-filtered subsets of the training data:
1. **Bull Specialist:** Train on top 30% return days
2. **Bear Specialist:** Train on bottom 30% return days
3. **Neutral Specialist:** Train on middle 40% days

**HPO:** Optuna, 15 trials, Objective = Calmar Ratio

## 5. Ensemble Methods to Compare

| Method | Description |
|:-------|:------------|
| **Baseline: Single Best** | Best agent from HPO (Validation Calmar) |
| **Baseline: Simple Avg** | Mean action of all 3 specialists |
| **Baseline: Regime Switch** | HMM-based regime detection → select specialist |
| **Synapse: Consensus** | Probabilistic mixture, consensus scoring |
| **Synapse: Profit** | Probabilistic mixture, profit-based scoring |

## 6. Transaction Costs
- Buy/Sell: 10 bps (0.1%)
- Turnover Penalty: 1% of position delta

## 7. Success Metrics

| Metric | Target |
|:-------|:-------|
| **Global Sharpe (2022-2024 OOS)** | > 1.0 |
| **Global Calmar** | > 0.5 |
| **Beat Single Best** | Yes |
| **Max Drawdown** | < 25% |

## 8. Compute Requirements
- **Platform:** Google Colab (T4 GPU)
- **Estimated Time:** ~2-3 hours total
- **Notebook:** `colab_phase10_synapse.ipynb`

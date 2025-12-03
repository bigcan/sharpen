# Phase 9 Experiment Manifest: "The Sonnet Protocol"

**Date:** November 26, 2025
**Status:** Preregistered
**Objective:** Evaluate the robustness of a PPO Agent using Walk-Forward Analysis with Regime-Aware Ensembling and strict Transaction Costs.

## 1. Universe Selection
**Criteria:** 20 Liquid, Blue-Chip US Equities with continuous data from 2005-01-01.
**Tickers:**
- **Consumer:** WMT, KO, PEP, MCD, NKE, COST, CL
- **Tech/Telecom:** CSCO, INTC, ORCL, IBM, VZ, T, ADBE, TXN
- **Industrial/Pharma:** BA, CAT, DIS, PFE, HON

## 2. Data Protocol
- **Source:** Yahoo Finance (via `yfinance`)
- **Range:** 2005-01-01 to 2025-01-01
- **Features:** Open, High, Low, Close, Volume (Standard OHLCV) + Technical Indicators (MACD, RSI, CCI, ADX) defined in `ProFeatureAssembler`.

## 3. Walk-Forward Design
To mitigate overfitting and capture regime changes:
- **Training Window:** 3 Years (Rolling)
- **Validation (HPO) Window:** 1 Year (Rolling)
- **Test (Out-of-Sample) Window:** 1 Year (Rolling)
- **Step Size:** 1 Year

**Sequence Example:**
1. **Train:** 2005-2007 | **Valid:** 2008 | **Test:** 2009
2. **Train:** 2006-2008 | **Valid:** 2009 | **Test:** 2010
...and so on.

## 4. Hyperparameter Optimization (HPO)
- **Method:** Optuna (TPE Sampler)
- **Trials:** 20-30 per window (constrained by compute)
- **Objective Function:** **Modified Calmar Ratio**
  - Formula: `Annualized Return / abs(Max Drawdown + epsilon)`
  - *Rationale:* Prioritizes survival and risk-adjusted returns over raw profit.

## 5. Ensemble Strategy (Risk Mitigation)
Instead of selecting the single best agent (Risk of overfitting):
- **Selection:** Select Top 3 Agents from HPO based on Validation Calmar.
- **Execution:** Weighted Soft Voting.
  - Weights = Softmax(Validation Calmar Ratios) of the Top 3.

## 6. Transaction Costs & Constraints
- **Transaction Cost:** 0.1% (10 bps) per trade (Buy & Sell).
- **Turnover Penalty:** `reward -= 0.01 * abs(delta_position)`
- **Risk Budget:** Max position size 100 shares (for simplicity/scaling).

## 7. Success Metric
The experiment is considered successful if the **Ensemble** achieves:
- **Global Sharpe Ratio (2009-2025):** > 1.0
- **Global Calmar Ratio:** > 0.5
- **Performance:** Outperforms `Buy & Hold (Equal Weight)` baseline.

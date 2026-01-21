# Phase 4: Risk, Costs, and Robustness

## Motivation
Strategies that perform well in idealized backtests often fail in live trading due to transaction costs, execution delays, and overfitting. Phase 4 focuses on stress-testing the "Phase 3 Winner" to ensure it is robust enough for multi-asset scaling.

## Requirements

### 1. Robustness Metrics
- **Deflated Sharpe Ratio (DSR):** Adjust Sharpe ratio for the number of trials (multiple testing bias). Target: DSR > 0 (95% confidence).
- **Probability of Backtest Overfitting (PBO):** Use Combinatorial Purged Cross-Validation (CPCV) to estimate the probability that the in-sample winner performs below the median out-of-sample. Target: PBO < 0.20.

### 2. Stress Tests
- **Cost Stress:** Evaluate performance at 2x and 3x standard transaction fees (2bps -> 4bps, 6bps) and slippage.
- **Execution Gap:** Simulate 1-tick and 2-tick execution delays.
- **Input Noise:** Add Gaussian noise to input features to test signal stability.

### 3. Structural Improvements
- **Action Smoothing:** Implement a wrapper to penalize rapid direction changes or smooth actions (e.g., `0.9 * prev_action + 0.1 * new_action`) to reduce turnover.

## Deliverables
1.  `scripts/run_phase4_robustness.py`: Driver script for PBO and DSR calculations.
2.  `reports/phase4/robustness_report.json`: JSON report containing DSR, PBO, and stress test results.
3.  `finrl_pro_ds/eval/robustness.py`: Library functions for DSR and PBO.

## Success Criteria
- **Pass:** DSR > 0, PBO < 0.20, and positive Sharpe under 2x cost stress.
- **Fail:** Strategy collapses (Sharpe < 0) under stress or shows high probability of overfitting.

## Artifacts
- `results/phase4/pbo_heatmap.png`
- `results/phase4/dsr_stats.txt`

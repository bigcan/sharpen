# Spec 600: HPO Driver & Ensemble Orchestrator

## 1. Motivation
Manual tuning is inefficient and prone to overfitting. "Best" parameters often change over time. We need a rigorous, automated system to find stable hyperparameters and combine diverse agents.

## 2. Goals
1.  **Efficiency:** Use Bayesian Optimization (Optuna TPE) to find optima 10x faster than grid search.
2.  **Stability:** Verify that optimal parameters are robust across time (Walk-Forward HPO).
3.  **Diversity:** Construct ensembles from agents that behave differently (Regime Specialists).

## 3. Architecture

### 3.1 The HPO Driver (`optuna_driver.py`)
- **Objective Function:** Maximizes `Sharpe * (1 - PBO_Penalty)`.
- **Sampler:** TPE (Tree-structured Parzen Estimator).
- **Pruner:** Hyperband (early stopping of bad trials).
- **Storage:** SQLite/Postgres for distributed sweeps.

### 3.2 Walk-Forward Validator
- **Loop:**
    - Train on `[T_start, T_end]`
    - Optimize HPO on `[T_end - Val_Size, T_end]`
    - Test on `[T_end, T_end + Test_Size]`
    - Slide window forward.
- **Output:** `param_stability_score` (Variance of optimal params over folds).

### 3.3 Ensemble Strategy
- **Voting:** Soft voting (average probabilities/actions).
- **Gating:**
    - `if ADX > 25: weight = TrendAgent`
    - `else: weight = MeanReversionAgent`

## 4. Acceptance Criteria
- **Speed:** 100 trials complete in < 12 hours on available hardware.
- **Stability:** Top 10% of params in Fold N overlap with Top 10% in Fold N+1.
- **Uplift:** Ensemble Sharpe > 1.1 * Best Single Sharpe.

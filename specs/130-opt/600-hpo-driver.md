# Phase 6: HPO/HPIO & Regime-Aware Ensembles

## Motivation
Manual tuning ("Alchemy") is inefficient and prone to bias. A single agent cannot optimally trade all market regimes (Bull, Bear, Sideways). Phase 6 introduces scientific hyperparameter optimization (HPO) and a "Mixture of Experts" ensemble approach.

## Requirements

### 1. Bayesian Optimization (HPO)
- **Algorithm:** Tree-structured Parzen Estimators (TPE) via `optuna`.
- **Search Space:** Learning Rate, Gamma, Clip Ratio, Entropy Coefficient, Batch Size.
- **Objective:** Val Sharpe Ratio (constrained by Max Drawdown).

### 2. Walk-Forward HPO
- **Protocol:** Instead of a static train/test split, use rolling windows (Train Y1-Y3, Val Y4 -> Test Y5) to test parameter stability.
- **Stability Check:** Optimal parameters should not fluctuate wildly between windows.

### 3. Regime-Aware Ensemble
- **Specialists:** Train separate agents for:
    - **Bull:** High Beta, trend-following.
    - **Bear:** Short-bias or cash-heavy, mean-reversion.
    - **Sideways:** Low-beta, mean-reversion.
- **Meta-Learner:** A mechanism (Voting, Gating Network, or Heuristic like VIX/ADX) to switch between specialists or weight their outputs.

## Deliverables
1.  `finrl_pro_ds/automl/hpo_driver.py`: Class to orchestrate Optuna trials.
2.  `scripts/tune_phase6_ppo.py`: Script to run the HPO sweep.
3.  `finrl_pro_ds/agents/ensemble.py`: Implementation of the `VotingEnsemble` or `RegimeSelector`.
4.  `finrl_pro_ds/configs/experiments/phase6_specialists/`: Config files for Bull/Bear/Sideways agents.

## Success Criteria
- **Ensemble Uplift:** Ensemble Test Sharpe > 1.2x best single agent.
- **Stability:** Top HPO parameters remain stable across > 3 walk-forward folds.
- **Regime Detection:** System correctly identifies major market regimes (e.g., 2020 crash, 2023 bull).

## Artifacts
- `results/phase6_hpo/parallel_coordinates.png`: HPO parameter interaction plot.
- `models/phase6/ensemble/`: Saved ensemble model artifacts.

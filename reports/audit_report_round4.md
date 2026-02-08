# DeepScalper Audit Report: Round 4 Findings & Fixes
**Date**: 2026-02-08
**Auditor**: Antigravity (Senior Data Scientist Persona)

## Executive Summary
This audit investigated the catastrophic performance observed in recent DeepScalper runs (Sharpe -97, 19:1 Sell Bias). We identified two critical structural flaws that were preventing the agent from learning:
1.  **Structural Sell Bias**: An asymmetric margin requirement in the environment that made Buying significantly harder than Selling.
2.  **Premature Pruning**: The Hyperparameter Optimization (HPO) process was terminating trials before the agent had begun to learn (due to extremely slow exploration decay), effectively optimizing for "lucky random walks."

We have implemented fixes for both issues and verified them with unit tests. The system is now ready for a robust HPO campaign.

---

## 1. Finding: The Structural "Sell Bias"
**Severity**: CRITICAL (Invalidates Policy)

### Observation
The agent exhibited a 19:1 ratio of Sells to Buys. Even random agents should have a near 1:1 ratio. This suggested a fundamental asymmetry in the Environment's transition dynamics, not just a learned behavior.

### Root Cause Analysis (`deep_scalper_env.py`)
We traced the order execution logic and discovered a critical discrepancy in how `Buy` vs `Sell` orders were validated:
*   **Long Orders (Buy)**: Were checked against `self.balance` (Cash). If `Order Cost > Cash`, the order was rejected. This effectively enforced a **100% Cash Requirement (1x Leverage)**.
*   **Short Orders (Sell)**: Were checked against `current_pos * margin_rate` (where implicit margin was 50%). This allowed the agent to open Short positions with **2x Leverage**.

Because the agent had 2x the buying power for Shorts compared to Longs, it learned that Shorting was more "reliable" and available, leading to a massive Sell Bias.

### The Fix: Symmetric Margin Logic
We refactored `deep_scalper_env.py` to use a consistent `_check_margin` method for both directions.
*   **New Logic**: `Required Equity = Order Value * margin_requirement`
*   **Configuration**: Added `margin_requirement: 1.0` (Default) to `deepscalper_rtx5090_production.yaml`. This enforces Spot-like (100% Cash) rules for *both* sides, ensuring perfect symmetry.
*   **Result**: Buying and Selling now have identical capital requirements.

---

## 2. Finding: Premature HPO Pruning ("The Random Walk Trap")
**Severity**: CRITICAL (Prevents Learning)

### Observation
HPO trials were being pruned at step 5,000 with a Sharpe of -2.0 to -4.0. The `epsilon_decay` parameter was set to `0.99999` (tuned for 1M+ steps).

### Root Cause Analysis (`deepscalper_trainer.py`)
At step 5,000 with `decay=0.99999`, the Epsilon value is `0.99999^5000 ≈ 0.95`.
*   **The Issue**: The agent was taking random actions **95% of the time** when it was being evaluated and pruned.
*   **The Consequence**: Optuna was effectively pruning based on the "luck" of the random seed, not the quality of the hyperparameters. Meaningful learning (Exploitation) doesn't start until Epsilon drops below ~0.5, which would take 70,000+ steps with the old setting—longer than the entire HPO trial!

### The Fix: Dynamic HPO Schedule
We modified `deepscalper_trainer.py` to be "HPO-Aware":
1.  **Dynamic Epsilon Decay**: When `hpo_mode=True`, the trainer automatically calculates a decay rate that ensures Epsilon drops to `0.01` by the *end of the trial* (e.g., 50k steps).
    *   *Old Decay*: 0.99999 (Epsilon ~0.60 at 50k steps) -> **Still Random**
    *   *New Decay*: ~0.9991 (Epsilon ~0.01 at 50k steps) -> **Fully Converged**
2.  **Minimum Pruning Steps**: Added a safeguard to prevent any pruning before step 20,000. This guarantees the agent has at least some time to collect rewards before being judged.

---

## Verification Results

### 1. Environment Symmetry Test (`tests/test_env_loading.py`)
| Test Case | Margin Req | Balance | Order Size | Result | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Buy 1 BTC** | 1.0 (Spot) | 100k | $20k | **ALLOWED** | Cost < Balance |
| **Sell 1 BTC** | 1.0 (Spot) | 100k | $20k | **ALLOWED** | Symmetric |
| **Buy 10 BTC** | 1.0 (Spot) | 100k | $200k | **REJECTED** | Insufficient Margin |
| **Sell 10 BTC** | 1.0 (Spot) | 100k | $200k | **REJECTED** | Symmetric Rejection |

### 2. HPO Trainer Test (`tests/test_trainer_hpo.py`)
*   **Scenario**: 50,000 Step Trial, 10 Envs.
*   **Expected Decay**: ~0.99908
*   **Actual Decay**: `0.999079` (Verified in Log)
*   **Result**: The trainer successfully overrides the configuration default to adapt to the short trial horizon.

---

## Recommendations & Next Steps

1.  **Run HPO**: Launch a new Optuna study. The trials will now accurately reflect the performance of the hyperparameters.
2.  **Monitor Win Rate**: Expect the Win Rate to climb above 45-50% reasonably quickly (within 20k steps) now that the agent is actually learning.
3.  **Monitor Buy/Sell Ratio**: This should converge towards 1.0 (or execute logic based on trend) rather than being stuck at 0.05.

**Status**: READY FOR DEPLOYMENT/TRAINING.

# DeepScalper V2 Improvement Plan

**Last Updated:** 2026-02-14

## Overview
This document tracks the incremental improvements to upgrade DeepScalper from V1 (Pilot) to V2 (Stabilized & Featurized).

## Sprints

### Sprint 1: Risk & Exploration Basics (COMPLETED)
- **Status:** ✅ Deployed & Verified
- **Outcome:** Fixed risk metrics, removed hold bonus, implemented basic epsilon decay.
- **Key Artifacts:** `v95_walk_forward.yaml`

### Sprint 2: Advanced Exploration (COMPLETED - REGRESSED)
- **Status:** ❌ Regression (Q-value Divergence)
- **Outcome:** Boltzmann exploration caused excessive Q-value growth (>1000).
- **Action:** Revert to Epsilon-Greedy in Sprint 3.

### Sprint 3: Stabilization & Features (READY FOR REVIEW)
- **Status:** 🟡 Verification Complete, Pending Audit
- **Objective:** Fix Sprint 2 regression and add remaining V2 features.
- **Implemented:**
    -   **Revert:** `exploration_mode` -> "epsilon_greedy".
    -   **Fix:** `gamma` -> 0.99 (was 0.995).
    -   **Safety:** `target_q_clip` (Limit Q-values to 5000).
    -   **Feature:** `RunningRewardNormalizer` (Normalize rewards to N(0,1)).
    -   **Feature:** Drawdown Penalty (5x for >10% DD).
- **Verification:**
    -   Unit Tests: `TestTargetQClipping` (PASSED).
    -   Smoke Run: `DeepScalper_V1_Local_20260214_0750` (Stable Q ~2.0).

### Sprint 4: Walk-Forward & Evaluation (PLANNED)
- **Status:** ⚪ Planned
- **Objective:** Implement rigorous walk-forward evaluation.
- **Items:**
    -   Multi-window backtesting.
    -   Out-of-sample metrics aggregation.

## Codebase Status
- **Current Branch:** (Main/V2-Dev)
- **Config:** `configs/deepscalper_rtx5090_production.yaml`
- **Tests:** `tests/test_audit_regression.py` covers critical fixes.

# Phase 2 Revamp: Expert Data Scientist Assessment & Proposal

**Date:** 2025-11-20
**Status:** Phase 2.8 Execution (Hybrid Baseline)
**Author:** Antigravity (AI Data Scientist)

## 1. Executive Summary & Diagnosis
... (Previous sections) ...

## 2. Proposed New Directions
... (Previous approaches) ...

### Approach E: The "FinRL Hybrid" Pivot
... (Description of Hybrid) ...

## 3. Recommended Action Plan (Phase 2.8)

### 3.1 Implement Hybrid Features
**Action:** Implemented `add_hybrid_features`.

### 3.2 Validation
Run `scripts/run_hybrid_baseline.py` on Seeds 41, 42, 43.
**Target:** Stability (Drawdown < 20%) and Positive Sharpe.

## 5. Results (2025-11-20)

### Real-Data Validation (Trend Baseline)
**Status:** Stable but Flat.
- Seed 41: Sharpe -0.01. No Crash.
- **Insight:** SMA distance features successfully prevented the agent from blowing up (unlike Log/Hybrid), likely by identifying downtrends. However, the agent was too conservative to make money.

### Real-Data Validation (Hybrid Baseline)
**Status:** FAILED (Seed 41 Crash).
- **Insight:** Smooth/Volume features (MACD/RSI/VWAP) reintroduced instability. The agent likely overtraded the noise.

**Final Verdict (Phase 2):**
- **Best Stability:** Trend Baseline (SMA Context).
- **Best Signal:** None found (All alpha attempts led to crashes).
- **Recommendation:** Carry forward the **Trend Baseline** features to Phase 3. Let the advanced algorithms (SAC/TD3) try to squeeze alpha from this stable foundation, rather than adding more volatile features that PPO can't handle.
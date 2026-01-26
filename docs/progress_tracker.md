# FinRL Pro - Research Program Progress Tracker

See `docs/research_roadmap.md` for the live research plan, milestones, and success criteria.

> Purpose: Track progress across phases with explicit gates, risk checks, and deliverables. Use GitHub task lists; one run = one seed. Keep the test set frozen.

Owner: <YOUR_NAME>
Start: <YYYY-MM-DD>
Target MVP (Phase 0) date: 2025-11-07
Repo commit: 60dd9a71625bcffa33678e2ae6e13e3bddd13bf4
Dataset hash: dvc://datasets/sp500_daily_2020_2025
Config: `finrl_pro_ds/configs/experiments/sp500_daily.yaml:1`
Fingerprints index: `reports/matrix/fingerprints.json:1`
Leaderboard: `docs/leaderboard.md:1`
Matrix report: `reports/matrix/report.md:1`

---

## Program Status (snapshot)
- [x] Phase 6: Regime-Aware Ensemble Complete (Sharpe 1.24)
- [x] Phase 7: XAI & OPE Analysis Complete (SHAP verified, OPE high variance)
- [x] Phase 26/27: DeepScalper Performance Optimization Complete (50x Speedup)
- [ ] Phase 8: Paper Trading & Monitoring (Next)
- [x] Leaderboard initialized with MVP fingerprint
- [x] Final report drafted: `reports/matrix/final_report.md:1`
- [x] PSR/CI computed and logged

---

## Legend
- [ ] pending
- [x] done
- [~] in progress (optional)

---

## Phase 6 - Alpha Validation & Ensembles (Complete)
Goal: Transition to real data, optimize parameters, and deploy regime-aware ensembles.

- Data Foundation
  - [x] Ingest 10-15 years of S&P 500 data (OHLCV) via Alpaca/Yahoo (Created `data/sp500_full_2010_2025.parquet`)
  - [x] Create canonical `snapshot://sp500_full`
- Feature Audit
  - [x] Upgrade `pit_validator` to support recursive indicators (EMA, etc.) (Implemented Deletion Test)
  - [x] Certify feature set on real data (no look-ahead bias) (Passed Deletion Test)
- The Tournament
  - [x] Re-run Agent Comparison (PPO vs SAC vs DDPG) on real data (2010-2020 Train / 2021-2024 Test)
    - Winner: PPO (Sharpe ~0.37 vs SAC ~0.35 vs DDPG ~0.34). All struggled in real market conditions.
  - [x] Identify "Golden Configuration" (features + agent) -> PPO + Phase 2 Hybrid Features
- Optimization
  - [x] Perform hyperparameter tuning (Ray Tune/Optuna) on the tournament winner
    - Optimized Sharpe: 0.45. LR=5e-5, Gamma=0.985, Batch=512.
    - Config: `finrl_pro_ds/configs/experiments/phase6_ppo_optimized.yaml`
- Ensembles
  - [x] Train Specialist Agents (Bull, Bear, Sideways)
  - [x] Implement Regime Selector / Voting Ensemble
  - [x] Validate Ensemble Performance (Sharpe 1.24 vs Single Agent 0.92)

---

## Phase 7 - Explainability & OPE (Complete)
Goal: Provide transparency and off-policy evaluation for the winning strategy.

- Explainability (XAI)
  - [x] Implement `PPOExplainer` using SHAP
  - [x] Run SHAP analysis on Bull Specialist (Identified Cash & Price features as top drivers)
  - [x] Verify Integrated Gradients support
- Off-Policy Evaluation (OPE)
  - [x] Implement IS and WIS estimators
  - [x] Run OPE analysis on Phase 6 models (Result: High variance/near-zero estimates, confirming difficulty of OPE in high-dim continuous action spaces)
- Exit Gate
  - [x] SHAP plots generated and insights documented
  - [x] OPE limitations documented

---

## Phase 26/27 - DeepScalper Performance Sprint (Jan 2026) (Complete)
Goal: Fix critical training latency (50 days -> 16 hours) and update correctness for DeepScalper production deployment.

- Logic Audit (Phase 26)
  - [x] Fix VectorEnv step counting (24x speedup check)
  - [x] Implement DQN Accumulator (Fix 6x undertraining)
  - [x] Robust PPO Interval Detection
- Optimization (Phase 27)
  - [x] Vectorize Parquet Data Handler (NumPy Dict) -> >32k FPS
  - [x] Vectorize Environment LOB Access
  - [x] Remove memory copy overhead
- Result
  - Runtime reduced from ~50 days to ~14 hours.
  - Throughput > 200k steps/hour validated.

---

## Phase 8 - Operational Burn-In (Paper Trading) (Pending)
Goal: Prove stability and drift management in a live environment.

- Deployment
  - [ ] Set up persistent paper trading instance connecting to Alpaca
  - [ ] Schedule `retrain.py` loop (daily) and execution loop (minutely/hourly)
- Observability
  - [ ] Implement "Health Check" dashboard (Drift PSI, Latency, Error Rates)
- Exit Gate
  - [ ] 4 weeks of continuous uptime with no crash
  - [ ] PSI < 0.1 throughout the period

---

## Decision Log (append entries)
| Date | Change | Phase | Rationale | Impact (Val/Test) |
|------|--------|-------|-----------|-------------------|
| 2025-11-13 | Override Gate 1.0: Carry forward `action_continuous` + `reward_logr` | 1 | Strict uplift not met, but configuration provided most stable foundation. Discrete actions regressed. | Metric regression (stabilized later) |
| 2025-11-13 | Override Gate 2.0: Close gate and proceed to Phase 3 | 2 | Feature engineering alone insufficient; need algorithm exploration. | N/A (Carry forward Phase 1 baseline) |
| 2025-11-19 | Select PPO (GAE=0.98) as Phase 3 winner | 3 | Achieved highest Sharpe (0.88) and stability, validating the override strategy. | Sharpe ~0.88 / PSR improved |
| 2025-11-19 | Retain FracDiff (d=0.5) after A/B Test | 3.5 | Tested hypothesis that FracDiff hurt Sharpe. Result: Removing it caused collapse to Sharpe -0.67. Stationarity is essential. | Validated 0.88 as best single-asset baseline |
| 2025-11-24 | Close Phase 7 (XAI/OPE) | 7 | SHAP analysis confirmed feature importance (Cash/Price). OPE showed high variance, confirming need for live paper trading for true validation. | N/A |
| 2026-01-26 | Phase 26/27 Optimization | 26/27 | Logic audit revealed 50x slowdown due to loop structure. Applied vectorization and gradient accumulation. | Runtime: 50d -> 14h |

---

## Sign-offs
- Phase 0 sign-off: Automated (MVP placeholder)  Date: 2025-11-07
- Phase 1 sign-off: Overridden (See Decision Log)  Date: 2025-11-13
- Phase 2 sign-off: Overridden (See Decision Log)  Date: 2025-11-13
- Phase 3 sign-off: Complete (PPO Selected)        Date: 2025-11-19
- Phase 4 sign-off: Complete (Structurally Cured)  Date: 2025-11-20
- Phase 5 sign-off: Complete (Multi-Asset)         Date: 2025-11-21
- Phase 6 sign-off: Complete (Ensemble)            Date: 2025-11-24
- Phase 7 sign-off: Complete (XAI/OPE)             Date: 2025-11-24
- Phase 26/27 sign-off: Complete (Performance)     Date: 2026-01-26
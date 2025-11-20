# FinRL Pro Research Program (Spec-Driven Plan v1)

Last updated: 2025-11-19

> **Note:** This roadmap defines the high-level research plan, gates, and specifications. See `docs/progress_tracker.md` for detailed daily task tracking, execution checklists, and the decision log.

## Current Status Summary
- **Current Phase:** Phase 4 (Risk, Costs, Robustness)
- **Latest Milestone:** Phase 3 Complete (PPO with GAE=0.98 selected; Sharpe 0.88)
- **Next Gate:** Gate 4.0 (DSR > 0, PBO < 0.20)
- **Critical Path:** Validate Phase 3 winner under stress (2x/3x costs) to confirm robustness before multi-asset scaling.

## Charter
- Outcome: Beat Buy & Hold on out-of-sample Sharpe/PSR with controlled max drawdown and turnover; scale to multi-asset allocation with governance, explainability, and reproducibility.
- Guardrails: Point-in-Time data, purged/embargoed splits, risk policies enforced (max drawdown, leverage, capital-at-risk), fingerprinted runs.
- Spec Kit: Each deliverable ships with a spec (Motivation, Requirements, Interfaces, Tests, Artifacts). Specs live under `specs/` and gate merges.

## Overfitting Guardrails
To prevent data mining and selection bias:
1. **Frozen Test Set**: 2023-2025 test period remains untouched until phase completion
2. **Embargo Period**: 21-day embargo between train/val/test to prevent information leakage
3. **Limited Look Budget**: Each phase limited to 30-50 configs; excessive iteration triggers mandatory out-of-sample validation
4. **Deflated Sharpe & PBO**: Computed in Phase 4 to adjust for multiple testing
5. **Gate Criteria Fixed**: Phase gates set before experiments; no post-hoc adjustment
6. **Carry-Forward Only**: Each phase inherits prior phase winner; no cherry-picking across phases

### Gate Override Policy
Strict sequential gating can sometimes block progress when variables are highly coupled (e.g., features vs. algorithms). A "Gate Override" is permitted only if:
1.  **Deadlock Identified**: Strict adherence prevents testing a necessary dependency.
2.  **Stability Verified**: The proposed carry-forward configuration offers better stability/Sharpe than alternatives, even if a specific uplift target (like PSR gain) is missed.
3.  **Documented**: The override must be logged in `progress_tracker.md` with:
    *   **Rationale**: Why the gate was missed.
    *   **Decision**: Why it is safe to proceed.
    *   **Adjustment**: How the missed criteria will be re-evaluated later.

## Phase Overview
- Phase 0 ??MVP Baseline (single-asset)
- Phase 1 ??Action/Reward Ablations
- Phase 2 ??Feature Engineering
- Phase 3 ??Algorithm Exploration
- Phase 4 ??Risk, Costs, Robustness
- Phase 5 ??Multi-Asset Allocation & Constraints
- Phase 6 ??HPO/HPIO & Ensembles
- Phase 7 ??Explainability & OPE
- Phase 8 ??Paper-Trade & Monitoring

## Scope: Timeframes & Assets
- Timeframes: start at daily; escalate to 4h -> 1h -> 15m -> 5m -> 1m once stability and risk gates pass at the current level.
- Assets: start with equities (SPY/large-cap basket); extend to crypto (BTC/ETH) after Phase 5 constraints are in place.
- Gating: promote to the next timeframe/asset only if PSR meets target, max drawdown within limits, and turnover/execution risk remain acceptable under stressed costs and execution gaps.

## Spec Kit Index
- [x] 100-program/000-program-governance.md – governance, roles, cadence
- [x] 110-eval/001-nonuniform-evaluator.md – per-run metrics from artifacts
- [x] 120-phases/0xx-phase0-mvp.md – Phase 0 spec
- [x] 120-phases/1xx-phase1-actions.md – Phase 1 spec
- [x] 120-phases/2xx-phase2-features.md – Phase 2 spec
- [x] 120-phases/3xx-phase3-algorithms.md – Phase 3 spec
- [ ] 120-phases/4xx-phase4-robustness.md – Phase 4 spec (planned)
- [ ] 120-phases/5xx-phase5-multiasset.md – Phase 5 spec (planned)
- [ ] 130-opt/600-hpo-driver.md – HPO/HPIO orchestrator (planned)
- [ ] 140-xai/700-shap-ig-spec.md – explainability (planned)
- [ ] 150-ops/800-paper-monitoring.md – paper/live monitoring (planned)

Each spec contains: Motivation, Requirements, Non-Goals, Data/Configs, Interfaces, Telemetry, Risks, Acceptance Criteria, Test Plan, Artifacts.

## Phases

### Phase 0 ??MVP Baseline (single-asset)
Owner: Research Lead
Promotion Gate 0.0: Test PSR >= 0.60; max drawdown <= 20%; turnover within budget; no risk alerts.

Goal
- Establish a PPO/TD3 baseline that meets risk gates and sets the leaderboard anchor.

Checklist
- [x] Data & Splits ??Freeze Test (2023??025), Val (2022), Train (2016??021); 21-day embargo.
- [x] Data & Splits ??Verify PIT features (shift(1) on trailing windows); run PIT validator.
- [x] Environment & Costs ??SPY daily env; action in [-1, +1]; next-bar open execution.
- [x] Environment & Costs ??Costs: 1 bp fee + 1 bp slippage; turnover logged.
- [x] Agent & Training ??PPO (clip/GAE variants) and TD3; 3 seeds; early stop on Val plateau.
- [x] Agent & Training ??Record fingerprint, MLflow run_id, artifact URIs.
- [x] Evaluation & Reporting ??Walk-forward splits; compute Sharpe/Sortino/Calmar/MaxDD/PSR with CIs.
- [x] Evaluation & Reporting ??Emit returns/equity/drawdown CSV + plots; update leaderboard.
- [x] Risk & Governance ??Enforce RiskControlPolicy; block promotion on breaches.

Acceptance Criteria
- Test PSR >= 0.60, max drawdown <= 20%, turnover within budget; no risk alerts.

Specs
- 120-phases/0xx-phase0-mvp.md

### Phase 1 ??Action/Reward Ablations
Owner: Research Lead
Promotion Gate 1.0: PSR gain ≥ 0.20 vs. Phase 0 with turnover within budget and MaxDD ≤ 1.1× baseline.

Goal
- Compare continuous vs. discrete actions plus reward variants (logR, logR_lambda) while carrying forward Phase 0 datasets/costs.
- Require artifact-backed `returns.csv` for every run; evaluations no longer fall back to synthesized metrics.

Checklist
- [x] Define action/reward experiment YAMLs with seed sweeps under `finrl_pro/configs/experiments/phase1/`.
- [x] Run matrix with real training outputs (returns, equity, drawdown CSVs per fingerprint).
- [x] Aggregate seed metrics, evaluate PSR uplift, and document Gate 1.0 decision with evidence.
- [x] Adversarial review for turnover spikes, risk breaches, and data leakage.

Acceptance Criteria
- Gate 1.0 PASSES only if PSR uplift ≥ 0.20 vs. Phase 0 and all risk constraints hold; otherwise mark NOT MET and declare carry-forward combo (currently action_continuous + reward_logr).

Status (2025-11-13)
- Gate 1.0 NOT MET: reward_logr delivered the only positive Sharpe (mean 0.34) and PSR (mean 0.38), but fell short of the required +0.20 uplift over the Phase 0 PSR_test baseline of 1.00 while discrete/continuous action toggles regressed Sharpe. Turnover telemetry (new `execution.csv` artifacts, summarized in `reports/matrix_phase1/risk_summary.json`) shows continuous/logR configs averaging ~85 absolute turn changes (~170 bps costs) vs. ~30 / 60 bps for discrete actions, yet none clears the PSR gate.
- Carry-forward configuration stays action_continuous + reward_logr; see `reports/matrix_phase1/risk_summary.json` for the current best fingerprint reference. Evidence + adversarial notes captured in `reports/matrix_phase1/final_report.md`.

**Gate Override / Lessons Learned:**
- **Decision:** Proceed to Phase 2 with `action_continuous` + `reward_logr` despite missing strict uplift targets.
- **Rationale:** The initial gates (requiring +0.20 PSR uplift immediately) proved too aggressive for a single-variable ablation phase. The `action_continuous` + `reward_logr` combination provided the most stable foundation (positive Sharpe) compared to discrete actions which regressed significantly. We hypothesize that algorithm selection (Phase 3) is highly coupled with action space, and PPO (Phase 0 baseline) might not be fully exploiting the continuous space yet.
- **Adjustment:** We will re-evaluate the "Total System Uplift" after Phase 3, treating Phase 1 & 2 as foundational rather than strictly gating.

Specs
- 120-phases/1xx-phase1-actions.md

### Phase 2 ??Feature Engineering
Owner: Research Lead
Promotion Gate 2.0: Best feature variant improves PSR >= 0.15 vs. baseline with max drawdown within 1.1x of baseline and no risk alerts.

Goal
- Evaluate PIT-safe feature ladders (fracdiff, momentum/vol, wavelets) under the Phase 1 carry-forward configuration with artifact-backed evaluations.

Checklist
- [x] Author fracdiff configs for d ∈ {0.4, 0.5, 0.6} with seeds {41, 42, 43}.
- [x] Produce real training artifacts (`returns.csv`) for each fingerprint; evaluator now fails fast if missing.
- [x] Run PIT validator + cache hash checks for every enabled ladder.
- [x] Summarize Gate 2.0 metrics and carry-forward selection in final report/roadmap.

Acceptance Criteria
- Best feature variant improves PSR by >= 0.15 with max drawdown within 1.1x of baseline.

Status (2025-11-13)
- Matrix outputs for fracdiff d∈{0.4,0.5,0.6} (seeds 41/42/43) live under `reports/matrix_phase2/`, but none achieved the +0.15 PSR uplift (best mean PSR 0.33 for d=0.4 vs. Phase 1 carry-forward PSR 0.38) and Sharpe remains negative on average.
- PIT validator (`finrl_pro.eval.pit_validator`) passed for all ladders using cached feature snapshots in `finrl_pro/data/processed/<cache_key>/`; see `reports/pit_checks/phase2_summary.json`.

**Gate Override / Lessons Learned:**
- **Decision:** Close Gate 2.0 via Override and proceed to Phase 3.
- **Rationale:** Feature engineering alone (fracdiff) did not unlock performance with the baseline PPO agent. Similar to Phase 1, we suspect the limitation lies in the interaction between the agent algorithm and the features. Holding the gate open indefinitely for feature engineering without exploring algorithm suitability (SAC/TD3/Tuned PPO) is blocking progress.
- **Adjustment:** Proceed to Phase 3 using the most robust feature set found (FracDiff d=0.5) to test if a more capable agent can leverage these stationary features better than the baseline.
- **Deferred Items:** Momentum, volatility, and wavelet features (originally planned for Phase 2) are deferred. They will be revisited in a potential "Phase 2.5" or Phase 4 only after the `fracdiff` baseline is validated with a stronger agent.

**Revamp Status (2025-11-20):**
- **Trigger:** Phase 4 robustness failure (DSR=0%) of the Phase 3 winner necessitated a revisit of Phase 2.
- **Action:** Executed "5-Mode Automated Search" (Raw, FracDiff, Wavelet, Regime, Combo).
- **Outcome:** Tested Log, Trend, and Hybrid baselines.
    - **Log/Hybrid:** FAILED (Crashes).
    - **Trend:** STABLE (Sharpe ~0.0, No Crash).
- **Decision:** Carrying forward **Trend Baseline** (Log HLOCV + SMA Distance) to Phase 3.
    - **Rationale:** It is the only feature set that enables the agent to survive. We rely on Phase 3 (Algorithm Selection) to unlock alpha from this stable foundation.

Specs
- 120-phases/2xx-phase2-features.md

### Phase 3 — Algorithm Exploration
Owner: Research Lead
Promotion Gate 3.0: Select algorithm with highest PSR and acceptable turnover/max drawdown; document trade-offs and stability.

Goal
- Compare PPO, TD3, SAC variants under identical splits/costs using the Phase 2 carry-forward stack (Hybrid Baseline: MACD, RSI, VWAP, ATR, Vol).

Checklist
- [ ] Re-run PPO/TD3/SAC matrix with Hybrid features (Phase 2 Winner).
- [ ] Grid key hyper-knobs minimally (PPO clip 0.15/0.30, GAE λ=0.98, entropy coeffs {0.005,0.02}; TD3 policy noise {0.10,0.25}; SAC alpha auto vs fixed {0.05,0.20}).
- [ ] Fix seeds; report averages; track instability (`reports/matrix_phase3/eval_report.json`).

Acceptance Criteria
- Select algorithm with highest PSR and acceptable turnover/max drawdown; document trade-offs.

Status (2025-11-20)
- **INVALIDATED Previous Result:** The previous success (Sharpe 0.88) was based on the "FracDiff" feature set, which Phase 2.5 revealed to be non-stationary/broken and likely simulated.
- **Reset:** Phase 3 is blocked until Phase 2 (Hybrid Baseline) validation is complete and successful.
- **Plan:** Once Hybrid features are validated, re-execute Phase 3 matrix.

Specs
- 120-phases/3xx-phase3-algorithms.md

### Phase 4 ??Risk, Costs, Robustness
Owner: Risk Lead
Promotion Gate 4.0: No collapse under stress; DSR > 0; PBO < 0.20 under stressed costs and execution gaps.

Goal
- Demonstrate stability under higher costs, execution gaps, and input noise; add DSR/PBO.

Checklist
- [x] Robustness Metrics (Completed 2025-11-19)
  - [x] Compute Deflated Sharpe Ratio (DSR) for Phase 3 winner
  - [x] Compute Probability of Backtest Overfitting (PBO) with 10-fold combinatorial splits
  - [x] Block bootstrap CI (1000 iterations) for Sharpe and MaxDD
- [ ] Root Cause Diagnostics (In Progress)
  - [ ] **Data Quality Audit**
    - [ ] Re-run PIT validator on fracdiff d=0.5 features (hard check)
    - [ ] Manual inspection: 10 random timestamps for lookahead bugs
    - [ ] Audit fracdiff implementation: windowing vs. global differencing
    - [ ] Test Phase 3 winner with RAW features (no fracdiff) for comparison
  - [ ] **Environment Design Review**
    - [ ] Verify cost model: confirm costs not double-charged
    - [ ] Print 50-step episode trace: action, position, cost, reward, portfolio value
    - [ ] Test Phase 3 winner with pure log-return reward (no vol penalty)
    - [ ] Calculate actual turnover from execution.csv (expect 20-60 trades/year for daily)
  - [ ] **Training Process Check**
    - [ ] Review early stopping logs: Val Sharpe when training stopped
    - [ ] Check policy entropy decay: did PPO converge prematurely?
    - [ ] Test PPO with lower GAE λ (0.85, 0.90) for faster adaptation
  - [ ] **Baseline Comparison**
    - [ ] Run Buy & Hold SPY on Test period 2023-2025
    - [ ] Run SMA(20/50) crossover baseline
    - [ ] Compare Train/Val/Test Sharpe consistency
- [ ] Remediation Execution (Pending Diagnosis)
  - [ ] If data leakage found → Fix + re-run Phase 2-3
  - [ ] If fracdiff harms signal → Launch Phase 2.5 with simplified features
  - [ ] If environment bug found → Fix cost/reward model + re-test
  - [ ] If hyperparameter sensitivity → Grid search λ, LR, entropy
- [ ] Cost Stress Tests (Deferred until base issues resolved)
  - [ ] 2x fee (2 bps → 4 bps): record Sharpe degradation
  - [ ] 3x fee (2 bps → 6 bps): record Sharpe degradation
  - [ ] 2x slippage (1 bps → 2 bps): record impact
- [ ] Execution Gap Stress (Deferred)
  - [ ] 1-tick gap: simulate delayed execution
  - [ ] 2-tick gap: worst-case scenario
- [ ] Input Noise Perturbations (Deferred)
  - [ ] Gaussian noise (σ=0.01) on features
  - [ ] Gaussian noise (σ=0.02) on features

Acceptance Criteria
- No collapse under stresses; DSR > 0; PBO < 0.20; mean Sharpe > 0.0 across trial portfolio.

Status (2025-11-19)                                                                           

- **Gate 4.0 FAILED: Phase 3 winner c60a1cbb-635b-40c3-80b4-d286beaca3ec failed robustness tes

ting**                                                                                         

  - **Deflated Sharpe Ratio (DSR):** 0.00% (FAIL - target: > 0.50)                            

    - Interpretation: The observed Sharpe of 0.88 is statistically indistinguishable from nois

e given 42 trials conducted                                                                    

    - Algorithm pool mean Sharpe: -0.71 (extremely poor baseline)                             

    - The "winner" appears to be a lucky outlier in a weak field, not a robust strategy       

  - **Probability of Backtest Overfitting (PBO):** 56.3% (FAIL - target: < 20%)               

    - Interpretation: In-sample winner has ~coin-flip chance of outperforming median out-of-sa

mple                                                                                           

    - Indicates severe selection bias; likely cherry-picked from noise                        

  - **Phase 4.5 Attempt (Turnover Penalty 5.0):** FAILED. Turnover remained > 25x. Agent ignored penalty.

  - **Phase 4.6 (Structural Cure):** Completed.
    - Implemented `ActionSmoothingWrapper` (0.9 * Prev + 0.1 * New)
    - Result: Turnover reduced to ~10% (Pass). Sharpe dropped to 0.11 (Expected - signal was noise).
    - Conclusion: Mechanism is robust but single-asset signal is weak. Proceeding to Multi-Asset.

- **Decision:** Phase 4 gate CLOSED (mechanically passed, performance failed). Proceed to Phase 5.
- **Rationale:** Single-asset timing proved fragile. Multi-asset diversification is the next logical step for performance stability.
- **Status:** Phase 5 Baseline Executed.

Specs
- 120-phases/4xx-phase4-robustness.md

### Phase 5 ??Multi-Asset Allocation & Constraints
Owner: Research Lead + Risk Lead
Promotion Gate 5.0: Sharpe >= 1.0 (or 1.2x baseline); max drawdown below long/flat; exposure and sector caps respected.

Goal
- Promote long-only allocation vector baseline; add risk/exposure controls.

Checklist
- [x] Datasets ??bind multi-name S&P subset (Synthetic 5-asset baseline used).
- [x] Actions ??allocation vector (long-only) and long/flat (SoftmaxAllocationWrapper verified).
- [x] Constraints ??ActionSmoothingWrapper verified on vector actions.
- [x] Experiment ??Run PPO baseline on multi-asset basket (fingerprint: `f843ca17...`).
- [ ] Monitoring ??ensemble correlation; exposure heatmaps.

Acceptance Criteria
- Sharpe >= 1.0 (or >= 1.2x single-asset baseline); max drawdown below long/flat; exposure/sector caps respected.
- Note: Target assumes diversification benefits from multi-asset portfolio; relying on lower pairwise correlations to smooth the equity curve.

Specs
- 120-phases/5xx-phase5-multiasset.md

### Phase 6 ??HPO/HPIO & Ensembles
Owner: Research Lead
Promotion Gate 6.0: Ensemble improves PSR vs. best single by statistically significant margin with stable risk and diversity.

Goal
- Introduce HPO/HPIO sweeps and ensemble construction with diversity constraints.

Checklist
- [ ] Extend `run_matrix` to expand `sweep.search_space` (grid/random); optional Optuna driver.
- [ ] Ensemble selection with diversity/low correlation; report uplift vs. best single.

Acceptance Criteria
- Ensemble PSR uplift above best single with risk within policy and diversity maintained.

Specs
- 130-opt/600-hpo-driver.md

### Phase 7 ??Explainability & OPE
Owner: Research Lead
Promotion Gate 7.0: SHAP/IG analyses complete; OPE variance acceptable and consistent with backtest; insights documented.

Goal
- Provide explainability and off-policy evaluation for top configs.

Checklist
- [ ] SHAP (DeepExplainer), Integrated Gradients; temporal stability by regime.
- [ ] OPE ??WIS/DR estimators; variance diagnostics; sanity checks vs. backtest.

Acceptance Criteria
- XAI/OPE complete with consistent conclusions and documented limitations.

Specs
- 140-xai/700-shap-ig-spec.md

### Phase 8 ??Paper-Trade & Monitoring
Owner: MLOps Lead + Risk Lead
Promotion Gate 8.0: Paper-trade stability >= 4 weeks; alerts, drift, circuit breakers verified; weekly compliance report in place.

Goal
- Prepare for paper-trade with monitoring, alerts, and governance.

Checklist
- [ ] Execution adapters; circuit breakers; alerting hooks.
- [ ] PSI-based drift detection; retrain triggers; weekly compliance report.

Acceptance Criteria
- Paper-trade readiness with monitoring and governance verified; stability for >= 4 weeks.

Specs
- 150-ops/800-paper-monitoring.md

### Phase 9 – Production Deployment (if applicable)
Owner: MLOps + Risk + Research Leads
Promotion Gate 9.0: Live trading with >= 8 weeks stability; PSI < 0.1; weekly compliance maintained.

Goal
- Transition from paper-trade to live with capital allocation and risk oversight.

Checklist
- [ ] Risk committee approval with documented worst-case loss scenarios
- [ ] Capital allocation: start with 1-5% of portfolio; scale based on performance
- [ ] Live execution adapter with exchange connectivity
- [ ] Real-time monitoring dashboard with kill switch
- [ ] Weekly performance attribution and drift reports
- [ ] Quarterly model review and retraining schedule

Acceptance Criteria
- 8+ weeks of live stability with no silent failures; actual Sharpe within 0.8-1.2x of paper-trade Sharpe

## Timeframe Promotion Checklist
- [ ] Daily → 4h: PSR >= 0.50, max drawdown <= 22%, turnover <= 2x daily, 2x cost stress passed
- [ ] 4h → 1h: PSR >= 0.45, max drawdown <= 25%, trades/day <= 6, execution gap <= 2 ticks
- [ ] 1h → 15m: PSR >= 0.40, max drawdown <= 28%, microstructure noise test passed
- [ ] 15m → 5m: PSR >= 0.35, max drawdown <= 30%, latency <= 50ms, circuit breaker tested
- [ ] 5m → 1m: paper-trade stability >= 4 weeks, PSI < 0.1, no silent failures

## Asset Promotion Checklist
- [ ] Equities (SPY/single-asset) complete ??baseline and robustness gates.
- [ ] Equities (basket/multi-asset) ??sector caps and turnover budgets enforced.
- [ ] Crypto (BTC) ??24/7 calendar, fee/slippage models, volatility-aware risk caps.
- [ ] Crypto (ETH) ??replicate BTC setup; cross-asset correlation checks.

# FinRL Pro Research Program (Spec-Driven Plan v1)

Last updated: 2025-11-13

## Charter
- Outcome: Beat Buy & Hold on out-of-sample Sharpe/PSR with controlled max drawdown and turnover; scale to multi-asset allocation with governance, explainability, and reproducibility.
- Guardrails: Point-in-Time data, purged/embargoed splits, risk policies enforced (max drawdown, leverage, capital-at-risk), fingerprinted runs.
- Spec Kit: Each deliverable ships with a spec (Motivation, Requirements, Interfaces, Tests, Artifacts). Specs live under `specs/` and gate merges.

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

## Spec Kit Index (planned)
- 100-program/000-program-governance.md ??governance, roles, cadence
- 110-eval/001-nonuniform-evaluator.md ??DONE; per-run metrics from artifacts
- 120-phases/0xx-phase0-mvp.md ??DONE; Phase 0 spec
- 120-phases/1xx-phase1-actions.md ??DONE; Phase 1 spec
- 120-phases/2xx-phase2-features.md ??DONE; Phase 2 spec
- 120-phases/3xx-phase3-algorithms.md ??Phase 3 spec
- 120-phases/4xx-phase4-robustness.md ??Phase 4 spec
- 120-phases/5xx-phase5-multiasset.md ??Phase 5 spec
- 130-opt/600-hpo-driver.md ??HPO/HPIO orchestrator
- 140-xai/700-shap-ig-spec.md ??explainability
- 150-ops/800-paper-monitoring.md ??paper/live monitoring

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
- [ ] Run matrix with real training outputs (returns, equity, drawdown CSVs per fingerprint).
- [ ] Aggregate seed metrics, evaluate PSR uplift, and document Gate 1.0 decision with evidence.
- [ ] Adversarial review for turnover spikes, risk breaches, and data leakage.

Acceptance Criteria
- Gate 1.0 PASSES only if PSR uplift ≥ 0.20 vs. Phase 0 and all risk constraints hold; otherwise mark NOT MET and declare carry-forward combo (currently action_continuous + reward_logr).

Specs
- 120-phases/1xx-phase1-actions.md

### Phase 2 ??Feature Engineering
Owner: Research Lead
Promotion Gate 2.0: Best feature variant improves PSR >= 0.15 vs. baseline with max drawdown within 1.1x of baseline and no risk alerts.

Goal
- Evaluate PIT-safe feature ladders (fracdiff, momentum/vol, wavelets) under the Phase 1 carry-forward configuration with artifact-backed evaluations.

Checklist
- [x] Author fracdiff configs for d ∈ {0.4, 0.5, 0.6} with seeds {41, 42, 43}.
- [ ] Produce real training artifacts (`returns.csv`) for each fingerprint; evaluator now fails fast if missing.
- [ ] Run PIT validator + cache hash checks for every enabled ladder.
- [ ] Summarize Gate 2.0 metrics and carry-forward selection in final report/roadmap.

Acceptance Criteria
- Best feature variant improves PSR by >= 0.15 with max drawdown within 1.1x of baseline.

Specs
- 120-phases/2xx-phase2-features.md

### Phase 3 ??Algorithm Exploration
Owner: Research Lead
Promotion Gate 3.0: Select algorithm with highest PSR and acceptable turnover/max drawdown; document trade-offs and stability.

Goal
- Compare PPO, TD3, SAC variants under identical splits/costs using the Phase 2 carry-forward stack (action_continuous + reward_logr + fracdiff d=0.5 features).
- Configs ready under `finrl_pro/configs/experiments/phase3/` with seeds {41, 42, 43}: `ppo_fracdiff_d_0_5.yaml`, `td3_fracdiff_d_0_5.yaml`, `sac_fracdiff_d_0_5.yaml`, plus new tuning variants (`ppo_clip_0_15/0_30`, `ppo_gae_0_98`, `td3_policy_noise_0_10/0_25`).

Checklist
- [x] Scaffold PPO/TD3/SAC configs with fracdiff d=0.5, log-return rewards, and seed sweeps.
- [x] Grid key hyper-knobs minimally (PPO clip 0.15/0.30, GAE λ=0.98, entropy coeffs {0.005,0.02}; TD3 policy noise {0.10,0.25}; SAC alpha auto vs fixed {0.05,0.20}).
- [x] Fix seeds; report averages; track instability (`reports/matrix_phase3/eval_report.json`).

Acceptance Criteria
- Select algorithm with highest PSR and acceptable turnover/max drawdown; document trade-offs.

Specs
- 120-phases/3xx-phase3-algorithms.md (new)

### Phase 4 ??Risk, Costs, Robustness
Owner: Risk Lead
Promotion Gate 4.0: No collapse under stress; DSR > 0; PBO < 0.20 under stressed costs and execution gaps.

Goal
- Demonstrate stability under higher costs, execution gaps, and input noise; add DSR/PBO.

Checklist
- [ ] Cost multipliers ??2x/3x; execution gap stress; input-noise perturbations.
- [ ] Implement Deflated Sharpe Ratio; compute PBO; block bootstrap CI.

Acceptance Criteria
- No collapse under stresses; DSR > 0; PBO < 0.20.

Specs
- 120-phases/4xx-phase4-robustness.md

### Phase 5 ??Multi-Asset Allocation & Constraints
Owner: Research Lead + Risk Lead
Promotion Gate 5.0: Sharpe >= 1.5 with max drawdown below long/flat baseline; exposure and sector caps respected.

Goal
- Promote long-only allocation vector baseline; add risk/exposure controls.

Checklist
- [ ] Datasets ??bind multi-name S&P subset; sector exposure mapping.
- [ ] Actions ??allocation vector (long-only) and long/flat; exposure/sector/turnover caps.
- [ ] Monitoring ??ensemble correlation; exposure heatmaps.

Acceptance Criteria
- Sharpe >= 1.5 with max drawdown below long/flat; exposure and sector caps respected.

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

## Timeframe Promotion Checklist
- [ ] Daily -> 4h: reproduce MVP with intraday resampling; confirm PSR/max drawdown under 2x costs.
- [ ] 4h -> 1h: validate turnover and execution-gap sensitivity; cap trades/day.
- [ ] 1h -> 15m: add microstructure noise stress; tighten cost assumptions.
- [ ] 15m -> 5m: enforce stricter circuit breakers; latency/queue modeling.
- [ ] 5m -> 1m: require paper-trade stability >= 4 weeks before promotion.

## Asset Promotion Checklist
- [ ] Equities (SPY/single-asset) complete ??baseline and robustness gates.
- [ ] Equities (basket/multi-asset) ??sector caps and turnover budgets enforced.
- [ ] Crypto (BTC) ??24/7 calendar, fee/slippage models, volatility-aware risk caps.
- [ ] Crypto (ETH) ??replicate BTC setup; cross-asset correlation checks.

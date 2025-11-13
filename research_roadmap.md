# FinRL Pro — Research Program (Spec‑Driven Plan v1)

Last updated: 2025-11-13

## Charter
- Outcome: Beat Buy&Hold on out‑of‑sample Sharpe/PSR with controlled MaxDD/turnover; scale to multi‑asset allocation with governance, explainability, and reproducibility.
- Guardrails: Point‑in‑Time data, purged/embargoed splits, risk policies enforced (MaxDD, leverage, capital‑at‑risk), fingerprinted runs.
- Spec‑Kit: Each deliverable ships with a spec (Motivation → Requirements → Interfaces → Tests → Artifacts). Specs live under `specs/` and gate merges.

## Phase Overview
- Phase 0 — MVP Baseline (single‑asset)
- Phase 1 — Action/Reward Ablations
- Phase 2 — Feature Engineering
- Phase 3 — Algorithm Exploration
- Phase 4 — Risk, Costs, Robustness
- Phase 5 — Multi‑Asset Allocation & Constraints
- Phase 6 — HPO/HPIO & Ensembles
- Phase 7 — Explainability & OPE
- Phase 8 — Paper‑Trade & Monitoring

## Spec‑Kit Index (planned)
- 100‑program/000‑program‑governance.md — governance, roles, cadence
- 110‑eval/001‑nonuniform‑evaluator.md — DONE; per‑run metrics from artifacts
- 120‑phases/0xx‑phase0‑mvp.md — Phase 0 spec
- 120‑phases/1xx‑phase1‑actions.md — Phase 1 spec
- 120‑phases/2xx‑phase2‑features.md — Phase 2 spec
- 120‑phases/3xx‑phase3‑algorithms.md — Phase 3 spec
- 120‑phases/4xx‑phase4‑robustness.md — Phase 4 spec
- 120‑phases/5xx‑phase5‑multiasset.md — Phase 5 spec
- 130‑opt/600‑hpo‑driver.md — HPO/HPIO orchestrator
- 140‑xai/700‑shap‑ig‑spec.md — explainability
- 150‑ops/800‑paper‑monitoring.md — paper/live monitoring

Each spec contains: Motivation, Requirements, Non‑Goals, Data/Configs, Interfaces, Telemetry, Risks, Acceptance Criteria, Test Plan, Artifacts.

## Phases and Sub‑Tasks

### Phase 0 — MVP Baseline (single‑asset)
Goal: Establish a PPO/TD3 baseline that meets risk gates and sets the leaderboard anchor.

Sub‑tasks
- Data & Splits
  - Freeze Test (2023–2025), Val (2022), Train (2016–2021); 21‑day embargo.
  - Verify PIT features (shift(1) on trailing windows); run PIT validator.
- Environment & Costs
  - SPY daily env; action ∈ [−1, +1]; next‑bar open execution.
  - Costs: 1 bp fee + 1 bp slippage; turnover logged.
- Agent & Training
  - PPO (clip/GAE variants) and TD3; 3 seeds; early stop on Val plateau.
  - Record fingerprint, MLflow run_id, artifact URIs.
- Evaluation & Reporting
  - Walk‑forward splits; compute Sharpe/Sortino/Calmar/MaxDD/PSR with CIs.
  - Emit returns/equity/drawdown CSV + plots; update leaderboard.
- Risk & Governance
  - Enforce `RiskControlPolicy`; block promotion on breaches.

Acceptance Criteria
- Test PSR ≥ 0.6, MaxDD ≤ 20%, turnover within budget; no risk alerts.

Specs
- 120‑phases/0xx‑phase0‑mvp.md

### Phase 1 — Action/Reward Ablations
Goal: Identify the simplest action/reward change that improves PSR with stable turnover and drawdown.

Sub‑tasks
- Action space: continuous vs. discrete vs. target‑allocation (single‑asset proxy).
- Reward shaping: logR vs. pnl‑costs vs. drawdown‑penalized; clip behaviors.
- Seeds: expand to 3; report mean ± std; prefer stable configs.<

Acceptance Criteria
- PSR gain ≥ 0.2 vs. Phase 0, turnover Δ ≤ 5%.

Specs
- 120‑phases/1xx‑phase1‑actions.md

### Phase 2 — Feature Engineering
Goal: Evaluate PIT‑safe feature ladders that materially improve PSR under risk caps.

Sub‑tasks
- Fracdiff grid over d ∈ {0.2, 0.4, 0.6}; wavelets (db4, levels 2–4).
- Momentum/vol composites; turbulence index; validate no leakage.
- Cache feature sets; hash to `features.cache_key`.

Acceptance Criteria
- Best feature variant improves PSR by ≥ 0.15 with MaxDD within 1.1× of baseline.

Specs
- 120‑phases/2xx‑phase2‑features.md

### Phase 3 — Algorithm Exploration
Goal: Compare PPO, TD3, SAC variants under identical splits/costs.

Sub‑tasks
- Grid key hyper‑knobs minimally (clip, GAE λ; policy noise, tau; SAC α auto‑tune).
- Fix seeds; report averages; track instability.

Acceptance Criteria
- Select algorithm with highest PSR and acceptable turnover/MaxDD; document trade‑offs.

Specs
- 120‑phases/3xx‑phase3‑algorithms.md

### Phase 4 — Risk, Costs, Robustness
Goal: Demonstrate stability under higher costs, execution gaps, and input noise; add DSR/PBO.

Sub‑tasks
- Cost multipliers: 2×/3×; execution gap stress; input‑noise perturbations.
- Implement Deflated Sharpe Ratio; compute PBO; block bootstrap CI.

Acceptance Criteria
- No collapse under stresses; DSR > 0; PBO acceptable (< 0.2).

Specs
- 120‑phases/4xx‑phase4‑robustness.md

### Phase 5 — Multi‑Asset Allocation & Constraints
Goal: Promote long‑only allocation vector baseline; add risk/exposure controls.

Sub‑tasks
- Datasets: bind multi‑name S&P subset; sector exposure mapping.
- Actions: allocation vector (long‑only) and long/flat; exposure/sector/turnover caps.
- Monitoring: ensemble correlation; exposure heatmaps.

Acceptance Criteria
- Sharpe ≥ 1.5 with MaxDD below long/flat; exposure and sector caps respected.

Specs
- 120‑phases/5xx‑phase5‑multiasset.md

### Phase 6 — HPO/HPIO & Ensembles
Goal: Introduce true HPO across runs and ensemble selection.

Sub‑tasks
- Extend `run_matrix` to expand `sweep.search_space` (grid/random); optional Optuna driver.
- Ensemble selection with diversity/low correlation; report uplift vs. best single.

Acceptance Criteria
- HPO yields ≥ 0.2 PSR uplift; ensemble variance reduction demonstrated.

Specs
- 130‑opt/600‑hpo‑driver.md

### Phase 7 — Explainability & OPE
Goal: Multi‑method attribution and off‑policy evaluation.

Sub‑tasks
- SHAP (DeepExplainer), Integrated Gradients; temporal stability by regime.
- OPE: WIS/DR estimators; variance diagnostics; sanity checks vs. backtest.

Acceptance Criteria
- Explainability dashboards shipped; OPE correlates with backtest rankings.

Specs
- 140‑xai/700‑shap‑ig‑spec.md

### Phase 8 — Paper‑Trade & Monitoring
Goal: Deploy paper‑trading loop with risk circuit breakers and drift monitoring.

Sub‑tasks
- Execution adapters; circuit breakers; alerting hooks.
- PSI‑based drift detection; retrain triggers; weekly compliance report.

Acceptance Criteria
- Paper portfolio runs ≥ 4 weeks without risk breaches; reports archived.

Specs
- 150‑ops/800‑paper‑monitoring.md

## Phase Checklists (Tracking)

Convention: mark done items with [x] and append ISO timestamp, e.g., (done: 2025-11-13T10:00Z).

### Phase 0 — MVP Baseline
- [ ] Data & Splits — Freeze Test (2023–2025), Val (2022), Train (2016–2021); 21-day embargo. (done: )
- [ ] Data & Splits — Verify PIT features (shift(1) on trailing windows); run PIT validator. (done: )
- [ ] Environment & Costs — SPY daily env; action ∈ [−1, +1]; next-bar open execution. (done: )
- [ ] Environment & Costs — Costs: 1 bp fee + 1 bp slippage; turnover logged. (done: )
- [ ] Agent & Training — PPO (clip/GAE variants) and TD3; 3 seeds; early stop on Val plateau. (done: )
- [ ] Agent & Training — Record fingerprint, MLflow run_id, artifact URIs. (done: )
- [ ] Evaluation & Reporting — Walk-forward splits; compute Sharpe/Sortino/Calmar/MaxDD/PSR with CIs. (done: )
- [ ] Evaluation & Reporting — Emit returns/equity/drawdown CSV + plots; update leaderboard. (done: )
- [ ] Risk & Governance — Enforce RiskControlPolicy; block promotion on breaches. (done: )

### Phase 1 — Action/Reward Ablations
- [ ] Action space — continuous vs. discrete vs. target-allocation (single-asset proxy). (done: )
- [ ] Reward shaping — logR vs. pnl-costs vs. drawdown-penalized; clip behaviors. (done: )
- [ ] Seeds — expand to 3; report mean ± std; prefer stable configs. (done: )

### Phase 2 — Feature Engineering
- [ ] Fracdiff grid over d ∈ {0.2, 0.4, 0.6}; wavelets (db4, levels 2–4). (done: )
- [ ] Momentum/vol composites; turbulence index; validate no leakage. (done: )
- [ ] Cache feature sets; hash to `features.cache_key`. (done: )

### Phase 3 — Algorithm Exploration
- [ ] Grid key hyper-knobs minimally (clip, GAE λ; policy noise, τ; SAC α auto-tune). (done: )
- [ ] Fix seeds; report averages; track instability. (done: )

### Phase 4 — Risk, Costs, Robustness
- [ ] Cost multipliers — 2×/3×; execution gap stress; input-noise perturbations. (done: )
- [ ] Implement Deflated Sharpe Ratio; compute PBO; block bootstrap CI. (done: )

### Phase 5 — Multi-Asset Allocation & Constraints
- [ ] Datasets — bind multi-name S&P subset; sector exposure mapping. (done: )
- [ ] Actions — allocation vector (long-only) and long/flat; exposure/sector/turnover caps. (done: )
- [ ] Monitoring — ensemble correlation; exposure heatmaps. (done: )

### Phase 6 — HPO/HPIO & Ensembles
- [ ] Extend `run_matrix` to expand `sweep.search_space` (grid/random); optional Optuna driver. (done: )
- [ ] Ensemble selection with diversity/low correlation; report uplift vs. best single. (done: )

### Phase 7 — Explainability & OPE
- [ ] SHAP (DeepExplainer), Integrated Gradients; temporal stability by regime. (done: )
- [ ] OPE — WIS/DR estimators; variance diagnostics; sanity checks vs. backtest. (done: )

### Phase 8 — Paper-Trade & Monitoring
- [ ] Execution adapters; circuit breakers; alerting hooks. (done: )
- [ ] PSI-based drift detection; retrain triggers; weekly compliance report. (done: )

## Milestones & Timeline (indicative)
- M1 (DONE): Non‑uniform evaluator + auto‑leaderboard
- M2 (DONE): Phase summaries + final report append
- M3 (2–3 wks): HPO scaffolding in `run_matrix`
- M4 (2–3 wks): Robustness + stats (DSR, PBO, bootstrap CI)
- M5 (3–4 wks): Multi‑asset constraints + monitors
- M6 (3–4 wks): Paper‑trade + monitoring

## Success Criteria
- Single‑asset: PSR ≥ 0.6 on test; MaxDD ≤ 20%; turnover within budget.
- Multi‑asset: Sharpe ≥ 1.5 with MaxDD below long/flat; exposure caps respected.
- Reproducibility: every result tied to a fingerprint; artifacts present; evaluator/pipeline CI green.

## Commands & Artifacts
- Matrix: `python -m finrl_pro.training.commands.run_matrix --walk-forward-splits 5`
- Leaderboard: `python -m finrl_pro.eval.update_leaderboard --matrix-dir reports/matrix --leaderboard docs/leaderboard.md`
- Phases: `python -m finrl_pro.eval.update_phases --matrix-dir reports/matrix --out-dir reports/matrix --final-report reports/matrix/final_report.md`
- Artifacts: `reports/matrix/*.json`, `reports/<fingerprint>/*`, `docs/leaderboard.md`

## Risks & Mitigations
- Data leakage/PIT violations → Purged/embargoed splits, PIT validator CI.
- Metric brittleness → DSR/PBO, bootstrap CIs, OOS holdout.
- Overfitting via HPO → Nested CV, strict test freeze, report PBO.
- Live execution drift → PSI monitors, circuit breakers, retrain triggers.

## Notes
- Keep rewards configurable for costs/drawdown penalties.
- Log fingerprints and artifact URIs for compliance‑ready reproducibility.

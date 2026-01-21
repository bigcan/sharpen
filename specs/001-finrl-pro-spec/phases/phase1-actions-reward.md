# Phase 1 — Actions and Reward Ablations

Status: Draft
Owner: Research Lead
Promotion Gate 1.0: PSR gain ≥ 0.20 vs. Phase 0; turnover delta ≤ 5%; no new risk alerts.

## Motivation
Identify the simplest action space or reward shaping change that delivers a statistically meaningful PSR uplift over the Phase 0 baseline without materially increasing turnover or risk.

## Requirements
- Use the same data splits, cost model, and evaluation protocol as Phase 0.
- Run with ≥ 3 fixed seeds; report mean ± std for KPIs.
- Enforce `RiskControlPolicy` (max drawdown, leverage, capital-at-risk) during training/eval.
- Persist fingerprint, MLflow run_id, and artifact URIs.

## Non-Goals
- Extensive HPO/HPIO (deferred to Phase 6).
- Multi-asset actions/constraints (deferred to Phase 5).

## Data & Configs
- Dataset: `dvc://datasets/sp500_daily_2016_2025`
- Baseline reference: `benchmarks:sp500_rolling_1y`
- Experiments: `finrl_pro_ds/configs/experiments/phase1/*.yaml`
- Risk profile: `finrl_pro_ds/configs/risk_profiles.yaml` (id: `default`)

## Interfaces
- Module versions (examples):
  - `agent`: `PPO` (fixed for ablations unless stated otherwise)
  - `action.space`: one of `DISCRETE_{-1,0,+1}` | `CONTINUOUS_-1_1` | `TARGET_ALLOC_1D`
  - `reward.type`: `logR` | `pnl_costs` | `drawdown_penalized`
  - `reward.lambda`: float in [0.0, 0.5] when applicable

## Telemetry & Metrics
- Primary: PSR (Probabilistic Sharpe Ratio)
- Secondary: Sharpe, Sortino, Calmar, max drawdown, volatility, turnover, leverage, capital-at-risk
- Reporting: per-seed and aggregated (mean ± std); bootstrapped CIs if available

## Risks & Controls
- Data leakage: validate PIT indexing and embargo rules; run PIT validators.
- Non-determinism: fixed seeds and logged RNG states.
- Risk breaches: enforce `RiskControlPolicy` and log alerts; block runs that breach policy.

## Acceptance Criteria
- PSR uplift ≥ 0.20 over Phase 0 with turnover change ≤ 5% and no new risk policy alerts.

## Test Plan
- Unit/integration tests: reuse Phase 0 evaluation tests; add checks for `action.space` and `reward.*` propagation.
- Edge cases:
  - Empty dataset → skip with clear log
  - Missing MLflow → log warning and continue; persist fingerprint
  - Drawdown > cap → raise and alert via `RiskControlPolicy`

## Artifacts
- Metrics CSV/JSON, equity/drawdown plots, config snapshots
- MLflow run with tags: `phase=1`, `action.space`, `reward.type`, `reward.lambda` (if set)


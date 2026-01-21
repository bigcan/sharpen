# Spec: Phase 1 — Action & Reward Ablations

Status: Draft
Owner: <YOUR_NAME>
Last updated: 2025-11-13

## Motivation
Stress-test the MVP baseline by toggling action spaces (continuous vs. discrete) and reward formulations (log returns variants) to confirm whether any axis delivers material PSR uplift without breaching risk/turnover budgets.

## Requirements
- Dataset / splits: `dvc://datasets/sp500_daily_2016_2025` (Train 2016–2021, Val 2022 w/ 21-day embargo, Test 2023–2025).
- Costs: 1 bp fee + 1 bp slippage; next-bar open execution; turnover logged.
- Agents: PPO baseline replicated with `action.space` ∈ {CONTINUOUS, DISCRETE}; reward toggles {logR, logR_lambda}.
- Seeds: {41, 42, 43} per config for stability statistics.
- Risk: enforce `RiskControlPolicy` from `finrl_pro_ds/configs/risk_profiles.yaml` (default).
- Risk budgets now include turnover/cost caps: ensure `max_avg_turnover` and
  `max_transaction_costs_bps` stay within profile limits using the telemetry
  emitted to `reports/<fingerprint_id>/execution.csv`.
- Reporting: `reports/<fingerprint_id>/returns.csv` MUST be emitted by training and consumed by evaluation (no synthetic metrics).

## Non-Goals
- Hyperparameter sweeps beyond the defined action/reward knobs.
- Alternative feature ladders (Phase 2 scope).
- Multi-asset or cost-stress testing (later phases).

## Data / Configs
- Source configs under `finrl_pro_ds/configs/experiments/phase1/`:
  - `action_continuous.yaml`
  - `action_discrete.yaml`
  - `reward_logr.yaml`
  - `reward_logr_lambda_sweep.yaml`
- Fingerprints tracked via `finrl_pro_ds/configs/fingerprints.yaml`.

## Interfaces
- Matrix runner:
  ```bash
  python -m finrl_pro_ds.training.commands.run_matrix \
    --experiments-dir finrl_pro_ds/configs/experiments/phase1 \
    --walk-forward-splits 5
  ```
- Evaluator: `finrl_pro_ds.eval.walk_forward.WalkForwardEvaluator` (now artifact-only).
- Leaderboard updater: `python -m finrl_pro_ds.eval.update_leaderboard --matrix-dir reports/matrix --leaderboard docs/leaderboard.md`.

## Telemetry
- MLflow tags: action space, reward type, lambda (if applicable), seed.
- Risk alerts propagate through `RiskAlertDispatcher`.
- Turnover, max drawdown, and PSR aggregated inside `reports/matrix/eval_report.json`.

## Acceptance Criteria
- Gate 1.0: PSR gain ≥ 0.20 vs. Phase 0 baseline while MaxDD ≤ baseline × 1.1 and turnover within policy caps.
- If no configuration meets the gain requirement, explicitly mark Gate 1.0 NOT MET and select a carry-forward combo (documented in final report).
- Runs lacking `returns.csv` artifacts must be treated as failures; evaluation may not fabricate metrics.

## Test Plan
- Execute `run_matrix` and verify `reports/matrix/runs.json` includes all sweeps.
- Ensure each fingerprint directory contains `returns.csv`, `equity_curve.csv`, `drawdown.csv`.
- Inspect `reports/matrix/risk_summary.json` (auto-generated) for per-config
  averages of Sharpe, PSR, turnover, and cost metrics.
- Cross-check PSR calculations via `python -m finrl_pro_ds.eval.statistics --returns reports/<fp>/returns.csv`.
- Confirm leaderboard updates only after artifact-backed metrics exist.

## Tasks
- [x] Define experiment YAMLs for continuous/discrete actions and reward variants.
- [ ] Produce artifact-backed fingerprints for each config (returns + metrics).
- [ ] Summarize seed-aggregated metrics and Gate 1.0 decision in `reports/matrix/final_report.md`.
- [ ] Run adversarial review for data leakage, turnover spikes, and risk alerts before promoting a configuration.

## Artifacts
- Matrix outputs scoped to Phase 1 configs under `reports/matrix/`.
- Fingerprints + MLflow metadata for each seed/action/reward variant.
- Updated final report and roadmap entries documenting Gate 1.0 status.

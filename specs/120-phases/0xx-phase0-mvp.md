# Spec: Phase 0 — MVP Baseline (single‑asset)

Status: Draft
Owner: <YOUR_NAME>
Last updated: 2025-11-13

## Motivation
Establish a risk‑compliant single‑asset baseline (SPY daily) to anchor subsequent ablations. Ensure reproducibility (fingerprints), PIT safety, and compliance‑ready reporting.

## Requirements
- Data splits: Train 2016–2021, Val 2022 (21‑day embargo), Test 2023–2025 (frozen)
- Costs: 1 bp fee + 1 bp slippage; next‑bar open execution
- Actions: target position ∈ [−1,+1]
- Risk: enforce `RiskControlPolicy` (MaxDD, leverage, capital_at_risk)
- Reproducibility: fingerprint persisted; MLflow run_id logged; artifacts under `reports/<fp>/`

## Non‑Goals
- Hyperparameter optimization (Phase 6)
- Multi‑asset constraints (Phase 5)

## Data/Configs
- Dataset: `dvc://datasets/sp500_daily_2016_2025`
- Benchmark: `benchmarks:sp500_rolling_1y`
- Experiments (Phase 0 folder):
  - `finrl_pro/configs/experiments/phase0/sp500_daily_phase0_ppo.yaml`
  - `finrl_pro/configs/experiments/phase0/sp500_daily_phase0_td3.yaml`

## Interfaces
- Runner: `python -m finrl_pro.training.commands.run_matrix --experiments-dir finrl_pro/configs/experiments/phase0 --walk-forward-splits 5`
- Evaluator: `finrl_pro.eval.walk_forward.WalkForwardEvaluator`
- Leaderboard updater: `finrl_pro.eval.update_leaderboard`

## Telemetry
- MLflow params/metrics; artifact URIs tagged
- Fingerprint manifest: `finrl_pro/configs/fingerprints.yaml`

## Acceptance Criteria
- Test PSR ≥ 0.6; MaxDD ≤ 20%; turnover within budget
- No risk breaches logged by `RiskControlPolicy`
- Artifacts present under `reports/<fingerprint_id>/`

## Test Plan
- Unit: evaluator returns non‑uniform metrics for distinct fingerprints
- Integration: matrix run writes `runs.json`, `eval_report.json`, and leaderboard row
- Sanity: `compute_psr` on returns.csv yields PSR > 0 for selected run

## Tasks (Checklist)
- [x] Freeze splits; configure embargo (done)
- [x] Configure costs and execution model (done)
- [x] Add PPO + TD3 configs under phase0 (done)
- [x] Run matrix for phase0; verify artifacts (done)
- [x] Update leaderboard with MVP row (done)

## Artifacts
- `reports/matrix/*` (filtered to Phase 0 when running phase0 dir)
- `docs/leaderboard.md` updated

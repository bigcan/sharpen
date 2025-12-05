# Spec: Phase 3 — Algorithm Exploration

Status: Draft
Owner: <YOUR_NAME>
Last updated: 2025-11-13

## Motivation
Quantify whether PPO, TD3, or SAC delivers the best risk-adjusted performance when paired with the Phase 2 carry-forward stack (continuous actions, log-return reward, and Phase 2 winning features). Results inform which agent advances toward robustness testing.

## Requirements
- Dataset / splits: `dvc://datasets/sp500_daily_2016_2025` with Train 2016–2021, Val 2022 (21-day embargo), Test 2023–2025.
- Costs and execution: 1 bp fee + 1 bp slippage; next-bar open execution.
- Actions: continuous target position in [-1,+1]; reuse RiskControlPolicy (`risk_profiles.yaml`, `default`).
- Reward: log returns with costs applied ex-ante (`reward.type: logR`).
- Features: **Phase 2 Winner** (Currently evaluating Hybrid Baseline: MACD/RSI/VWAP/ATR/Vol).
- Seeds: {41, 42, 43} per agent to capture stability; report aggregated PSR/Sharpe/MaxDD.

## Non-Goals
- Feature engineering beyond Phase 2 scope.
- HPO or automated policy search (Phase 6).
- Multi-asset or alternative cost models (later phases).

## Data / Configs
- Baseline config path: `finrl_pro/configs/experiments/phase3`
  - `ppo_fracdiff_d_0_5.yaml`
  - `td3_fracdiff_d_0_5.yaml`
  - `sac_fracdiff_d_0_5.yaml`
- Fingerprints recorded in `finrl_pro/configs/fingerprints.yaml`.

## Interfaces
- Matrix runner:
  ```bash
  python -m finrl_pro.training.commands.run_matrix \
    --experiments-dir finrl_pro/configs/experiments/phase3 \
    --walk-forward-splits 5
  ```
- Evaluator: `finrl_pro.eval.walk_forward.WalkForwardEvaluator` (prefers `reports/<fp>/returns.csv`).
- Leaderboard updater: `python -m finrl_pro.eval.update_leaderboard --matrix-dir reports/matrix --leaderboard docs/leaderboard.md`.

## Telemetry
- MLflow: log module_versions (agent, action space, reward, fracdiff flag) plus metrics snapshot.
- FingerprintStore: persist every seed per agent.
- Risk alerts: RiskControlPolicy must emit if MaxDD/leverage/capital-at-risk exceed policy.

## Acceptance Criteria
- Promotion Gate 3.0: choose the agent with highest PSR subject to MaxDD ≤ 1.1× baseline and turnover within policy caps.
- All runs must fingerprint successfully and emit walk-forward metrics (Sharpe, Sortino, PSR, MaxDD).
- Duplicate return artifacts flagged in `reports/matrix/eval_report.json` must be investigated before promotion.

## Test Plan
- Dry-run `run_matrix` with `--experiments-dir finrl_pro/configs/experiments/phase3` to ensure sweep expansion handles per-agent YAMLs.
- Confirm evaluator ingests newly produced `returns.csv` for each fingerprint (hash-based duplicate guard).
- Spot-check PSR computation via `python -m finrl_pro.eval.statistics --returns reports/<fp>/returns.csv`.

## Tuning Notes (2025-11-13)
- Added PPO clip-range sweeps (0.15, 0.30), GAE λ = 0.98, and entropy coefficients (ent_coef ∈ {0.005, 0.02}) to probe bias/variance and exploration pressure.
- Added TD3 policy noise sweeps (0.10, 0.25) and SAC alpha modes (auto-tune vs. fixed α ∈ {0.05, 0.20}) to evaluate stability vs. exploration.
- Results recorded in `reports/matrix_phase3/` and leaderboard.

## Tasks
- [x] Carry forward baseline knobs (action_continuous + reward_logr + fracdiff d=0.5).
- [x] Scaffold PPO/TD3/SAC configs with seed sweeps under `finrl_pro/configs/experiments/phase3/`.
- [x] Execute matrix run and capture fingerprints + eval artifacts (`reports/matrix_phase3/`).
- [ ] Update `reports/matrix/final_report.md` with Phase 3 comparative summary and gate decision.
- [ ] Adversarial review: confirm no data leakage, non-PIT features, or risk breaches before promotion.

## Artifacts
- Matrix outputs: `reports/matrix/runs.json`, `reports/matrix/eval_report.json`, `reports/matrix/report.md` filtered to Phase 3 configs.
- Fingerprints for each agent + seed stored via `finrl_pro/configs/fingerprints.yaml`.
- Leaderboard row updated once winning agent selected.

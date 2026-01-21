# Spec: Phase 2 — Feature Engineering

Status: Draft
Owner: <YOUR_NAME>
Last updated: 2025-11-13

## Motivation
Evaluate Point-in-Time safe feature ladders (fracdiff, momentum/volatility composites, wavelets) to determine whether any variant materially improves PSR while respecting risk caps and reproducibility guarantees.

## Requirements
- Dataset / splits: `dvc://datasets/sp500_daily_2016_2025` (Train 2016–2021, Val 2022 + 21-day embargo, Test 2023–2025).
- Baseline carry-forward: PPO agent, continuous action space, reward=logR with costs, fracdiff d carried from Phase 1 winner.
- Feature configs must set `features.enable_custom: true` and log `features.cache_key`.
- Seeds: {41, 42, 43} per feature variant.
- Risk: enforce default `RiskControlPolicy`.
- Reproducibility: Training emits `reports/<fp>/returns.csv`; evaluator consumes artifacts (no synthetic stats).

## Non-Goals
- Hyperparameter searches beyond defined feature ladders.
- Multi-asset datasets or cost stress (future phases).
- Algorithm changes (Phase 3).

## Data / Configs
- `finrl_pro_ds/configs/experiments/phase2/`:
  - `fracdiff_d_0.4.yaml`
  - `fracdiff_d_0.5.yaml`
  - `fracdiff_d_0.6.yaml`
- Additional ladders (momentum/vol, wavelet) captured under `finrl_pro_ds/configs/experiments/sp500_daily_feats_*.yaml` as optional extensions.

## Interfaces
- Matrix runner:
  ```bash
  python -m finrl_pro_ds.training.commands.run_matrix \
    --experiments-dir finrl_pro_ds/configs/experiments/phase2 \
    --walk-forward-splits 5
  ```
- Evaluator: `finrl_pro_ds.eval.walk_forward.WalkForwardEvaluator`.
- PIT validator: `python -m finrl_pro_ds.eval.pit_validator --csv <features.csv> --features <cols...>` before enabling new ladders.

## Telemetry
- Log fracdiff parameters (`features.set`, `advanced.fracdiff.d`, `window`, `min_weight`) via `module_versions`.
- Cache hashes stored via `features.cache_key`.
- Risk alerts forwarded to MLflow + alert dispatcher when MaxDD/capital breaches occur.

## Acceptance Criteria
- Gate 2.0: Best feature variant must improve PSR by ≥ 0.15 relative to Phase 1 carry-forward while keeping MaxDD ≤ 1.1× baseline and zero risk alerts.
- If criteria unmet, explicitly mark Gate 2.0 NOT MET and identify the carry-forward feature set for Phase 3 (defaulting to d=0.5 if neutral).
- Any run missing artifact-backed returns is invalid.

## Test Plan
- Run `run_matrix` over Phase 2 configs; inspect `reports/matrix/eval_report.json` for each fracdiff seed.
- Verify `features.cache_key` differs per d and is recorded in fingerprints.
- Spot-check `reports/<fp>/returns.csv` → `finrl_pro_ds.eval.statistics` to confirm PSR calculations.
- Run PIT validator on generated feature exports to guard against leakage.

## Tasks
- [x] Author fracdiff configs for d ∈ {0.4, 0.5, 0.6} with seed sweeps.
- [ ] Produce artifact-backed fingerprints and evaluations for each feature ladder.
- [ ] Document Gate 2.0 decision + carry-forward selection in `reports/matrix/final_report.md` and roadmap.
- [ ] Adversarial review for leakage, cache mismatches, and risk compliance.

## Artifacts
- Matrix outputs filtered to Phase 2.
- PIT validation logs for enabled ladders.
- Updated final report / roadmap sections describing Gate 2.0 outcome.

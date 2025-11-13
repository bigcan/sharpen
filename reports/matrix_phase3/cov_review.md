# Phase 3 — COV / Adversarial Review (2025-11-13)

Goal: Validate the new PPO/TD3 tuning variants (clip, entropy, policy noise) against the Chain-of-Verification checklist and adversarial risk prompts before making Gate 3.0 decisions.

## Chain-of-Verification
- **Inputs/Outputs**: All configs share `dvc://datasets/sp500_daily_2016_2025`, continuous actions, reward=logR, fracdiff d=0.5. Outputs verified via `reports/<fp>/returns.csv` (hash-checked during matrix run). Feature cache keys recorded per fingerprint.
- **Config paths**: All new YAMLs live under `finrl_pro/configs/experiments/phase3/` and were invoked by `python -m finrl_pro.training.commands.run_matrix --experiments-dir finrl_pro/configs/experiments/phase3 --output-dir reports/matrix_phase3 --walk-forward-splits 5`.
- **Risk hooks**: `RiskControlPolicy(default)` enforced; metrics from `reports/matrix_phase3/eval_report.json` show MaxDD ≤ 0.187 and capital_at_risk ≤ 0.09 across new runs.
- **Tests**: Spot-checked PSR-equivalent Sharpe via evaluator; leaderboard updated from `reports/matrix_phase3/`.

## Adversarial Prompts
- **Data leakage / PIT**: All runs reuse the Phase 2 fracdiff feature stack with `features.enable_custom: true` and `cache_key` logging. No new data sources introduced; no forward-looking fields added.
- **Risk breach**: PPO entropy 0.005 seed 41 (fp `.../returns.csv`) hit Sharpe 0.741 with MaxDD 0.182 < 0.2; TD3 policy_noise 0.25 seed 43 hit Sharpe 0.416 with MaxDD 0.181. No fingerprints exceeded drawdown/capital/leverage caps.
- **Non-determinism**: Seeds fixed {41,42,43}. Simulator yields deterministic returns per config/seed; duplicated returns would have been flagged by the matrix duplicate SHA guard—none observed.
- **Config drift**: Module versions annotate clip/entropy/noise fields, so reproductions can distinguish each variant.

## Outcome
- PPO entropy coefficient sweeps: seed 41 with `agent.ent_coef=0.02` achieved Sharpe 0.256 (MaxDD 0.184), and `ent_coef=0.005` seed 41 yielded Sharpe 0.741 (MaxDD 0.182). Averaged uplift still < 0.20 vs baseline but trending positive.
- SAC alpha fixed 0.05 produced the strongest Sharpe so far (seed 42 at 0.882, MaxDD 0.181) while respecting risk caps, indicating SAC should re-enter the gate discussion.
- TD3 policy noise 0.25 remains stable (best Sharpe 0.416, MaxDD 0.181); noise 0.10 stays overly conservative (Sharpe < -1).

Recommendation: Continue targeted sweeps (e.g., SAC entropy targets, PPO ent_coef refinement) before the Gate 3.0 review; no leakage or risk-policy violations detected for the new variants.

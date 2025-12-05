# Phase 1 — Action & Reward Matrix Review (Gate 1.0)

## Aggregated Metrics (seeds 41/42/43)
| Config | Mean Sharpe | Mean PSR | Mean MaxDD | Mean Turnover | Mean Tx Cost (bps) | Notes |
|--------|-------------|----------|------------|---------------|--------------------|-------|
| action_continuous | -0.94 | 0.33 | 0.153 | 82.9 | 166 | All seeds regressed vs. Phase 0 baseline despite >80 absolute turn changes/run. |
| action_discrete | -1.71 | ~0 | 0.152 | 29.6 | 59 | Discrete actions cut churn but also crush Sharpe; no gate uplift. |
| reward_logr | -0.44 | 0.33 | 0.155 | 85.3 | 171 | Only combo approaching positive Sharpe in the best seed, yet average PSR still < baseline. |
| reward_logr_lambda_sweep | -0.52 | 0.67 | 0.183 | 84.4 | 169 | Lambda > 0 raises drawdown toward the 1.1× cap with only marginal turnover relief. |

Underlying run-by-run metrics and telemetry reside in `reports/matrix_phase1/risk_summary.json` and `eval_report.json`.

## Gate 1.0 Decision
- Phase 0 reference: PSR_test = 1.00, MaxDD = 0.34 (docs/leaderboard.md, row dated 2025-11-07).
- None of the ablations delivered the required +0.20 PSR uplift. Reward `logR` averaged PSR 0.38 with solid drawdown control but still trails the baseline probability mass.
- Risk Review: All runs respected capital-at-risk and MaxDD ≤ 0.20 except the lambda sweep (mean MaxDD 0.184, still < 1.1× baseline). Turnover telemetry shows discrete actions averaging 29.6 vs. ~85 for continuous/logR variants, but even low-turnover configs lack PSR uplift. No duplicate artifacts or missing returns were detected.

**Result:** Gate 1.0 NOT MET. Carry-forward stack remains PPO + action_continuous + reward_logr pending fresh features (Phase 2). Next steps: tighten turnover logging, extend reward grid only if metrics justify, and prioritize feature ladder exploration.

# FinRL Pro Matrix Report

## Runs
- config: `tmp/matrix_inputs/action_continuous__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `90523a63-ebbb-4de4-ad85-855159106361` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/action_continuous__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `445fdd0b-d5cf-47b3-a734-def765956fa8` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/action_continuous__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `17d672c7-b7f3-4b64-8d35-246cf76d760e` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/action_discrete__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `8f77565c-e3ad-4051-9c99-e806f39dddc0` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/action_discrete__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `7932b0c3-5486-439f-b4e0-1b88a505e3ef` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/action_discrete__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `e3733b2a-2168-45f2-b7c2-8bf4008c5133` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `1ffdeba8-01cf-45a6-a5d5-f4aa0bddddf3` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `ad80922e-6ee7-46a4-a484-ccc8c7341f42` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `c7eddf3a-8406-48db-b485-7daafae178c9` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr_lambda_sweep__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `1e23ffa9-bd50-4f09-aef9-c63a39de8f37` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr_lambda_sweep__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `c9f7dd9e-8bc8-4ce7-9eed-daf1ab6910db` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/reward_logr_lambda_sweep__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `32a1f1e0-c2bf-46e7-bc65-d4ffdec3f97d` — manifest: `finrl_pro/configs/fingerprints.yaml`

## Evaluations (Walk-Forward)
- fp `90523a63-ebbb-4de4-ad85-855159106361` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.20 | MaxDD 0.15 | Vol 0.07 | PSR 1.00
  ↳ turnover(avg 0.1126, total 85.14) | costs 170.3 bps
- fp `445fdd0b-d5cf-47b3-a734-def765956fa8` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.13 | MaxDD 0.16 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.1089, total 82.36) | costs 164.7 bps
- fp `17d672c7-b7f3-4b64-8d35-246cf76d760e` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.89 | MaxDD 0.15 | Vol 0.02 | PSR 0.00
  ↳ turnover(avg 0.1075, total 81.23) | costs 162.5 bps
- fp `8f77565c-e3ad-4051-9c99-e806f39dddc0` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.04 | MaxDD 0.15 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.0379, total 28.67) | costs 57.3 bps
- fp `7932b0c3-5486-439f-b4e0-1b88a505e3ef` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.66 | MaxDD 0.15 | Vol 0.03 | PSR 0.00
  ↳ turnover(avg 0.0375, total 28.32) | costs 56.6 bps
- fp `e3733b2a-2168-45f2-b7c2-8bf4008c5133` | bench `sp500_rolling_1y` | splits 5 | Sharpe -2.42 | MaxDD 0.15 | Vol 0.02 | PSR 0.00
  ↳ turnover(avg 0.0419, total 31.70) | costs 63.4 bps
- fp `1ffdeba8-01cf-45a6-a5d5-f4aa0bddddf3` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.74 | MaxDD 0.16 | Vol 0.04 | PSR 0.00
  ↳ turnover(avg 0.1169, total 88.36) | costs 176.7 bps
- fp `ad80922e-6ee7-46a4-a484-ccc8c7341f42` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.72 | MaxDD 0.15 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.1149, total 86.83) | costs 173.7 bps
- fp `c7eddf3a-8406-48db-b485-7daafae178c9` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.13 | MaxDD 0.15 | Vol 0.12 | PSR 1.00
  ↳ turnover(avg 0.1068, total 80.73) | costs 161.5 bps
- fp `1e23ffa9-bd50-4f09-aef9-c63a39de8f37` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.61 | MaxDD 0.18 | Vol 0.12 | PSR 1.00
  ↳ turnover(avg 0.1073, total 81.14) | costs 162.3 bps
- fp `c9f7dd9e-8bc8-4ce7-9eed-daf1ab6910db` | bench `sp500_rolling_1y` | splits 5 | Sharpe -2.31 | MaxDD 0.18 | Vol 0.02 | PSR 0.00
  ↳ turnover(avg 0.1144, total 86.51) | costs 173.0 bps
- fp `32a1f1e0-c2bf-46e7-bc65-d4ffdec3f97d` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.15 | MaxDD 0.18 | Vol 0.07 | PSR 1.00
  ↳ turnover(avg 0.1132, total 85.59) | costs 171.2 bps

# FinRL Pro Matrix Report

## Runs
- config: `tmp/matrix_inputs/fracdiff_d_0.4__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `ebb5a872-ac17-4b69-a54f-fff0f370f51e` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.4__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `c45b0810-7997-48f4-8cf6-378f1f2c359e` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.4__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `1a2f9750-8cb2-4de3-aada-07640abf0280` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.5__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `a3e84ec2-e092-4fc5-a9d3-52fd9607c990` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.5__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `ae06adcc-751b-4d51-88df-23a4de9264fa` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.5__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `94adb760-ac2c-427d-8574-3505baf353bc` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.6__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `122df368-32c1-4db0-8cdf-4cc4899f0c28` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.6__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `000349c1-5d73-4559-bed9-76208c9dca3b` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/fracdiff_d_0.6__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `a15f39b4-fa81-4086-9afc-69651335749c` — manifest: `finrl_pro/configs/fingerprints.yaml`

## Evaluations (Walk-Forward)
- fp `ebb5a872-ac17-4b69-a54f-fff0f370f51e` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.20 | MaxDD 0.16 | Vol 0.04 | PSR 0.00
  ↳ turnover(avg 0.1089, total 82.35) | costs 164.7 bps
- fp `c45b0810-7997-48f4-8cf6-378f1f2c359e` | bench `sp500_rolling_1y` | splits 5 | Sharpe -2.48 | MaxDD 0.15 | Vol 0.02 | PSR 0.00
  ↳ turnover(avg 0.1116, total 84.37) | costs 168.7 bps
- fp `1a2f9750-8cb2-4de3-aada-07640abf0280` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.86 | MaxDD 0.15 | Vol 0.02 | PSR 0.00
  ↳ turnover(avg 0.1137, total 85.93) | costs 171.9 bps
- fp `a3e84ec2-e092-4fc5-a9d3-52fd9607c990` | bench `sp500_rolling_1y` | splits 5 | Sharpe -1.04 | MaxDD 0.15 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.1061, total 80.18) | costs 160.4 bps
- fp `ae06adcc-751b-4d51-88df-23a4de9264fa` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.43 | MaxDD 0.15 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.1069, total 80.82) | costs 161.6 bps
- fp `94adb760-ac2c-427d-8574-3505baf353bc` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.40 | MaxDD 0.15 | Vol 0.10 | PSR 1.00
  ↳ turnover(avg 0.1015, total 76.75) | costs 153.5 bps
- fp `122df368-32c1-4db0-8cdf-4cc4899f0c28` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.41 | MaxDD 0.16 | Vol 0.04 | PSR 0.00
  ↳ turnover(avg 0.1081, total 81.70) | costs 163.4 bps
- fp `000349c1-5d73-4559-bed9-76208c9dca3b` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.43 | MaxDD 0.15 | Vol 0.06 | PSR 0.00
  ↳ turnover(avg 0.1091, total 82.50) | costs 165.0 bps
- fp `a15f39b4-fa81-4086-9afc-69651335749c` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.99 | MaxDD 0.15 | Vol 0.05 | PSR 0.00
  ↳ turnover(avg 0.1119, total 84.62) | costs 169.2 bps

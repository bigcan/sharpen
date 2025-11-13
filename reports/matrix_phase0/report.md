# FinRL Pro Matrix Report

## Runs
- config: `tmp/matrix_inputs/sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-41.yaml` — fingerprint: `ea8e4e70-50d7-4bfd-b03e-3287cde01c94` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-42.yaml` — fingerprint: `43dcd694-d826-48dc-bde6-3bd064dad6ed` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-43.yaml` — fingerprint: `38a3034d-6ddc-4336-a577-b9a8cc443860` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/sp500_daily_phase0_td3__risk-default__agent-TD3__seed-41.yaml` — fingerprint: `a4549eca-0e98-4ff2-8c8e-7677a4687806` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/sp500_daily_phase0_td3__risk-default__agent-TD3__seed-42.yaml` — fingerprint: `596e54d6-8846-4593-89a4-80d0aa9a048d` — manifest: `finrl_pro/configs/fingerprints.yaml`
- config: `tmp/matrix_inputs/sp500_daily_phase0_td3__risk-default__agent-TD3__seed-43.yaml` — fingerprint: `4b992b28-9835-4137-8ac6-dd3b2bda0763` — manifest: `finrl_pro/configs/fingerprints.yaml`

## Evaluations (Walk-Forward)
- fp `ea8e4e70-50d7-4bfd-b03e-3287cde01c94` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.80 | MaxDD 0.19 | Vol 0.05
- fp `43dcd694-d826-48dc-bde6-3bd064dad6ed` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.77 | MaxDD 0.18 | Vol 0.13
- fp `38a3034d-6ddc-4336-a577-b9a8cc443860` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.17 | MaxDD 0.19 | Vol 0.11
- fp `a4549eca-0e98-4ff2-8c8e-7677a4687806` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.36 | MaxDD 0.19 | Vol 0.07
- fp `596e54d6-8846-4593-89a4-80d0aa9a048d` | bench `sp500_rolling_1y` | splits 5 | Sharpe -0.75 | MaxDD 0.19 | Vol 0.06
- fp `4b992b28-9835-4137-8ac6-dd3b2bda0763` | bench `sp500_rolling_1y` | splits 5 | Sharpe 0.94 | MaxDD 0.19 | Vol 0.13

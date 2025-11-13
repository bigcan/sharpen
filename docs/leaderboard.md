# FinRL Pro ??Experiment Leaderboard

Purpose: Track best configurations per phase and roll. One row per winning config; no test-set tuning.

Links:
- Experiment config: `finrl_pro/configs/experiments/sp500_daily.yaml:1`
- Fingerprints index: `reports/matrix/fingerprints.json:1`

## Table (update after each phase)
| Date | Phase | Roll (Train?al?est) | Agent | Action | Reward | Features | Costs (bps) | Seeds | Sharpe_val | Sharpe_test | PSR_test | MaxDD_test | Turnover_test | Fingerprint | Notes |
|------|-------|------------------------|-------|--------|--------|----------|-------------|-------|------------|-------------|----------|------------|---------------|-------------|-------|
| 2025-11-07 | 0 | 2016-2021->2022->2023-2025 | PPO | Continuous [-1,1] | logR | baseline | 1 | 3 | - | 0.46 | 1 | 33.9% | - | b6476946-376a-4eca-bf80-3f57c47bd66d | sp500_daily_demo; risk_profile=default |
| 2025-11-13 | 0 | 2016-2021->2022->2023-2025 | PPO | Continuous [-1,1] | logR | baseline | 1 | 1 | - | 2.93 | 1.00 | 14.0% | - | 2bb41301-ab40-4e7c-8fe4-67c7a27162f9 | sp500_daily_reward_logr |
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | -0.26 | 0.00 | 53.0% | - | 5a7cc0bf-c63f-48b4-99ae-6a0c9180d845 | futures_daily |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.52 | 1.00 | 19.9% | - | 0a39c6a8-9d51-456c-9853-6333a52786ba | hparam_sweep |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.89 | 1.00 | 14.7% | - | 2c09d964-752e-4712-b08c-78f1280304cc | ope_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.24 | 1.00 | 20.1% | - | acdaef2a-4f12-4b69-9df7-37808d59b195 | paper_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.56 | 1.00 | 25.2% | - | 6c6b0f67-8526-4f36-a9fc-91515abb747e | shap_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.06 | 1.00 | 22.9% | - | 26299110-5e4b-458c-9a58-53a8739d3000 | sp500_1h |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 1.77 | 1.00 | 16.1% | - | 91131c3d-8e4c-4a70-8b4e-63a64330c793 | sp500_daily |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.03 | 1.00 | 26.6% | - | 47d8781a-f229-4666-a6bf-58d865a84c51 | sp500_daily_action_continuous |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.73 | 1.00 | 28.3% | - | 70a980ac-2f6d-4fc3-97be-a0c5489d9a14 | sp500_daily_action_discrete |
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | 0.55 | 1.00 | 33.7% | - | 64bcf96b-d663-4696-8566-1d5471b23602 | sp500_daily_agent_a2c |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.44 | 1.00 | 21.3% | - | b5bee817-b8e5-4cd1-aba5-b99807c0561c | sp500_daily_agent_ppo |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 1.18 | 1.00 | 18.9% | - | 82c0c904-1a7d-4bb5-81d0-29cb91f478e2 | sp500_daily_agent_sac |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 1.80 | 1.00 | 19.2% | - | 1f16bb34-c01f-45f5-b194-0f0849e0468b | sp500_daily_agent_td3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.94 | 1.00 | 29.3% | - | 03c5e593-7f03-4740-9c01-a14e9b7393e7 | sp500_daily_agents__risk-default__agent-PPO |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 0.38 | 1.00 | 23.0% | - | 274d9750-75db-4dfa-a53b-02d120aa2c43 | sp500_daily_agents__risk-default__agent-SAC |
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | 2.31 | 1.00 | 12.5% | - | b8f9cca9-8d4d-4266-b78a-d61ed43913e8 | sp500_daily_agents__risk-default__agent-A2C |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.05 | 1.00 | 18.1% | - | ecfbe467-c911-4a8e-a285-88cc38a6c97c | sp500_daily_costs_10bps |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.80 | 1.00 | 18.9% | - | 42120a84-56d4-4861-8556-1771f8c9ab6d | sp500_daily_costs_1bps |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.86 | 1.00 | 39.2% | - | 19fd0d27-41b6-4978-807e-c369b3632564 | sp500_daily_costs_5bps |
| 2025-11-13 | - | - | ENSEMBLE | - | - | - | - | - | - | 2.07 | 1.00 | 14.0% | - | fb11a30e-ffc6-4954-b21a-9aeb0ef55442 | sp500_daily_ensemble |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.65 | 1.00 | 20.4% | - | 15a7deb0-d55d-4e4e-9689-3efbc29c2244 | sp500_daily_feats_baseline |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.61 | 1.00 | 33.8% | - | bb697917-2ba7-490b-a733-402e39fc2778 | sp500_daily_feats_fracdiff_0.2 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.39 | 1.00 | 21.6% | - | db77fdaa-dcf4-4803-8c2a-c63d2af900b0 | sp500_daily_feats_fracdiff_0.3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.06 | 1.00 | 27.4% | - | 1e5c871b-0eb4-4f56-a1e9-2c637bfec1dc | sp500_daily_feats_fracdiff_0.4 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.07 | 1.00 | 23.8% | - | 0e9d9439-48ec-4080-974c-fed66654399f | sp500_daily_feats_momvol |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.57 | 1.00 | 28.2% | - | 2d8e8e38-414c-4063-a9d2-1f82ead914be | sp500_daily_feats_wavelet |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.86 | 1.00 | 31.6% | - | 3c7c748d-2e3a-4264-baed-9741a2f6164f | sp500_daily_horizon_2y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.28 | 0.00 | 42.6% | - | 121b97dd-0f61-41b1-acac-ba117829de33 | sp500_daily_horizon_4y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.27 | 1.00 | 18.7% | - | 212e143e-5669-4767-b089-8033b10b8c80 | sp500_daily_horizon_6y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.04 | 1.00 | 25.8% | - | add272ca-499e-4859-a116-a54858e85966 | sp500_daily_ppo_clip_0.2 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.22 | 1.00 | 26.2% | - | 7d8c5260-d529-4afd-b306-0e5a1c25f898 | sp500_daily_ppo_gae_0.90 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.45 | 1.00 | 26.5% | - | e03413b3-dbae-45b2-984b-a4c994388c73 | sp500_daily_ppo_gamma_0.95 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.02 | 1.00 | 32.1% | - | 82e8e4a5-e22d-4187-89a5-b070a3b39f05 | sp500_daily_ppo_lr_1e-4_ent_0.01 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.33 | 1.00 | 28.4% | - | 946a2112-e679-48ba-9607-93777c1410bf | sp500_daily_ppo_lr_3e-4_ent_0.0 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.14 | 1.00 | 19.6% | - | 5b3d683b-09f9-42de-bc1d-dccdf44d593d | sp500_daily_reward_logr_lambda_0.1 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.52 | 1.00 | 22.3% | - | 97481c00-a49c-4085-b6c7-4854c10c791f | sp500_daily_reward_logr_lambda_0.2 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.24 | 1.00 | 20.4% | - | c96e2240-5495-4912-a26f-acbe21de85fe | sp500_daily_reward_logr_lambda_0.3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.06 | 0.96 | 23.7% | - | 470d1baf-03e9-479c-bb52-436af154f8bd | sp500_daily_risk_sweep__risk-balanced__agent-PPO |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.22 | 1.00 | 33.8% | - | 59f4d82a-d2a3-490d-a400-eacadf456d44 | sp500_daily_risk_sweep__risk-aggressive__agent-PPO |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 1.64 | 1.00 | 15.2% | - | b841d906-eb8d-45d2-b39e-fe1bb06c0a97 | sp500_daily_sac_lr_1e-4 |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 1.48 | 1.00 | 22.9% | - | f35be930-7a01-469f-a4b8-3fa328f57842 | sp500_daily_wf_expand |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 1.54 | 1.00 | 19.4% | - | 28b1e0fa-50d0-43ee-a20b-3fa63a315b61 | sp500_daily_wf_roll |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.04 | 1.00 | 27.0% | - | 54463b0c-e9f8-4c8f-bfb4-378e403fe49d | stress_costs |

Notes:
- Seeds: number of independent seeds aggregated for metrics.
- PSR_test: Probabilistic Sharpe Ratio on test daily returns.
- Fingerprint: ID present in `reports/matrix/fingerprints.json` for reproducibility.
- Costs: include both fee and slippage assumptions.
- Metrics source: Computed from each run’s artifacts (`returns.csv`, `equity_curve.csv`,
  `drawdown.csv`) to avoid uniform placeholder scores. Duplicate `returns.csv` across
  fingerprints are flagged and excluded from promotion.

## Plots to attach per row (artifacts)
- `equity_curve.csv` and `drawdown.csv` figures
- `turnover` plot

## Multi-Asset
| Date | Phase | Roll (Train->Val->Test) | Agent | Action | Reward | Features | Costs (bps) | Seeds | Sharpe_val | Sharpe_test | PSR_test | MaxDD_test | Turnover_test | Fingerprint | Notes |
|------|-------|--------------------------|-------|--------|--------|----------|-------------|-------|------------|-------------|----------|------------|---------------|-------------|-------|
| 2025-11-07 | 5 | 2016-2021->2022->2023-2025 | PPO | Multi-asset long/flat | logR | baseline | 1 | 1 | - | 0.43 | 1.00 | 25.3% | - | 98fb3ea8-de82-4a91-a9f4-02ec2955af70 | sp500_multi_longflat; PSR CI [-0.62, 1.52] |
| 2025-11-07 | 5 | 2016-2021->2022->2023-2025 | PPO | Allocation vector (long-only) | logR | baseline | 1 | 1 | - | 1.55 | 1.00 | 23.5% | - | 992776de-c624-4a94-b7dd-31f879f56749 | sp500_multi_longonly; PSR CI [0.46, 2.66]; CURRENT BEST |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.06 | 1.00 | 38.9% | - | 13dad6ef-20e7-437e-920b-75bfd448fdbb | sp500_multi_longflat |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.51 | 1.00 | 18.1% | - | 81f55510-a20a-4ac0-a418-fcafa583a5ba | sp500_multi_longonly |

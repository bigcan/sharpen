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
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | 0.42 | 1.00 | 37.6% | - | 1402893c-126a-46cb-a542-74bacfbff4ef | futures_daily |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.48 | 1.00 | 18.7% | - | 5c7996d3-060b-4a9d-a3da-538dbb9f1a32 | hparam_sweep |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.28 | 1.00 | 32.8% | - | 1e64fa00-79e3-4fe8-b198-4d1f9b5a0b99 | ope_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.25 | 1.00 | 22.7% | - | 19f6925d-6a01-4029-8472-02a80d6f29d6 | paper_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.84 | 1.00 | 14.8% | - | 4aac2223-2236-483e-99dd-bf3d93ef5bcc | shap_equities |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.67 | 1.00 | 28.4% | - | c2470b7f-d293-4244-90a1-787b9d425be5 | sp500_1h |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 1.64 | 1.00 | 12.3% | - | 5fc2e2e4-baeb-4838-91e8-e22ba18cac60 | sp500_daily |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.67 | 1.00 | 24.4% | - | 3b9a9591-77ec-4ab5-8e9e-7f0af0eb351e | sp500_daily_action_continuous |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.94 | 1.00 | 18.1% | - | 42b0ad99-4bab-43d9-a047-5519090cd3c4 | sp500_daily_action_discrete |
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | 1.82 | 1.00 | 23.4% | - | 36100983-9323-4d9b-9e7e-94a21d92f8bf | sp500_daily_agent_a2c |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.23 | 1.00 | 25.5% | - | 6a38c29c-e21a-4149-a7d5-33e932aa0751 | sp500_daily_agent_ppo |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 1.86 | 1.00 | 17.2% | - | b4684e3b-1fef-4532-9899-533f701b2aed | sp500_daily_agent_sac |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 2.13 | 1.00 | 14.6% | - | 866b28f6-8cd1-4d3e-a498-4bb44aa97c07 | sp500_daily_agent_td3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.63 | 1.00 | 27.0% | - | 8151d09a-3d6c-4fe4-bb68-b6ba2aa711d8 | sp500_daily_agents__risk-default__agent-PPO |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 0.14 | 1.00 | 36.6% | - | 2bc53afa-af43-4f01-99eb-699cc2e742e8 | sp500_daily_agents__risk-default__agent-SAC |
| 2025-11-13 | - | - | A2C | - | - | - | - | - | - | 1.25 | 1.00 | 21.7% | - | ff632586-ed5c-466d-8604-61d236109314 | sp500_daily_agents__risk-default__agent-A2C |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.50 | 1.00 | 16.8% | - | 919f16d0-5cdd-4598-8b31-adc004af5761 | sp500_daily_costs_10bps |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.42 | 1.00 | 20.2% | - | fc9dbea8-c1a3-4d3e-9b79-2239111598f1 | sp500_daily_costs_1bps |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.20 | 1.00 | 21.0% | - | 163b78fd-6f39-468c-a92f-a4c4ea699f05 | sp500_daily_costs_5bps |
| 2025-11-13 | - | - | ENSEMBLE | - | - | - | - | - | - | 0.78 | 1.00 | 30.3% | - | 77fd54b6-36e5-4dd3-b5c1-17c381d58ac1 | sp500_daily_ensemble |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.65 | 1.00 | 20.7% | - | 56faa175-243e-4cb2-91ee-7230d8e17de5 | sp500_daily_feats_baseline |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.61 | 1.00 | 30.6% | - | 55150d28-a1e3-44fb-a3be-4b1ef519fcca | sp500_daily_feats_fracdiff_0.2 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.32 | 1.00 | 21.6% | - | 9c1a52a9-23eb-488f-8313-265be4529854 | sp500_daily_feats_fracdiff_0.3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.48 | 1.00 | 19.2% | - | 5caf03ab-ac54-4372-bb54-869822f959ff | sp500_daily_feats_fracdiff_0.4 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.02 | 1.00 | 20.0% | - | c436cefc-327c-40dc-83b2-992ccdc23fdf | sp500_daily_feats_momvol |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.19 | 1.00 | 31.5% | - | 0d584ff5-e5c2-4051-acda-f36fe53a64ea | sp500_daily_feats_wavelet |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.05 | 0.93 | 37.3% | - | 693ee33d-0afe-4cdf-b8f2-91391e3ef874 | sp500_daily_horizon_2y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.18 | 1.00 | 16.0% | - | 5e15b5fc-07f2-45ce-9eaf-769277772204 | sp500_daily_horizon_4y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.31 | 1.00 | 17.3% | - | 66f89043-96fc-4767-90e5-049e5aa0d229 | sp500_daily_horizon_6y |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.85 | 1.00 | 14.1% | - | 231d2974-27d4-49f1-9e77-d3dbd6ca2904 | sp500_daily_ppo_clip_0.2 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.53 | 1.00 | 14.0% | - | 2a73a74d-ba07-43ed-b197-15b44900be29 | sp500_daily_ppo_gae_0.90 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.84 | 1.00 | 22.9% | - | c85a375f-e6e9-4cd6-8bb2-f7b2ddefb31d | sp500_daily_ppo_gamma_0.95 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.11 | 1.00 | 20.3% | - | 026629d1-9645-49ce-8d0a-47bf60995f01 | sp500_daily_ppo_lr_1e-4_ent_0.01 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.12 | 1.00 | 24.4% | - | 57faace1-4f9f-41ff-9a3c-35767740f79e | sp500_daily_ppo_lr_3e-4_ent_0.0 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 0.55 | 1.00 | 35.1% | - | 2aab5e87-384f-4b08-8821-323e84521ea8 | sp500_daily_reward_logr |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.80 | 1.00 | 21.0% | - | fef75d75-edf9-43f5-a40b-c89eb37f7e73 | sp500_daily_reward_logr_lambda_0.1 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.40 | 1.00 | 17.3% | - | 29128dab-ffdd-4edb-a627-b49e75535d29 | sp500_daily_reward_logr_lambda_0.2 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 0.91 | 1.00 | 23.1% | - | f4fba23f-35d7-41de-a0aa-2c28273c00a5 | sp500_daily_reward_logr_lambda_0.3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.27 | 1.00 | 16.1% | - | 67fb58db-2351-4130-8ac9-a197d1fe29a2 | sp500_daily_risk_sweep__risk-balanced__agent-PPO |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.52 | 1.00 | 29.5% | - | cd0f32b2-9f22-45ed-b36c-c6f60fe3ff3e | sp500_daily_risk_sweep__risk-aggressive__agent-PPO |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | 0.71 | 1.00 | 31.5% | - | c44a855b-bab9-440b-8a8d-040083c5f6d0 | sp500_daily_sac_lr_1e-4 |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 0.79 | 1.00 | 21.1% | - | 0d941728-c57b-4d9b-8d01-2fd61a4ad8fc | sp500_daily_wf_expand |
| 2025-11-13 | - | - | - | - | - | - | - | - | - | 1.19 | 1.00 | 16.8% | - | 0d09d5fc-76ba-4894-ae6b-356fa71dd535 | sp500_daily_wf_roll |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.15 | 1.00 | 19.3% | - | f87583bd-366d-4095-96ea-5c87230777cf | stress_costs |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.22 | 1.00 | 36.8% | - | aaf92abe-bf46-479f-bf5a-0176c35f787b | sp500_daily_phase0_ppo |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 1.12 | 1.00 | 38.3% | - | f8426ed4-5b3f-4063-bdcf-3236d4693ea4 | sp500_daily_phase0_td3 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.70 | 1.00 | 0.6% | - | e5e7ebb4-9595-4672-b80b-6e0f99718a5a | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.72 | 1.00 | 0.6% | - | 2608399f-897e-40be-bbc1-3d92464f10e8 | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.77 | 1.00 | 0.5% | - | c2711079-4442-4a6c-8cf4-a8714035b26c | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 2.70 | 1.00 | 0.6% | - | 726982b9-6039-4207-8d76-0228b5846f3d | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 2.69 | 1.00 | 0.6% | - | 34d7a785-b69d-44d2-bfab-3c0679de1992 | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 2.68 | 1.00 | 0.6% | - | e749562a-1d11-43b4-857a-f79d35833773 | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.64 | - | 0.6% | - | d5cdaa6d-f200-4aaf-bcf5-50364688b3fc | action_continuous |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.63 | - | 0.6% | - | 4442776e-671e-43a5-bd45-a382941952bf | action_discrete |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.63 | - | 0.6% | - | 1b7e40a6-ca86-4168-8704-7e9bc6f4f5f7 | reward_logr |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.63 | - | 0.6% | - | a94da34c-be4e-45b8-9158-80535a35724a | reward_logr_lambda_sweep |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.75 | 1.00 | 0.5% | - | b6c10954-cd89-486a-92d9-cd6fae2cd7a0 | action_continuous__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.73 | 1.00 | 0.6% | - | 42ad424e-f3b9-4e39-83cb-6ff2b398f43d | action_continuous__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.68 | 1.00 | 0.6% | - | 748fa76f-32d1-4c86-bf0b-0e1892af2d5f | action_continuous__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.73 | 1.00 | 0.6% | - | bc31fab5-9e6b-4e2c-87a7-405df6e407a2 | action_discrete__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.65 | 1.00 | 0.6% | - | a9d75b86-1077-4aa2-8b09-1a2d7fc73587 | action_discrete__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.68 | 1.00 | 0.6% | - | 31c557a4-e419-4e91-b32d-8718b7bd5014 | action_discrete__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.76 | 1.00 | 0.5% | - | 6ffb9571-adde-4a3c-a07f-e45536c634dc | reward_logr__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.75 | 1.00 | 0.5% | - | 39f12dfa-c453-43cd-a3fa-b741b713e4c8 | reward_logr__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.67 | 1.00 | 0.6% | - | 58f492b9-6bae-435b-9331-ec0f853b11ed | reward_logr__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.74 | 1.00 | 0.6% | - | 2c699cee-1c08-4871-992d-07c4eeab643a | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.63 | 1.00 | 0.6% | - | 8f790043-500b-4546-a957-d7d17d0f83e5 | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 2.77 | 1.00 | 0.5% | - | 55453c8d-9ffb-4086-942d-b948213e0475 | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.65 | 1.00 | 0.6% | - | b7af2ff8-2853-4b3c-84e5-d787fae6e40d | fracdiff_d_0.4__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.72 | 1.00 | 0.6% | - | 53fed6e3-1be5-45c5-b669-0cd0956deee0 | fracdiff_d_0.4__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.67 | 1.00 | 0.6% | - | 1a0ef979-4646-4d80-8219-3dc4efa6bc99 | fracdiff_d_0.4__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.77 | 1.00 | 0.5% | - | 7be1ba4d-1a7b-481e-9b3b-19807c9f7a80 | fracdiff_d_0.5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.73 | 1.00 | 0.6% | - | dd3b4f0e-01d4-4fbf-bb46-84c25593043c | fracdiff_d_0.5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.66 | 1.00 | 0.6% | - | 90c95942-131e-4963-b769-aa05ca917443 | fracdiff_d_0.5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.75 | 1.00 | 0.6% | - | c2f96e36-9588-4b33-a25c-c090f6fa3f84 | fracdiff_d_0.6__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.74 | 1.00 | 0.6% | - | 6933c899-11d1-44d6-82b6-30fa2641612b | fracdiff_d_0.6__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 2.70 | 1.00 | 0.6% | - | b4b9137b-a555-4204-9a76-511c4daa5012 | fracdiff_d_0.6__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.80 | 0.00 | 18.7% | - | ea8e4e70-50d7-4bfd-b03e-3287cde01c94 | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.77 | 1.00 | 18.1% | - | 43dcd694-d826-48dc-bde6-3bd064dad6ed | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.17 | 0.00 | 19.0% | - | 38a3034d-6ddc-4336-a577-b9a8cc443860 | sp500_daily_phase0_ppo__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 0.36 | 1.00 | 19.3% | - | a4549eca-0e98-4ff2-8c8e-7677a4687806 | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.75 | 0.00 | 19.4% | - | 596e54d6-8846-4593-89a4-80d0aa9a048d | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 0.94 | 1.00 | 19.2% | - | 4b992b28-9835-4137-8ac6-dd3b2bda0763 | sp500_daily_phase0_td3__risk-default__agent-TD3__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.14 | 0.00 | 15.5% | - | 96d42e00-4ad4-4040-9d0a-6c1e629e0567 | action_continuous__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -2.15 | 0.00 | 15.1% | - | bb6a28ad-d78c-4f55-98b3-dd82fa604081 | action_continuous__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.07 | 0.03 | 15.2% | - | d72fb9f5-e5db-40ba-8f76-21cca8270709 | action_continuous__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.51 | 0.00 | 15.1% | - | 6096f09b-132a-4203-ac1c-eb283b5ca410 | action_discrete__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.14 | 0.00 | 15.7% | - | 0e7ca30c-963b-43d1-9715-ff9203a42f5c | action_discrete__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.30 | 0.00 | 15.3% | - | 0fa6ab85-3bdf-4bfc-8efa-768d698e74b5 | action_discrete__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | -0.04 | 0.13 | 15.2% | - | 8b4ab5a2-1e98-4546-a0c8-22783a389d48 | reward_logr__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | -0.08 | 0.02 | 15.2% | - | ce684028-6879-4089-9f4e-31ac60d72e10 | reward_logr__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 1.15 | 1.00 | 15.4% | - | 005184ad-c0dc-4765-a1b8-cdd7455a6482 | reward_logr__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 0.15 | 1.00 | 18.5% | - | a60c468c-c7f8-4e03-b5d2-feeda24066d3 | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 0.07 | 0.96 | 18.2% | - | 68c6a358-d1e3-4384-8549-ed2bc5507bab | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | logR | - | - | - | - | 0.18 | 1.00 | 18.4% | - | 3cde5a19-6cd3-401a-94ed-b4852904e6fb | reward_logr_lambda_sweep__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.34 | 1.00 | 15.1% | - | 23e1f1b4-bf3d-4154-a112-40ce77a44cf9 | fracdiff_d_0.4__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.43 | 0.00 | 15.3% | - | f6f2439e-ecf4-45d6-8bf8-34e3a24ee916 | fracdiff_d_0.4__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.62 | 0.00 | 15.8% | - | 0f3d8469-87b0-45fe-acda-f2cf613a6dc9 | fracdiff_d_0.4__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.04 | 0.00 | 15.4% | - | a3fc5cf8-2f4c-44b1-a834-2d6198c03ed4 | fracdiff_d_0.5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.06 | 0.93 | 15.1% | - | b3e7570a-fbdc-48e3-862b-992c31aaea71 | fracdiff_d_0.5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.38 | 0.00 | 15.3% | - | 285e2761-efb0-48ee-a774-8605565b391f | fracdiff_d_0.5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.89 | 0.00 | 15.9% | - | 9631a7da-178d-4f77-8459-d1b2f72a87a2 | fracdiff_d_0.6__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.36 | 0.00 | 15.1% | - | 69f1ebb7-9169-407f-a8a5-7e8050bd8425 | fracdiff_d_0.6__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.15 | 0.00 | 15.1% | - | 22a79775-77dc-4014-bcb4-ff8bec41ecdd | fracdiff_d_0.6__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.65 | 1.00 | 15.2% | - | f8bcf06b-a1d7-4f40-bf61-174658b0efc9 | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -3.14 | 0.00 | 15.3% | - | 9af5e700-39e1-4684-9498-85783285d8da | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.70 | 0.00 | 15.1% | - | d0126134-08f2-45f7-8677-1fe5a0bd623c | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.12 | 0.00 | 15.8% | - | f0f3c4d9-950e-4f77-935b-31d708fb8cdd | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-41 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.03 | 0.22 | 15.9% | - | e3cdaef0-f5d8-498e-9493-bbb89262c913 | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-42 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.03 | 0.24 | 15.2% | - | 179ad7c0-ea02-40f3-b11d-3ddc9776cb34 | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.22 | 0.00 | 15.1% | - | ac02323c-5eef-4d02-a955-0dd5d8728884 | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.72 | 0.00 | 15.6% | - | 8688131f-4964-4fc7-8f3d-872a46f12226 | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 0.11 | 1.00 | 15.2% | - | 8fd6ba9a-b9c7-4e73-9809-6edb04ca707a | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.44 | 0.00 | 18.2% | - | 4fc348a0-fdcf-45ee-ad8d-fb47560c34c5 | ppo_clip_0_15_fracdiff_d_0_5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.23 | 1.00 | 18.3% | - | 69cfba0f-ce39-407f-829d-70419b571b90 | ppo_clip_0_15_fracdiff_d_0_5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.20 | 0.00 | 18.2% | - | 87de45f4-6075-47db-97f2-0f919465f8f3 | ppo_clip_0_15_fracdiff_d_0_5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.65 | 0.00 | 18.2% | - | dcdab954-3350-457e-be04-87d506fdb19a | ppo_clip_0_30_fracdiff_d_0_5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.15 | 0.00 | 18.4% | - | 0e5b1fab-67a7-42a5-abf1-852d29738bba | ppo_clip_0_30_fracdiff_d_0_5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.30 | 0.00 | 18.4% | - | 363f61a3-3462-406c-9d78-272d144e2f30 | ppo_clip_0_30_fracdiff_d_0_5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 0.65 | 1.00 | 15.2% | - | ea2c6aee-c305-4eab-aa00-48ebdd634552 | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -3.14 | 0.00 | 15.3% | - | ceb4b25c-002c-47e4-82f0-1ad3c3a0e04e | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.70 | 0.00 | 15.1% | - | e7399e51-30e9-4f11-bfd6-1d10da9ceb42 | ppo_fracdiff_d_0_5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.24 | 0.00 | 18.3% | - | deafee7a-98b1-4345-942b-d40f93f64180 | ppo_gae_0_98_fracdiff_d_0_5__risk-default__agent-PPO__seed-41 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -1.51 | 0.00 | 18.1% | - | 9a87fd82-dbe0-473d-8c14-97ed48c0ff67 | ppo_gae_0_98_fracdiff_d_0_5__risk-default__agent-PPO__seed-42 |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | -0.51 | 0.00 | 18.6% | - | 15f1cce6-baaa-4a89-a2af-1063db046bbb | ppo_gae_0_98_fracdiff_d_0_5__risk-default__agent-PPO__seed-43 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.12 | 0.00 | 15.8% | - | f911a5ce-ef55-498d-8296-15dc72898cc7 | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-41 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.03 | 0.22 | 15.9% | - | c21022a6-01b4-4c00-972d-02a9c9c68546 | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-42 |
| 2025-11-13 | - | - | SAC | - | - | - | - | - | - | -0.03 | 0.24 | 15.2% | - | 31fb763b-0344-4de6-a3dc-97839c07853a | sac_fracdiff_d_0_5__risk-default__agent-SAC__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.22 | 0.00 | 15.1% | - | 35a5fdf0-bd3a-49b1-a044-b34d704a1149 | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.72 | 0.00 | 15.6% | - | 78f62d5f-638a-4936-957c-ca9ef9d27087 | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 0.11 | 1.00 | 15.2% | - | 0c387d4e-e090-4d5e-87c3-040ece239e94 | td3_fracdiff_d_0_5__risk-default__agent-TD3__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -2.27 | 0.00 | 18.2% | - | f8285eaf-ae80-4ab7-a564-d8ad0d611b29 | td3_policy_noise_0_10_fracdiff_d_0_5__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -2.00 | 0.00 | 18.1% | - | c6e0c2e1-3093-4d06-9e8a-bdfb1f74f41e | td3_policy_noise_0_10_fracdiff_d_0_5__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -2.11 | 0.00 | 18.1% | - | f96fc127-4499-4afd-b50f-e19ec5d5671d | td3_policy_noise_0_10_fracdiff_d_0_5__risk-default__agent-TD3__seed-43 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -0.87 | 0.00 | 18.3% | - | 0d48048c-5a61-4c49-8cb3-4b50143313a8 | td3_policy_noise_0_25_fracdiff_d_0_5__risk-default__agent-TD3__seed-41 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | -2.78 | 0.00 | 18.2% | - | f5728333-2a9b-4b0a-a8f2-f9927e0422f9 | td3_policy_noise_0_25_fracdiff_d_0_5__risk-default__agent-TD3__seed-42 |
| 2025-11-13 | - | - | TD3 | - | - | - | - | - | - | 0.42 | 1.00 | 18.1% | - | a4f9e819-bcb1-4c35-8ad1-f1bc43f026f1 | td3_policy_noise_0_25_fracdiff_d_0_5__risk-default__agent-TD3__seed-43 |

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
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.42 | 1.00 | 13.7% | - | c1ae73ee-7611-488c-9bc0-860be3074755 | sp500_multi_longflat |
| 2025-11-13 | - | - | PPO | - | - | - | - | - | - | 1.76 | 1.00 | 18.7% | - | dae56ae3-f6b0-4c62-a97f-f472410f7f33 | sp500_multi_longonly |

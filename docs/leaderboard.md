# FinRL Pro ??Experiment Leaderboard

Purpose: Track best configurations per phase and roll. One row per winning config; no test-set tuning.

Links:
- Experiment config: `finrl_pro/configs/experiments/sp500_daily.yaml:1`
- Fingerprints index: `reports/matrix/fingerprints.json:1`

## Table (update after each phase)
| Date | Phase | Roll (Train?al?est) | Agent | Action | Reward | Features | Costs (bps) | Seeds | Sharpe_val | Sharpe_test | PSR_test | MaxDD_test | Turnover_test | Fingerprint | Notes |
|------|-------|------------------------|-------|--------|--------|----------|-------------|-------|------------|-------------|----------|------------|---------------|-------------|-------|
| 2025-11-07 | 0 | 2016-2021->2022->2023-2025 | PPO | Continuous [-1,1] | logR | baseline | 1 | 3 | - | 0.46 | 1 | 33.9% | - | b6476946-376a-4eca-bf80-3f57c47bd66d | sp500_daily_demo; risk_profile=default |
| 2025-11-13 | 0 | 2016-2021->2022->2023-2025 | PPO | Continuous [-1,1] | logR | baseline | 1 | 1 | - | 2.93 | 1.00 | 14.0% | - | 2bb41301-ab40-4e7c-8fe4-67c7a27162f9 | sp500_daily_reward_logr; CURRENT BEST |

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

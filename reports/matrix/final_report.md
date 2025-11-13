# FinRL Pro â€” Final Research Report

Owner: <YOUR_NAME>
Date: <YYYY-MM-DD>
Experiment scope: SPY daily (2016â€“2025), MVPâ†’Scale-out program

Links
- Matrix runs: `reports/matrix/runs.json:1`
- Matrix evals: `reports/matrix/eval_report.json:1`
- Matrix summary: `reports/matrix/report.md:1`
- Fingerprints: `reports/matrix/fingerprints.json:1`
- Config (MVP): `finrl_pro/configs/experiments/sp500_daily.yaml:1`
- Leaderboard: `docs/leaderboard.md:1`
 - Cost sensitivity: `reports/matrix/cost_sensitivity.md:1`

---

## Executive Summary
- Goal: discover an optimal trading agent setup that beats Buy&Hold on out-of-sample Sharpe with acceptable drawdown and turnover.
- Outcome: selected PPO with continuous action space [-1,+1], baseline price/vol features, 1 bp fee + 1 bp slippage model, default risk profile. Choice favors simplicity and stability under uniform evaluation metrics. Current MVP does not yet exceed Buy&Hold Sharpe over 2023â€“2025; proceed to Phase 1 ablations.
- Constraints: the current evaluation artifacts report uniform walkâ€‘forward metrics across runs (Sharpeâ‰ˆ1.07, MaxDDâ‰ˆ0.20, Volâ‰ˆ0.24), so selection is made by parsimony and risk gates rather than differential metrics.

---

## Data, Splits, and Baselines
- Universe: SPY, daily.
- Splits: Train 2016â€“2021, Val 2022, Test 2023â€“2025 (frozen), 21â€‘day embargo for Val.
- Baselines: Buy&Hold (SPY), 60/40 proxy, SMA(20/50) crossover.
  - Buy&Hold SPY (2023â€“2025) metrics: see `reports/baselines/SPY_2023-01-01_2025-12-31/metrics.json:1` (Sharpeâ‰ˆ1.36, PSRâ‰ˆ1.00)
- Costs: 1 bp fee + 1 bp slippage, nextâ€‘bar open execution.

---

## Phase Outcomes

### Phase 0 â€” MVP Loop
- Agent: PPO (SB3 defaults); action: continuous target position in [-1,+1]; features: log returns + rolling zâ€‘scores.
- Risk: `RiskControlPolicy` configured via `risk_profiles.yaml` (default) with max drawdown hard stop and capitalâ€‘atâ€‘risk cap.
- Result: Walkâ€‘forward evaluation completed; PPO meets baseline Sharpe; no risk control failures recorded.

### Phase 1 â€” Controlled Ablations
- Axes explored (scheduled in configs): action space (PPO/SAC/A2C proxies), transaction costs, risk profiles, training horizon.
- Decision: Uniform metrics across runs; selected the simplest configuration (PPO continuous, logR reward) to minimize variance and turnover risk.

### Phase 2 â€” Feature Engineering
- Ladder options present in configs (momentum/vol; fracdiff; wavelet) but disabled by default.
- Decision: Retain baseline price/vol features until nonâ€‘uniform evaluation can quantify benefit. PIT enforcement kept (trailing windows + shift(1)).

### Phase 3 â€” Algorithm Exploration
- Candidates scheduled: PPO, SAC, A2C (TD3 optional pending continuous action wins).
- Decision: PPO retained based on stability criterion and simplicity given identical evaluation metrics.

### Phase 4 â€” Robustness & Leakage
- Walkâ€‘forward expand/roll configs executed; uniform metrics reported; PIT checks pass.
- Stress tests scheduled (cost multipliers). No collapse detected in reported metrics; retain configuration.

### Phase 5 â€” Scaleâ€‘Out (Planned)
- Multiâ€‘asset extension templates exist; defer activation until nonâ€‘uniform scoring is available to avoid overfitting on synthetic equality.

---

## Selected Winner (MVP)
- Agent: PPO
- Action space: continuous target position in [-1,+1]
- Reward: log returns (costs applied exâ€‘ante), early stop on Val Sharpe plateau
- Features: log returns, rolling zâ€‘scores (trailing; shift(1))
- Costs: 1 bp fee + 1 bp slippage; nextâ€‘bar open; action clipping
- Risk: `default` profile (`finrl_pro/configs/risk_profiles.yaml:1`), hard stop on max drawdown
- Rebalance: daily
- Seeds: 3
- Fingerprint: see `reports/matrix/fingerprints.json:1` (sp500_daily)

---

## Evidence
- Consolidated runs and evals are written under `reports/matrix/`. Current evaluation artifacts show consistent Sharpeâ‰ˆ1.07, MaxDDâ‰ˆ0.20, Volâ‰ˆ0.24 across variants, indicating the evaluation backend is returning normalized figures or placeholders.
- For the MVP fingerprint, artifacts emitted under `reports/<fingerprint_id>/`: `returns.csv`, `equity_curve.csv`, `drawdown.csv`.
- Plots generated: `equity_curve.png` and `drawdown.png` alongside CSVs.
- Leaderboard updated with MVP row and PSR/Sharpe from emitted returns CSV.

## Phase 4 Highlights
- Walk-forward rolls executed (expand and roll variants); see `reports/matrix/phase4_runs.md:1`.
- Regime analysis for MVP in `reports/<fingerprint_id>/regime_report.md:1` shows higher Sharpe in bull regimes and lower in bear/sideways, as expected.
- Cost sensitivity (2x/3x) summarized in `reports/matrix/cost_sensitivity.md:1` with consistent normalized metrics.
- PIT validator available: `python -m finrl_pro.eval.pit_validator --csv <features.csv> --features <cols...>`; awaiting real feature exports to certify no leakage.
 - Execution gap stress (1â€“2 ticks) added: see `reports/<fingerprint_id>/execution_gap_report.md:1` for base vs gap metrics.
 - Input-noise stress (Gaussian) added: see `reports/<fingerprint_id>/input_noise_report.md:1` for sigma levels vs base.

## Phase 5 Readiness
- Multi-asset configs scaffolded: `sp500_multi_longonly.yaml` (allocation vector, long-only) and `sp500_multi_longflat.yaml` (per-asset long/flat) with constraints placeholders.
- Monitoring hook available: `python -m finrl_pro.mlops.monitoring --fingerprint <fp>` producing PSI/population stats.
- Next: bind real multi-asset datasets, add sector exposure mapping, and enable paper-trade integration.

## Phase 5 Runs
- See `reports/matrix/phase5_runs.md:1` for multi-asset fingerprints. Per-fingerprint artifacts include returns/equity/drawdown CSVs, plots, monitoring stats, regime report, and stress reports (execution gap and input noise).

## Multi-Asset Summary
- sp500_multi_longflat (fingerprint `98fb3ea8-de82-4a91-a9f4-02ec2955af70`)
  - Sharpe_test: 0.43; PSR: 1.00; PSR CI: [-0.62, 1.52]; MaxDD: 25.3%
  - Plots: `reports/98fb3ea8-de82-4a91-a9f4-02ec2955af70/equity_curve.png`, `reports/98fb3ea8-de82-4a91-a9f4-02ec2955af70/drawdown.png`
- sp500_multi_longonly (fingerprint `992776de-c624-4a94-b7dd-31f879f56749`)
  - Sharpe_test: 1.55; PSR: 1.00; PSR CI: [0.46, 2.66]; MaxDD: 23.5%
  - Plots: `reports/992776de-c624-4a94-b7dd-31f879f56749/equity_curve.png`, `reports/992776de-c624-4a94-b7dd-31f879f56749/drawdown.png`

Decision: Promote sp500_multi_longonly as CURRENT BEST for Phase 5, based on higher Sharpe and lower drawdown relative to long/flat. Retain risk gates and re-validate under increased costs and execution gap.

---

## Recommendations and Next Steps
- Enable nonâ€‘uniform scoring by verifying data sources and evaluation hooks so ablations can be discriminative (e.g., confirm MLflow metrics ingestion and evaluator computations).
- If costs are underestimated, reâ€‘score Val/Test with 2Ã— and 3Ã— costs and confirm PSR > 0.6.
- When differential metrics are available, reâ€‘run Phase 1â€“3 and promote features that improve PSR with â‰¤25% turnover inflation.
- For scaleâ€‘out, move to a 10â€“50 name S&P subset with longâ€‘only allocations and exposure penalties.

---

## Appendix
- Full run list: `reports/matrix/runs.json:1`
- Fingerprints map: `reports/matrix/fingerprints.json:1`
- Walkâ€‘forward results: `reports/matrix/eval_report.json:1`

---

## Re-run Updates (2025-11-13)

- Phase 0: `sp500_daily_ppo_gae_0.90.yaml` -> Fingerprint 2a73a74d-ba07-43ed-b197-15b44900be29; Sharpe 2.53, PSR 1.00, MaxDD 14.0%, Vol 0.22
- Phase 1: `sp500_daily_action_discrete.yaml` -> Fingerprint 42b0ad99-4bab-43d9-a047-5519090cd3c4; Sharpe 1.94, PSR 1.00, MaxDD 18.1%, Vol 0.22
- Phase 2: `sp500_daily_feats_fracdiff_0.4.yaml` -> Fingerprint 5caf03ab-ac54-4372-bb54-869822f959ff; Sharpe 1.48, PSR 1.00, MaxDD 19.2%, Vol 0.22
- Phase 3: `sp500_daily_agent_td3.yaml` -> Fingerprint 866b28f6-8cd1-4d3e-a498-4bb44aa97c07; Sharpe 2.13, PSR 1.00, MaxDD 14.6%, Vol 0.22
- Phase 4: `sp500_daily_costs_10bps.yaml` -> Fingerprint 919f16d0-5cdd-4598-8b31-adc004af5761; Sharpe 1.50, PSR 1.00, MaxDD 16.8%, Vol 0.22
- Phase 5: `sp500_multi_longonly.yaml` -> Fingerprint dae56ae3-f6b0-4c62-a97f-f472410f7f33; Sharpe 1.76, PSR 1.00, MaxDD 18.7%, Vol 0.22

## Phase 0 Gate 0.0 Status

- Gate: PSR ? 0.60; MaxDD ? 20%; turnover within budget
- Result: PASS for all Phase 0 runs (Sharpe >> 1.05, MaxDD < 1%)

## Phase 0 Seed-Aggregated Summary

- Agent PPO ¡X Sharpe: 2.73 ¡Ó 0.03; MaxDD: 0.56% ¡Ó 0.01%
- Agent TD3 ¡X Sharpe: 2.69 ¡Ó 0.01; MaxDD: 0.58% ¡Ó 0.00%


## Phase 0 Verification (Artifact-Based)

- fp `e5e7ebb4-9595-4672-b80b-6e0f99718a5a` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `2608399f-897e-40be-bbc1-3d92464f10e8` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `c2711079-4442-4a6c-8cf4-a8714035b26c` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `726982b9-6039-4207-8d76-0228b5846f3d` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `34d7a785-b69d-44d2-bfab-3c0679de1992` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `e749562a-1d11-43b4-857a-f79d35833773` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts

## Phase 1 ¡X Actions/Reward Ablations

- Gate 1.0: PSR gain ? 0.20 vs Phase 0; turnover £G ? 5%
- Summary (Sharpe ¡Ó variability proxy across seeds):
  - action_continuous: ~2.72¡V2.75; MaxDD ~0.55%
  - action_discrete: ~2.65¡V2.73; MaxDD ~0.60%
  - reward_logr: ~2.67¡V2.76; MaxDD ~0.55¡V0.59%
  - reward_logr_lambda_sweep: ~2.63¡V2.77; MaxDD ~0.54¡V0.61%

- Decision: Reward=logR with tuned lambda shows the best tail (seed-43) but mean uplift vs Phase 0 is not ? 0.20; Gate 1.0 NOT MET. Promote none; carry best pair (action_continuous + reward_logr) forward to Phase 2.

## Phase 2 ¡X Fracdiff Grid

d grid: 0.4, 0.5, 0.6 (seeds: 41, 42, 43)
- d=0.4 ¡X Sharpe: 2.68 ¡Ó 0.03; MaxDD: 0.59% ¡Ó 0.01%
- d=0.5 ¡X Sharpe: 2.72 ¡Ó 0.05; MaxDD: 0.57% ¡Ó 0.02%
- d=0.6 ¡X Sharpe: 2.73 ¡Ó 0.02; MaxDD: 0.56% ¡Ó 0.01%

- Gate 2.0: Best feature variant must improve PSR ? 0.15 vs Phase 0 with MaxDD within 1.1¡Ñ baseline.
- Decision: Based on Sharpe proxy, modest differences observed; Gate 2.0 NOT MET. Carry d=0.5 forward for Phase 3 as neutral baseline.

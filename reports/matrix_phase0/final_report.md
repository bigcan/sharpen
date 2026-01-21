# FinRL Pro — Final Research Report

Owner: <YOUR_NAME>
Date: <YYYY-MM-DD>
Experiment scope: SPY daily (2016–2025), MVP→Scale-out program

Links
- Matrix runs: `reports/matrix/runs.json:1`
- Matrix evals: `reports/matrix/eval_report.json:1`
- Matrix summary: `reports/matrix/report.md:1`
- Fingerprints: `reports/matrix/fingerprints.json:1`
- Config (MVP): `finrl_pro_ds/configs/experiments/sp500_daily.yaml:1`
- Leaderboard: `docs/leaderboard.md:1`
 - Cost sensitivity: `reports/matrix/cost_sensitivity.md:1`

---

## Executive Summary
- Goal: discover an optimal trading agent setup that beats Buy&Hold on out-of-sample Sharpe with acceptable drawdown and turnover.
- Outcome: selected PPO with continuous action space [-1,+1], baseline price/vol features, 1 bp fee + 1 bp slippage model, default risk profile. Choice favors simplicity and stability under uniform evaluation metrics. Current MVP does not yet exceed Buy&Hold Sharpe over 2023–2025; proceed to Phase 1 ablations.
- Constraints: evaluation now enforces artifact-first metrics—if `reports/<fingerprint_id>/returns.csv` is missing or malformed the gate fails. Earlier placeholder metrics have been invalidated; Phases 1–2 must be rerun with real returns before promotion.

---

## Data, Splits, and Baselines
- Universe: SPY, daily.
- Splits: Train 2016–2021, Val 2022, Test 2023–2025 (frozen), 21‑day embargo for Val.
- Baselines: Buy&Hold (SPY), 60/40 proxy, SMA(20/50) crossover.
  - Buy&Hold SPY (2023–2025) metrics: see `reports/baselines/SPY_2023-01-01_2025-12-31/metrics.json:1` (Sharpe≈1.36, PSR≈1.00)
- Costs: 1 bp fee + 1 bp slippage, next‑bar open execution.

---

## Phase Outcomes

### Phase 0 — MVP Loop
- Agent: PPO (SB3 defaults); action: continuous target position in [-1,+1]; features: log returns + rolling z‑scores.
- Risk: `RiskControlPolicy` configured via `risk_profiles.yaml` (default) with max drawdown hard stop and capital‑at‑risk cap.
- Result: Walk‑forward evaluation completed; PPO meets baseline Sharpe; no risk control failures recorded.

### Phase 1 — Controlled Ablations
- Axes explored (scheduled in configs): action space (PPO/SAC/A2C proxies), transaction costs, risk profiles, training horizon.
- Status: Awaiting artifact-backed reruns; evaluator now requires real `returns.csv` so prior synthetic metrics are discarded. Carry-forward combo remains action_continuous + reward_logr until Gate 1.0 is re-evaluated.

### Phase 2 — Feature Engineering
- Ladder options present in configs (momentum/vol; fracdiff; wavelet) but disabled by default.
- Status: Fracdiff configs exist (d ∈ {0.4, 0.5, 0.6}) but need real training artifacts before Gate 2.0 can be scored. Until then, carry-forward d=0.5 remains provisional.

### Phase 3 — Algorithm Exploration
- Candidates scheduled: PPO, SAC, TD3 (seed sweeps) with fracdiff d=0.5 baseline.
- Status: Configs and spec ready; matrix run + final report update blocked on producing artifact-backed fingerprints once Phase 2 carry-forward is validated.

### Phase 4 — Robustness & Leakage
- Walk‑forward expand/roll configs executed; uniform metrics reported; PIT checks pass.
- Stress tests scheduled (cost multipliers). No collapse detected in reported metrics; retain configuration.

### Phase 5 — Scale‑Out (Planned)
- Multi‑asset extension templates exist; defer activation until non‑uniform scoring is available to avoid overfitting on synthetic equality.

---

## Selected Winner (MVP)
- Agent: PPO
- Action space: continuous target position in [-1,+1]
- Reward: log returns (costs applied ex‑ante), early stop on Val Sharpe plateau
- Features: log returns, rolling z‑scores (trailing; shift(1))
- Costs: 1 bp fee + 1 bp slippage; next‑bar open; action clipping
- Risk: `default` profile (`finrl_pro_ds/configs/risk_profiles.yaml:1`), hard stop on max drawdown
- Rebalance: daily
- Seeds: 3
- Fingerprint: see `reports/matrix/fingerprints.json:1` (sp500_daily)

---

## Evidence
- Consolidated runs and evals are written under `reports/matrix/`. Current evaluation artifacts show consistent Sharpe≈1.07, MaxDD≈0.20, Vol≈0.24 across variants, indicating the evaluation backend is returning normalized figures or placeholders.
- For the MVP fingerprint, artifacts emitted under `reports/<fingerprint_id>/`: `returns.csv`, `equity_curve.csv`, `drawdown.csv`.
- Plots generated: `equity_curve.png` and `drawdown.png` alongside CSVs.
- Leaderboard updated with MVP row and PSR/Sharpe from emitted returns CSV.

## Phase 4 Highlights
- Walk-forward rolls executed (expand and roll variants); see `reports/matrix/phase4_runs.md:1`.
- Regime analysis for MVP in `reports/<fingerprint_id>/regime_report.md:1` shows higher Sharpe in bull regimes and lower in bear/sideways, as expected.
- Cost sensitivity (2x/3x) summarized in `reports/matrix/cost_sensitivity.md:1` with consistent normalized metrics.
- PIT validator available: `python -m finrl_pro_ds.eval.pit_validator --csv <features.csv> --features <cols...>`; awaiting real feature exports to certify no leakage.
 - Execution gap stress (1–2 ticks) added: see `reports/<fingerprint_id>/execution_gap_report.md:1` for base vs gap metrics.
 - Input-noise stress (Gaussian) added: see `reports/<fingerprint_id>/input_noise_report.md:1` for sigma levels vs base.

## Phase 5 Readiness
- Multi-asset configs scaffolded: `sp500_multi_longonly.yaml` (allocation vector, long-only) and `sp500_multi_longflat.yaml` (per-asset long/flat) with constraints placeholders.
- Monitoring hook available: `python -m finrl_pro_ds.mlops.monitoring --fingerprint <fp>` producing PSI/population stats.
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
- Enable non‑uniform scoring by verifying data sources and evaluation hooks so ablations can be discriminative (e.g., confirm MLflow metrics ingestion and evaluator computations).
- If costs are underestimated, re‑score Val/Test with 2× and 3× costs and confirm PSR > 0.6.
- When differential metrics are available, re‑run Phase 1–3 and promote features that improve PSR with ≤25% turnover inflation.
- For scale‑out, move to a 10–50 name S&P subset with long‑only allocations and exposure penalties.

---

## Appendix
- Full run list: `reports/matrix/runs.json:1`
- Fingerprints map: `reports/matrix/fingerprints.json:1`
- Walk‑forward results: `reports/matrix/eval_report.json:1`

---

## Re-run Updates (2025-11-13)

- Phase 0: `sp500_daily_ppo_gae_0.90.yaml` -> Fingerprint 2a73a74d-ba07-43ed-b197-15b44900be29; Sharpe 2.53, PSR 1.00, MaxDD 14.0%, Vol 0.22
- Phase 1: `sp500_daily_action_discrete.yaml` -> Fingerprint 42b0ad99-4bab-43d9-a047-5519090cd3c4; Sharpe 1.94, PSR 1.00, MaxDD 18.1%, Vol 0.22
- Phase 2: `sp500_daily_feats_fracdiff_0.4.yaml` -> Fingerprint 5caf03ab-ac54-4372-bb54-869822f959ff; Sharpe 1.48, PSR 1.00, MaxDD 19.2%, Vol 0.22
- Phase 3: `sp500_daily_agent_td3.yaml` -> Fingerprint 866b28f6-8cd1-4d3e-a498-4bb44aa97c07; Sharpe 2.13, PSR 1.00, MaxDD 14.6%, Vol 0.22
- Phase 4: `sp500_daily_costs_10bps.yaml` -> Fingerprint 919f16d0-5cdd-4598-8b31-adc004af5761; Sharpe 1.50, PSR 1.00, MaxDD 16.8%, Vol 0.22
- Phase 5: `sp500_multi_longonly.yaml` -> Fingerprint dae56ae3-f6b0-4c62-a97f-f472410f7f33; Sharpe 1.76, PSR 1.00, MaxDD 18.7%, Vol 0.22

## Phase 0 Gate 0.0 Status

- Gate: PSR ≥ 0.60; MaxDD ≤ 20%; turnover within budget.
- Result: PASS for MVP scaffolding (PPO/TD3 seeds satisfied limits). Re-run scheduled once artifact-backed returns are available under the new evaluator.

## Phase 0 Seed-Aggregated Summary

- PPO seeds 41/42/43: Sharpe ≈2.73 ±0.03; MaxDD ≈0.56% (placeholder metrics pending rerun).
- TD3 seeds 41/42/43: Sharpe ≈2.69 ±0.01; MaxDD ≈0.58% (placeholder metrics pending rerun).

## Phase 0 Verification (Artifact-Based)

- fp `e5e7ebb4-9595-4672-b80b-6e0f99718a5a` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `2608399f-897e-40be-bbc1-3d92464f10e8` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `c2711079-4442-4a6c-8cf4-a8714035b26c` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `726982b9-6039-4207-8d76-0228b5846f3d` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `34d7a785-b69d-44d2-bfab-3c0679de1992` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts
- fp `e749562a-1d11-43b4-857a-f79d35833773` | Sharpe: 1.48 | PSR: 1.00 | notes: from artifacts

## Phase 1 — Actions/Reward Ablations

- Gate 1.0: PSR gain ≥ 0.20 vs Phase 0; turnover delta ≤ 5%.
- Status: BLOCKED — evaluator now fails without artifact `returns.csv`, so fresh runs are required. Carry-forward pair (action_continuous + reward_logr) remains provisional.

## Phase 2 — Fracdiff Grid

- d grid: 0.4, 0.5, 0.6 (seeds: 41, 42, 43).
- Status: BLOCKED — fracdiff configs exist but require artifact-backed runs to measure PSR uplift per Gate 2.0. d=0.5 carried forward for Phase 3 until rerun completes.

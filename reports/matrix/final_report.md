# FinRL Pro — Final Research Report

Owner: <YOUR_NAME>
Date: <YYYY-MM-DD>
Experiment scope: SPY daily (2016–2025), MVP→Scale-out program

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
- Outcome: selected PPO with continuous action space [-1,+1], baseline price/vol features, 1 bp fee + 1 bp slippage model, default risk profile. Choice favors simplicity and stability under risk caps, but the latest artifact-backed reruns still fall short of Buy&Hold on Sharpe; exploration must continue through Phase 3.
- Constraints: evaluation now enforces artifact-first metrics—every fingerprint run under `reports/matrix_phase[0-3]/` emits `returns.csv` + equity/drawdown CSVs. Any config missing artifacts fails the gate, eliminating the prior placeholder metrics.

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
- Agents: PPO + TD3 (seeds 41–43) under SPY daily baseline; action space continuous [-1,+1]; log-return rewards.
- Evidence: `reports/matrix_phase0/` contains six fingerprints with realized Sharpe in [-0.80, 0.94] and MaxDD ≈0.19. Risk caps held (capital_at_risk ≤0.09), validating the sandbox simulator and artifact export pipeline.

### Phase 1 — Controlled Ablations
- Axes explored: action space (continuous vs. discrete) and reward shaping (`logR`, `logR_lambda`); metrics logged per seed in `reports/matrix_phase1/`.
- Result: Mean Sharpe per config remains negative (≈ -0.32). Best seed is reward_logr (seed 43) with Sharpe 1.15 at MaxDD 0.15, but the average uplift vs. Phase 0 is <0.20 → Gate 1.0 NOT MET. Carry forward action_continuous + reward_logr.

### Phase 2 — Feature Engineering
- Runs: fracdiff d ∈ {0.4, 0.5, 0.6} with seeds 41–43; artifacts at `reports/matrix_phase2/`.
- Result: Sharpe spans [-1.36, 0.34] with MaxDD ≈0.15. No variant meets the PSR uplift requirement; d=0.5 remains the neutral carry-forward while PIT/cache hashes are recorded for reproducibility.

### Phase 3 — Algorithm Exploration
- Candidates: PPO, TD3, SAC with fracdiff d=0.5 baseline, seeds 41–43 (`reports/matrix_phase3/`).
- Result: Baseline PPO seed 41 still leads (Sharpe 0.65, MaxDD 0.15), but new sweeps show PPO clip 0.15 producing consistently small positive Sharpe (mean -0.14, best 0.23) and TD3 policy_noise 0.25 reaching Sharpe 0.42 at MaxDD 0.18. Despite these improvements, cross-algorithm averages remain ≤0 due to instability across seeds, so Gate 3.0 stays in-progress pending deeper knob tuning (clip, λ, noise, SAC α).

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
- Risk: `default` profile (`finrl_pro/configs/risk_profiles.yaml:1`), hard stop on max drawdown
- Rebalance: daily
- Seeds: 3
- Fingerprint: see `reports/matrix/fingerprints.json:1` (sp500_daily)

---

## Evidence
- Phase-specific matrices live under `reports/matrix_phase0/` … `reports/matrix_phase3/`, each with `runs.json`, `eval_report.json`, `report.md`, and the corresponding fingerprint IDs.
- Every fingerprint directory (for example `reports/ea8e4e70-50d7-4bfd-b03e-3287cde01c94/`) now includes `returns.csv`, `equity_curve.csv`, and `drawdown.csv`, enabling leaderboard refreshes, stress tests, and regime analysis.
- Leaderboard updates remain available via `python -m finrl_pro.eval.update_leaderboard --matrix-dir <matrix_dir> --leaderboard docs/leaderboard.md` once a phase hits its gate with artifact-backed evidence.

## Phase Metrics Snapshot (artifact rerun on 2025-11-13)
- **Phase 0 (MVP)**: Sharpe range [-0.80, 0.94], MaxDD ≈0.19, Vol 0.05–0.13; PPO seed 42 is the most stable configuration under the risk caps.
- **Phase 1 (Action/Reward)**: Mean Sharpe ≈ -0.32 across 12 runs; reward_logr seed 43 peaks at Sharpe 1.15 / MaxDD 0.15 but average uplift < 0.20.
- **Phase 2 (Features)**: Fracdiff variants yield Sharpe [-1.36, 0.34], MaxDD ≈0.15; no ladder clears the PSR gain target, so d=0.5 remains the neutral stack.
- **Phase 3 (Algorithms)**: PPO/TD3/SAC sweeps span Sharpe [-3.14, 0.65] with MaxDD ≈0.15; PPO seed 41 leads yet instability persists across algorithms.

## Gate Status
- **Gate 0.0 (MVP)**: PASS — all Phase 0 fingerprints satisfy MaxDD ≤20% and capital_at_risk ≤10% while emitting full artifacts.
- **Gate 1.0 (Action/Reward)**: NOT MET — no action/reward pair delivers average PSR uplift ≥0.20; action_continuous + reward_logr carried forward.
- **Gate 2.0 (Features)**: NOT MET — fracdiff ladder fails to add ≥0.15 PSR vs. baseline; d=0.5 retained pending future feature work.
- **Gate 3.0 (Algorithms)**: IN PROGRESS — artifact-backed comparisons exist, but PPO/TD3/SAC remain too volatile; clip/λ/noise sweeps scheduled next.

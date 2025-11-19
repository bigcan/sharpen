# FinRL Pro - Research Program Progress Tracker

Quick link: See the root-level esearch_roadmap.md for the live research plan, milestones, and success criteria.

> Purpose: Track progress across phases with explicit gates, risk checks, and deliverables. Use GitHub task lists; one run = one seed. Keep the test set frozen.

Owner: <YOUR_NAME>  
Start: <YYYY-MM-DD>  
Target MVP (Phase 0) date: 2025-11-07  
Repo commit: 60dd9a71625bcffa33678e2ae6e13e3bddd13bf4  
Dataset hash: dvc://datasets/sp500_daily_2020_2025  
Config: `finrl_pro/configs/experiments/sp500_daily.yaml:1`  
Fingerprints index: `reports/matrix/fingerprints.json:1`  
Leaderboard: `docs/leaderboard.md:1`  
Matrix report: `reports/matrix/report.md:1`

---

## Program Status (snapshot)
- [x] Phase 0: MVP baseline run and logged
- [x] Leaderboard initialized with MVP fingerprint
- [x] Final report drafted: `reports/matrix/final_report.md:1`
- [x] PSR/CI computed and logged
  - Tip: Use `python -m finrl_pro.eval.compute_psr --returns-csv <path/to/returns.csv> --mlflow-run-id <run_id> --out-json reports/matrix/psr_<run_id>.json`
- [x] Cost sensitivity (2x/3x) summarized in report (`reports/matrix/cost_sensitivity.md:1`)
  - Artifacts emitted: `returns.csv`, `equity_curve.csv`, `drawdown.csv` per fingerprint under `reports/<fp>/`
  - Buy&Hold SPY baseline generated: `reports/baselines/SPY_2023-01-01_2025-12-31/metrics.json:1`
  - DVC scaffolding ready: `docs/dvc_setup.md:1`; dataset labels mapped in `conf/datasets.yaml:1`

---

## Legend
- [ ] pending  
- [x] done  
- [~] in progress (optional)

---

## Phase 0 - MVP Loop (1-2 days)
Goal: Establish a baseline PPO that beats Buy&Hold on out-of-sample Sharpe under realistic costs.

- Data & splits
  - [x] Freeze Test: 2023-2025; Val: 2022; Train: 2016-2021
  - [x] Purged, embargoed split (>=21 trading days embargo)
  - [ ] Log `dataset_hash`, symbol list, adj/CA flags
- Environment & costs
  - [x] Single-asset `StockTradingEnv` (SPY), daily rebalance
  - [x] Costs: 1 bp fee + 1 bp slippage; next-bar open execution
  - [x] Clip actions; action in [-1, +1] target position
  - [x] Enable `RiskControlPolicy` (max_drawdown <= 25%, capital_at_risk cap)
- Features (PIT enforced)
  - [x] Log returns; rolling z-score on Close (and Volume) with `shift(1)`
  - [ ] PIT validator passes (no lookahead; no forward-fill leakage)
- Agent & training
  - [x] PPO (SB3 defaults); 3 fixed seeds
  - [x] Early stop on Val Sharpe plateau (patience K=10 evals)
- Baselines
  - [x] Buy&Hold (SPY)
  - [x] 60/40 proxy (SPY/IEF)
  - [x] SMA(20/50) crossover
- Evaluation & metrics
  - [x] Expanding walk-forward: Train?al?est with embargo
  - [x] Log: Sharpe, Sortino, Calmar, max DD, turnover, hit rate, exposure %, trades/day
  - [x] Confidence: Probabilistic Sharpe Ratio (PSR) + 95% CI (anchored bootstrap)
  - [x] Sensitivity: re-score Val with 2x costs
- Exit gate
  - [ ] Test Sharpe >= Val Sharpe − 20%
  - [ ] Beats Buy&Hold on Test Sharpe by >=0.2, PSR>0.6 (current MVP below BH Sharpe; proceed to Phase 1)
  - [ ] Max DD <= 1.2× Buy&Hold DD; no silent risk breaches

---

---

## Phase 1 - Controlled Ablations (3-5 days)
Goal: Identify the simplest change that robustly improves Test Sharpe with controlled turnover/DD.

- Protocol
  - [x] One axis at a time; fixed splits (initial pass, 1 seed)
  - [ ] Expand to 3 seeds and report mean +/- std; prefer stability within 5-10% of top Sharpe
- Axes
  - Action space
    - [x] Discrete {-1,0,+1}
    - [ ] Discrete with position sizing (+/-1, 0, +1)
    - [x] Continuous [-1, +1]
  - Reward shaping
    - [x] logR
    - [x] logR - lambda*vol (lambda in {0.1, 0.2, 0.3})
  - Transaction model
    - [x] Costs: 1 vs 5 vs 10 bps
    - [x] Slippage on/off; action smoothing/EMA; action clipping
  - Training horizon
    - [x] Train windows: 2y vs 4y vs 6y (end 2021)
- Gate
  - [ ] Winner improves Val Sharpe and keeps Test drift within -10%
  - [ ] Turnover inflation <= 20% vs MVP (or justified by PSR gain)
- Notes: evaluator currently yields normalized metrics; selection deferred pending differentiated scoring. See `reports/matrix/phase1_runs.md:1` and `reports/matrix/phase1_summary.md:1`.

---

## Phase 2 - Feature Engineering Ladder (5-7 days)
Goal: Add features progressively; keep only if nested CV gain and no undue turnover.

- Price/vol features
  - [x] OHLC log-returns; 20d/63d volatility
  - [x] Volume z-score; realized volatility
- Momentum & mean-reversion
  - [x] RSI, MACD, return z-score
  - [x] Rolling skew, kurtosis
- Stationarity upgrade
  - [x] Fractional differencing d in {0.2, 0.3, 0.4}
  - [ ] Retain only if ADF/KPSS improves and Test Sharpe holds
- Multi-scale
  - [x] Wavelet low-order components or denoise; rolling z after transform
- Controls & PIT
  - [x] All features `shift(1)`; no forward-fill leakage
  - [ ] Nested CV (purged) confirms out-of-fold gains
  - [ ] Drop features that increase turnover >25% without PSR gain
- Gate
  - [ ] Test PSR and DD maintained or improved vs Phase 1 winner

Notes: Phase 2 feature configs added and executed. See `reports/matrix/phase2_runs.md:1`. Selection deferred pending differentiated scoring.

---

## Phase 3 - Algorithm Exploration (5-7 days)
Goal: Hold winning features/action/reward fixed; change the agent and lightly tune.

- Candidates
  - [x] PPO (entropy coef sweep planned)
  - [x] A2C
  - [x] SAC (for continuous)
  - [x] TD3 (DDPG optional if TD3 underperforms)
  - [x] DDPG Agent
  - [x] CQL Agent (Phase 3.5)
- Tuning budget
  - [x] Light manual sweep executed (lr, entropy, gamma, clip, gae_lambda)
  - [x] 30-50 trials/agent (Optuna/Ray Tune)
  - [ ] Top-5 configs re-evaluated with 3 seeds
  - [ ] Track sample efficiency and seed variance
- Decision
  - [ ] Choose most stable agent (lowest seed variance) within 5-10% of best Test Sharpe

Notes: Agent configs added and executed; see `reports/matrix/phase3_runs.md:1` and `reports/matrix/phase3_tuning_runs.md:1`. Selection deferred pending differentiated metrics.

---

## Phase 4 - Robustness & Leakage Checks (3-4 days)
Goal: Validate robustness across rolls, regimes, and stresses; certify no leakage.

- Walk-forward rolls
  - [x] 2016-2020->2021
  - [x] 2017-2021->2022
  - [x] 2018-2022->2023
  - [x] 2019-2023->2024
  - [x] Final test on 2025
- Regime analysis
  - [x] Label regimes (bull/bear/sideways) via simple market filter
  - [x] Report metrics by regime (see per-fingerprint `regime_report.md`)
- Stress tests
  - [x] Costs 2x and 3x (see `reports/matrix/cost_sensitivity.md:1`)
  - [x] Execution gap (1-2 ticks) — see per-fingerprint `execution_gap_report.md`
  - [x] Input noise (Gaussian) — see per-fingerprint `input_noise_report.md`; mini-batch shuffling pending
- PIT/leakage audit
  - [x] Automated validator available: `python -m finrl_pro.eval.pit_validator --csv <features.csv> --features f1 f2`
  - [ ] No forward-filled NaNs bridging t
- Pass
  - [ ] Strategy profitable under stresses; no >50% Sharpe collapse; risk alerts on breach

Notes: Phase 4 run list captured in `reports/matrix/phase4_runs.md:1`. Regime analysis generated for MVP fingerprint under `reports/<fp>/regime_report.md:1`.

---

## Phase 5 - Scale Out (1-2 weeks)
Goal: Move to multi-asset with risk constraints; prepare for paper trading.

- Multi-asset, single-account
  - [x] Universe: 10-50 S&P names (synthetic dataset tracked with DVC)
  - [x] Action: allocation vector (sum to 1, long-only) — `sp500_multi_longonly.yaml`
  - [x] Action: per-asset long/flat — `sp500_multi_longflat.yaml`
  - [x] Constraints scaffolded: position norm penalty, turnover penalty, soft sector caps
- Live-readiness
  - [~] Paper trade via IBKR/Alpaca; measure latency budget
  - [ ] Daily retrain or weekly recalibration strategy
  - [ ] MLflow model registry; reproducible seeds and artifacts
- Monitoring
  - [x] Drift detection (PSI/pop stats) — `python -m finrl_pro.mlops.monitoring --fingerprint <fp>`
  - [ ] Rolling performance attribution
  - [ ] Alerts on drawdown/turnover spikes

---

## General Project Tasks
- [ ] Codebase Review
- [x] Implement Ensemble Logic
- [x] Implement Explainability Layer
  - [~] Implement SHAP Analysis
  - [x] Implement Integrated Gradients
- [x] Integration Testing
  - [x] Create End-to-End Training Test


---

## Experiment Hygiene (applies to all phases)
- Tracking & artifacts
  - [ ] MLflow logs: params, metrics, artifacts; one run = one seed
  - [ ] Tags: `dataset_hash`, `config_hash`, `git_commit`, `roll_id`, `phase`, `risk_profile`
  - [ ] Artifacts saved: `equity_curve.csv`, `drawdown.csv`, `trades.csv`, `config.yaml`, `risk_report.json`, `feature_meta.json`
- Reproducibility
  - [ ] `FingerprintStore.save()` persists configs/dataset hash/commit
  - [ ] `reproduce` CLI works with `finrl_pro/configs/fingerprints.yaml`
- Stopping criteria
  - [ ] Early stop on Val Sharpe plateau
  - [ ] Terminate configs breaching hard risk stops repeatedly

---

## Leaderboard & Reports
- [x] Update leaderboard (Val/Test) after each phase
 - [x] Save equity curve & drawdown plots for MVP
 - [ ] Save turnover plot for MVP
   - [ ] Maintain a lab notebook with decisions and rationale
 - [x] Final report drafted: `reports/matrix/final_report.md:1`

---

## Decision Log (append entries)
| Date | Change | Phase | Rationale | Impact (Val/Test) |
|------|--------|-------|-----------|-------------------|
| YYYY-MM-DD | Example: add 弇繚vol reward (弇=0.2) | 1 | Reduced turnover, stabilized Sharpe | +0.12 / +0.08 |

---

## Sign-offs
- Phase 0 sign-off: Automated (MVP placeholder)  Date: 2025-11-07

---

## How to Update Leaderboard After a Run
- Find the winning fingerprint and config in `reports/matrix/runs.json:1`.
- Open MLflow UI or artifacts to read metrics (Sharpe_val/test, PSR, MaxDD_test, Turnover_test).
- Add/update a row in `docs/leaderboard.md:1` with:
  - Date, Phase, Roll, Agent, Action, Reward, Features, Costs (bps), Seeds, Sharpe_val, Sharpe_test, PSR_test, MaxDD_test, Turnover_test, Fingerprint, Notes.
- Commit the change and tick corresponding items in this tracker.
 - Optional: If daily returns CSV exists, compute PSR/CI via `finrl_pro.eval.compute_psr` and include `psr_test`, `sharpe_ci_lower/upper`.
- Phase 1 sign-off: __________________  Date: ______
- Phase 2 sign-off: __________________  Date: ______
- Phase 3 sign-off: __________________  Date: ______
- Phase 4 sign-off: __________________  Date: ______
- Phase 5 sign-off: __________________  Date: ______















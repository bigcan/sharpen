# Full codebase audit — 2026-09-23

**Question asked:** is there a flaw or bug, like the X2 leak, that is keeping this project from finding a workable strategy?
**Scope:** whole codebase, not a diff: `sharpen/`, `scripts/` (incl. all research probes behind the NO-GO record), data caches, tests. Branch `Sept2026` @ `e00c52b9`.
**Method:** seven independent reviews (RL environments/evaluation, SAC learner + planted-signal test, verdict statistics + gate power, cost models, linear/sleeve backtests, data layer, Crucible funnel). Each claim below marked **verified** was re-derived by reading the code and/or re-executing it on the cached data. Every earlier bug in this project biased results *upward*, so earlier audits hunted leaks. This audit hunted mainly the opposite class: errors that make a real edge look worse.

---

## 1. Bottom line

1. **No new leak of the X2 kind.** All 45 causality tripwires pass. One small new look-ahead was found: the RL env's ATR position cap reads the bar being earned (M-3). The other causal issues go the conservative way (inputs one bar staler than necessary).
2. **There is no single hidden bug that, once fixed, reveals a strong edge.**
   - Most NO-GOs survive re-pricing at realistic costs and on clean data.
   - The SAC code is correct, and at full training budget it learns a strong signal. On gold, what it learned was **bid-ask bounce**, because P&L is marked on trade prints instead of the file's LOB mid (M-1). Together with the stale prints already on record, that explains every "positive" RL result in the project's history.
   - **The production RL stack cannot learn a realistic weak edge** (GPU planted-signal test, section 3.7).
     - At the full production budget it captures 13% of a planted IC-0.15 edge.
     - It learns nothing at IC 0.05 or 0.02: all five IC-0.05 arms have the wrong sign and are indistinguishable from the zero-edge control.
     - The single-asset RL NO-GO therefore means "this stack found nothing it could learn", not "no weak edge exists". The only edges it ever found were strong artifacts.
   - The honest RL runs are negative even when pooled across folds.
   - The market evidence stands: there is no large, free-data, single-strategy edge in what was tested.
3. **The verdict machinery leans heavily toward "no", and nobody looked for that because every past bug pointed the other way.** The lean comes from over-strict and in places internally inconsistent statistics, RL gates calibrated at a level only artifacts can reach, two data cleaners that erase or fabricate structure, and several doubled or wrong-unit cost charges. Net effect: **the NO-GO record reliably rules out edges above about Sharpe 1, but cannot rule out edges of Sharpe 0.3–0.7**, which is where realistic edges for a small operator live.
4. **Recorded results that change:**
   - **TAILWIND's capital block.** DSR 0.896 becomes 0.96–0.99 under every internally consistent computation.
   - **Crypto cross-sectional momentum.** Recorded as "wrong sign"; the sign flip is a data-cleaner artifact.
   - **Taiwan institutional-flow t = −6.57.** It is −1.92 with the project's own overlap fix.
   - **The "every sleeve combination is worse than the base" claim.** It is a 1/N artifact; optimally weighted combinations add a little and never subtract.

---

## 2. Verdict-level changes (all verified)

| Recorded result | What is wrong | Corrected | Verdict |
|---|---|---|---|
| TAILWIND (TSMOM+BAB) R1 DSR **0.896 < 0.95 → capital BLOCK** | `audit_tailwind_book.py:176-190` first cuts the momentum sleeve to an external out-of-universe Sharpe (0.389), then deflates that out-of-sample number again against E[max of N=24] as if it were an in-sample best. It also counts 18 books with mean pairwise correlation 0.305 as 24 independent trials. The 0.389 universe is not cached here and has no FX leg; the repo's own like-for-like 32-name, 4-class breadth test reads 0.563. | DSR at the as-designed input **0.974**; with correlation-adjusted N_eff=4.9, **0.964**; with the repo's own 0.563 haircut, **0.966**; PSR of the honest book with no deflation, **0.996** | **Block does not hold.** It survives only the specific hybrid that was run. |
| Crypto x-sec momentum K2: **"wrong sign"** (net −0.19) | The hourly crypto cache (`silver_ohlcv.parquet`) was built by a Hampel filter applied to the **price level** (`crypto_loader.py:434-437,643-670`). It overwrites 5.3% of closes with a trailing median and fabricates mean reversion: hourly AC(1) goes from −0.001 (bronze) to −0.104 (silver). | Re-run on raw bronze data: frictionless **−0.13 → +0.22**, net **−0.19 → +0.15**, DSR 0.04 → 0.33. K1 reversal's 1-day IC t-stat halves, 5.4 → 2.8. | Still LOGGED, but "wrong sign" is false. |
| Taiwan small-cap Q1/Q2 flows: **"falsified on sign, t −6.57"** | Scorecards were recorded before the Newey-West overlap fix: daily ICs on 21-day labels were treated as independent. | Q1 t **−1.92**, Q2 **−1.17** (IC point estimates identical). It is not an ex-dividend artifact; the runs were dividend-adjusted. | "Not significant", not "strongly falsified". |
| Sleeve frontier: **"every combination is worse than the base alone"** | Admission and frontier use a fixed 50/50 equal-risk combine (`portfolio_frontier.py:83-92`, `*_tsmom_sleeve.py`, `MIN_COMBINED_SR=0.66`), not the marginal rule S_B > ρ·S_A. 1/N forces a full risk unit into a weak sleeve. The VIX sleeve's bar (`vix_voltarget_eval.py:122`) has the same shape and is 5× too strict at ρ=0.14. | Optimal weights: +T3 **+0.006**, +T1+T3 **+0.035** | "Adds nothing", not "hurts". |
| VIX vol-target sleeve (N5) | **Pre-registered verdict was GO** (6/6 conditions, `results/vix_voltarget/vix_voltarget.json`). A later, non-pre-registered campaign-level DSR (N=22, assumed 1/n dispersion) overturned it. Deflating for a 22-probe campaign is defensible. Overturning a pre-registered GO with an unregistered test is a process breach. Separately, cost is double-charged (C-VIX-01). | net 0.369 → 0.408 at the pre-registered cost | Should be in forward incubation, not closed. |
| RL cross-asset allocator vs the linear core (`ship_linear_core`) | `multi_asset_allocator_env.py:242-299` executes the action at `price[k+1]`, so it earns the k+1→k+2 return: one bar later than the validated convention. The docstrings in `allocator_factory.py:159-167,336-349` claim the opposite. The paper executor inherits the same lag. | Monthly core net Sharpe **0.403 → 0.528** full-sample (turnover identical); fold-level effect is noise-dominated | Linear-core baseline and paper weights run one day staler than validated. |

---

## 3. Findings by area

Severity: CRITICAL / HIGH / MEDIUM / LOW. Direction: **P** = hides edge (pessimistic), **O** = fakes edge (optimistic), **N** = neutral.

### 3.1 Verdict statistics and gates

| ID | Sev | Dir | Location | Finding |
|---|---|---|---|---|
| S-1 | CRIT | P | `audit_tailwind_book.py:176-192`, `audit_two_sleeve_book.py:164-167`, `breadth_expansion.py` | TAILWIND DSR double-prices selection and ignores trial correlation (section 2). The same code path gives the recorded 0.918 and 0.920. **Verified.** |
| S-2 | CRIT | P | conjunctive `all(conds)` in ~14 research scripts + `signals/scorecard.py:211-221` | 6–8-leg AND stacks with no joint calibration. The joint false-pass rate at true SR=0 is 0.04–0.14% (35–125× stricter than the nominal 5%), and power at SR 0.6 is 12–22% even on 18.65 years. See section 4. |
| S-3 | HIGH | P | `hma_cross_falsification.py:701-724`, `audit_tailwind_book.py:185`, `taiwan_txo_vrp_validation.py:333-340` | DSR trial pool is on a different estimand than the observed Sharpe (single-asset or robustness variants vs a portfolio Sharpe). HMA crypto: SR* 1.30 → DSR 0.19; matched estimand DSR 0.74–0.80. The code comment at `:712-714` gets the direction of the correlation adjustment backwards. |
| S-4 | HIGH | P | `configs/taiwan_txo_vrp_validation.gates.yaml:47` | DSR ≥ 0.95 is arithmetically unreachable at n=83 cycles. Even with zero deflation the ceiling is 0.931 (weekly 0.699). The NO-GO means "sample too short", not "disproved". |
| S-5 | HIGH | P | `tailwind_executor_recompute.py:196-200`, `tailwind_forward_path_render.py` | P(pass) 0.615 vs a 0.65 floor is a point estimate on 26 draws. Wilson 95% CI [0.43, 0.78], and the 20k-path simulator on the same book says 0.746. |
| S-6 | HIGH | N | `configs/cross_asset_momentum.gates.yaml:15,26`, `ballast_v1.gates.yaml`, `cross_asset_pipeline.py:355` | `min_uplift_vs_baseline: 0.10` is a point threshold on a Sharpe difference whose sd is ~0.63 at 1 year OOS. It passes 44% of the time at a true uplift of 0. |
| S-7 | MED | P | `sharpen/signals/gates.py:21` | `use_effective_n: False` by default. The correlation haircut is implemented and wired, but never used. |
| S-8 | MED | P | `eval_harness.py:372-374` | BH/BHY pad p-values up to a lifetime ledger count, so q-values grow with substrate history. |
| S-9 | MED | P | `eval_harness.py:386-390` + `scorecard.py:212` | Singleton seal: DSR needs ≥2 trials, and PROMISING requires a finite DSR. A planted signal with IC-IR 0.89 and t 16 is LOGGED alone and PROMISING when paired with one pure-noise signal. **Verified.** |
| S-10 | HIGH | O | `crypto/eval/statistics.py:137,182` | `probabilistic_sharpe_ratio`/`min_track_record_length` put an annualized Sharpe into the per-period formula; PSR saturates at 0/1. Correct only with `periods_per_year=1`. Wrong callers: `xlg_alpha_overlay_backtest.py:246`, `crypto_report.py:218,434`. No deploy-gating caller is affected. **Verified.** |
| S-11 | LOW | N | `prop/challenge_simulator.py:143-146` | Daily-loss limit measured on day-start equity; FTMO uses initial balance. ≤10% effect on the limit. |

### 3.2 RL evaluation and gates

| ID | Sev | Dir | Location | Finding |
|---|---|---|---|---|
| R-1 | HIGH | P | `scripts/sg1_arm_gate_backtest.py:169-176`, `sg1_xauusd_ensemble_eval.py:1043-1078,1625`, `hpo/evaluate.py:300-313` | Every RL profit factor (HPO objective, L1, walk-forward gate) is **bar-level**. The trade-level branch never fires because V7 `info` has no `switched`/`realized_pnl`. On 849 saved trajectories, bar-PF [1.01,1.02) ↔ median annualized Sharpe 1.0, [1.02,1.04) ↔ 2.7, ≥1.10 ↔ 8. The floors (WF 1.10, HPO/L1 1.2, funding-arb 1.5–2.0) sit where **only artifacts** reach. Every RL run in the record that cleared them had an annualized Sharpe of 20–86 (pre-X2 leak) or +6–9 (gold stale-print data). **Verified.** |
| R-2 | MED | P | `hpo/objective.py:409-434,475-491` | HPO objective = median bar-PF over 3 random 500-bar slices (12% of the validation window). Trial ranking is mostly noise. |
| R-3 | HIGH | P | `multi_asset_allocator_env.py:242-299` | Allocator extra lag (section 2). **Verified in code.** |
| R-4 | HIGH | P | `envs/risk_shaping_wrapper.py:257-262` | The drawdown penalty (up to −5.0, fires on 58% of bars, 28% of reward variance) and DD termination depend on equity drawdown, which is **not in the observation** (`augment_obs: "off"` in every active config). From the agent's view that share of the reward is noise. |
| R-5 | MED | P | `data/multiscale_handler.py:141,158` | `log_return`/`parkinson` channels are left raw (std 0.0017/0.009) next to tanh-normalized channels (std ~0.4). At init they carry 0.09%/0.46% of the encoder's input signal. **Verified.** |
| R-6 | MED | P | `data/multiscale_handler.py:374-377` | The X2 fix maps on `base_ts − scale_ns` (start of bar). The decision is at bar close, so coarse features are one base bar staler than causally available on 25% (60-min) / 6% (240-min) of bars. |
| R-7 | HIGH | P | `eval/obs_noise.py:190-286` | The obs-noise gate perturbs the observed close **and** the execution price, which induces lag-1 autocorrelation −0.18/−0.47. It structurally fails trend-followers. Latent (`obs_noise_required: false`). |
| R-8 | HIGH | P | `envs/signal_gated_wrapper.py:190-194` | The hold action is not leverage-normalized. With `max_leverage=0.5` (sensitivity audit), 1 trade becomes 1,426 trades and +20% becomes −7%. Latent. |
| R-9 | MED | O | `config_utils.py:105-161` | `_prep_backtest_config` never applies the fee_schedule final tier, so the shared eval engine can evaluate at `taker_fee: 0.0` for 30 configs. |
| R-10 | — | — | pooled check | Pooled across folds (daily-resampled): SG-1-BTC clean WF Sharpe −0.1 to −2.1 (SE ≈1.5); GMGP1-BTC clean canary −4.8 to −6.0 (−61 to −65% in 114 days, cost-destroyed churn); de-leaked gold +6 to +9 (known stale-print artifact). **No honest positive RL result was rejected by R-1**, but the protocol could never have confirmed a realistic Sharpe-1–2 edge. |

*RL learning machinery and planted-signal test: see section 3.7.*

### 3.3 Cost and execution models (probes re-run; all reproduce to 4 d.p. before correction)

| ID | Sev | Dir | Location | Finding → corrected |
|---|---|---|---|---|
| C-1 | HIGH | P | `taiwan_intraday_momentum_falsification.py:52,175` | The gate uses 2.0 bp round trip; the file's own model says ~1.0 bp (tick + 0.002%/side tax + fee ≈ 1.0–1.3 bp). Recent-half net **−0.535 → +0.24 to +0.38** vs a 0.50 floor. The closest call in the record; NO-GO holds narrowly. **Verified.** |
| C-2 | HIGH | P | `vwap_intraday_falsification.py:42` | XAUUSD charged 2 bp/side vs realistic 0.5–1.0. Trend arm net **−2.91 → +0.35 @0.5 bp, ≈+0.06 @0.63 bp**. Hyper cost-sensitive; NO-GO holds. **Verified.** |
| C-3 | MED | P | `fx_majors_reversal_eval.py:40,46` | Charges the full Dukascopy spread per side on mid bars (sibling scripts halve it). Half-spread only: −3.83 → −1.68; with a realistic ECN commission (~0.35 bp/side) −3.13. NO-GO holds; the "7× unharvestable" figure is closer to 4×. **Verified.** |
| C-4 | MED | P | `gmgp1_btc_conviction_probe.py:63`, `gmgp1_btc_meanrev_maker_probe.py:70` | Taker cost 5.5 bp + 5 bp "slippage" per side on BTC perp (spread ~0.1 bp). Meanrev PF 0.415 → 0.657, maker 0.775 → 0.946. NO-GO holds. |
| C-5 | LOW | P | `vix_term_structure_eval.py:80`, `vix_voltarget_eval.py:58` | 2× the pre-registered cost (each transition charged a round trip). N4 0.591 → 0.625 (N4 still fails on DD −50% and on its shuffled-signal control); N5 0.369 → 0.408. |
| C-6 | MED | P | `smb_macd_scalp_eval.py:337,419` | Round-trip cost used as one-way on the `macd_pure_*` variants (2×). No verdict impact. |
| C-7 | LOW | P | `xsec_momentum_falsification.py:177,183` | 331 pre-rebalance zero-return days are included in Sharpe. TSMOM 0.601 → 0.621. No verdict change. |
| C-8 | MED | O | `funding_arb_carry_falsification.py:42,53,147` | Spot fee 1 bp (real 7.5–10); `basis = 0.0` removes all price risk. The probe's Sharpes are meaningless; the NO-GO is strengthened. |
| C-9 | LOW | O | `etf_pairs_arb_probe.py`, `vol_managed_overlay.py:47-53` | Costs generous (pairs) or under-charged ~12× (overlay). NO-GOs strengthened. |
| — | — | — | cleared | No double execution lag in any probe; `signals/costs.py` units correct; R1 illiquidity robust to 2× (max `pf_net_half` 0.90 < 1.10); futures-basis extra lag is deliberate and documented. |

### 3.4 Data layer (measured on the cached files)

| ID | Sev | Dir | Location | Finding |
|---|---|---|---|---|
| D-1 | HIGH | P/O | `crypto/data/crypto_loader.py:434-437,643-670` | The Hampel filter fabricates hourly mean reversion (section 2). Worst case: the Oct-2025 liquidation cascade is overwritten with pre-crash medians for hours. Affects every hourly crypto result (Sync-1H, Funding-Arb, cmgp1 probes, K1/K2); daily-rebalanced T2 is unaffected (−0.067 silver vs −0.087 bronze). **Verified.** |
| D-2 | HIGH | P | `scripts/clean_ohlcv.py:75-199` via all daily loaders | A cleaner calibrated for 1-minute bars (5% wick clamp) is applied to **daily** bars. It erases the crisis-day ranges: 2010 flash crash (SPY low 78.49 → 84.42), 2015-08-24, 2008, 2020-03, 2025-04. It also truncates ~95,000 Taiwan small-cap bars. Closes are never touched, so close-to-close TSMOM is unaffected; ATR/Parkinson/range features and high/low WQ101 alphas are distorted (Parkinson vol −9 to −40% in crisis months). **Verified.** |
| D-3 | MED | P | `crucible/data/taiwan_smallcap_panel.py:436`, `taiwan_panel_loader.py:402` | ADV = dividend-adjusted close × raw volume. Inflated up to 21× (median 2.3×) for high-payout names; 13 of the true top-50 are mis-ranked. Feeds ~20 WQ101 alphas and the size control. |
| D-4 | MED | P | `taiwan_xsec_momentum_eval.py:155` | Large-cap momentum on raw unadjusted prices (median dividend yield 4.65%, 458 ex-dates). Modest effect: IC 0.072 → 0.075. |
| D-5 | MED | N | `data/taiwan_smallcap/prices.parquet` | 777 tradeable bars with open=0 & low=0. 15,270 beyond-limit returns from unadjusted capital reductions and bad prints. |
| D-6 | MED | P (latent) | `taiwan_smallcap_panel.py:427` | `dividend_adjusted = div_path.exists()` silently falls back to raw prices (−482 bp mean on ex-dates). The provenance note says "RAW unadjusted" even when adjusted. |
| — | — | — | cleared | Dukascopy divisors correct for all 15 instruments; ETF panels are adjusted and never mixed with raw; US-equity PIT membership and ADV are causal; bar-label conventions uniform; small-cap probes ran on adjusted data. |

### 3.5 Crucible funnel

The IC estimator and book builder recover a planted signal **exactly** (IC, gross Sharpe and cost drag match independent references). The losses are all in the verdict layer, on top of the known power problem (`project_crucible_zero_alpha_root_cause`).

| ID | Sev | Dir | Location | Finding |
|---|---|---|---|---|
| X-1 | HIGH | P | `crucible/lockbox/incubation.py:187-195`, `lockbox.py:105-107` | The lockbox, the last gate before human review, scores `annSharpe(b_aug − b_base)`, which equals `α·(r_c − b_base)`: a long-candidate/short-base stream, not ΔSR. A low-vol diversifier with ΔSR +1.0 scores +0.11 and is REJECTED (terminal). This is the same seal already removed from `marginal_t`, left in place here. **Verified.** |
| X-2 | HIGH | P | `signals/eval_harness.py:421` | `_ls_weights(min_names=10)` is hardcoded. On panels with <10 names (9 FX majors, 10-ETF Taiwan) the book is identically zero and the card claims "loses money at zero cost". No recorded FX verdict used this path. |
| X-3 | HIGH | P | `library/operators.py:28,33,40` + union-grid panels | `rank`/`scale`/`indneutralize` include untradeable names: 49.9% of priced us_equity cells are inactive. 37/84 WQ101 alphas change order; alpha016 t 2.77 → 3.05 (crosses the 3.0 bar). |
| X-4 | MED | P | `generation/evolve.py:632`, `fitness.py:121-133` | The holdout combiner restarts its 63-bar warm-up on holdout rows only, so the holdout book is largely equal-weight. ΔSR is understated 20–46% at H=126–252. |
| X-5 | MED | P | `generation/fitness.py:186-201` | CPCV silently drops paths when embargo ≥ group size: 10/15 paths lost on cross_asset H=126. |
| X-6 | MED | P | `generation/fitness.py:393-395` | Turnover penalty soft cap 12/yr is below what a random rank book pays at H=21 (~16/yr). The median penalty of 0.14 exceeds the 0.10 uplift floor. |
| X-7 | MED | P | `generation/evolve.py:126-135` | Warm-up bars enter as 0.0 returns in the train pre-filter; ΔSR shrinks ~20%. |
| X-8 | MED | O | `generation/evolve.py:677` | `n_holdout_tested` over-counts (includes degenerate candidates). |
| — | — | — | cleared | Forward-label timing; rebalancing already at the hold horizon (no 21× over-trading); cost model exact; all 17 time-series operators causal and correct vs WQ101; holdout embargo; BH/BHY. `pytest tests/signals tests/crucible`: 710 passed. |

### 3.6 Test suite and repo hygiene

| ID | Sev | Finding |
|---|---|---|
| T-1 | LOW | Full suite: 11 order-dependent failures (deepscalper, PPO, DSAC train steps). Cause: `sharpen/alphaseek/trainer.py:109,164` calls `torch.set_grad_enabled(False)` process-wide and never restores it. All 11 pass in isolation; reproduced by running `tests/alphaseek/test_trainer.py` first. No production impact (separate processes). The FINRL.md "baseline EMPTY" covers only 3 test directories and excludes `tests/alphaseek`. |
| T-2 | LOW | `ruff`: `sharpen/` clean; `scripts/research` 88 style hits, none hiding a dropped correction. |
| T-3 | — | Reporting-only metric errors: `analytics/wandb_evaluator.py:163` annualizes with √525600 regardless of bar size (×3.87 Sharpe on 15-min); `pyfolio_analyzer` assumes 24/7 bars (×1.23 on Gold/SPX); Omega adds 1. |

### 3.7 RL learning machinery and planted-signal test

**The SAC learner is correct and does learn.** Verified by execution:
- Buffer `(s,a,r,s′)` alignment: 3,936/3,936 transitions exact, 0 misaligned.
- Truncations bootstrap; phantom auto-reset transitions are filtered.
- Critic target, twin/target networks, Polyak and tau auto-scale, the temperature loss sign and the tanh log-prob correction all match Haarnoja et al.
- Every deploy-gating inference path is deterministic, with `eval()` on (the dropout question is cleared).

**Real-data capability test.** Six shipped `gmgp1-gold-steadystate` L1 checkpoints (the Stage-2 set, recorded test PF 2.45–2.66, later promoted to the paper ensemble), deterministic, on the held-out test window (3,760 bars):

| | corr(action, −log_return) | gross Sharpe, close-marked | gross Sharpe, **mid**-marked |
|---|---|---|---|
| one-line oracle | 0.95 | 9.52 | — |
| 6 SAC seeds | +0.357 to +0.370 | 5.98 to 8.65 | **1.02 to 3.33** |

The agents reach 63–91% of the oracle out of sample, but what they learned is **bid-ask bounce** (next table, M-1).

| ID | Sev | Dir | Location | Finding |
|---|---|---|---|---|
| M-1 | HIGH | O | `data/multiscale_handler.py:135-140`, `envs/continuous_swing_env.py:318,410` | The gold file's `close` sits on the bid 38.5% and on the ask 38.5% of minutes. The env marks P&L on `close` and never reads the file's true LOB `mid_price`. 15-min AC(1): close −0.297 vs mid −0.065. Mid-marking removes 70–86% of the trained policies' gross edge; at 2 bp they lose 8–15%. Together with the stale prints already on record, this is what the gold "edge" was. Affects `gc_2025_lob1_1min_stitched.parquet`, the file behind the gold L1 multiseed, the live-paper ensemble and `live_gmgp1_gc_ib.yaml`. BTC, EURUSD and XAUUSD files are clean. **Verified.** |
| M-2 | MED | N | `continuous_swing_env.py:460-468`, `scripts/clean_ohlcv.py:653` | PF-XCHECK marks at `(H+L)/2` of the same trade prints, which is *more* bounce-contaminated than close (AC1 −0.47). `rederive_downstream` overwrites `mid_price` with `(H+L)/2`. The invariant is blind to the defect it exists to catch. |
| M-3 | LOW | O (look-ahead) | `continuous_swing_env.py:321-360` | The ATR position cap is evaluated after `current_atr` is loaded from bar *p*, the bar whose return the new position earns. That is a small LEAK-2 violation: cap indicator ↔ \|return\| corr +0.082 vs +0.050 causal; fires on 11% of bars. **Verified in code.** |
| M-4 | MED-HIGH | P | `envs/dsr.py:70-98` + `episode_length: 500` | The differential-Sharpe EMA resets every episode, but `eta=0.001` needs ~2,300 bars to warm up. The first 25 bars (5% of steps) carry 32% of the critic's squared-target energy, 26–60% of them hit the ±10 clip, and 44.5% of reward variance is not a function of the step return. Evaluation never sees this regime (`episode_length=0`). |
| M-5 | MED-HIGH | P | `hpo/evaluate.py:136,233,309-313`, `hpo/objective.py:479-491` | Each HPO trial is scored by bar-PF on one 500-bar episode (median of 3). Simulated at production reward scale, a trial with a real IC 0.05 edge wins a 50-trial search 28–31% of the time (IC 0.02: 7–9%). **Measured fix:** scoring by Sharpe over the full ~5,600-bar validation window lifts this to 92% (IC 0.05) and 50% (IC 0.03). |
| M-6 | MED | P | `hpo/objective.py:152-154`, `training/sac_trainer.py:128-135` | Search-space floors: 19% of trials get a target-net time constant ≥4× the trial length, and 26% cannot move `log α` by >0.5 all trial. About 40% of trials are structurally crippled. |
| M-7 | MED | P | `agents/sac/networks.py` | 708k parameters vs 15,883 training bars. In a supervised probe on an IC-0.156 signal the production encoder reaches test IC 0.08 (train 0.34); linear OLS on the same inputs reaches 0.145. Not shown to bind in the RL runs. |
| M-8 | LOW-MED | N | `hpo/objective.py:181-190`; `run_full_pipeline.py:233-241,276`, `distributed_hpo_coordinator.py:739-750,789` | `max_leverage` is sampled but missing from both routing tables, so a leverage-axis HPO raises `ValueError` after all trials are spent. |
| M-9 | MED | N | `tests/` | No test asserts SAC learning math: Bellman target, log-prob correction, temperature sign, Polyak direction or buffer alignment. |

| M-10 | MED | P | learner dynamics | At a small budget SAC barely learns a planted IC-0.15 edge (ladder below). The critic's action preference is mostly spurious: the Q spread across actions is 0.81 in the zero-edge control vs 0.96 at IC 0.15. Q drifts down to about −20, consistent with compounding twin-min underestimation. |
| M-11 | LOW-MED | P | `agents/sac/sac_agent.py:494-507` | Trained critics reach \|Q\| ≈ 20, where the bf16 rounding step (0.125) is about the size of the signal-driven Q-advantage (~0.15). Rounding is zero-mean and averaged over a 512 batch, so this is a residual risk, not a proven failure. Fix: keep the Q head in fp32. |

**Synthetic planted-signal ladder.** An AR(1) edge is planted in the 15-min returns and run through the real handler → env → `SACTrainer` → `SACAgent` at CPU budget: 34k gradient steps, shrunk network, one seed per arm, fp32. Oracle Sharpe is 20.2 at IC 0.15 and 6.0 at IC 0.05. The Sharpe SE per arm is about ±1.33.

| arm | IC | change vs shipped | Sharpe | corr(action, signal) |
|---|---|---|---|---|
| A | 0.15 | shipped defaults | +1.48 | +0.120 |
| B | 0.15 | initial α 0.01 | −1.04 | −0.008 |
| C | 0.15 | α 0.01, no deadband, P&L reward | +1.70 (long bias, not timing) | +0.015 |
| D (control) | 0 | α 0.01 | −1.16 | −0.066 |
| E | 0.05 | α 0.01 | +0.28 | −0.022 |

Only the shipped-default arm shows alignment with the signal, about 2× the control's spurious alignment. No arm is statistically distinguishable from the control on Sharpe. In the supervised probe, the production encoder reaches test IC 0.08 on an IC-0.156 signal, where linear OLS gets 0.145 (M-7).

**GPU planted-signal ladder (full production budget), 2026-09-23/24.**
- **Setup:** 8 arms on 3× RTX 4090. Production geometry and hyperparameters, bf16 + `torch.compile`, 3M steps, 599k gradient steps and ~13 h per arm, frictionless.
- **Scoring:** deterministic evaluation on a 2-year held-out window (70k bars), with the linear oracle and a flat policy run through the same env.
- **Code:** harness in `scripts/research/rl_planted_signal/`, configs in `configs/rl_planted_signal/`, collated table at `results/rl_planted_signal/collated/`.

| arm | planted IC | change vs production | SAC Sharpe ± SE | oracle Sharpe | SAC / oracle | corr(action, signal) | test bars |
|---|---|---|---|---|---|---|---|
| P15_s1 | 0.15 | — | **+3.42 ± 0.71** | 26.66 | 13% | +0.084 | 70,081 |
| P05_s1 | 0.05 | — (seed 1) | +0.83 ± 0.71 | 9.29 | 9% | −0.063 | 70,081 |
| P05_s2 | 0.05 | — (seed 2) | −1.88 ± 0.99 | 9.29 | — | −0.067 | 36,036 (30% DD stop) |
| X05_ep5000 | 0.05 | episode length 5000 (DSR warm-up fix) | −2.25 ± 1.02 | 9.29 | — | −0.069 | 33,902 (DD stop) |
| X05_pnl | 0.05 | raw P&L reward | −2.18 ± 1.21 | 9.29 | — | −0.158 | 23,756 (DD stop) |
| X05_fp32 | 0.05 | fp32 instead of bf16 | −1.57 ± 1.01 | 9.29 | — | −0.045 | 34,010 (DD stop) |
| P02_s1 | 0.02 | — | +0.55 ± 0.71 | 3.98 | 14% | −0.106 | 70,081 |
| **P00_s1 (control)** | 0 | — | −1.76 ± 0.94 | 0.62 | — | −0.141 | 39,700 (DD stop) |

**Reading:**
- **Strong edge.** At full budget the production stack learns a planted IC-0.15 edge only weakly: 13% of the oracle's Sharpe, with action–signal correlation 0.08 against the oracle's 0.95.
- **Weak edges.** At IC 0.05 and 0.02 it learns nothing. All six arms have a *negative* action–signal correlation, like the zero-edge control. Four of the five IC-0.05 arms hit the 30% drawdown stop on a frictionless test. The two positive Sharpes (+0.83, +0.55) are within about one SE of zero and carry the wrong sign of correlation.
- **Single-change fixes.** Fixing the DSR warm-up, switching to P&L reward and running the critic in fp32 each fail to rescue IC 0.05.
- **Summary.** The stack overfits noise with a consistent spurious sign (M-7, M-10) instead of learning a weak linear edge that a one-line rule turns into Sharpe 9.3. A realistic per-bar edge is IC 0.01–0.05, so the shipped stack could not have found one. That is consistent with the record: every "edge" it ever found was a strong artifact (look-ahead, stale prints, bid-ask bounce).

**Caveats:**
- One seed per rung except IC 0.05, which has two seeds plus three variants.
- The planted signal lives in the raw-scale `log_return` channel, the one R-5 shows is 240× under-scaled. The same information is also present through `Δclose_z` (corr 0.87), so the result stands for the stack as shipped. But no arm isolated input scaling, so a standardized-channel arm is the natural next test.

---

## 4. What the NO-GO record can and cannot rule out

Monte Carlo of the gates as coded (20,000 paths per cell; every leg computed on the same path, so joint probabilities respect dependence; fat tails change nothing material). Full table: 637 rows × {normal, t5}.

**Minimum true annualized net Sharpe to pass**

| Sample | Gate stack | 50% pass | 80% pass | P(pass) at true SR 0.5 | False pass at SR 0 |
|---|---|---|---|---|---|
| 20.4 y daily | TAILWIND stack | 0.66 | 0.89 | 0.23 | 0.10% |
| 18.65 y daily | 7-leg generic | 0.82 | 1.11 | 0.12 | 0.04% |
| 18.65 y daily | multispeed-style | 0.84 | 1.08 | 0.07 | 0.00% |
| 5.1 y holdout | 7-leg | 1.41 | 1.89 | 0.02 | 0.16% |
| 3.6 y holdout | 7-leg | 1.69 | — | 0.02 | 0.14% |
| 53-day OOS | any | unreachable | unreachable | ≤0.06 | — |
| RL bar-PF ≥ 1.10 (15-min) | single leg | ≈ SR 6–8 | — | ≈0 | ≈0 |

Single legs at 18.65 y (P(pass) at SR 0.5): PSR≥0.95 0.69 · bootstrap CI excludes 0 0.57 · HLZ t≥3 0.20 · DSR≥0.95 (N=24, measured) 0.22 · DSR (N=23, assumed 1/n) 0.08 · 4/4 subperiods 0.54 · PF≥1.10 in ≥3/4 folds (daily) 0.19.

**Reading:** the platform's best measured edge (SR ~0.6) passes its own stacks 9–37% of the time on the longest sample available, and ~2% on the 3.6–5.1-year holdouts Crucible uses. A clean record of 76 NO-GOs is what this machinery produces whether or not modest edges exist. It is strong evidence against SR > ~1 edges and weak evidence about everything below.

---

## 5. What still stands

- **Look-ahead:** nothing of the X2 kind; tripwires pass; sim↔live parity holds at mid-interval. The ATR-cap look-ahead (M-3) is small and not covered by any tripwire.
- **RL (single-asset directional):** clean runs are negative pooled across folds (SG-1-BTC ≈ −0.1 to −2.1; GMGP1-BTC canary ≈ −5 to −6 annualized, cost-destroyed churn). The falsification stands on the evidence; the gates were not what killed it. Its scope is "nothing this stack can learn", and the GPU ladder shows the stack cannot learn a planted IC-0.05 edge at all (3.7).
- **Cost-killed probes:** FX reversal, EURUSD 3h, ETF pairs, TAIEX basis, R1 illiquidity, BTC conviction/maker, gold VWAP. All still fail at realistic cost. TX intraday momentum is the narrowest (+0.24 to +0.38 vs 0.50).
- **Crypto TSMOM (T2), Taiwan large-cap momentum (LOGGED), value, carry (OOS reversal), options VRP:** unchanged by any finding here.
- **SPY vs TSMOM** (the "rather buy SPY" comparison): the memo's "TSMOM 0.60 vs stocks 0.56, blend 0.79" mixes total return and excess return. On one consistent (total-return) basis, 2008–2026: TSMOM 0.601, SPY 0.647, 50/50 blend 0.849. Max drawdown at a matched 10% vol: −22.5% / −32.3% / −16.9%. SPY alone did beat TSMOM alone; the blend beat both.

---

## 6. Recommendations (ordered by leverage)

1. **Fix the verdict math before running more research.**
   - DSR: deflate only in-sample best-of-N Sharpes; use PSR for genuinely out-of-sample numbers; set `use_effective_n: true`; match the trial-pool estimand to the observed one.
   - Replace AND-stacks with one pre-registered primary test at a stated joint false-positive rate, plus diagnostics. Publish power at SR 0.5 for every gate before running it.
   - Report intervals for P(pass) and uplift. Use the marginal admission rule (S_B > ρ·S_A).
   - Keep pre-registered verdicts binding; route contested GOs (VIX) to forward incubation.
2. **Crucible:** replace the lockbox statistic with ΔSR or residual IR (X-1); use PSR when a batch has one pre-registered hypothesis (S-9); pass `min_names` through (X-2); mask inactive names before cross-sectional operators (X-3); warm the holdout combiner from the full panel (X-4); fix CPCV path dropping (X-5); scale the turnover cap with the hold horizon (X-6).
3. **RL protocol:**
   - Mark P&L at the LOB mid where the file has one, and make PF-XCHECK use it (M-1, M-2). Add a variance-ratio / on-touch-fraction check to `clean_ohlcv.py`.
   - Retire bar-level PF gates in favor of daily-aggregated Sharpe with CIs over pooled OOS (R-1).
   - Add a plausibility tripwire: any single-asset intraday result above annualized Sharpe ~3 is presumed a leak or data artifact until audited. Every such result in this record was one.
   - Evaluate HPO trials on the full validation window (R-2, M-5), and tighten the tau/`lr_alpha` search floors (M-6).
   - Carry the DSR EMA across episodes or lengthen episodes (M-4). Put drawdown in the observation or drop the DD penalty (R-4). Standardize the return channels (R-5).
   - Make the ATR cap causal (M-3). Add SAC-math tripwire tests (M-9); the four written for this audit are drop-in.
   - The GPU ladder (3.7) shows the shipped stack cannot learn IC ≤ 0.05. Do not spend RL budget on signal *discovery* until a planted IC-0.05 edge is learnable. Candidates to test on the same ladder, one at a time: standardized input channels (R-5), a far smaller network or linear-policy baseline (M-7), γ ≈ 0.9 (A2-12), and the HPO/search-space fixes (M-5, M-6). Until then, find signals with linear models and keep RL to sizing and execution on top of them, which is the project's own "linear core first" direction.
4. **Data:** rebuild the crypto silver layer with the filter on returns (or disabled) and re-read every hourly crypto verdict (D-1). Stop applying the 1-minute wick clamp to daily bars (D-2). Compute ADV from raw close × volume (D-3). Make dividend adjustment fail closed (D-6).
5. **Costs and timing:** fix C-1 through C-6 and the allocator lag (R-3, which also feeds the paper executor).
6. **Re-run only what these change:** TAILWIND R1 DSR, K1/K2 on bronze, the Taiwan small-cap cards with overlap-aware t, TX intraday at 1.0–1.3 bp, and the sleeve admissions under the marginal rule.

---

## 7. Incident during this audit

One review called `load_cross_asset_panel("2007-01-01", ...)`, which re-fetched from yfinance and **overwrote three gitignored cache files**: `data/raw/cross_asset_panel/ohlcv_daily.parquet`, `ohlcv_daily_raw.parquet`, `ohlcv_daily.manifest.json`. They were 2008-01-02 → 2026-06-30 and are now 2007-01-03 → 2026-09-22 (same pipeline, superset window). No tracked file or result changed. Back-adjustment rescales earlier prices uniformly, so returns inside the old window are unchanged up to vendor revisions, but the old bytes cannot be recovered. Scripts that read this path directly and will now see a longer window: `multispeed_tsmom_eval.py`, `turn_of_month_eval.py`, `vix_voltarget_eval.py`, `session_decomposition_eval.py`, `planted_sweep.py`, `fable/fable_data_audit_etf.py`. To rebuild the old window: `load_cross_asset_panel("2008-01-01", end="2026-07-01")` (network).

---

## 8. Audit checklist (Tier-1 phases, run whole-codebase)

- Phase 1 lint: `ruff check sharpen` PASS; research scripts style-only.
- Phase 2 tests: full suite green except T-1 (order-dependent, root-caused). `tests/signals` + `tests/crucible` 710 passed.
- Phase 3.7 leakage: 45/45 tripwires pass. One small uncovered look-ahead (M-3, ATR cap).
- Tier-2: not a capital promotion. If TAILWIND is reconsidered after S-1, it needs the Tier-2 lifecycle audit on the challenge config.

Reproduction scripts and outputs live in the session scratchpad; the key ones are reproducible from the repo: the DSR sensitivity (re-running `audit_tailwind_book.py` with the input swapped), `crypto_xsec_eval.py --data data/crypto_cache/bronze/ohlcv.parquet`, and the bar-PF census (daily-resample the saved `results/**/*trajectory*.parquet`).

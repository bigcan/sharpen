# Negative results

Every strategy this project built, measured and closed, with the number that closed it.

**How to read this.** A NO-GO here means *this* implementation, on *this* data, at *this* sample
size and cost, did not clear its gate. It is not a proof that the underlying idea can never work.
Where a result was bounded by statistical power rather than refuted, the row says so. Reopen any
of these with new evidence (a different venue, better data, a structurally new signal), not with a
re-run.

**Evidence.** Each row points at what ships in this repository: a research document under
[`docs/research/`](docs/research/README.md), or the script that produced the number under
`scripts/research/`. Run artifacts (`results/`) and the raw R&D log are not published, and several
early pre-registrations lived in internal notes that are not included. A row marked *not shipped*
is recorded here from those notes and cannot be re-run from this repository as published.

**Bugs are listed separately.** Three results were manufactured by defects rather than markets;
see [docs/LEAKS_FOUND.md](docs/LEAKS_FOUND.md). The rows below are measured on the fixed code.

**The one survivor** is not in this file: cross-asset time-series momentum, net Sharpe ≈ 0.60, and
the book built on it (TSMOM plus a betting-against-beta sleeve), which failed deflation at 0.896
against a 0.95 bar. See [`tailwind_v1_R1_dsr_pbo_2026-07-01.md`](docs/research/tailwind_v1_R1_dsr_pbo_2026-07-01.md).

---

## Summary

| Family | Closed | What killed it, mostly |
|---|---|---|
| [1. Directional RL and its rescues](#1-directional-rl-and-its-rescues) | 14 | No edge after removing leaks; rescues added nothing |
| [2. Options and volatility premium](#2-options-and-volatility-premium) | 7 | Data contamination, lucky subsamples, tail risk |
| [3. Arbitrage, carry and market making](#3-arbitrage-carry-and-market-making) | 7 | Real gross structure that costs or execution cannot harvest |
| [4. Equity cross-section and factors](#4-equity-cross-section-and-factors) | 12 | Survivorship, sector beta, arbitraged to cost |
| [5. Sleeves and overlays on the momentum book](#5-sleeves-and-overlays-on-the-momentum-book) | 11 | Dilutive, correlated, or one-subperiod gains |
| [6. Retail technical and intraday](#6-retail-technical-and-intraday) | 10 | Cost-killed; win rate is not edge |
| [7. Pre-registered free-data probes](#7-pre-registered-free-data-probes) | 11 | Artifacts, deflation, unharvestable effects |
| [8. Automated mining substrates](#8-automated-mining-substrates) | 4 | Underpowered at the horizons costs force |
| **Total** | **76** | |

Counts are rows, and a row is one pre-specified question with its own verdict. Some are variants of
one idea (three arbitrage mechanisms, six Taiwan small-cap signals); they are listed separately
because each was tested and closed on its own evidence.

---

## 1. Directional RL and its rescues

Reinforcement-learning traders, mostly single-asset directional SAC on multi-timeframe OHLCV
features, plus every attempt to rescue them.

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| SG-1-BTC (signal-gated SAC) | Volatility-gated swing trading on BTC has an edge | Cost-corrected walk-forward, 8 folds; ≥7/8 folds at PF ≥ 1.10 | **Leak-borne.** PF 1.689 → 1.016 once the look-ahead was fixed; 0/8 folds; every seed loses money | [strategy audit](docs/research/sg1_btc_strategy_audit_2026-05-29.md) · [LEAKS_FOUND §1](docs/LEAKS_FOUND.md) |
| GMGP1-BTC clean canary | De-leaked SAC swing on BTC 15m is profitable | HPO → 4-fold WF → stress gates; net WF PF < 1.1 kills | Best-solo median PF **0.977**; all 20 solo PFs 0.834–0.997; worst DD −33%. Edge absent even before fees | [lifecycle audit](docs/research/gmgp1-btc_deep_lifecycle_audit_2026-06-03.md) · [reinvestigation](docs/research/gmgp1_btc_canary_reinvestigation_2026-06-09.md) · `reverify_gmgp1_btc_canary.py` |
| GMGP1-BTC clean re-baseline | A fresh HPO on clean code finds what the canary missed | Protocol v2 HPO probe on de-leaked env | NO-GO: directional single-asset RL is unprofitable, not merely unproven | [runbook](docs/research/gmgp1_btc_clean_rebaseline_runbook_2026-07-19.md) |
| GMGP1-Gold | The gold policy's edge survives the leak fix | De-leaked 4-fold WF, same config as the paper ensemble | Edge real but regime-dependent, decays to breakeven; 2/4 folds, uplift gate fails | *not shipped* |
| GMGP1-XAUUSD regime gate | A regime detector can gate a leak-free price signal | Causal regime separation test | Detector backward-looking and never accuracy-validated; exact use failed 9/9 | [R0 regime spec](docs/research/r0_regime_spec_2026-06-02.md) · `r0_regime_separation.py` |
| GMGP1-SPX500 (long/short vs long-only) | Swing RL on the S&P 500 index adds skill | A/B across all protocol stages + WF multiseed | Both NO-GO. What survived was beta; an apparent regime effect disappeared under multiseeding | *not shipped* |
| PPO-GAE vs SAC | The failure is SAC-specific | Same env, PPO with GAE | Same no-edge result: the failure is the signal, not the algorithm | [PPO-GAE screen](docs/research/ppo_ge_gmgp1_btc_screen_2026-06-19.md) · `ppo_ge_gmgp1_btc_screen.py` |
| High-confidence reward | Rewarding only confident trades exposes a conditional edge | Step-0 conditional-alpha probe (CPU) | No directional signal to rescue. The one positive-IC signal, short-horizon reversal, fails even as a passive maker with rebate | `gmgp1_btc_conviction_probe.py` · `gmgp1_btc_meanrev_maker_probe.py` |
| BTC loss-regime overlay | Avoiding loss regimes (vol, volume, weekend, hour, loss streaks) makes it profitable | Discovery / confirmation split | Loss is diffuse; the only OOS-stable slice is BTC beta and seasonality | `gmgp1_btc_loss_regime_discovery.py` · `…_confirm.py` |
| Risk overlays on GMGP1 and SG-1 | Stops, take-profit, time stops, vol targeting or DD throttles rescue the P&L | 54 arms, each re-scored with the edge sign flipped | **0/54 reach PF ≥ 1.** Every helpful feature reverses when the sign flips; stops are cost-killed | `risk_overlay_lab.py` |
| PRISM regime model | Regime features (L1) or regime sizing (L2) improve a gold policy | Gate test of both layers | Both layers fail their gates; a dormant daily→intraday look-ahead also found | [PRISM eval spec](docs/research/prism_regime_eval_spec_2026-06-18.md) |
| Sync-1H multi-asset crypto RL | Portfolio RL over 20 crypto perps at 1h | Pilot runs | Pilot v1 **−62.84%**, v2 **−34.30%**; workstream closed | *not shipped* |
| AlphaSeek DQN ensemble | A crypto-contest DQN ensemble is tradeable | Fee audit of all 12 checkpoints | Median PF **0.07–0.27** at realistic fees vs 2.02 at contest fees; a v3 redesign scored PF 0.00 | [fee audit report](docs/alphaseek_fee_audit_report.md) |
| BALLAST v1 (long-only S&P 500 RL) | RL managing a long-only S&P 500 core beats SPY | Free-data build, survivorship bound from index-exit cohort | The whole +0.104 margin over SPY is survivorship inflation (bound +0.237); OOS never opened | [design](docs/research/ballast_v1_design.md) · `ballast_g1b_inflation.py` |

## 2. Options and volatility premium

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| Crypto options VRP (short vol) | Systematic short-vol on crypto options earns a premium | Tail audit → cost gate → paper wiring → decontamination re-audit | **Cleared, then rescinded.** The 0.61 deploy anchor was USDC-linear data contamination; the clean real chain was negative across every structure | [lifecycle audit](docs/research/options_vrp_linear_core_deep_lifecycle_audit_2026-06-11.md) · [re-audit](docs/research/options_vrp_decontamination_reaudit_2026-06-23.md) · `options_vrp_falsification.py` |
| Taiwan TXO VRP, delta-hedged | Delta-hedged index-option VRP on TAIFEX pays | Scout → validation → pre-registered 2002–2018 backward extension | The 2019–2025 +0.63 was a lucky subsample: extension Sharpe **+0.26**, DSR at N=50 **0.59** | [backward extension](docs/research/txo_vrp_backward_extension_2026-07-02.md) · `taiwan_txo_vrp_extension.py` |
| 0DTE SPX/XSP option selling | High-win-rate same-day selling works for small accounts | Literature + cost-inclusive simulation | Negative net of costs, unconditionally | *not shipped* |
| "Top 3 0DTE strategies" | Credit spread, iron condor and calendar each have an edge | Rules as published | All three NO-GO | `zerodte_3strategies_eval.py` |
| Wheel (cash-secured put → covered call) | Wheel income is alpha | Canonical and single-name simulation vs holding the index | Down-scaled, capped equity beta; the index version roughly matches SPY risk-adjusted; protective puts and collars never help | `wheel_canonical_eval.py` · `wheel_singlename_sim.py` |
| Gamma / dealer positioning (GEX) | Dealer gamma predicts TX intraday momentum | Levels, conditioning, then six rescue filters | GEX levels NO-GO; conditioning NO-GO; 0/6 rescues | `taiwan_intraday_momentum_gamma_conditioned.py` · `…_rescue_filters.py` |
| VIX front-vs-mid carry (N4) | Term-structure carry on VIX futures | Pre-registered, 4 subperiods, tail gate | +0.734 gross, 4/4 subperiods, but **DD −50.1%**, worst day −19.9% | [prereg](docs/research/vix_term_structure_preregistration_2026-08-01.md) · `vix_term_structure_eval.py` |

## 3. Arbitrage, carry and market making

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| Crypto market making (Avellaneda-Stoikov) | A-S quoting earns spread on liquid crypto books | Baseline on recorded books, PF ≥ 1.05 | Failed at **zero fees** on Binance BTC/SOL/XRP and at 1.5 bps maker on Hyperliquid BTC; books are tick-pinned, fills are adverse | *not shipped* |
| Funding-rate arbitrage (standalone) | Delta-neutral perp funding carry is a strategy | DSAC pipeline to walk-forward, then two un-shelf investigations | WF 4/8 windows profitable, gates fail; funding regime has not recovered; cross-sectional dispersion variant also falsified | [un-shelf 06-06](docs/research/funding_arb_unshelf_2026-06-06.md) · [06-19](docs/research/funding_arb_unshelf_2026-06-19.md) · `funding_arb_carry_falsification.py` |
| ETF same-index pairs | Two ETFs on one index mean-revert | Pre-registered spread probe | Valid gross signal, killed by taker cost | [prereg](docs/research/etf_pairs_arb_preregistration_2026-07-12.md) · `etf_pairs_arb_probe.py` |
| TAIEX cash–futures basis | The basis reverts profitably | Stage 0 daily, Stage 1 intraday | Gross Sharpe +1.37, but the cash index is stale: the tradeable futures leg is ≈ 0 | [prereg](docs/research/futures_basis_arb_preregistration_2026-07-12.md) · `futures_basis_stage1_intraday_probe.py` |
| ETF vs underlying (0050 vs TSMC + basket) | ETF mispricing against constituents | Pre-registered probe | No gross edge | [prereg](docs/research/etf_underlying_arb_preregistration_2026-07-12.md) · `etf_underlying_arb_probe.py` |
| Rates carry sleeve | Carry adds to the momentum book | Honest out-of-sample re-evaluation | The in-sample +0.21 uplift **reverses** out of sample | [carry spec](docs/research/carry_falsification_spec_2026-06-12.md) · [Tier-2 audit](docs/research/momentum_rates_carry_tier2_audit_2026-06-18.md) · `carry_falsification.py` |
| Commodity carry sleeve | Energy roll-spread carry diversifies | Admission test against the book | Uncorrelated but dilutive; broad carry data unavailable free | `eval_commodity_carry_sleeve.py` |

## 4. Equity cross-section and factors

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| Cross-sectional equity momentum | Large-cap momentum is deployable | Survivorship-controlled, cost-inclusive | Correctly signed and survivorship-robust, but arbitraged to about cost and regime-fragile | `signal_momentum_confirmed.py` · `xsec_momentum_falsification.py` |
| Low-turnover reversal + distress filter | Filtering distress rescues the reversal lead | The named delisting-exit escape hatch, tested | Reversal edge and survivorship exposure are the same trade | `signal_lead_distress_filter.py` |
| PEAD, post-pandemic | Post-earnings drift survives 2023–2026 | Event study + literature | Dead in the investable universe; survives only in untradeable microcaps | *not shipped* |
| XLG entry timing (10 rules) | Timing rules beat buy-and-hold on a mega-cap ETF | 2005–2026, survivorship-clean | 8/10 are cash drag; best active IR negative, DSR 0.043; two survive only as drawdown insurance | `xlg_alpha_overlay_backtest.py` · `xlg_pit_validation.py` |
| RL fundamental reweighting of XLG | RL reweighting beats cap weight | Stage-0 linear falsification first | The reweight is beta, not alpha | `xlg_megacap_ic_gate.py` · `xlg_sleeve_robustness.py` |
| Value factor sleeve | Cross-asset value adds to momentum | Pre-registered falsification | Pooled net Sharpe **−0.364** vs momentum +0.545; decayed after 2011; combining halved Sharpe and roughly doubled drawdown | [spec](docs/research/value_falsification_spec_2026-06-18.md) · `value_falsification.py` |
| "ETFs that beat SPY" | Long-run ETF outperformance is persistent skill | 354 funds, factor attribution, FDR | Concentrated tech/semiconductor beta; **0/351 survive BH-FDR**; the lookback selection rule has zero predictive power | [research](docs/research/etf_outperformance_research_2026-08-14.md) · `etf_outperformance_*.py` |
| Taiwan large-cap momentum | Cross-sectional momentum on TWSE | First pass, then size control | The first-pass PROMISING was a size confound | `taiwan_xsec_momentum_eval.py` |
| Taiwan small-cap P1 (monthly revenue drift) | Revenue surprises drift in small caps | Round-1 alt-data probe; longer hold to beat the cost wall | The only PROMISING ever recorded; the 63-day hold rescues the cost wall but not the Sharpe (net ≈ 0.52 flat) | [prereg](docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md) · [lower turnover](docs/research/taiwan_smallcap_p1_lower_turnover_2026-07-31.md) |
| Taiwan small-cap price signals (R1 reversal, R2 IVOL) | Classic price anomalies in small caps | Pre-registered | Frictionless Sharpe **−0.182** and **−0.627** | [prereg](docs/research/taiwan_smallcap_price_probes_preregistration_2026-07-31.md) · `taiwan_smallcap_price_eval.py` |
| Taiwan small-cap flows and short interest (Q1, Q2, S1) | Institutional flow and short interest predict returns | Pre-registered with committed sign | Q1 and Q2 **falsified on sign** (t −6.57 against a committed +); S1 CI straddles zero and flips sign at 63d | [flow prereg](docs/research/taiwan_smallcap_institutional_flow_preregistration_2026-07-31.md) · [short interest](docs/research/taiwan_smallcap_short_interest_preregistration_2026-07-31.md) |
| Illiquidity premium (R1) | Illiquid assets carry exploitable structure | 92 asset × horizon cells, cost-inclusive | Real structure (TRY, ZAR, mid-cap alts), but **0/92 cells survive cost** | [spec](docs/research/r1_illiquidity_probe_spec_2026-06-12.md) · `r1_illiquidity_probe.py` |

## 5. Sleeves and overlays on the momentum book

Candidates to add to, lever or time the surviving TSMOM book. The admission bar is marginal: a
sleeve must raise the book's Sharpe given its correlation.

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| Country TSMOM sleeve (T1) | Country-ETF trend adds breadth | Closed-form admission rule | Sharpe 0.409 but correlation 0.693: dilutes | [prereg](docs/research/country_tsmom_preregistration_2026-07-31.md) · `country_tsmom_sleeve.py` |
| Crypto TSMOM sleeve (T2) | Crypto trend diversifies | Same | Correlation −0.063 but Sharpe −0.067: nothing to add | [prereg](docs/research/crypto_tsmom_preregistration_2026-07-31.md) · `crypto_tsmom_sleeve.py` |
| Commodity TSMOM sleeve (T3) | Commodity trend diversifies | Same | Sharpe 0.356, correlation 0.485: cleared both bars, still short of admission | [prereg](docs/research/commodity_tsmom_preregistration_2026-07-31.md) · `commodity_tsmom_sleeve.py` |
| Multi-sleeve frontier | Some combination of sleeves beats the base | Frontier over every deployable combination | Every combination is worse than the base alone | [frontier](docs/research/multisleeve_frontier_2026-07-31.md) · `multisleeve_frontier_2026.py` |
| Multi-speed TSMOM blend (N6) | Blending lookbacks beats one speed | Paired test | Difference +0.145, CI [−0.325, +0.594], DSR 0.090 | [prereg](docs/research/multispeed_tsmom_preregistration_2026-08-01.md) · `multispeed_tsmom_eval.py` |
| Vol-managed overlay (V1) | Vol scaling improves the book | Per-subperiod | The +0.07 gain is one crisis; 1/4 subperiods positive | [prereg](docs/research/vol_managed_overlay_preregistration_2026-07-31.md) · `vol_managed_overlay.py` |
| Leveraged-ETF substitution | 3× ETFs raise return efficiently | Probe on the momentum book | Decay tax; leverage is Sharpe-neutral; not implementable on commodity/FX legs | `letf_substitution_probe.py` |
| Gross-exposure cap / DD sizing | Capping gross exposure improves risk-adjusted return | Sizing null with matched exposure | Vol targeting is real; the gross cap is de-levering, nothing more | `tailwind_sizing_null_lab.py` · [sizing reconciliation](docs/research/tailwind_sizing_reconciliation_2026-08-01.md) |
| RL sleeve allocator (v1) | RL weighting of sleeves beats the linear core | Stage-3 walk-forward vs the frozen core | Real gross alpha (frictionless median 0.80 vs core 0.34) lost entirely to cost: net 0.135, cost gap 0.41 vs 0.15 gate ⇒ ship the linear core | `allocator_turnover_lever_probe.py` |
| RL execution overlay | RL execution scheduling saves ≥ 2 bps | Action-space ceiling measured before training | **Unreachable by construction:** the best achievable saving is +1.704 bps against a 2.0 bps gate | `execution_overlay_action_ceiling.py` |
| Defensive / BAB as a standalone diversifier | BAB adds robust uplift on its own | CPCV uplift | Uplift fragile; additive only at small weight (kept inside the survivor book as a crash hedge) | `eval_defensive_sleeve.py` |

## 6. Retail technical and intraday

Strategies taken from books, courses and videos, implemented as published and tested net of cost.

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| HMA(16/65) + RSI + regression crossover | The published rules produce alpha | 18 ETFs × 18.65y; circular-shift timing null | Median per-asset alpha −0.005%/yr (t −0.01); **fails its timing null at p = 0.582**: randomly re-timed copies of the same trades earn more | [falsification](docs/research/hma_cross_falsification_2026-09-04.md) · `hma_cross_falsification.py` |
| Opening-range breakout | ORB has positive expectancy on stocks | Diversified universe, including a faithful Zarattini-Aziz replication | Zero-to-negative expectancy in every regime; **0/40** variants positive | *not shipped* |
| VWAP / anchored VWAP (swing and intraday) | VWAP reclaim and rejection rules work | Literature review + own falsification | NO-GO both: redundant with TSMOM at swing, cost-killed intraday | `vwap_avwap_falsification.py` · `vwap_intraday_falsification.py` |
| VWAP and MACD scalps | Intraday scalps on liquid instruments are profitable | All variants, cost-inclusive | All NO-GO; win rate is not edge | `smb_macd_scalp_eval.py` |
| Volume breakout | Volume-confirmed breakouts continue | Cost-inclusive evaluation | NO-GO | *not shipped* |
| Earnings volatility crush | Selling pre-earnings implied vol pays | Cost-inclusive evaluation | NO-GO | *not shipped* |
| Taiwan TX intraday RL canary | TAIEX futures intraday has an RL-harvestable edge | CPU falsification before any GPU | The frictionless edge is bid-ask bounce: beats null, sub-economic | `taiwan_tx_canary_falsification.py` |
| Taiwan TX intraday momentum | Last-half-hour momentum (Gao et al. 2018) holds on TX | Price-only step before any gamma conditioning | Thin and decaying on a single market | `taiwan_intraday_momentum_falsification.py` |
| EURUSD 3h TSMOM (F1) | Sub-daily FX trend | Free 22.5y history | Gross −0.020 | [prereg](docs/research/eurusd_3h_preregistration_2026-07-31.md) · `eurusd_3h_eval.py` |
| EURUSD 3h reversal (G1) | Sub-daily FX reversal | Same | Gross +0.272, CI straddles zero; net −0.457 | [prereg](docs/research/eurusd_3h_reversal_preregistration_2026-08-01.md) · `eurusd_3h_reversal_eval.py` |

## 7. Pre-registered free-data probes

A campaign of cheap, pre-registered probes on free data, each with a written kill criterion and a
negative control.

| Strategy | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| US-equity overnight premium (N1) | Returns accrue overnight on 18 ETFs | 7 gates | Fails 6/7: the signature was **dividends** (adjusted +0.083, price-only −0.003) | [prereg](docs/research/session_decomposition_preregistration_2026-08-01.md) · `session_decomposition_eval.py` |
| Commodity session premium (N2) | A session premium in commodity ETFs | 16 OOS ETFs | +0.93, 14/16, 4/4 — then **reverses in spot gold**: a measurement artifact | [prereg](docs/research/commodity_session_preregistration_2026-08-01.md) · `commodity_session_eval.py` |
| Cross-asset turn of month (N3) | Returns concentrate around month turns | Random-window null | The canonical window sits at the **78th percentile of random 4-day windows** | [prereg](docs/research/turn_of_month_preregistration_2026-08-01.md) · `turn_of_month_eval.py` |
| Vol-targeted VIX sleeve (N5) | A vol-targeted VIX sleeve is admissible | 6 gates + deflation | Passed all six, **failed deflation**: DSR 0.315; net CI [−0.112, +0.869] | [prereg](docs/research/vix_voltarget_preregistration_2026-08-01.md) · `vix_voltarget_eval.py` · `vix_deflation_check.py` |
| FX majors short-term reversal (probe 17) | Sub-daily FX reversal is harvestable | Pre-registered, 4 OOS splits | The one real effect found: gross +0.579, 4/4 OOS — structurally unharvestable after cost | [prereg](docs/research/fx_majors_reversal_preregistration_2026-08-01.md) · `fx_majors_reversal_eval.py` |
| Country cross-sectional momentum (C1) | Momentum across 30 country ETFs | Pre-registered | Real but Sharpe 0.081: too small | [prereg](docs/research/country_momentum_preregistration_2026-07-31.md) · `country_momentum_eval.py` |
| Crypto cross-sectional reversal (K1) | Reversal across 10 perps | Pre-registered | Fails t, CI and economics | [prereg](docs/research/crypto_xsec_preregistration_2026-07-31.md) · `crypto_xsec_eval.py` |
| Crypto cross-sectional momentum (K2) | Momentum across 10 perps | Pre-registered | Wrong sign | same |
| Dynamic sleeve timing | Time-varying sleeve weights beat static weights (the cheap gate before an RL allocator) | Linear dynamic allocation vs static, train/OOS split | Does not beat static weights: an RL sleeve allocator is not justified | `sleeve_timing_falsification.py` |
| Portfolio frontier (100%/yr target) | An honest book can reach the north-star return | Frontier over sleeves | Not honestly reachable; best honest Sharpe 0.601 | `portfolio_frontier.py` · [multi-sleeve report](docs/research/multi_sleeve_strategy_report_2026-06-18.md) |
| Breadth expansion of the momentum book | More instruments raise the book's Sharpe | Breadth test | No deployable improvement over the base | [breadth expansion](docs/research/tailwind_v1_breadth_expansion_2026-07-01.md) · `breadth_expansion.py` |

## 8. Automated mining substrates

The Crucible miner closes a data substrate when it can measure that no honest test on it has power.

| Substrate | Hypothesis | Method / kill criterion | Result | Evidence |
|---|---|---|---|---|
| Cross-asset daily | The miner can find alphas across asset classes | Measured detectable effect vs realistic edge | Refused on power: MDE 1.68 annualised ΔSharpe vs ~0.50 realistic | [zero-alpha root cause](docs/research/crucible_zero_alpha_root_cause_2026-08-09.md) · `crucible_crossmarket_power.py` |
| US equity (WorldQuant-101 cross-section) | Formulaic alphas on top-K US equities | Real cohort run, turnover and power windows | Powered only at hold ≤ 2 days; tradeable hold ≈ 21 days. The windows are disjoint; no holdout split bridges them | `crucible_equity_breadth_measure.py` · `crucible_hold_horizon_tradeoff.py` |
| Taiwan small-cap | Higher breadth (n_eff 38.1) makes Taiwan minable | Same, with the 0.30% sell tax | The tax forces a 21-day hold, where 0/100 candidates clear the IC floor | `crucible_substrate_eligibility.py` |
| Intraday (FX and cross-asset) | More bars buy power | Stage-0 pre-registration, power confirmation | MDEs are per 252-*bar* year: more bars buy no power. Closed | [intraday prereg](docs/research/crucible_intraday_stage0_preregistration_2026-07-13.md) · [result](docs/research/crucible_intraday_stage0_partA_result_2026-07-13.md) · `crucible_intraday_power_confirm.py` |

---

## What the rows have in common

- **Structure without capture.** Many rows found a real statistical effect (positive IC, a gross
  Sharpe, an OOS-stable pattern) that costs, staleness or execution could not harvest. A gross
  result is a hypothesis, not a strategy.
- **Beta in costume.** Wheel income, ETF outperformance, index RL and several overlays were equity
  beta or de-levering. Score a candidate against matched exposure, not against zero.
- **Artifacts look like alpha.** Dividends credited to the overnight leg, USDC-linear option data, a
  size confound and a stale cash index each produced a convincing result. The fix was a different
  venue or a different series, not a re-run.
- **Power bounds the silence.** In families 7 and 8 a NO-GO often means the test could not have
  seen a realistic edge. That is a statement about the apparatus, not the market.

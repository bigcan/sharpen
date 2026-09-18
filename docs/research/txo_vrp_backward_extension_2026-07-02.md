# TXO Delta-Hedged VRP — Backward-Extension Confirmatory Test (2002-2018)

**Session:** 2026-07-02 · **Status:** COMPLETE — **VERDICT: NO-GO-FINAL** (single-shot, pre-registered)
**Design source (frozen):** `results/taiwan_txo_vrp_validation/` (cont-88 corrected primary)
**Gates:** `configs/taiwan_txo_vrp_extension.gates.yaml` (hash recorded below before data fetch)
**Entry point:** `python scripts/research/taiwan_txo_vrp_extension.py --gates configs/taiwan_txo_vrp_extension.gates.yaml`
**Results:** `results/taiwan_txo_vrp_extension/vrp_extension_report.json` (+ cycles parquet)

## Plain-English summary

The closest any options strategy has come to surviving this project's falsification funnel was
a delta-hedged short monthly ATM straddle on Taiwan index options (TXO): net Sharpe +0.63 on
2019-2025, killed only on statistical power (bootstrap p=0.0501, DSR 0.767 vs 0.95). This test
ran the exact frozen design — one shot, pre-registered, hash-locked gates — on 14 additional
years (2002-2018, 184 clean monthly cycles including the 2008 GFC) that the design search never
touched. **The edge did not replicate: extension net Sharpe +0.26 (PSR 0.845, block-bootstrap
p=0.163; +0.12 at period-correct pre-2013 taxes), combined 2002-2025 Sharpe +0.36 with
DSR@N=50 = 0.59.** The 2019-2025 result was the lucky subsample, not the signal. The variance
premium itself is real in every era (IV>RV in 72% of cycles) and the daily delta-hedge does
collapse the crash tail (worst cycle −4.5% *through 2008 and 2015*) — but the harvestable
net edge is economically thin, regime-unstable (2011-2018 half: Sharpe ≈ 0.06), and
statistically indistinguishable from zero. **The TXO VRP book — and with it the last open
premium-harvesting options thread in this project — closes permanently.** Options-as-alpha now
stands 0-for-everything tested here; the remaining unexplored corners are data-blocked (paid
intraday SPX chains, Polymarket LOB history) or capability-blocked (market-maker seats), not
idea-blocked.

---

## 1. Why this direction (and not the others)

The mission listed four open threads. Chosen: **thread #1 (TXO), backward extension**.

| Thread | Why / why not |
|---|---|
| **TXO statistical power** (chosen) | The only candidate whose binding failure was *sample size*, not a dead edge. FinMind `TaiwanOptionDaily` reaches back to ~2005 (chains verified to 2005-06 in the availability probe; settlements to 2002) while the stage-1 sample used only 2019-2025 — the pre-2019 data was never fetched, never touched by the 50-trial design search. A pre-registered replication on it is exactly the "legitimately extending the sample / tightening researcher degrees-of-freedom" move the brief sanctions. All infra (fetcher, engine, gates, canonical statistics) already exists and is audited. |
| Polymarket maker/latency-arb MM | No historical order-book/queue data is freely available for Polymarket CLOB; a credible backtest of a queue-position/latency edge cannot be built from trades alone, and the project's own nearest analog (R1 illiquidity probe) went 0/92 cells on realistic execution. Without LOB replay this can only produce the kind of naive mid-price backtest the brief explicitly forbids. Left unevaluated (correctly marked OUTSIDE the killed family). |
| Vilkov conditional 0DTE (10:00 ET) | Well-specified but requires paid intraday SPX/XSP option data (ThetaData ~$40/mo or Databento OPRA credit). The sanctioned probe (cont-61) remains the right vehicle; it is a data-purchase decision for the operator, not something to half-test on daily bars, which would say nothing about an intraday timing rule. |
| FX / commodity options, dispersion, vol-of-vol | No free point-in-time chain data with history (OTC FX vol surfaces, CME settles are paid). Consistent with the small-operator reframe: the blocker is data, not ideas. Noted as untested, not vetted. |

## 2. Hypothesis and structural story

**H:** The Taiwan index option (TXO) variance risk premium, harvested as a short front-monthly
ATM straddle delta-hedged daily with the TX future, is a persistent, positive, tradeable-net-
of-costs premium — not a 2019-2025 fluke.

**Who is on the other side, and why it isn't arbitraged away:**
- TXO is one of the most retail-dominated index-option markets in the world; retail flow is
  net *long* options (lottery + insurance demand). Liquidity providers who absorb it demand a
  variance premium for inventory and hedging costs. The stage-1 measurement is consistent with
  crash-risk compensation: unhedged short straddle *loses* (−0.47 net Sharpe, worst −17.7%)
  while the delta-hedged version monetizes the same premium with the tail collapsed (worst
  −3.4% on 2019-2025).
- The corner is structurally protected: a 0.1% transaction tax per side *on premium* (0.125%
  pre-2006), FINI access friction for offshore vol arbitrage, and small capacity relative to
  global vol books. This is a fee-engineered moat that keeps the marginal global arbitrageur
  out while remaining harvestable at small-operator size — precisely the capacity-constrained
  niche the project's small-operator doctrine targets.
- Honest counter-story: vol-selling institutionalized globally post-2010 and Taiwan retail
  share has drifted down; the premium may have been *larger* in 2002-2018 for non-recurring
  reasons (and taxes were larger too — handled by the period-correct co-primary). The
  extension tests *persistence across regimes*, including 2008; the modern regime is already
  measured by stage 1.

## 3. Pre-registration (frozen before any extension P&L was computed)

### 3.1 Two-stage statistical frame

- **Stage 1 (done, 2019-2025, n=83):** design selected from a ~50-trial family; net Sharpe
  +0.63; FAILED at the capital bar (block of failures: bootstrap p=0.0501, DSR@N=50=0.767).
- **Stage 2 (this test, 2002-2018, expected n≈160):** the *exact frozen design*, evaluated
  once. No selection occurs on this sample ⇒ honest multiplicity N=1 ⇒ the confirmatory
  statistic is the skew/kurtosis-adjusted **PSR ≥ 0.95** (Bailey & López de Prado 2012),
  plus a circular block-bootstrap p(SR≤0) < 0.05 and the same economic gates as stage 1.
- **Combined 2002-2025:** still carries the stage-1 search ⇒ **DSR deflated at the original
  N=50** (BLdP 2014, canonical implementation `sharpen/crypto/eval/statistics.py`).
  The extension adds observations, not trials.
- Power check (from stage-1 moments, before extension data): per-cycle SR 0.183, skew −1.17,
  ex-kurt 1.93 ⇒ at n=160 the PSR bar needs per-cycle SR ≳ 0.14 (≈0.49 annualized); if the
  true edge is stage-1's 0.63, P(pass) ≈ 0.68. A fair test with real power, not a formality.
  Note the same arithmetic shows the combined DSR@N=50 cannot reach 0.95 unless the extension
  Sharpe materially *exceeds* 0.63 (needs ≈0.73 combined over ~250 cycles) — this is why the
  verdict ladder separates full-GO from CONDITIONAL-GO.

### 3.2 Frozen design (identical to stage-1 corrected primary)

Entry first trading day after prior monthly TXO settlement; short 1 ATM straddle (nearest
strike to TAIEX close with both legs quoted >0 within ±600 pts) at entry-day regular-session
closes; hold to cash settlement; daily Black-Scholes delta hedge at constant entry IV
(Brenner-Subrahmanyam back-out) executed in the TX future (same `contract_month`, expires at
cycle end — no intra-window roll); unwind at expiry; `vol_floor` 0.0014; costs (frozen
primary): option 2×1.0 pt half-spread + 2×0.4 pt fee + 0.001×(premium+intrinsic) tax; hedge
1.0 pt × turnover; cpy=12.

### 3.3 Pre-registered deviations (declared before fetch, all with rationale)

| # | Deviation | Rationale / direction of bias |
|---|---|---|
| D1 | Hedge marked at TX **13:45** regular-session close (per-contract `TaiwanFuturesDaily`), not the 13:30 1-min mark | No 13:30 TX mark exists pre-2019 on free data. Measured on 2019-2025: −0.09 Sharpe (0.63→0.54) ⇒ **conservative**. An engine tripwire (§3.5) reproduces the 0.54 before the extension is read. |
| D2 | Data-quality floors: ATM leg volume ≥10, ≥15 TX path points, 20≤DTE≤45 | 2002-2004 TXO infancy: stale prints / sparse strikes are data defects, not evidence. Uniform, not outcome-based. |
| D3 | `delta_robust_min_sharpe` 0.5 → **0.30** for the ±20% IV mis-spec variants (primary `entry_iv` still gated at 0.5) | cont-88 flagged the 0.5-under-stress gate as over-strict and non-load-bearing (perturb_hi 0.41 while realistic deltas passed). A 20% vol mis-spec is a stress, not the base case. |
| D4 | **Period-correct co-primary cost variant** must also clear 0.5 | The frozen block charges 2025 frictions to 2002-2013; the TX-hedge transaction tax was up to ~12× higher pre-2006 (levied schedule in gates yaml, anchored on the 2005-12-14 / 2008-08-06 Act amendments and the verified 2013-04-01 halving). Biases **against** the strategy. |
| D5 | Bootstrap switched to canonical **circular block bootstrap** (block=3 cycles) + Newey-West t (3 lags) reported | Mission bar requires autocorrelation-robust significance; stage 1 used iid. More conservative. |
| D6 | Tail gates added: worst cycle ≥ −10%, CVaR5 ≥ −5% | 2008 is in-sample by design. A premium that pays for its tail must show it *here*; −10%/month also matches prop-firm static-DD viability. |
| D7 | Settlement-table holes (FinMind TXO missing 2006/2012/2018) filled from the exchange-published `TaiwanFuturesFinalSettlementPrice` (TX) — same settlement print — cross-checked on all overlap months (tolerance 1.0 pt, ≥98% match required, else HALT) | TX and TXO share the final-settlement day and print. The cross-check is a hard data gate, not a modeling choice. |

**Amendments during data-prep (before any extension P&L was computed):**

- **A1 (D7 fill source).** The first fill attempt derived the print from
  `TaiwanFuturesDaily`'s final-day `settlement_price`; the pre-registered cross-check
  **HALTED it** (match 4.2%, n=71) — that field is 0.0 on the final day, and pre-2008 TXO
  settles the business day *after* the last trading day (regime change verified in-data:
  2005 settlement dates are Thursdays, 2013+ Wednesdays; next-morning TAIEX average in the
  old regime). Fill source replaced with `TaiwanFuturesFinalSettlementPrice`
  (exchange-published; exact TXO match on probed months). The TX-futures dataset has its own
  holes (2006/2008/2014), complementary to TXO's except **2006, un-fillable on FinMind** →
  2006 cycles (~13) drop as a documented data hole (calm year; exclusion direction-neutral).
- **A2 (old-regime final hedge mark).** Pre-2008 cycles have no TX mark on the settlement
  day; the TX future itself cash-settles to the same print, so the engine appends
  (settle_date, settlement_print) as the final path mark — the faithful "hold hedge to
  settlement" behavior. Modern cycles unaffected. Unwind turnover still charged the full
  half-spread (conservative: real cash settlement pays no spread).

### 3.4 Verdict ladder (locked)

- **GO** — all extension gates pass AND combined DSR@N=50 ≥ 0.95.
- **CONDITIONAL-GO ("PROMISING, needs Tier-2")** — all extension gates pass AND combined
  DSR@N=50 ≥ 0.85 (two-stage frame: the N=1 replication carries the confirmatory weight;
  Tier-2 must explicitly adjudicate the multiplicity frame before any capital).
- **NO-GO-FINAL** — anything else. The book closes permanently; no third look without a
  structurally different signal (not another re-sampling of the same premium).
- **DATA-INSUFFICIENT** — fewer than 100 clean extension cycles assemble; report and stop.

### 3.5 Engine tripwire (runs before the extension sample is read)

The extension engine re-run on 2019-2025 with `TaiwanFuturesDaily` per-contract 13:45 closes
must land within ±0.10 of the known 13:45-mark net Sharpe 0.54 (cont-88 audit note). Failure
halts the test (engine bug), it does not produce a verdict.

### 3.6 Reported-but-not-gated (exploratory, declared now)

Per-year Sharpe table; VRP level by era; hedge turnover; 13:30-vs-13:45 delta on 2019-2025;
±10% synthetic settlement-gap stress (hedge frozen); worst-10 cycles table; early-tax
worst-case diagnostic (0.05% pre-2006); margin-capital return approximation; correlation to
TAIEX and to the TSMOM+BAB book over the overlap; DSR grid at N∈{10,20,50,100}; combined PSR.

### 3.7 Pre-registration integrity

- FinMind **availability probe** (2026-07-02, before this file): coverage/schema only —
  settlement year counts, chain row counts/columns on 4 sample days, TX futures schema,
  TAIEX day counts. **No option P&L, premium level, IV, or return was computed.**
- Gates file SHA-256 (frozen): `2b16e95416d8805445d8dbb9ede4955f574bedd885c646402ea23329df5a463f (amended pre-fetch: data_quality block added, values unchanged from D7; supersedes ecff3f58...)`
- Any change to the gates file after the first extension evaluation invalidates the test.

---

## 4. Results (single shot, 2026-07-02)

### 4.1 Data assembly

190 extension cycles fetched (contracts 200202..201901; entries 2002-01-18 → expiry
2019-01-16), 184 survive the pre-registered quality floors. Settlement hole-fill cross-check:
**146/146 overlap months match exactly (max diff 0.00 pt)** — the TX-futures final-settlement
print is byte-identical to TXO's, validating the fill. Un-fillable 2006 hole: ~13 cycles
dropped (documented, direction-neutral). Failed fetches: 200401, 200701 only.
Manifest: `data/taiwan_options_ext/TXO_vrp_ext.manifest.json` (status PASS).

### 4.2 Engine tripwire (§3.5): PASSED

2019-2025 re-run on `TaiwanFuturesDaily` per-contract 13:45 closes → net Sharpe **0.483** vs
expected 0.54 ± 0.10 (n=83). Engine and data source are sound; the extension read is valid.
Residual −0.06 vs the 1-min-derived close ≈ per-contract vs continuous-series differences.

### 4.3 Extension 2002-2018 (n=184) — the confirmatory test

| Statistic | Value | Gate | Pass? |
|---|---|---|---|
| Net Sharpe (ann, frozen costs) | **+0.259** | ≥ 0.50 | **FAIL** |
| Gross Sharpe (ann) | +0.519 | — | (even frictionless-gross fails to replicate +0.63) |
| PSR (skew/kurt-adj, N=1 bar) | **0.845** | ≥ 0.95 | **FAIL** |
| Block-bootstrap p(SR≤0), block=3 | **0.163** | < 0.05 | **FAIL** |
| Newey-West t (3 lags) | 0.98 | report | — |
| Period-correct co-primary Sharpe | **+0.116** | ≥ 0.50 | **FAIL** |
| First / second half Sharpe | +0.40 / +0.06 | 2nd > 0 | pass (barely) |
| Equity beta (corr) | +0.057 (0.257) | \|β\| ≤ 0.15 | pass |
| VRP positive fraction | 72.3% (IV 0.199 / RV 0.175) | ≥ 50% | pass |
| Delta modes (entry_iv/lo/hi/smile) | 0.26 / 0.32 / 0.21 / 0.24 | all ≥ 0.30 | **FAIL** |
| Worst cycle / CVaR5 | −4.5% / −2.8% | ≥ −10% / −5% | pass |

Extension gates: **4 of 11 FAIL, including all four load-bearing ones** (Sharpe floor, PSR,
bootstrap, period-correct). Per-year Sharpe swings sign across regimes: 2002 −0.58, 2004
+1.24, 2009 +1.44, 2011 +1.12, **2015 −1.15**, 2018 −0.07 — not a stable premium.

### 4.4 Combined 2002-2025 (n=267)

Net Sharpe **+0.359**, block-bootstrap p 0.051, PSR 0.953, **DSR@N=50 = 0.592**
(grid: N=10 → 0.75, N=20 → 0.68, N=100 → 0.53; stage-1-frozen trial-variance sensitivity:
0.568). All combined gates FAIL. At the combined per-cycle SR (0.104), even the *undeflated*
PSR-0.95 bar needs ≈ 326 cycles (~27 years); the DSR@N=50 bar is unreachable at any
realistic horizon.

### 4.5 Exploratory diagnostics (declared §3.6, not gated)

- **Hedge-cost sweep** (pt/rebalance → net SR): 0.0 → +0.39, 0.5 → +0.32, 1.0 → +0.26,
  2.0 → +0.13, 4.0 → −0.13. Cost is not the killer; the gross edge is thin.
- **Tail behavior (the one positive finding):** worst-10 cycles are all vol-spike months
  (2015-08 CNY deval, 2007-08 quant quake, 2009-05, 2002-08…) yet every one is contained to
  ≤ 4.5% of notional — through 2008 — confirming the daily delta-hedge genuinely converts
  crash-risk compensation into a bounded-tail stream. The design's risk engineering works;
  its alpha doesn't.
- **Synthetic settlement-gap stress (±10%, hedge frozen):** worst −18.9% of notional
  (201107, +10% gap). The bounded historical tail is conditional on no overnight
  limit-gap at expiry — this residual tail always was the Tier-2 blocker and is now moot.
- **rv_oracle (look-ahead) delta mode: +0.005** — hedging at the *true* realized vol removes
  nearly all P&L. Same pattern as stage 1 (0.10). Read: a large share of the measured
  "hedged VRP" P&L is the entry-IV-mis-specified hedge accidentally momentum-trading the
  underlying drift, not variance harvesting. This decomposition alone should temper any
  future enthusiasm for the design family.
- **Correlation to the TAILWIND book:** declared, skipped as moot for a permanently closed
  book (script provided: `scripts/research/txo_ext_book_correlation.py`, runnable in ~2 min
  if ever wanted). TAIEX correlation of the strategy: 0.26 (extension), 0.30 (combined).

## 5. Verdict

**NO-GO-FINAL** under the pre-registered ladder (§3.4): the extension fails 4/11 gates
including every load-bearing statistical gate, and the combined DSR@N=50 (0.59) is not within
sight of even the conditional floor (0.85). Per the ladder, **the book closes permanently — no
third look without a structurally different signal** (different information set or execution
layer, not another re-sampling of the same premium).

**What this run actually established (the epistemic payoff):**
1. The stage-1 "+0.63, blocked only by power" reading is now resolved: given 3.2× more
   independent cycles, the point estimate collapsed to +0.26 instead of the CI tightening
   around 0.6. The binding constraint was never sample size — it was that the 2019-2025
   subsample was lucky. This is the file-drawer effect caught in the act by a pre-registered
   replication, exactly as designed.
2. The Taiwan VRP is *real but not harvestable at taker costs with a daily futures hedge* —
   consistent with the project's global VRP verdict on clean data (Deribit straddle −0.25)
   and with the un-hedged TXO result (−0.47). The premium compensates crash risk and hedging
   frictions almost exactly; the residual after real frictions is noise.
3. The structural story ("retail-dominated flow + fee moat ⇒ persistent premium") survives
   for the *market-maker* who earns the spread and the tax asymmetry — not for a taker-side
   harvester. The moat that protects the corner from global vol arb also prices out the
   small taker. (Same conclusion as the cont-85 STF cost scout: the friction edge is the
   market-maker seat, an infra/membership moat outside this project's capability.)

**Honest caveats — what could, in principle, be criticized:**
- The extension hedge marks at 13:45 (D1) drag ≈ −0.09 SR vs the frozen 13:30 convention;
  granting the full correction back (+0.09 → ≈ 0.35) still fails every load-bearing gate.
  The frictionless-gross ceiling (+0.52) also fails to replicate +0.63 — no cost or mark
  assumption explains the collapse.
- 2002-2004 TXO microstructure (wider effective spreads than the modeled 1.0 pt/leg) would
  bias the early years *further down*, strengthening the kill, not weakening it.
- The 2006 data hole (~13 cycles) is direction-neutral (calm year) and too small to flip
  n=184 statistics.
- Regime non-stationarity ("old Taiwan isn't today's Taiwan") cuts the other way here: the
  *modern-half* of the extension (2011-2018) is the weakest stretch (SR ≈ 0.06).

**What would flip this verdict — and what explicitly would not:**
- NOT more of the same data (weekly cadence already falsified separately; backward extension
  now falsified; combined-sample power arithmetic unreachable).
- NOT vol-regime/trend filters (closed project-wide, cont-81 filter-rescue).
- Structurally different only: (a) intraday-rebalanced hedging on paid tick data changing the
  hedge-frequency economics — a new design family with a new (unfunded) data cost and no
  prior evidence in its favor here; (b) the market-maker seat (out of capability); (c) a
  genuinely different underlying with a real access moat AND free point-in-time chains —
  none identified (KOSPI/Nifty chains are paid or access-blocked).

## 6. Tier-2 checklist

Not applicable — NO-GO-FINAL. Nothing here may proceed toward paper or live capital. The
pre-registered ladder forbids re-opening this design family; the only Tier-2-relevant output
is the tail-engineering observation (§4.5) if a *different* short-vol sleeve ever reaches a
gate, and the settlement-print cross-check pattern (D7/A1), which is reusable data infra.

## 7. Reusable assets produced

- `scripts/data/fetch_taiwan_options_finmind_ext.py` — TXO/TX extension ingestion with
  exchange-print settlement fill + hard cross-check (2002-2026 panel now on disk:
  `data/taiwan_options_ext/`, 190 cycles + 30,794 TX daily rows).
- `scripts/research/taiwan_txo_vrp_extension.py` — pre-registered two-stage confirmatory
  evaluator (reconciliation tripwire, PSR/DSR/block-bootstrap/NW-t, period-correct cost
  schedules, gap stress). Pattern reusable for any future discovery→replication test.
- `tests/research/test_taiwan_txo_vrp_extension.py` — 8 tripwires (schedule boundaries,
  stage-1 hedge-loop equivalence, LEAK-2 causality, pick-then-filter volume floor, NW-t,
  gap-stress arithmetic, hole-fill halt).
- `configs/taiwan_txo_vrp_extension.gates.yaml` — the frozen prereg (hash in §3.7).
- Verified period-correct Taiwan futures/options transaction-tax schedule 2002-2026 (in the
  gates yaml), reusable for any future Taiwan-derivatives cost model.
- `scripts/research/txo_ext_book_correlation.py` — cycle-vs-TAILWIND-book correlation
  (unused here; runnable for any future Taiwan sleeve).

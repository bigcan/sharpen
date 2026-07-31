# PRE-REGISTRATION — EURUSD 3-hour TSMOM (F1) + negative control (F2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the data finished downloading and before any
Sharpe was computed.** At commit time the EURUSD history was still being fetched (12,510 of ~140,000
bars). Nothing in this file was chosen after seeing a result.

## 0. Why this substrate and this holding

`free_data_unblock_2026-07-31.md` established, from **measured** inputs, that exactly one
(substrate, holding) cell on free data has power and economics coinciding:

**EURUSD at 3-hour holding** — N_eff 7,159, MDE **0.448** ≤ 0.50 ceiling, cost drag **0.31** < MDE.
Measured spread 0.256 bp, measured annualised vol 12.03%, ~23 years of free Dukascopy history.

Every other cell tested this session failed one side or the other. This is the first place worth
running a probe at all.

## 1. What is tested (LOCKED)

Single instrument, so this is a **time-series** test, not cross-sectional — consistent with the
session's finding that time-series is the style that works on this stack (8 cross-sectional probes
produced nothing ≥0.30; the first time-series probe produced 0.409).

Bars: EURUSD **3-hour**, built from Dukascopy ticks by `scripts/data/fetch_dukascopy.py`
(mid = (bid+ask)/2), full available history.

### F1 — house TSMOM at 3-hour scale · sign **+1**
- **Signal (causal):** the project's own `tsmom_signal` construction, unchanged — mean of
  `sign(trailing return)` over lookbacks **63 / 126 / 252 bars** with `skip = 5` bars. At 3-hour
  bars those lookbacks are ~8 / 16 / 32 trading days. Uses only bars ≤ t−skip.
- **Position:** sign of the trend score, vol-targeted on trailing realised vol (the house's
  `vol_scaled_weights` convention), leverage capped at 3.
- **Expected sign: +1** (trend continuation). A negative Sharpe is a FAIL.
- **Cost:** charged on |Δposition| at the **measured** 0.256 bp median spread, plus a 0.5 bp
  and 1.0 bp stress. An edge that only survives at the median spread is fragile and is reported so.

### F2 — NEGATIVE CONTROL · deterministic pseudo-random position
- No price information. **Expected Sharpe ≈ 0.** Validated instrument: it returned DSR 0.0002 and
  0.0054 on two independent substrates earlier today.
- **If F2 shows a significant Sharpe, the run is INVALIDATED and F1 is discarded** regardless of how
  it looks.

## 2. Pre-committed pass conditions — ALL must hold

1. **net Sharpe ≥ 0.30** after the measured 0.256 bp cost (the same economic bar used for every
   sleeve today — below it a sleeve cannot move a book already holding a 0.60 edge);
2. **block-bootstrap CI on the Sharpe excludes zero** — not merely a large t. Overlapping and
   autocorrelated intraday returns inflate naive t-stats; today's S1 had t = +5.00 with a CI that
   straddled zero;
3. **positive in ≥3 of 4 subperiods** — V1's gain was entirely one crisis window and would have read
   as a +0.07 headline improvement without this check;
4. **still positive at 0.5 bp** (roughly double the measured spread) — a slippage stress;
5. **F2 control fails.**

## 3. Honest priors, written before the run

**For:** the cell is the only one all session where the funnel could detect an edge that survives
costs; trend is the one signal family with a validated track record on this stack; 23 years spans
many FX regimes.

**Against, and these are serious:**
- **The detection bar is high relative to what intraday FX plausibly offers.** MDE 0.448 means only
  a fairly strong edge is findable here. Sub-daily FX trend is not a well-documented anomaly — FX
  momentum is documented at **monthly+** horizons, and at sub-daily horizons the literature leans
  toward mean reversion and order-flow effects. **A negative is the more likely outcome and would
  not be surprising.**
- EURUSD is the most liquid, most arbitraged instrument on earth; a persistent 3-hour trend edge
  surviving 23 years is a strong claim.
- The 0.70-turnover assumption behind the drag figure is generic; a trend signal flipping often at
  3-hour scale could turn over far more and blow through the cost budget.

## 4. Stop rule

One hypothesis, one control, one parameterisation (the house's, unchanged). **No F3.** If F1 fails,
sub-daily FX trend on free data is recorded closed and is not re-probed with other lookbacks, other
bar sizes, a session filter, or a vol-regime conditioner.

## 5. Promotion ceiling

A pass earns **forward-incubate + Tier-2 before any capital**, and does not by itself authorise
changing TAILWIND. The panel is a single instrument, so a pass would also owe a breadth check across
other FX majors before it could be called a strategy rather than a curiosity.

## 6. Results

*(Empty at commit time on purpose — verifiable from git history, and the data was still downloading.)*

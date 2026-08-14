# What is the edge of the ETFs that beat the S&P 500 for 10-20 years?

**Date:** 2026-08-14 · **Status:** COMPLETE — measured, not argued
**Scripts:** `scripts/research/etf_outperformance_{universe,factors,baserate,window_sensitivity,attribution,predictability,robustness}.py`
**Artifacts:** `results/etf_outperformance/`
**Context:** follow-up to BALLAST v1 ([[project_ballast_v1_sp500_long_only_rl]]), which was killed by free-data
survivorship. The ETF framing avoids that specific defect — an ETF's own NAV history is a real,
net-of-fee, survivorship-free record *of that fund*, including everything it held that went to zero.

## Answer in one paragraph

The consistent outperformers are real, and their edge is **not skill, security selection, or fund
construction — it is concentrated exposure to one sector (US large-cap technology, and inside it,
semiconductors), measured over a window that begins after the last time that exposure collapsed.**
Across 351 ETFs and three model specifications, **zero funds have a positive alpha that survives
multiple-testing correction** once market beta and a tech-sector factor are controlled for. Extend
the window to a full cycle and the outperformance disappears: QQQ's CAGR since Jan 2000 is 8.66% vs
SPY's 8.38%, with a **worse** Sharpe (0.372 vs 0.415) and an 83% drawdown that took **14.9 years** to
recover. The funds did not beat the index; they held a levered slice of it during that slice's decade.

## 1. Universe and method

357 US-listed ETFs across 19 categories, deliberately including categories expected to *lose*
(inverse, single-country, commodity, bond) so the denominator is not a recalled list of winners.
354 downloaded, dividend- and split-adjusted (total return). Factors from the Ken French library
(FF5 + momentum, daily, through 2026-06).

**Survivorship caveat, stated up front:** yfinance retains almost nothing that delisted, so the
universe is survivors-only. Published industry data puts closures at 1-3%/yr, ~244 in 2023 alone,
with **~30% of ETFs launched in the last decade already shut**, concentrated in leveraged/inverse
and thematic products. Every base rate below is therefore an **upper bound** on the true win rate.

## 2. Base rate — how many actually beat SPY

| Window | Funds existing at start | Beat SPY (raw) | Beat SPY (Sharpe) | Beat in ≥70% of rolling 3y | ≥90% |
|---|---|---|---|---|---|
| 20y (2006-2026) | 196 | 38 (**19.4%**) | 27 (13.8%) | 20 (10.2%) | **1 (0.5%)** |
| 15y (2011-2026) | 322 | 54 (16.8%) | 28 (8.7%) | 36 (11.2%) | 13 (4.0%) |
| 10y (2016-2026) | 343 | 57 (16.6%) | 36 (10.5%) | 28 (8.2%) | 13 (3.8%) |

The premise is true but thin: ~1 in 5 beat SPY on raw return, ~1 in 8 on risk-adjusted return, and
over 20 years exactly **one fund of 196** (QQQ, 95.6% of rolling 3y windows) is near-unbroken.

The winners are one trade. Top of the 20y list: QLD (2x QQQ), SMH, SOXX, PSI, XSD (semis), VGT, IYW,
XLK, QQQ, QTEC, IXN (tech), SSO/DDM (levered beta), then growth (SPYG, VUG, IWF, IVW).

## 3. The window is doing the work

**Start-date sweep.** Recomputing for every start year 2000-2016, the fraction beating SPY is stable
at 15-18%, but the *magnitude* is entirely start-dependent — SMH's excess CAGR runs +2.7%/yr from a
2000 start, +9.7% from 2007, +13.9% from 2011, +20.0% from 2015.

**The 2000 cohort, full 26-year record (n=67).** Only **39.4%** beat SPY, and the tech complex is
effectively flat against it: **QQQ +0.5%/yr, XLK +0.7%/yr, IYW +0.6%/yr.** SMH ranks #1 at +2.7%/yr —
with a **Sharpe of 0.426 vs SPY's 0.424**, i.e. the entire margin is compensation for bearing 2x the
volatility and an **−85.3%** drawdown.

**Time underwater from the 2000 peak:** QQQ 14.9 years · IYW 16.5 · XLK 16.9 · SMH 17.3.

That is the mechanism. A 20-year lookback from today starts in 2006 — after the crash, before the
recovery. It books the entire rebound and none of the loss. "Consistent 20-year outperformance" is a
statement about the sample window, not the fund.

**Risk-adjusted leaders over the full 26 years are defensives, not tech:** IYK (staples) Sharpe 0.507,
IJJ 0.470, XLY 0.454, XLV 0.448, XLP 0.444 — all with *lower* raw returns than SPY or close to it.
SPY itself ranks 19th of 67. Whether a fund "outperforms" depends entirely on whether you mean
return or return-per-unit-risk, and the two lists barely overlap.

## 4. Attribution — the edge is a sector, and it is not statistically alpha

Daily excess returns, HAC (Newey-West, 21 lag) errors, three nested models: CAPM; FF5+MOM; and
FF5+MOM plus an orthogonalised **TECH** factor (XLK excess return with its FF5+MOM slope exposure
hedged out, intercept retained so the factor keeps its own mean return).

> Methodological note worth keeping: the first cut built TECH as a *demeaned* residual. A demeaned
> factor contributes zero expected return to anything loading on it, so the entire tech premium was
> being dumped back into the intercept and every tech ETF looked skilled (20 "survivors"). Retaining
> the intercept is what makes the factor an attribution of return rather than of variance.

**Results across 351 funds:**

| Window | Positive CAPM alpha | Positive FF6 alpha | FF6 alpha surviving BH-FDR q=0.10 | +TECH alpha surviving FDR |
|---|---|---|---|---|
| since 2000 | 140 | 109 | **0** | **0** |
| last 15y | 111 | 97 | **0** | **0** |

Return decomposition (annualised, since 2000, sums to excess-over-cash):

| Fund | Total | Market beta | TECH | SMB/HML/RMW/CMA/MOM | Residual alpha | t |
|---|---|---|---|---|---|---|
| SMH | 15.5% | 10.6% | 5.4% | −4.3% | 4.0% | 1.46 |
| SOXX | 17.8% | 12.4% | 5.2% | −2.9% | 3.1% | 1.21 |
| QTEC | 16.4% | 12.2% | 3.8% | −1.3% | 1.8% | 1.15 |
| USD (2x semis) | 44.8% | 28.1% | 12.9% | −2.0% | 5.7% | 1.00 |
| TECL (3x tech) | 60.3% | 48.4% | 14.2% | −0.5% | −1.7% | −1.63 |

Market beta plus one sector factor accounts for essentially all of it. What remains is
indistinguishable from zero, individually and after correcting for having searched 351 funds.

**Leverage is purchased beta, not edge.** Over 20 years SSO (2x SPY) returns a Sharpe of 0.538
against SPY's 0.575, and QLD (2x QQQ) 0.715 against QQQ's 0.749 — leverage buys return and gives back
slightly more than proportional risk, plus −83% drawdowns. The levered funds top the 15y and 10y
tables and are absent from the top of the 20y table for exactly one reason: 2008.

## 5. Could you have picked them in advance?

The only question that matters for building anything. At each month-end, rank equity ETFs (n=281) by
trailing excess return vs SPY over lookback L, hold quintiles for horizon H, measure realised excess.

| Lookback | Hold | Q5−Q1 | t (NW) | Rank IC | IC > 0 |
|---|---|---|---|---|---|
| 12m | 36m | +11.1% | 2.36 | 0.137 | 70% |
| 24m | 36m | +14.6% | 2.24 | 0.196 | 71% |
| **36m** | **36m** | **+15.8%** | **2.48** | **0.223** | **75%** |
| 60m | 60m | +13.3% | 0.76 | 0.139 | 65% |
| **120m** | **36m** | **+1.3%** | **0.11** | **0.073** | 62% |
| **120m** | **60m** | **−1.3%** | **−0.05** | **0.028** | 55% |

**The 10-year lookback — the exact rule implied by "buy the ETFs that have outperformed for the past
10-20 years" — has no predictive power at all, and turns negative at a 5-year hold.** The horizon
that works is 12-36 months, which is momentum, not persistence.

### Robustness of the one cell that works (36m/36m)

- **Sub-period:** first half (2003-2015) +17.5% t=2.67 → second half (2016-2023) **+12.8% t=1.02**. Decays.
- **Composition:** Q5 is 50% → **63%** tech/sector funds across the two halves. It keeps re-selecting the same bet.
- **Sector-neutral:** ranking *within* category collapses the spread to +6.4%, t=1.34 — **60% of the
  spread was picking categories, not funds.**
- **Long-only leg:** Q5 alone is **+1.49%/yr** gross vs SPY (Q1 −3.89%). The spread is mostly the short
  leg, which a long-only mandate cannot harvest. Before turnover costs, and before discounting for
  the 10-cell (lookback × hold) grid searched.

## 6. Conclusions

1. **There is no fund-level edge to reverse-engineer.** The outperformance is sector concentration
   plus beta. Nothing survives FDR correction in any window or specification.
2. **The premise is window-dependent.** Over a full cycle the tech complex matches SPY with double
   the drawdown and 15-17 years underwater. Selecting on 10-20 year trailing performance is
   selection on the dependent variable — the same error that killed BALLAST, on a different axis.
3. **The one thing that would have worked is not what the premise suggests.** Ex-ante ranking works
   at 12-36 months and fails completely at 120 months. That is TSMOM/cross-sectional momentum — which
   this project has *already* validated (18 ETFs × 4 classes, net SR ~0.60,
   [[project_cross_sectional_relative_value_lever_s553]]).
4. **The honest reframing of "how do I beat SPY":** you take a concentrated factor or sector exposure
   and you underwrite its drawdown. That is a risk-budgeting decision, not an alpha. The measured
   menu: tech/semis (higher return, −85% DD, 17y underwater), defensives (better Sharpe, lower
   return), mid/small value (won 2000-2010, lost 2010-2026).

## 7. Recommendation

**Do not build an "identify the consistent outperformer ETF" strategy** — the predictability test
above is its falsification, pre-registered and run.

The defensible extension of this work is the one that connects to the validated edge: the 36m/36m
result is a *slower, long-only, ETF-level* cousin of the TSMOM sleeve already in TAILWIND. Its
standalone case is weak (t decays to 1.02 out of sample, 60% is category selection, +1.49%/yr gross
long-only). It should be tested **only** as a candidate overlay inside the existing TSMOM
admission rule `s2 > s1·(√(2+2ρ)−1)` ([[project_alpha_search_16_probes_2026_07_31]]) against the
current sleeve, not as a new workstream.

## Data caveats

- Survivors-only universe; base rates are upper bounds (§1).
- SMH's pre-2011 history is the Merrill Lynch HOLDRS basket (fixed 2000-vintage composition), not the
  current VanEck ETF; the series is continuous and tradeable throughout but the vehicle changed.
- BH-FDR across 351 highly correlated funds is conservative; raw t-stats are reported alongside so
  the correction can be reconsidered. No fund exceeds t=2.7 on FF6 alpha in any case.
- Predictability grid searched 10 (lookback, hold) cells; the reported best cell is not discounted
  for that search.

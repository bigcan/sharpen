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

---

# Addendum (2026-08-15) — the overlay was tested, and two operator probes

Follow-on to §7. Scripts: `etf_overlay_vs_tsmom_admission.py`,
`etf_overlay_admission_stress.py`, `etf_concentration_family_gradient.py`.

## A1. The 36m/36m overlay does NOT earn admission to TAILWIND

Applied the pre-existing admission rule `s2 > s1·(√(2+2ρ)−1)` exactly as it was applied to the
country/crypto/commodity sleeves, against the rebuilt base book (cross-asset TSMOM, s1 = **0.601**).

| Variant | s2 | ρ | threshold | verdict | combined (equal-risk) |
|---|---|---|---|---|---|
| A dollar-neutral Q5−Q1 | 0.374 | +0.002 | 0.250 | **ADMIT** (+0.125) | 0.689 (+0.088) |
| B long-only Q5 − SPY | 0.171 | +0.051 | 0.270 | REJECT (−0.099) | 0.533 (−0.068) |

Variant A's pass is a **false positive**, and the reason is the finding worth keeping:

```
beta_SPY       +0.075  (t +3.90)
beta_TECH_rel  +0.107  (t +4.70)
alpha ann      +0.0038 (t +0.34)      R2 0.137
Sharpe after hedging SPY + tech:  0.080   (was 0.374)  =>  REJECT
```

The sleeve is the same tech-beta trade in long/short clothing. Its ρ ≈ 0 against the base book is
not evidence of a distinct premium — it only says the cross-asset TSMOM book carries no tech tilt.
Verdict robust to ρ ∈ [−0.10, +0.10] (threshold never exceeds 0.290 vs hedged 0.080).

> ⚠ **DEFECT IN THE ADMISSION RULE (generalises beyond this candidate).** The rule reads only
> `(s1, s2, ρ)`. **A candidate can be uncorrelated with the base book and still be pure beta.** It
> tests diversification against *one book*, not against the factor space. Same defect class as the
> Crucible gate where a pure-beta candidate passed 3 of 5 legs with `guards.max_market_beta` inert.
> **Fix: apply the rule to the FACTOR-HEDGED candidate return.** Under that correction this
> candidate rejects cleanly and the rule still admits genuinely orthogonal sleeves.

Secondary stresses, each independently damaging: breakeven **borrow cost is only 1.27%/yr** on the
50%-short leg (survives 1%, dies at 2%; the Q1 leg is small thematic funds); margin halves across
sample halves (+0.166 → +0.078). Short-leg breadth is fine (median 51 names) — that check passes.

## A2. Operator probes: XLG and IWY — concentration and growth are inception-date artifacts

Both are fair challenges: neither fund is a sector bet or levered beta (XLG's beta is **0.937,
below the market**, with a smaller drawdown than SPY in every window).

**XLG** (S&P 500 Top 50): excess +0.32%/yr (20y), +1.28%/yr (10y), Sharpe 0.595 vs 0.575, maxDD
−52.4% vs −55.2%, and it beat SPY in **90.5%** of rolling 3y windows over 10y. But over its full
21.1 years: **FF6 alpha −0.05%/yr (t=−0.08)**, −0.62%/yr once tech is controlled. Its mega-cap tilt
(SMB −0.247) contributed **−0.0001** to return — the loading is real, the premium wasn't.

**IWY** (Russell Top 200 Growth): the best non-levered fund measured, +2.52%/yr, Sharpe 0.851 vs
0.797. Also the luckiest inception available — **2009-09-28**, six months after the bottom. Of its
16.2%/yr excess-over-cash, **87% is plain market beta**; residual +0.57%/yr at **t=0.97**.

**The decisive test — order each family by inception, measure every fund on its own window:**

| concentration | inception | excess || growth | inception | excess |
|---|---|---|---|---|---|---|
| OEF (S&P 100) | 2000-10 | **−0.25%** || SPYG (S&P 500 Gr) | 2000-10 | **−1.01%** |
| XLG (Top 50) | 2005-05 | +0.12% || IWF (R1000 Gr) | 2000-05 | **−0.27%** |
| MGC (Mega Cap) | 2007-12 | +0.36% || IVW (S&P 500 Gr) | 2000-05 | +0.30% |
| IWY (Top 200 Gr) | 2009-09 | +2.52% || VUG / MGK | 2004-01 / 2007-12 | +1.38% / +2.34% |
| | | || IWY / SCHG | 2009-09 / 2010-01 | +2.52% / +2.12% |

**corr(inception order, excess) = +0.886 (concentration), +0.829 (growth).** Every fund in either
family that spans the dot-com bust is at or below SPY; every fund launched after 2004 beats it, and
the later the launch the bigger the margin. These are not funds of differing merit — they are one
strategy sampled from different start dates. OEF, the only concentration vehicle covering a full
mega-cap cycle, **underperforms over 25.8 years** with a −50.9% drawdown and 6.5 years underwater.

The gradient runs forward too: measured from 2022, IWY is **−0.26%/yr**, IWF −0.84%, VUG −0.39%.

**Mechanism.** Cap-concentration is mechanically *momentum-of-market-cap* — you hold more of
whatever already grew, and more of it than the cap-weighted index does. It compounds while mega-cap
leadership persists and inverts when it breaks. XLG remains a defensible *holding* (mild quality
tilt, better Sharpe, lower DD); it is not an alpha.

> Method notes. IWY was a **false positive in the first attribution run** (t=2.632, "survived" FDR)
> — an artifact of the demeaned-TECH-factor bug in §4; with the mean retained, t = 0.97. A second
> bug was caught and fixed here: the within-fund start-year grid compared a fund's post-inception
> series against SPY measured from the requested year, inflating IWY's "2006" cell to +5.82%. Cells
> where the fund did not yet exist are now NaN. The family table was never affected (it uses
> inception for both legs).

Prior project work on XLG (S553-cont-75) tested 10 entry/timing rules against buy-and-hold XLG (all
10 negative IR) but never asked whether XLG beats SPY. This fills that gap.

## A3. PPA (aerospace & defense) — the last non-tech name, closed by one event

Applying the strictest filter available (beat SPY on **return AND Sharpe in all three windows**)
leaves **16 of 196 funds**. Thirteen are the same trade (semis → tech → growth → levered); XLG and
OEF are the concentration bet from A2. **PPA is the only survivor that is none of those** — and its
tech loading is **negative (−0.11)**, so it is genuinely orthogonal to every other winner. Its
sibling ITA is the same shape (β 0.96, alpha +3.18%/yr t 1.19, tech −0.16). It therefore needed its
own falsification rather than an inherited one. Script: `etf_defense_ppa_closure.py`.

**It passes the inception-date test** — the one that closed growth and concentration:
`corr(inception order, excess) = +0.587` across PPA/ITA/XAR/DFEN, and **non-monotone** — ITA
launched *after* PPA and earns less (+1.94% vs +2.49%). Compare +0.886 / +0.829. Defense is not a
start-date artifact. It is a different failure mode.

**The binding test is the event split:**

| Window | Excess CAGR | FF5+MOM alpha | t | +XLI alpha | b_XLI | b_mkt given XLI |
|---|---|---|---|---|---|---|
| full (20.8y) | +2.49%/yr | +2.83%/yr | 1.26 | +2.91% | +0.74 | +0.18 |
| **ex-rearmament (16.2y)** | **+0.28%/yr** | **−0.17%/yr** | **−0.07** | +0.96% | +0.67 | +0.26 |
| rearmament only (4.6y) | +10.73%/yr | +8.75%/yr | 1.55 | +6.56% | **+0.94** | −0.06 |

**Over the first 16 years of its life PPA returned 28 bps/yr over SPY with an alpha of exactly
zero.** The whole record is one 4.6-year geopolitical repricing. It is also ~3/4 industrials —
adding XLI drops market beta 0.91 → 0.18, and in 2022+ PPA is 0.94×XLI with *negative* market beta.

Episodes (non-overlapping): +5.09%/yr (2005-08, GFC) · +0.69% (2009-13) · +2.69% (2014-18) ·
**−11.45%** (2019-21) · **+10.73%** (2022-26). Calendar years beating SPY: **12 of 22** — a coin
flip, with the mean carried by a few enormous years (+17.1% 2007, +18.0% 2013, +27.7% 2022, +19.4%
2025) against −17.9% 2020 and −21.8% 2021.

**Crash-hedge hypothesis TESTED and REJECTED.** Both winning episodes coincide with equity stress,
and TAILWIND has the precedent of reclassifying BAB from a premium into a crash hedge on exactly
this evidence. It does not hold: in the **worst** SPY decile PPA *underperforms* (−0.47%/mo), and
only **27.3%** of total active return comes from the worst two deciles (20.3% of months) versus
BAB's 383%-from-crash-days signature. Mildly bad-month tilted; not insurance.

**COMPOSITION — and a correction to "one event".** Verified from fund data: 63 holdings,
cap-weighted, top names **RTX 8.26% · GE Aerospace 7.10% · Boeing 7.07% · Lockheed 6.37% · General
Dynamics 4.80%**. Regressing PPA on an equal-weight defense-prime basket (LMT/NOC/GD/LHX) vs a
commercial-aero basket (BA/GE/TDG/HEI):

```
PPA ~= 0.49 x defense primes + 0.42 x commercial aero      R2 0.892
```

| Window | b_PRIME | b_COMM | primes ann. | commercial ann. | SPY |
|---|---|---|---|---|---|
| full | 0.49 | 0.42 | +14.8% | +18.8% | +11.4% |
| pre-COVID (→2019-12) | 0.54 | 0.38 | +16.3% | +19.2% | +9.5% |
| COVID (2020-21) | 0.49 | 0.40 | +6.7% | **+3.0%** | **+23.4%** |
| post-2022 | 0.41 | **0.51** | +14.1% | **+25.3%** | +12.7% |

**PPA is ~half a government contractor and ~half an air-travel-cycle play** — the ampersand in
"Aerospace & Defense" is doing real work, and the two halves have opposite drivers. That is the
structural reason the record is episodic rather than smooth.

> ⚠ **CORRECTION to the framing above.** A3 originally attributed the post-2022 run to rearmament
> alone. The decomposition shows the **commercial leg outran the defense leg** in that window
> (+25.3% vs +14.1%/yr) with PPA's loading shifting toward it (0.51 vs 0.41). The surge is **TWO
> independent cycles coinciding** — European rearmament *and* the post-COVID commercial-aero
> recovery off a collapsed base. "One geopolitical repricing" was too narrow. The 2020-21 collapse
> is the same structure inverted: COVID destroyed the commercial half while a tech-led melt-up ran
> away from the defensive half. **Verdict unchanged** — episodic, not a premium.

> ⚠ **CAVEAT on these baskets:** they are **survivor-selected** (names large *today*), so their
> return LEVELS are inflated — the tell is that both baskets "beat SPY" by more than the actual
> 63-holding fund did. The loadings and the cross-era comparison are the usable output; the basket
> return levels are not a clean sector measure.

⇒ **PPA is an episodic cyclical spanning two unrelated cycles, on industrials beta — not a premium
and not a hedge.** Holding it is a joint forecast on defense budgets and air travel.

**The one genuinely open thread:** an **exclusion premium**. Many ESG mandates cannot hold weapons
manufacturers; if a persistent buyer base is structurally absent the sector should clear at a higher
expected return (the tobacco/gambling "sin premium" argument). This is the only mechanism proposed
here that would produce a *premium* rather than an episode, it is **untested**, and testing it needs
an exclusion-flow proxy rather than a price series. Logged, not claimed.

The outperformer list is now fully accounted for.

## Data caveats

- Survivors-only universe; base rates are upper bounds (§1).
- SMH's pre-2011 history is the Merrill Lynch HOLDRS basket (fixed 2000-vintage composition), not the
  current VanEck ETF; the series is continuous and tradeable throughout but the vehicle changed.
- BH-FDR across 351 highly correlated funds is conservative; raw t-stats are reported alongside so
  the correction can be reconsidered. No fund exceeds t=2.7 on FF6 alpha in any case.
- Predictability grid searched 10 (lookback, hold) cells; the reported best cell is not discounted
  for that search.

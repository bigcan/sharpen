# PRE-REGISTRATION — N6: multi-speed TSMOM blend vs single-speed, paired improvement test

**Written:** 2026-08-01, **before any multi-speed blend was constructed or scored.**

## 0. Why this, and why it is a better-powered question than probes 1-22

Twenty-two probes have hunted a *new standalone* edge and found none. N5 showed why that is now
structurally hard: after 22 trials the deflated-Sharpe hurdle is **SR\* ≈ 0.494 annualised**, so a
standalone candidate must clear ~0.5 net on its own to survive multiplicity. Nothing free has.

This probe asks a different question with far more statistical power: **does a multi-speed blend
improve the edge this project has already validated?**

The key is that it is a **paired** test. The single-speed and blended books are highly correlated,
so their *difference* has much lower variance than either book alone — a 0.10 Sharpe improvement in
a paired difference is far more detectable than a 0.10 Sharpe standalone strategy. The estimand is
the difference, not the level.

**Prior:** blending trend speeds is standard institutional practice (AQR, Man AHL and others run
fast/medium/slow ensembles) and is documented to add roughly 0.1-0.2 Sharpe over a single lookback,
because different speeds capture different reversal horizons and their errors partially cancel.
This project's deployed book uses a **single** speed and the blend has never been tested here.

**Honest framing up front:** if this passes it is an **improvement to an existing validated edge**,
not a new alpha. It will be reported that way.

## 1. Construction — fixed in advance

Universe: the 18-ETF panel, 2008-01-02 → 2026-06-30, close-to-close, already DATA-CLEAN.

**Baseline (single-speed):** 12-month (252-day) time-series momentum, `sign(P_t / P_{t-252} - 1)`,
inverse-vol weighted by trailing 63-day realised vol, weights normalised to sum-abs 1, **lagged one
day**.

**Candidate (multi-speed):** the equal-weight average of the identical construction at **63, 126
and 252 days** — fast, medium, slow. Equal weights, **not optimised**; the three speeds are the
conventional quarterly/semi-annual/annual set and no other combination will be tried.

Both books carry the same universe, same weighting scheme, same lag, same rebalance. **The only
difference is the lookback set.** Costs: 2 bp round trip on turnover, charged identically to both.

**Committed sign: +1** — the blend's net Sharpe exceeds the single-speed book's.

## 2. Pre-committed pass conditions — ALL must hold

1. **Paired difference (blend − baseline) mean > 0** with **CI95 excluding zero** (block bootstrap
   on the difference series).
2. **The difference survives multiplicity: DSR ≥ 0.95 on the paired difference**, with
   `n_trials = 23`. This gate is written in **because N5 failed exactly here** and its absence was
   the defect in that pre-registration. It is not optional and not negotiable after the fact.
3. **Blend net Sharpe > baseline net Sharpe** after 2 bp costs — i.e. the improvement is not an
   artifact of the blend simply trading less.
4. **≥3 of 4 subperiods** show a positive paired difference.
5. **Blend max drawdown ≤ baseline max drawdown × 1.1** — the improvement must not be bought with
   materially more tail risk.
6. **Turnover check:** blend turnover reported; if the blend's Sharpe gain vanishes at 3× cost
   (6 bp), it fails.

Anything less than all six is a **NO-GO**.

## 3. Honest priors

**For:** the paired design has genuinely more power than any standalone test run so far; the effect
is standard practice rather than a data-mined pattern; equal weights mean nothing is fitted; and
both books share a universe, so universe selection cannot drive the difference.

**Against, and decisive if they hold:** (a) faster trend speeds are **more cost-sensitive**, so the
63-day leg may add turnover faster than it adds signal — condition 6 targets exactly that;
(b) the three speeds are highly correlated, so the blend may be nearly identical to the baseline
and the difference indistinguishable from zero; (c) 18 assets is thin for a trend book and the
diversification benefit of blending speeds is largest in wide universes (60+ futures), not narrow
ETF panels; (d) **22 consecutive falsifications**; (e) the DSR gate in condition 2 is strict and
N5 — a much larger raw effect — failed it.

## 4. Stop rule

**One draw.** No other speed set, weighting, universe, or cost assumption. If N6 fails, the trend
construction question is closed and the search stands at 23 probes.

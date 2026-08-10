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

## 5. Results — **NO-GO** (fails the two statistical gates), and my power argument was wrong

Run 2026-08-01, `results/multispeed_tsmom/multispeed_tsmom.json`. 4,652 days x 18 assets.

| | baseline (252d) | blend (63/126/252) |
|---|---|---|
| net Sharpe @2bp | +0.2350 | **+0.3691** |
| turnover | 14.1x/yr | 18.8x/yr |
| max drawdown | -16.8% | **-13.2%** |
| net Sharpe @6bp | +0.1298 | +0.2110 |

| pre-committed bar | result |
|---|---|
| 1. paired CI excludes zero | **FAIL — +0.1449, CI95 [-0.3246, +0.5935]**, P(<=0)=0.27 |
| 2. DSR >= 0.95 (n_trials=23) | **FAIL — DSR 0.0902**, hurdle SR* +0.4565 vs observed +0.1449 |
| 3. blend net SR > baseline | PASS — +0.369 vs +0.235 |
| 4. >=3 of 4 subperiods positive | PASS — 3/4 (+1.58%, +0.41%, +0.87%, **-1.48%**/yr) |
| 5. DD not worse than 1.1x | PASS — -13.2% vs -16.8%, blend is *better* |
| 6. gain survives 3x cost | PASS — +0.0812 at 6bp |

**VERDICT: NO-GO.**

### 5.1 The power argument in section 0 was wrong, and that is the finding

I claimed a paired test would have "far more statistical power" because the two books are "highly
correlated", so the difference would have low variance. **Measured correlation is 0.778, not the
~0.95 that argument implicitly assumed.** At rho = 0.78 the difference series carries substantial
variance, and the paired design bought much less power than advertised: the difference Sharpe is
+0.1449 with a CI spanning [-0.32, +0.59].

**The paired-test advantage is real but it scales with correlation.** At rho = 0.95 the difference
variance is ~10% of the average book variance; at rho = 0.78 it is ~44%. I asserted the mechanism
without measuring the input it depends on — the same error class as reading GLD/IAU/SGOL agreement
as independent evidence in N2.

### 5.2 What the numbers do and do not support

**Descriptively the blend looks better on every practical axis:** higher net Sharpe (+0.369 vs
+0.235), *lower* drawdown (-13.2% vs -16.8%), and the gain survives 3x cost (+0.081 at 6bp). If a
speed set had to be chosen on judgement, the blend is the better default.

**Statistically it is not distinguishable from the baseline.** CI spans zero, DSR = 0.09, and the
most recent subperiod (2023-2026) is **negative** at -1.48%/yr, SR -0.50 — the one window that
matters most for a forward deployment decision.

### 5.3 A caveat that limits what this probe could ever have shown

The reconstructed baseline scores **+0.235**, well below the deployed book's documented net
**~0.60**. So this compares two *proxies* of the deployed construction, not the deployed book
itself, and the deployed book's construction differs in ways this reconstruction does not capture.
A blend improvement measured against a weaker baseline does not transfer automatically. Testing
this properly requires the deployed book's own return series, which is not stored (`results/` is
gitignored).

### 5.4 Ledger (durable)

- **Multi-speed TSMOM blend (63/126/252) vs single-speed 252: NO-GO.** Paired difference +0.145,
  CI [-0.325, +0.594], **DSR 0.090**. Descriptively better (SR +0.369 vs +0.235, DD -13.2% vs
  -16.8%, survives 3x cost) but not statistically separable, and **negative in 2023-2026**.
- **A paired test's power gain scales with the correlation between the two arms — measure it, do
  not assume it.** Here rho = 0.778, not ~0.95, and the difference was noisy.
- **The reconstructed TSMOM baseline is +0.235, not the deployed ~0.60.** Any future construction
  comparison needs the deployed book's stored returns, not a reconstruction.
- Family **CLOSED** per section 4. Free-data search stands at **23 probes, zero deployable alpha**.

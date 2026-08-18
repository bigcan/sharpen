# PRE-REGISTRATION — N5: vol-targeted VIX term-structure sleeve, and does it improve the deployed book

**Written:** 2026-08-01, **after** N4's verdict and **before** any statistic on any vol-scaled
series, any drawdown of it, or any correlation against a trend book was computed.

## 0. The new information that justifies a fifth pre-registration

N4's §6 stop rule bars re-cutting N4 — no alternative hedge ratio, threshold, holding rule or
instrument. This is a different construction with its own bars, and the new information is specific
and measured:

**N4 failed on tail alone.** Gross **+0.7338** (CI95 [+0.2823, +1.2075]), net **+0.5907** at 3%/yr
borrow, **4 of 4** subperiods positive, **borrow wall 14.6%/yr**. Existence, persistence and cost
all cleared. The single failure was **max DD −50.1% / worst day −19.9%** against a 35% / 15% gate.

That is a *risk-sizing* failure, not an absence of edge — a distinction none of the previous 20
probes offered, because they all failed on existence or on cost.

**This is the last probe on this family. If N5 fails, volatility carry is closed for this project.**

## 1. The trap I must not fall into, and how it is avoided

The tempting move is to fit something to the crashes I have now seen — a hedge ratio, a stop level,
a crash filter. **Any of those would be fitting to the observed tail and the result would be
worthless.** Three commitments prevent it:

1. **The tail gate is UNCHANGED from N4: max DD ≤ 35% and worst day ≤ 15%.** Not loosened, not
   renegotiated. If this construction cannot reach the same bar N4 missed, it fails.
2. **The target volatility is an EXTERNAL number, not fitted here.** It is **10% annualised**,
   taken from this project's own TAILWIND forward-path work, where 10% effective vol was the
   configuration that produced `RENDER_CLEAR` after 15% failed four legs
   (`project_tailwind_forward_path_render_s553_cont_146`). It is imported from a different
   workstream on a different asset set, so it cannot encode anything about VIX crashes.
3. **The vol-estimation window is fixed at 63 trading days** (one quarter, the conventional
   choice) and **no other window will be tried**, per §6.

## 2. Construction

Identical signal, legs and window to N4: short 1 VIXY / long 1 VIXM when `VIX3M > VIX` at the prior
close, else flat; 2011-01-04 → 2026-06-30.

**The only change** is position size:

```
scale[t] = min( 0.10 / realised_vol_63d(spread)[t-1] , 1.0 )
```

Realised vol is computed on the spread book's own returns over a trailing 63 days and **lagged one
day**. The cap at **1.0 means the sleeve can only DE-risk, never lever** — the unscaled book runs
~37% vol, so the scale is typically ~0.27 and the cap will rarely bind.

Costs as in N4: 10 bp round trip per leg on every position change, borrow at **3%/yr** primary and
**10%/yr** stress, charged on the **scaled** short notional.

**Committed sign: +1.**

## 3. The question that actually matters — does it improve the deployed book?

A standalone Sharpe is not the deliverable. This project already has one validated edge (cross-asset
TSMOM, net SR ≈ 0.60), and the campaign's closed-form admission rule is
**`s2 > s1·(√(2+2ρ) − 1)`**.

Volatility carry should be **negatively correlated** with trend: trend-following is long-convexity
and earns in crises, short-vol is the crisis loser. At ρ = 0 the bar is 0.249; at ρ = −0.3 it is
**0.110**; at ρ ≤ −0.5 **any positive Sharpe** is admissible. So the correlation is as important as
the Sharpe, and it is measured, not assumed.

**Benchmark honesty:** the deployed book's return series is not stored (`results/` is gitignored),
so I reconstruct a **TSMOM proxy** from the same 18-ETF panel — 12-month time-series momentum,
inverse-vol weighted, monthly rebalance. It is explicitly a **proxy, not the deployed book**, and
every correlation and combination number is reported as such.

## 4. Pre-committed pass conditions — ALL must hold

1. **Gross Sharpe CI95 excludes zero.**
2. **Net Sharpe ≥ 0.30** at 10 bp + 3%/yr borrow, and **≥ 0** at 10 bp + 10%/yr borrow.
3. **≥3 of 4 subperiods gross-positive.**
4. **TAIL — max DD ≤ 35% AND worst day ≤ 15%.** Unchanged from N4.
5. **PORTFOLIO ADMISSION:** measured ρ against the TSMOM proxy must satisfy the admission rule
   `net_SR > 0.60·(√(2+2ρ) − 1)`, **and** the equal-risk combination (TSMOM proxy + sleeve) must
   have a **higher Sharpe than the TSMOM proxy alone** on the common window.
6. **TAIL CONTROL MUST COME FROM SIZING, NOT FROM THE SIGNAL.** The *always-on* vol-targeted
   variant (signal ignored, permanently short front / long mid) must **also** satisfy gate 4. N4
   measured the signal as nearly inert (+0.640 shuffled vs +0.734), so if the drawdown improvement
   depended on the signal rather than on sizing, it would be an artifact of 92%-on exposure rather
   than risk management.

Anything less than all six is a **NO-GO**.

## 5. Honest priors, and what a pass would and would not mean

**For:** vol targeting is the one risk primitive that is standard, parameter-light and *not* fitted
to this data; the unscaled book runs ~37% vol against a 10% target, so the mechanical DD reduction
should be roughly 3.7x if scaling were perfect, taking −50% toward the mid-teens; and the premium
underneath is already established as real and cost-surviving.

**Against, and decisive if they hold:** (a) **trailing vol lags spikes** — realised vol is *low*
right before a volatility explosion, so the sleeve will be at its LARGEST size going into exactly
the events that hurt it, which is the well-known failure mode of vol targeting on short-convexity
books and could leave the worst day above 15% regardless of average DD; (b) 2024-08-02 and
2021-11-26 both followed calm stretches, so this is the likely case, not the unlikely one;
(c) scaling to 10% cuts the return proportionally, so the *net-of-borrow* Sharpe may fall below
0.30 once a fixed 3%/yr borrow is charged against a much smaller position — **borrow does not
scale down as cleanly as return does**; (d) ρ against trend may be closer to zero than the theory
suggests over a 15-year window dominated by one regime.

**What a pass would mean:** the **variance risk premium**, a *known* premium, validated for this
project as a tail-controlled sleeve — the project's **second** validated edge after TSMOM. **It
would not be a novel alpha and will not be reported as one.**

## 6. Stop rule

**Terminal for this family.** No other target vol, estimation window, cap, hedge ratio, signal
threshold or instrument. If N5 fails, volatility carry is closed and the free-data search stands at
22 probes.

## 7. Results — passes all six pre-committed bars, then **FAILS DEFLATION**. Verdict: **NO-GO**.

Run 2026-08-01, `results/vix_voltarget/vix_voltarget.json`.

Vol targeting worked mechanically: realised vol **9.9%/yr** against the 10% target, mean scale
**0.278**, the leverage cap never binds (0.0% of days).

| pre-committed bar | result |
|---|---|
| 1. gross CI95 excludes zero | PASS — +0.5231, CI95 [+0.0455, +1.0233] |
| 2. net >= 0.30 @3% borrow, >= 0 @10% | PASS — +0.3687 and +0.1906 |
| 3. >=3 of 4 subperiods positive | PASS — 4/4 (+0.35, +0.40, +0.83, +0.51) |
| 4. max DD <= 35%, worst day <= 15% | PASS — **-21.0%** and **-5.9%** |
| 5. admission + improves combo | PASS — rho +0.141, bar 0.306, SR 0.369; combo 0.526 vs 0.426 |
| 6. always-on also passes tail | PASS — -25.4% DD, -7.2% worst day |

**All six pass. The verdict is still NO-GO, on a stricter test applied afterwards.**

### 7.1 The deflation, which I did not pre-commit and which decides it

This is **probe 22**. Applying this project's own standard — multiplicity-corrected inference,
which Crucible mandates everywhere else and which this pre-registration omitted:

| statistic | plain 95% | Bonferroni (22 probes) |
|---|---|---|
| gross | [+0.0455, +1.0233] | **[-0.2092, +1.3139]** — includes zero |
| **net@3% borrow** | **[-0.1118, +0.8685]** — *already includes zero* | [-0.3638, +1.1563] |

**Deflated Sharpe (Bailey / Lopez de Prado), n_trials = 22: DSR = 0.3147.** The SR* hurdle is
**+0.4942** annualised against an observed net **+0.3687** — the strategy sits *below* the
multiplicity hurdle. Return skew **-1.82**, kurtosis **12.7**, so the heavy left tail is penalised
correctly rather than ignored.

**Verdict: NO-GO.** DSR 0.31 is nowhere near the 0.95 this project requires.

### 7.2 Two defects in my own pre-registration, both material

**(a) I gated the CI on GROSS and the threshold on NET, so nothing ever required the tradeable
series' CI to exclude zero.** It does not: net@3% is **[-0.1118, +0.8685]** at a plain 95%, before
any multiplicity correction. Condition 1 and condition 2 each looked reasonable and together left a
hole exactly where it mattered — the series you would actually trade was never required to be
distinguishable from zero.

**(b) No deflation gate, in a campaign of 22 probes.** Under a global null, the expected number of
probes clearing a 95% bar is ~1.1. **One marginal pass is precisely what chance produces**, and
N5's gross lower bound was +0.0455. A pre-registration that omits multiplicity in a long search is
not a rigorous pre-registration, whatever its other conditions say.

Both are corrections in the **stricter** direction, applied after seeing a pass — the legitimate
direction, and the one this project's record explicitly endorses.

### 7.3 Two priors I got wrong, recorded

- **"Trailing vol lags spikes, so the worst day may still exceed 15%."** Wrong. Worst day
  **-5.9%** (2021-11-26); 2018-02-02 -4.7%, 2024-08-02 -4.7%. Vol targeting handled the tail
  materially better than I predicted: DD -50.1% -> **-21.0%**, worst day -19.9% -> **-5.9%**.
- **"Vol carry should be negatively correlated with trend."** Wrong in sign. Measured rho against
  the TSMOM proxy is **+0.141**, not negative. The admission bar is therefore 0.306 rather than the
  ~0.11 I expected, and the sleeve clears it only by 0.06 — a margin that does not survive 7.1.

### 7.4 What is and is not established

**Established:** the front-vs-mid VIX roll premium is real (N4: gross +0.734, CI [+0.282, +1.208],
4/4 subperiods, borrow wall 14.6%/yr) and **vol targeting at an externally-set 10% brings its tail
inside conventional risk limits at a cost of ~0.21 gross Sharpe** (0.734 -> 0.523), because it
de-risks after spikes when the premium is richest. That mechanism is a genuine, reusable finding.

**NOT established:** that the resulting sleeve is a deployable edge. After borrow, its 95% CI spans
zero; after multiplicity across 22 probes, DSR = 0.31. **It is not an alpha and must not be
reported as one.** It is a known risk premium whose tradeableresidual, at retail borrow cost and
honest multiple-comparison accounting, is not distinguishable from zero.

### 7.5 Ledger (durable)

- **Vol-targeted VIX term-structure sleeve: NO-GO on DEFLATION.** Passes all six pre-committed
  bars; fails DSR (0.3147, hurdle SR* 0.494 vs observed 0.369) and its net 95% CI [-0.112, +0.869]
  spans zero before any correction.
- **Vol targeting DOES fix the short-vol tail** (-50.1% -> -21.0% DD; -19.9% -> -5.9% worst day) at
  ~0.21 gross Sharpe. Reusable mechanism; the target came from TAILWIND, not from this data.
- **Volatility carry rho vs trend is POSITIVE (+0.141), not negative.** The "short-vol diversifies
  trend" intuition is wrong on this window; do not assume a low admission bar.
- **Pre-registration defect to avoid repeating: never gate the CI on one series and the threshold
  on another.** Require the CI on the series you would actually trade.
- **Any campaign past ~10 probes needs a deflation gate written INTO the pre-registration.**
- Family **CLOSED** per section 6. Free-data search stands at **22 probes, zero deployable alpha**.

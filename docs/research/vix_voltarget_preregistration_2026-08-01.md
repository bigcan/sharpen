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

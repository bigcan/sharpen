# PRE-REGISTRATION — N2: commodity session premium, OOS instruments + independent-venue test

**Written:** 2026-08-01, **after** N1's pre-registered commodity *secondary* was observed and
**before** any statistic on any out-of-sample instrument, any dividend-stripped series, any
subperiod, or any spot-gold series was computed.

## 0. The new information that justifies a second pre-registration

N1's §7 stop rule closed the equity session family and forbade re-cutting N1. It permitted a
separate pre-registration that **states what new information justifies it**. That information is:

N1 pre-committed a sign for four asset classes and required them reported separately. Three
behaved as nulls (equity +0.097, bond +0.082, currency −0.423 — CIs spanning zero or negative).
The commodity class did not: **gross Sharpe +0.5008, CI95 [+0.1024, +0.9248], excluding zero**,
net@1bp +0.215, with **4 of 5 instruments positive** (GLD +0.569, SLV +0.559, DBA +0.666,
DBC +0.351, USO −0.035). GLD's decomposition is extreme: **+9.73%/yr overnight vs −0.17%/yr
intraday** — essentially all of gold's 2008-2026 return accruing while the US ETF is shut.

**Those five instruments are now in-sample and cannot be used to confirm anything.** This
pre-registration is a confirmatory test on instruments and a venue that had no part in generating
the observation.

## 1. The three ways this is probably an artifact, and the test for each

I take the artifact explanations as the leading hypotheses, not the alpha.

**(A) Bid-ask bounce — the most likely explanation by far.** If closing prints land nearer the bid
and opening prints nearer the ask, close→open is mechanically positive and open→close mechanically
negative by roughly one spread per day, with **no capturable return** — you pay exactly the amount
you appear to earn. This would produce precisely the observed sign pattern in every instrument.
*Test:* the artifact predicts the effect measured **in return terms** scales with the instrument's
**relative spread**. DBA (~$25/share) is many times wider than GLD (~$300+/share); if bounce drives
this, DBA's annualised spread return must be far larger than GLD's in proportion to spread.
Reported as a diagnostic, and the cost model in condition 5 charges each instrument its own
estimated spread, which is the direct defence.

**(B) ETF-specific microstructure** — NAV premium/discount cycling, stale opening prints in thin
funds, opening-auction imbalance. *Test:* **condition 4**, the independent-venue bar. Spot gold
(XAUUSD, Dukascopy tick-derived hourly, 2008-2026) has no ETF, no NAV, no opening auction and no
closing auction. If the pattern is ETF plumbing it cannot appear there.

**(C) Distribution artifact** — the mechanism that killed N1. *Test:* condition 3, price-only
series. GLD/SLV are physical trusts that pay no cash distribution and DBC/DBA/USO are partnerships,
so this is a weaker prior here than in equities, but it is measured, not assumed.

## 2. Instruments (LOCKED before any download or statistic)

**Primary OOS set — commodity ETFs absent from the N1 panel**, chosen on underlying coverage and
issuer diversity, never on outcome:

`IAU, SGOL, SIVR, PPLT, PALL, GSG, DJP, USCI, PDBC, UNG, BNO, UGA, CORN, WEAT, SOYB, CPER`

IAU and SGOL are gold (different issuers/structures from GLD) and SIVR is silver — deliberately
included so the same underlying is tested through *different* ETF plumbing, which separates
"gold has a session premium" from "GLD has a session premium".

**Inclusion rule, fixed now:** an instrument enters the pooled statistic iff it has **≥1,000
usable daily bars** in 2008-01-01→2026-06-30. Instruments failing this are reported and excluded.
**No instrument is added or dropped after seeing its result.**

**Independent venue:** `XAUUSD` hourly from Dukascopy, 2008-2026, tick-derived, with its own
measured spread.

**Mechanism diagnostic (reported, NOT gating):** single-country equity ETFs `EWJ, EWG, EWU, FXI,
EWA, EWZ`. If the mechanism were simply "the underlying trades while the US ETF is closed", these
should show it strongly — but N1 found EFA **−0.450** and EEM −0.001, which argues against that
mechanism. Committing the diagnostic now prevents a post-hoc story either way.

## 3. Construction — identical to N1, still zero parameters

ETFs: `r_on = Open[t]/Close[t-1] − 1`, `r_id = Close[t]/Open[t] − 1`, book `= r_on − r_id`,
equal-weighted across included instruments, 2 round trips/day.

Spot gold: hourly bars converted to `America/New_York`; **US-ETF session** = bars with ET start
hour in {10,…,15} (10:00–16:00 ET), **non-US session** = all other bars. Book = long non-US
session, short US session, same 2 round trips/day. *(The 09:30–10:00 ET half-hour falls in the
non-US bucket; hourly resolution cannot split it. Stated, not hidden.)*

**Committed sign: +1** on every instrument and on both pooled books.

## 4. Costs — estimated from the data, not chosen by me

Per-instrument effective spreads are estimated with the **Corwin–Schultz (2012) high-low
estimator**, which infers the effective bid-ask spread from daily high/low ranges alone. This is
used instead of a number I pick, because N1's campaign lesson was that generalising one
instrument's spread across a basket understates cost (the H1 error, 4.7×). Each instrument is
charged **its own** estimate; negative estimates are floored at zero and the cross-sectional
median is reported. Spot gold is charged its **own measured** Dukascopy spread. The **cost wall**
in bp is reported for every book so no verdict depends on the estimator being exactly right.

## 5. Pre-committed pass conditions — ALL must hold

1. **Pooled OOS commodity gross Sharpe CI95 (block bootstrap) excludes zero.**
2. **≥60% of included OOS instruments individually gross-positive.**
3. **Price-only (distribution-stripped) pooled gross Sharpe > 0.**
4. **INDEPENDENT VENUE — spot gold gross Sharpe > 0 with CI95 excluding zero.** No ETF, no NAV,
   no auction. *If this fails, the ETF result is plumbing and N2 is a NO-GO regardless of
   everything else.*
5. **Pooled net Sharpe ≥ 0.30** charging each instrument its own Corwin–Schultz spread, **and**
   still ≥ 0 at 2× that.
6. **≥3 of 4 subperiods gross-positive** (2008-2012, 2013-2017, 2018-2022, 2023-2026).
7. **Negative control CI includes zero**, scored on the same cost basis.

Anything less than all seven is a **NO-GO**.

## 6. Honest priors

**For:** commodity price discovery genuinely is global and largely outside US ETF hours (London
gold fixes, Shanghai, 24h futures), so a session-concentrated risk premium is economically
coherent rather than a data-mined coincidence; the N1 signature was pre-registered, not a
post-hoc slice; it appeared in 4 of 5 instruments spanning metals, broad commodities and
agriculture, which is hard for a single-instrument artifact to produce; the rule has zero
parameters; and the book is beta-neutral.

**Against, and decisive if they hold:** (a) **bid-ask bounce explains the entire sign pattern**
and is the single most likely cause — the effect is measured on closing and opening *prints*,
exactly where bounce lives; (b) the cost wall is **0.70 bp round trip** for a 0.30 net Sharpe,
which is *tight* — GLD may clear it but DBA and USO almost certainly do not, so even a real effect
may be harvestable in only one or two names; (c) gold's 2008-2012 bull could carry the whole
result, and N1 already showed the equity version was a single-regime artifact; (d) commodity ETFs
carry roll and contango effects (USO catastrophically) that have nothing to do with sessions;
(e) ~21 consecutive falsifications in this project is the strongest prior available.
**A null is still the most likely outcome, and condition 4 is where I expect it to die.**

## 7. Stop rule

**One draw.** N2 is confirmatory; there is no N3 on this family. No re-cut by instrument subset,
holding window, session boundary, or conditioning variable. If the pooled test fails but spot gold
passes condition 4, that is recorded as a *phenomenon* (as sub-daily FX reversal was) and **not**
as an alpha, and it still requires its own future pre-registration to become tradeable.

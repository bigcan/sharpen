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

## 8. Results — **NO-GO**. The gating test fired and the ETF signature is a measurement artifact.

Run 2026-08-01, `results/commodity_session/commodity_session.json`. Spot gold: 110,856 hourly bars
2008-2026, per-year uniformity verified (no thin years) after two data bugs were fixed mid-run.

| pre-committed bar | result |
|---|---|
| 1. pooled OOS gross CI excludes zero | PASS — +0.9295, CI95 [+0.4961, +1.3708] |
| 2. >=60% individually gross-positive | PASS — 14/16 (88%) |
| 3. price-only (distribution-stripped) > 0 | PASS — +0.9091 |
| **4. spot gold CI excludes zero** | **FAIL — +0.1505, CI95 [-0.2553, +0.5692]** |
| 5. net >= 0.30 and >= 0 at 2x | FAIL — cost wall 2.63 bp vs a ~5 bp honest spread |
| 6. >=3 of 4 subperiods positive | PASS — 4/4 (+1.09, +1.06, +1.05, +0.52) |
| 7. control CI includes zero | PASS — +0.171, CI [-0.297, +0.634] |

**VERDICT: NO-GO.**

### 8.1 The ETF result reverses in spot gold — per hour, the US session earns MORE

The ETF panel says gold's return accrues almost entirely while the US market is shut: GLD
**+9.12%/yr overnight against +0.16%/yr intraday**, i.e. 98.3% of the total in the overnight
window. Spot gold, over the same calendar and comparable windows, says the opposite once window
length is controlled for:

| | window | return | **return per hour** | variance per hour |
|---|---|---|---|---|
| non-US session | 17.2 h | +5.37%/yr | **+0.312 %/yr/h** | 4.24e-6 |
| US session | 5.9 h | +2.86%/yr | **+0.484 %/yr/h** | 7.53e-6 |

**In the venue with no ETF, the US session earns 1.55x MORE per hour than the rest of the clock.**
The ETF measurement implies the US session earns ~0.025 %/yr/h — understating it by roughly 20x.
Gold does not have an overnight return premium. GLD's prints do.

Subperiods confirm the spot book is not an effect: -0.18, -0.20, +0.33, +0.91 — only 2 of 4
positive, and both positives are recent.

### 8.2 Which artifact it was, and why the other tests could not see it

Section 1 named three candidate artifacts and built a test for each. **(A) bid-ask bounce is
falsified** — corr(spread, annual spread-return) = **-0.267** with slope -25 bp per bp, where pure
bounce requires about +1 and about +252. **(C) distributions are falsified** — price-only +0.9091
against total-return +0.9295. The survivor is **(B), ETF-specific measurement structure**, which is
precisely what condition 4 was built to catch, and it is the one that fired.

### 8.3 The correction I owe: cross-issuer agreement is NOT independent evidence

Mid-run I checked GLD against IAU and SGOL — three issuers, three custodians, three share prices —
and found overnight-return correlations of **0.9967 and 0.9928**, and read that as strong evidence
the effect was real rather than fund plumbing. **That inference was wrong**, and the spot-gold test
is what shows it.

All three funds share the *same measurement geometry*: a 09:30 open print and a 16:00 close print
imposed on an underlying that trades ~23 hours. Any bias in that geometry is common to all of them,
so their agreement is guaranteed whether or not the effect exists. The 0.997 correlation tested
**fund plumbing** and found none — it could never have tested **measurement structure**, because
that is held constant across the whole instrument class.

**Durable rule: replication across instruments of the same TYPE controls for issuer, not for
measurement. Only a different VENUE — a different price-formation mechanism — tests the latter.**
This has the same shape as the campaign's earlier lesson that a synthetic fixture gave the wrong
physics: an internally consistent set of measurements agreeing with each other is not evidence, if
they share the assumption under test.

### 8.4 It was cost-dead as well, independently

Even granting the ETF signature, the pooled book breaks even at **2.63 bp** round trip and reaches
SR 0.30 at **1.78 bp**, against an honest Roll-estimated median of ~5 bp (the pre-committed
Corwin-Schultz figure of 23.7 bp is itself unusable here — see 8.5). Spot gold is worse: its cost
wall is **0.50 bp** against its own **measured** 2.18 bp tick spread — dead by more than 4x. So
condition 5 fails on any defensible cost model, not only the committed one.

### 8.5 The pre-committed cost model was circular — record it, do not reuse it

Corwin-Schultz infers the spread from the one-day versus two-day high/low relationship, so a large
**overnight gap** inflates the two-day range and is booked as "spread". This probe's hypothesis
selects precisely for instruments with large overnight moves, so **the estimator's bias is maximal
on exactly the instruments under test** — it returned 16.5 bp for GLD, whose quoted spread is a
cent or two on a ~$300 share. Roll (1984) on 5-minute intraday returns, with the first bar of each
session dropped so no gap can enter the autocovariance, gives a ~5 bp median instead.

**Never price a session strategy with a range-based spread estimator.** The measurement and the
hypothesis are the same quantity.

### 8.6 Ledger (durable)

- **Commodity-ETF overnight session premium: NO-GO, and specifically an ARTIFACT** — it does not
  reproduce in spot gold, where the per-hour ordering reverses. Do not re-propose for any
  US-listed ETF on a 24-hour underlying.
- **The country-ETF diagnostic (pre-registered, non-gating) is consistent**: EWJ -0.616,
  EWU -0.433, FXI -0.327, EWG -0.318, EWA -0.273 — five of six negative. The naive "the underlying
  trades while the US ETF is closed" mechanism is falsified in both directions.
- **Cross-issuer replication does not test measurement structure** (8.3). Venue does.
- **Range-based spread estimators are invalid for session/overnight strategies** (8.5).
- Per section 7 this family is **CLOSED**. There is no N3: the phenomenon clause in section 7 is
  not triggered, because condition 4 failed — there is no confirmed phenomenon to carry forward,
  only a refuted one.

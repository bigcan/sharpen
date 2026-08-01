# PRE-REGISTRATION — N4: VIX term-structure carry, front vs mid, dollar-neutral

**Written:** 2026-08-01, **before any Sharpe, drawdown or subperiod statistic on any VIX instrument
was computed.** Only data *availability* (row counts and date ranges) was checked, to pick
instruments with continuous history — the same procedure N2 used with its ≥1,000-bar rule.

## 0. Why this, after 20 falsifications

The **variance risk premium** is the largest and most persistent premium in liquid markets that
this project has **never validated**. The ledger's `options-as-alpha CLOSED` and
`Options-VRP CLOSED` entries do not settle it: the VRP work was closed because its 0.61 anchor was
traced to **USDC contamination in crypto options data**, i.e. a data defect, and the Taiwan TXO
thread was closed on its own instruments. Neither tested a listed US VIX-futures expression, and
neither is evidence about whether the premium exists.

It also fits the constraint every previous probe died on: **turnover is low** (the term structure
flips sign only ~15-20 times a year), so cost cannot be the executioner — and N3 established that
removing the cost constraint does not by itself produce an edge, so this must stand on signal.

**I am not claiming novelty.** If this passes it is a *known* risk premium newly validated for this
project, not a discovery. It will be reported that way.

## 1. Instruments (LOCKED)

- **Signal:** `^VIX` (2004+) and `^VIX3M` (2006+). Contango when `VIX3M > VIX`.
- **Front leg:** `VIXY` — 1x long VIX short-term futures, **continuous since 2011-01-04**, no
  leverage change.
- **Mid leg:** `VIXM` — 1x long VIX mid-term futures, continuous since 2011-01-04.

**Explicitly rejected and why, before seeing any result:** `SVXY` changed from −1x to −0.5x in
February 2018, a structural break mid-sample; `VXX`'s current ETN series only begins 2018-01-25.
Using either as the primary would confound the leverage change or throw away 7 years. `UVXY` is
1.5x/2x levered. None of these will be substituted in later.

Window: **2011-01-04 → 2026-06-30** (~3,894 days), set by VIXY/VIXM inception.

## 2. Construction — zero free parameters

**Position:** when `VIX3M > VIX` at the close of day *t−1*, hold **short 1 unit VIXY and long 1
unit VIXM** on day *t*; otherwise flat. Dollar-neutral, 1:1, signal lagged one day.

The spread form is chosen **on tail grounds, stated in advance**: an outright short-front position
is the classic XIV trade and loses catastrophically in a spike. Both legs rise together in a spike,
so the mid leg absorbs part of it. No hedge ratio is estimated — 1:1 is fixed, because any
fitted ratio would be a free parameter and this campaign's rule is zero.

**Committed sign: +1** (front-month roll yield exceeds mid-term, so the spread earns).

## 3. Costs — including borrow, which is real here

- **Trading:** 10 bp round trip on each leg, charged on every position change.
- **Borrow on the short VIXY leg:** **3%/yr** primary, **10%/yr** stress. VIXY is periodically
  hard-to-borrow and ignoring this would flatter the result. A **borrow wall** (the annual borrow
  rate at which net Sharpe reaches 0.30 and 0) is reported so the verdict does not depend on my
  choice of 3%.

## 4. Pre-committed pass conditions — ALL must hold

1. **Gross Sharpe CI95 (block bootstrap) excludes zero.**
2. **Net Sharpe ≥ 0.30** at 10 bp + 3%/yr borrow, and **≥ 0** at 10 bp + 10%/yr borrow.
3. **≥3 of 4 subperiods gross-positive** (2011-2014, 2015-2018, 2019-2022, 2023-2026).
4. **TAIL GATE — max drawdown ≤ 35% AND worst single-day loss ≤ 15%.** This project's stated
   objective includes prop-firm challenges, where a single catastrophic day is disqualifying
   regardless of Sharpe. A short-volatility book that clears a Sharpe bar but can lose half its
   capital in a day is **not** a deployable alpha here, and this gate says so in advance rather
   than after seeing the equity curve.
5. **2018-02-05 reported explicitly** — the XIV event. The book must be **positive-or-survivable**
   that day under gate 4; I note now that the term structure *was in contango* going into it, so
   the signal does **not** filter it out and this is a live risk, not a hypothetical.
6. **Negative control:** the identical book on a randomly shuffled contango signal (block-shuffled
   to preserve run lengths) must have a CI including zero.

Anything less than all six is a **NO-GO**.

## 5. Honest priors

**For:** the premium is real, large and mechanically grounded (VIX futures roll down toward spot in
contango, which is ~75-80% of days); the spread form removes the pure-beta objection; turnover is
low so cost cannot decide it; and both legs are liquid listed ETFs with 15 years of continuous
history.

**Against, and decisive if they hold:** (a) **the tail gate is where I expect this to die** — on
2018-02-05 VIXY rose far more than VIXM and a 1:1 dollar-neutral spread still takes a very large
one-day loss, and gate 4 was written to fail exactly that; (b) 1:1 dollar weights are **not**
vega- or beta-neutral, so the spread is structurally short-vol rather than truly neutral;
(c) borrow on VIXY can spike far above 10%/yr precisely when the trade is crowded; (d) the front-vs-mid
roll differential has compressed since 2018 as the trade became well known; (e) **20 consecutive
falsifications**. **A null or a tail-gate failure is the most likely outcome.**

## 6. Stop rule

**One draw.** No alternative hedge ratio, signal threshold, holding rule, instrument substitution,
or subperiod re-cut. If N4 fails, the VIX term-structure family is closed for this project and the
free-data search stands at 21 probes.

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

## 7. Results — **NO-GO** (fails 2 of 6), but it fails differently from every prior probe

Run 2026-08-01, `results/vix_term_structure/vix_term_structure.json`. 3,894 days 2011-01-04 to
2026-06-30. Contango on **92.3%** of days; 193 switches (**12.5/yr**).

| pre-committed bar | result |
|---|---|
| 1. gross CI excludes zero | PASS — **+0.7338**, CI95 [+0.2823, +1.2075] |
| 2. net >= 0.30 @3% borrow and >= 0 @10% | PASS — **+0.5907** and **+0.4149** |
| 3. >=3 of 4 subperiods positive | PASS — **4/4** (+0.56, +0.59, +0.99, +0.75) |
| **4. max DD <= 35% and worst day <= 15%** | **FAIL — max DD -50.1%, worst day -19.9%** |
| 5. 2018-02-05 survivable | PASS — book **-0.2%** (see 7.2, my prediction was wrong) |
| **6. control CI includes zero** | **FAIL — +0.6398, CI95 [+0.5125, +0.7825]** |

**VERDICT: NO-GO.**

### 7.1 What is actually true here — the premium is real, the signal is not

This is the first probe in 21 that fails on **neither existence nor cost**. Gross +0.73 with a CI
excluding zero, positive in **all four** subperiods, and a **borrow wall of 14.6%/yr** for a 0.30
net Sharpe — five times the primary assumption. The front-vs-mid VIX roll differential is a real,
large, persistent premium, now measured for this project rather than assumed.

**But the term-structure signal is nearly inert.** The block-shuffled control earns **+0.6398**
against the real signal's +0.7338 — the timing rule adds roughly **0.09 Sharpe**. Because contango
holds 92.3% of days, the book is in the market 92.2% of the time, and essentially all of the return
is *being exposed to the roll*, not *choosing when*.

**Honest decomposition: ~0.64 of premium (beta) + ~0.09 of signal (alpha), minus a disqualifying
tail.**

### 7.2 CORRECTION — my stated fact about the XIV event was wrong

Section 4 asserted, as a fact and in advance: *"the term structure **was in contango** going into
[2018-02-05], so the signal does **not** filter the XIV event and this is a live risk."*

**That is false.** At the 2018-02-02 close `VIX3M > VIX` was **False** — the curve had already
inverted — so the book was **flat** on 2018-02-05 and lost **-0.2%** while VIXY rose **+34.2%** and
VIXM **+13.5%**. The signal *did* filter the event, and condition 5 passes for the opposite reason
to the one I gave.

This matters beyond bookkeeping: I chose the spread construction "on tail grounds" partly to
survive an event the signal already avoided, and I wrote a confident factual claim about a specific
date without checking it. **The gate passed despite my reasoning, not because of it.**

### 7.3 And the tail still disqualifies it — from a different direction

Avoiding February 2018 did not make this safe. The worst days are **2021-11-26 (-19.9%)**,
2020-06-11 (-19.8%), 2024-08-02 (-15.1%), 2016-06-24 (-14.4%), 2021-01-27 (-13.9%) — Omicron, a
mid-COVID vol spike, the yen-carry unwind, Brexit, and the meme squeeze. **Max drawdown -50.1%**,
and every subperiod carries a 29-39% drawdown.

Under this project's stated objective — prop-firm challenges where a single day can disqualify — a
-19.9% day and a -50% drawdown are not deployable at any Sharpe. Gate 4 was written in advance
precisely so this could not be argued away after seeing a 0.73.

### 7.4 Condition 6 was MIS-SPECIFIED, and that is my error, not a result

Requiring the control's CI to *include zero* was wrong for a signal that is on 92% of the time. A
block-shuffled control inherits nearly the full exposure, so it inherits the premium and **cannot**
be centred on zero however inert the signal is. Condition 6 therefore tested *"does the premium
exist"* (it does) rather than *"does the signal add value"* (it barely does).

The correct control for signal value is the **always-on** book, not zero. Reading the numbers that
way — +0.7338 signalled against +0.6398 shuffled — gives the right answer anyway, so the NO-GO
stands on gate 4 regardless. But the condition as written was not the test I intended.

**Durable rule: a control must be able to come out negative. If a control inherits the exposure
whose premium is under test, its null is not zero, and comparing it to zero measures nothing.**
Same family as this project's NULL-DEGEN-01 finding — a null that could not vary.

### 7.5 Ledger (durable)

- **VIX front-vs-mid term-structure carry: NO-GO on TAIL.** Gross +0.734 (CI [+0.282, +1.208]),
  net +0.591 at 3% borrow, 4/4 subperiods, borrow wall 14.6%/yr — **the premium is REAL and
  survives cost**. It fails on **max DD -50.1% / worst day -19.9%**, both beyond the pre-committed
  gates.
- **The term-structure timing signal is nearly inert**: shuffled control +0.640 vs +0.734. Contango
  holds 92.3% of days, so this is ~88% beta and ~12% timing. Do not call it a timing strategy.
- **2018-02-05 was avoided because the curve had already inverted** — contrary to what this
  pre-registration asserted. Do not repeat the claim that the contango filter fails to catch XIV.
- **A control that cannot come out negative measures nothing** (7.4).
- Family **CLOSED** per section 6. Free-data search stands at **21 probes, zero deployable alpha**.
- WARNING for any revisit: the live question is **not** whether the premium exists — it does — but
  whether a tail-controlled expression can bring DD inside 35% **without fitting the hedge ratio to
  the observed crashes**. That needs its own pre-registration and is **not** implied by this result.

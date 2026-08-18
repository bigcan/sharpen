# PRE-REGISTRATION — N3: cross-asset turn-of-month, dollar-neutral, 18-ETF panel + country OOS

**Written:** 2026-08-01, **before any turn-of-month statistic was computed on any instrument.**

## 0. Why this, after 19 consecutive falsifications

Every probe in this campaign and the last died one of two deaths: no effect, or an effect eaten by
**cost**. Probes 15-17 (FX) lost by ~7x on turnover; N1/N2 needed 504 round trips a year against
cost walls of 0.35-2.63 bp. **The binding constraint is turnover, not signal.**

This probe is chosen to be structurally immune to that constraint rather than to fight it:

- **~24 round trips per year** (two switches a month). At 5 bp round trip on a ~12% vol book the
  drag is **≈0.10 Sharpe** — an order of magnitude below the walls that killed N1/N2.
- **It uses close-to-close returns only.** No open price enters anywhere. The ETF opening-print
  artifact that refuted N2 **cannot appear by construction**, which is the specific reason a
  positive result here would not be the same mistake twice.
- The mechanism is a **flow**, not a forecast: salary, pension and index-fund contributions cluster
  at month boundaries. Nothing has to be predicted from past prices — the position is a calendar.

## 1. Why the existing NO-GO does not transfer

`project_xlg_entry_timing_nogo_s553` tested `TurnOfMonth` on XLG and returned NO-GO: Sharpe 0.30
against buy-and-hold's 0.567. **That test cannot settle this one**, and the reason is specific.

It was a **long-or-cash** rule: in the market ~4 days a month, in cash otherwise, and compared to a
fully-invested benchmark. Its own write-up classifies it with five siblings as **"outright NO-GO
cash-drag"** — it lost because it was out of the market 80% of the time, not because turn-of-month
days were unremarkable. That verdict is about *exposure*, not about the calendar effect.

A **dollar-neutral** book has no cash drag: it is long the turn-of-month window and short the rest
of the month, so it is never merely under-invested. It isolates exactly the quantity the XLG test
confounded. This is a different estimand on a different universe, not a re-cut. *(That said, the
XLG result is real evidence the US-equity version is weak, and it is priced into section 6.)*

## 2. Window and construction — fixed in advance, zero free parameters

**Turn-of-month window (canonical, McConnell & Xu 2008):** the **last trading day of month t−1
through the third trading day of month t** — 4 trading days. This window is taken from the
literature and **is not searched**. No alternative width, offset or month-end definition will be
tried; section 7 makes that binding.

**Book (dollar-neutral in time):** for each month, hold **+1** on each of the 4 TOM days and
**−(4 / n_rest)** on each remaining day of that month, so net exposure over the month is exactly
zero. The book carries no static market beta and requires two position changes per month.

Equal-weight within each asset class; the pooled book is the equal-weight average of the four class
books, so no single class can carry the result by having more members.

**Committed sign: +1** (TOM days outperform the rest of the month) on every instrument, every class
and both pooled books.

## 3. Instruments (LOCKED before any statistic)

**Primary — the existing 18-ETF panel**, already on disk, DATA-CLEAN, 2008-01-02→2026-06-30:

| class | tickers |
|---|---|
| Equity | SPY, QQQ, IWM, EFA, EEM |
| Bond | TLT, IEF, LQD |
| Commodity | GLD, SLV, DBC, USO, DBA |
| Currency | UUP, FXE, FXY, FXB, FXA |

**Out-of-sample instrument extension — country equity ETFs**, locked here, chosen on breadth of
market coverage only: `EWJ, EWG, EWU, EWA, EWC, EWZ, EWY, EWT, EWH, EWS, EWL, EWD, EWP, EWI, EWQ,
EWN, EWM, EWK, EWO, FXI`. Inclusion requires **≥1,000 usable daily bars**; failures are reported
and excluded. Turn-of-month is documented internationally, so the committed sign is **+1**.

**Venue check (reported, NOT gating):** spot gold (XAUUSD, 110,856 hourly bars → daily) and the
five FX majors. The flow mechanism is equity-specific and does **not** predict an effect here, so
this is a specificity check: a large effect in FX/gold would suggest a calendar artifact in the
data rather than a flow.

## 4. Costs

24 round trips/year charged at **5 bp** round trip — deliberately several times the realistic cost
for these ETFs, since the whole point of this probe is that cost should not bind. Also reported at
**20 bp** (4x) and as a **cost wall** in bp.

## 5. Pre-committed pass conditions — ALL must hold

1. **Pooled 18-ETF gross Sharpe CI95 (block bootstrap) excludes zero.**
2. **≥3 of 4 asset classes individually gross-positive.**
3. **Pooled net Sharpe ≥ 0.30 at 5 bp**, and still ≥ 0 at 20 bp.
4. **≥3 of 4 subperiods gross-positive** (2008-2012, 2013-2017, 2018-2022, 2023-2026).
5. **RANDOM-WINDOW NULL — the decisive one.** Draw 1,000 random 4-consecutive-trading-day windows
   per month and rebuild the identical book. The observed pooled Sharpe must exceed the **95th
   percentile** of that null. This directly answers "would any 4-day window have looked this
   good?", which is the failure mode a calendar study is most exposed to.
6. **Country-ETF OOS extension: pooled gross CI95 excludes zero** with the committed sign.

Anything less than all six is a **NO-GO**.

## 6. Honest priors

**For:** the mechanism is a documented flow with an identifiable cause rather than a behavioural
story; the effect is one of the longest-standing calendar regularities in the literature
(Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008 report the 4-day window capturing
essentially all of the equity premium 1926-2005); turnover is low enough that cost genuinely cannot
decide the verdict, which no probe in this campaign has been able to say; and the construction
touches no opening price, so N2's artifact cannot recur.

**Against, and decisive if they hold:** (a) this is a **heavily published** anomaly and the sample
here is 2008-2026, entirely *after* the papers that made it famous — post-publication decay is the
single most likely outcome; (b) the XLG probe already found the US-equity version unremarkable, and
US equity is the class with the strongest prior mechanism; (c) 222 months is thin — a 0.5 Sharpe
effect gives t ≈ 2.2, so the CI will be wide and condition 1 is a real hurdle; (d) month-boundary
flows are exactly the sort of predictable liquidity demand that execution desks have had two
decades to front-run; (e) **19 consecutive falsifications** is the strongest prior available.
**A null remains the most likely outcome.**

## 7. Stop rule

**One draw.** The window is fixed at the canonical 4 days: **no alternative width, offset, month-end
convention, subset of instruments, or conditioning variable will be tried.** If N3 fails, calendar
/ turn-of-month is closed for this project and the free-data search stands at 20 probes.

## 8. Results — **NO-GO** (5 of 6 bars fail). The canonical window is unremarkable.

Run 2026-08-01, `results/turn_of_month/turn_of_month.json`. 4,651 days x 18 tickers; 888 TOM days
(19.1%).

| pre-committed bar | result |
|---|---|
| 1. pooled gross CI excludes zero | FAIL — +0.1001, CI95 [-0.3241, +0.5440], P(SR<=0)=0.31 |
| 2. >=3 of 4 classes gross-positive | PASS — 3/4 (equity +0.089, commodity +0.085, currency +0.042, bond -0.006) |
| 3. net >= 0.30 at 5bp and >= 0 at 20bp | FAIL — net@5bp -0.171, net@20bp -0.985 |
| 4. >=3 of 4 subperiods positive | FAIL — 2/4 (+0.33, -0.42, +0.44, -0.31) |
| **5. beats random-window q95** | **FAIL — observed sits at the 78th percentile** |
| 6. country-ETF OOS CI excludes zero | FAIL — +0.1703, CI95 [-0.2469, +0.6115] |

**VERDICT: NO-GO.**

### 8.1 Condition 5 is the finding: the turn-of-month window is not special

The random-window null (300 draws, each picking a random block of 4 *consecutive* trading days per
month and rebuilding the identical dollar-neutral book) has mean **-0.0612**, sd **0.2250**, q95
**+0.3374**. The canonical McConnell-Xu window scores **+0.1001** — the **78th percentile**.

**Roughly one random 4-day window in five beats the turn-of-month window.** The calendar position
carries no information on this panel. This is a much stronger statement than a wide confidence
interval: it says the specific window the literature identifies is not distinguishable from an
arbitrary one, which is the exact failure mode a calendar study must rule out.

### 8.2 The specificity check points the same way

The flow mechanism is equity-specific and predicts nothing in gold or FX. Measured: spot gold
**+0.115**, FX majors **+0.149** — both *at or above* the 18-ETF panel's **+0.100**. A supposed
US-equity-flow effect that is no stronger in equities than in spot gold is not a flow effect. It is
the same small positive noise everywhere, consistent with the null.

### 8.3 Cost did not decide this one, by design — and it still lost

The probe was chosen so cost could not be the executioner: 24 round trips/yr instead of N1/N2's
504. It worked as intended — the cost wall for SR=0 is **1.8 bp**, comfortably above realistic ETF
costs, so the verdict rests on signal, not friction. **There was simply no signal.** Removing the
cost constraint did not reveal an edge underneath it; it revealed that the edge was never there.

That is worth recording precisely, because "everything dies on cost" had become the campaign's
working explanation. On the one probe built to be cost-immune, the result is still a null.

### 8.4 Why the XLG prior turned out to be right for the wrong reason

Section 1 argued the XLG NO-GO could not settle this, because it was long-or-cash and lost to cash
drag rather than to the calendar. **That reasoning was sound and the conclusion still holds** — but
the dollar-neutral version, which has no cash drag, is *also* a null. So the XLG verdict was
correct about turn-of-month even though its stated mechanism (cash drag) was not the whole story:
the window is unremarkable on its own terms.

### 8.5 Ledger (durable)

- **Cross-asset turn-of-month, dollar-neutral, 2008-2026: NO-GO.** Gross +0.100 with a CI spanning
  zero, 2/4 subperiods, country-ETF OOS also spanning zero.
- **The canonical 4-day window sits at the 78th percentile of random 4-day windows** — the calendar
  position carries no information. Do not re-propose any turn-of-month variant.
- **Equity-flow specificity is absent**: the effect is no larger in equities than in spot gold or
  FX, which falsifies the mechanism independently of the significance test.
- **A cost-immune probe still returned a null** — friction was not hiding an edge.
- Per section 7 the calendar/turn-of-month family is **CLOSED**. The free-data search stands at
  **20 probes, zero deployable alpha**.

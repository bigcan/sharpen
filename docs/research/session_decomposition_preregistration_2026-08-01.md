# PRE-REGISTRATION — N1: overnight vs intraday session decomposition, 18-ETF cross-asset panel

**Written:** 2026-08-01, **before any Sharpe on any session-split series was computed.** The panel
(`data/raw/cross_asset_panel/ohlcv_daily.parquet`) has been on disk since the TSMOM work and its
*close-to-close* returns are extensively studied here; its **open** column has never been used.

## 0. Why this is a new family, not a re-cut of a barred probe

The 17-probe free-data campaign (`project_alpha_search_16_probes_2026_07_31`) is closed, and its
stop rules are honoured: **no** further FX-reversal probe, **no** Taiwan small-cap re-cut, **no**
new instruments/holding periods for H1. This tests a different object entirely.

Every one of the 20-odd falsified families in this project's ledger asks the same question — *can
past prices forecast future returns?* — and the answer here has been no, ~20 times. This probe asks
a **different** question: **not when to be in the market, but which hours of the day the market's
return actually accrues in.** No forecast is involved. The position is a fixed calendar rule.

That distinction matters economically, because it changes what has to be true for the strategy to
work. A forecasting alpha needs an inefficiency that survives arbitrage. A session-decomposition
alpha needs only that the *risk premium is unevenly distributed across the trading day* — which
is a compensation question, not a mispricing one, and is not competed away by the same mechanism.

**Literature basis (external, not this repo):** Lou/Polk/Skouras (2019, JFE, "A Tug of War"),
Cliff/Cooper/Gulen (2008), Berkman et al. (2012) all document that in US equities essentially the
entire equity premium accrues **overnight** (close→open), with the intraday (open→close) component
approximately zero or negative, over multiple decades and internationally. This has never been
measured on this project's own panel, in any form.

## 1. Instruments (LOCKED — the existing panel, no selection)

All **18** ETFs already in `cross_asset_panel`, chosen because they are the panel the deployed
TSMOM book uses. **None is selected on outcome; nothing will be added or dropped after seeing a
result.**

| class | tickers |
|---|---|
| Equity (5) | SPY, QQQ, IWM, EFA, EEM |
| Bond (3) | TLT, IEF, LQD |
| Commodity (5) | GLD, SLV, DBC, USO, DBA |
| Currency (5) | UUP, FXE, FXY, FXB, FXA |

Window 2008-01-02 → 2026-06-30, 4,652 daily bars/name, `auto_adjust=true`, already DATA-CLEAN
(manifest `status: PASS`, 80 outliers repaired, zero stale-flagged tickers).

**SPY is the hypothesis generator** (it is the security the cited literature is built on) and is
therefore **excluded from the primary pooled statistic**, mirroring the H1 design that excluded
EURUSD. The out-of-sample equity set is **QQQ, IWM, EFA, EEM**.

## 2. Construction — a fixed calendar rule, no parameters

For each name, decompose the daily total return into two disjoint, exhaustive components:

```
r_overnight[t] = Open[t]  / Close[t-1] - 1
r_intraday[t]  = Close[t] / Open[t]    - 1
(1 + r_on)(1 + r_id) = Close[t]/Close[t-1]        # exact by construction
```

**Primary book — dollar-neutral in time, "long overnight / short intraday":**
hold +1 unit from the close to the next open, and −1 unit from that open to that close. Net
exposure over a full day is zero, so the book carries **no static market beta**. Its return is
`r_on[t] − r_id[t]` to first order (compounded exactly in code).

**There are no free parameters.** No z-window, no lookback, no threshold, no vol target, no
filter. Every name is equal-weighted within its class and each class is reported separately. This
is the entire specification.

**Committed sign: +1** (overnight > intraday) for every equity name, and **+1** for the pooled
out-of-sample equity book. The sign is committed for the other three classes too, but those are
**secondary** and are reported separately — they do not enter the primary verdict.

## 3. The dividend discriminator — pre-committed, and it can kill this on its own

An overnight-only holder owns the shares across the ex-dividend open, so on an `auto_adjust=true`
(total-return) series the dividend is credited **entirely to the overnight leg**. For SPY that is
~1.3%/yr arriving mechanically, not from price discovery.

**This must be separated or the result is worthless.** The run therefore computes the identical
statistic on `ohlcv_daily_raw.parquet` (price-only, dividends NOT added back), which *penalises*
the overnight leg by the ex-dividend drop and so is a strict lower bound on the price effect.

- If the effect exists on **adjusted only** → it is a dividend-accounting artifact, **NO-GO**,
  regardless of what the adjusted Sharpe says.
- If it survives on **raw** prices → it is a genuine price-discovery effect and the adjusted number
  is the tradeable one.

## 4. Pre-committed pass conditions — ALL must hold

1. **Pooled gross Sharpe CI95 (stationary bootstrap) excludes zero** on the four out-of-sample
   equity names (SPY excluded — it generated the hypothesis).
2. **≥3 of 4 out-of-sample names individually gross-positive.**
3. **Survives the dividend discriminator (§3):** pooled gross Sharpe on RAW prices > 0.
4. **Pooled net Sharpe ≥ 0.30** at 1.0 bp round-trip, 2 round trips/day (504/yr) — the stated
   liquid-US-ETF cost. The **cost wall** (breakeven bp) is reported alongside so the verdict does
   not hinge on my choice of 1.0 bp.
5. **Net still ≥ 0 at 2× cost** (2.0 bp).
6. **≥3 of 4 subperiods gross-positive** (2008-2012, 2013-2017, 2018-2022, 2023-2026) — method
   rule 6: an overlay result driven by one regime is not an effect.
7. **Negative control is a clean null:** the same construction where each day's *total* return is
   split between two pseudo-sessions at a random fraction (preserving the daily total exactly)
   must produce a pooled gross CI that **includes zero**. Scored on the **same cost basis** as the
   metric it controls (method rule 4).

Anything less than all seven is a **NO-GO**.

## 5. Sleeve-admission arithmetic, committed in advance

The deployed TSMOM book runs net SR ≈ 0.60. The closed-form admission rule from the campaign is
`s2 > s1·(√(2+2ρ) − 1)`. This book is beta-neutral by construction and is a *session-timing* rule,
so ρ against a daily-rebalanced trend book should be near zero; at ρ = 0 the bar is
**s2 > 0.60 × 0.414 = 0.249**. Condition 4 (≥ 0.30) is therefore set *above* the admission bar, not
at it — a pass is a sleeve, not a curiosity.

## 6. Honest priors

**For:** the effect is one of the most replicated regularities in empirical finance, documented
across decades, countries and asset classes by top-journal work; it needs no forecasting skill; the
position rule has zero parameters, so there is nothing to overfit; and the book is beta-neutral, so
a positive result cannot be equity beta in disguise. Turnover is high (504/yr) but US ETF spreads
are ~10-30× tighter *relative to volatility* than the FX cell that killed probe 17 — the cost
regime is genuinely different, which is the specific reason this is worth a draw.

**Against, and decisive if they hold:** (a) the **dividend artifact** in §3 is the single most
likely explanation for any positive adjusted result, and it is a mechanical accounting effect, not
alpha; (b) 504 round trips/yr is still a lot — at 1 bp on a book whose vol may be well under 14%
the drag could exceed the effect, exactly as it did in probes 15-17; (c) the literature's estimates
are pre-2018 and this window is 2008-2026, so decay is likely and the effect may already be
arbitraged; (d) the open print is the least reliable price of the day and any measured effect may
be microstructure noise that is not capturable at the opening auction; (e) ~20 consecutive
falsifications in this project is the strongest prior available. **A null is the most likely
outcome.**

## 7. Stop rule

**One draw.** If N1 fails, the session-decomposition family is closed for the ETF panel — no
re-cuts by holding window, no conditioning variables, no subset re-selection, no intraday-bar
version on the same instruments. A different construction would need its own pre-registration and
would have to state what new information justifies it.

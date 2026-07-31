# PRE-REGISTRATION — Cross-sectional COUNTRY-EQUITY momentum (C1) + negative control (C2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE either signal was computed or any IC was seen.**
**Market:** international / US-listed country ETFs (the "international (US) markets" half of the goal).
**Data:** yfinance daily adjusted closes — free, no quota. **New substrate**, not Taiwan.

## 0. Why this substrate, after five Taiwan negatives

Five probes on Taiwan small/mid-cap all failed today. The measured base rate there is 1-in-6 and the
one survivor (P1) is cost-blocked and decaying, so a sixth draw from the same well is
negative-expected-value. This moves market **and** asset class.

What the NO-GO ledger closes for the US is **single-stock** liquid large-cap cross-section
(momentum, PEAD, XLG timing, distress reversal, LETF), retail intraday/options, and
options-as-alpha. **Country-level cross-sectional equity momentum is none of those** and is not in
the ledger. The validated house edge (TSMOM, net SR 0.60) is *time-series* — each asset against its
own trend. Ranking countries **against each other** is a different signal with its own literature
(Asness-Moskowitz-Pedersen; Bhojraj-Swaminathan).

Depth measured before writing this: **7,641 daily bars** (1996-2026) for 13 core country ETFs, 20
tickers with ≥4,000 — deeper than the Taiwan panel (5,292) and the cross-asset ETF panel (5,133).
Costs are ETF-normal (~5bp round trip) with **no 0.30% transaction tax**, which is what killed
capture in every Taiwan probe today.

## 1. The two signals (LOCKED)

Both: `horizons = (1,5,10,21,63)`, `primary_horizon = 21` (monthly rebalance),
`neutralization = ("winsor","zscore","size")`. Sector neutralization is **deliberately omitted** —
every name is a country equity ETF, so a sector control is degenerate here; `size` (log trailing
dollar ADV) is kept to prevent a liquidity confound, the same control that killed the June
large-cap mirage.

### C1 — Cross-sectional country momentum · `ctry_xs_momentum` · sign **+1**
- **Signal (causal):** 12-1 momentum — `close[t−21]/close[t−252] − 1`, i.e. the twelve-month return
  **skipping the most recent month** (the standard construction, which drops the short-term reversal
  contamination). Uses only bars ≤ t−21.
- **Expected sign: +1** (long past winners).
- **Mechanism:** country-level information diffuses slowly across borders; cross-sectional
  under-reaction produces continuation.

### C2 — NEGATIVE CONTROL · `ctry_null_control` · sign **+1**
- **Signal:** a deterministic pseudo-random score, seeded per (date, ticker) and containing **no
  price information whatsoever**. Regenerated identically on every run.
- **Expected result: IC ≈ 0 and a FAIL.**
- **Why it is here — this is the point of the run.** Five Taiwan probes today produced nominally
  significant ICs (t from +4.2 to +6.7) that were all non-capturable, several with a positive
  rank-IC and a *negative* decile spread. That pattern raises a question none of those probes could
  answer: **does this panel/harness manufacture significance?** C2 answers it directly.
  - C2 shows IC ≈ 0 and fails ⇒ the harness is calibrated, and C1's result — pass or fail — means
    what it says.
  - **C2 shows a "significant" IC ⇒ the harness is broken on this substrate and C1 must be
    discarded regardless of how good it looks.** That outcome would invalidate the run, not produce
    a finding, and is committed as such here.
- C2 also makes the deflation computable: today's S1 run returned `deflation undefined (batch too
  small for DSR)` because a single-signal batch has no trial Sharpe distribution. Two signals fix it.

## 2. Pre-committed pass conditions for C1

- realized sign == **+1**;
- IC-IR ≥ 0.05 and |IC t| ≥ 3.0;
- **the block-bootstrap CI must exclude zero** — not merely a large t. Today's S1 had IC t = +5.00
  with a CI of [−0.0027, +0.0184] straddling zero, because overlapping forward returns inflate the
  t. The CI is the binding statistic;
- **frictionless Sharpe > 0 AND net Sharpe at the standard cost model > 0.** Capturability is a pass
  condition, not a caveat — the funnel's own `PROMISING` verdict never consults it
  (`scorecard.py:136`), which today let a signal with frictionless Sharpe −0.627 carry the label;
- **decile spread must share the sign of the IC.** Five probes today showed positive rank-IC with
  negative decile spread — correlation outside the extremes the book actually trades. This is now an
  explicit bar;
- C2 (the control) must FAIL.

## 3. Multiplicity

Declared **2** for this batch. This is a **new substrate** (international country ETFs), so the
Taiwan cumulative count does not carry over — the hypotheses are not drawn from the same data. The
Taiwan campaigns remain charged at 8 on their own substrate.

## 4. Stop rule

Complete at 1 hypothesis + 1 control. **No C3.** If C1 fails, cross-sectional country momentum is
recorded falsified on free daily data and is not re-probed with a different lookback (6-1, 9-1),
a different skip, or a volatility scaling.

## 5. Promotion ceiling

A pass earns **"PROMISING, forward-incubate"** only. Survivorship: country ETFs that delisted are
not in the current yfinance query, so this panel is **not survivorship-free** and results are
UPPER BOUNDS — flagged in advance, same status as every Taiwan probe.

## 6. Results

*(Empty at commit time on purpose — the sign commitment is verifiable from git history.)*

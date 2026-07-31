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

## 6. Results — C1 **statistically real, economically negligible**; C2 control **clean**

Run 2026-07-31, `results/country_momentum/scorecard.json`. Panel **N=30, T=7,641**
(1996-03-18 → 2026-07-30), 7,636 usable days. C1 coverage 96.2%, C2 100.0%.

### 6.1 The control passed — and that is the load-bearing result

| C2 `ctry_null_control` (21d) | |
|---|---|
| IC-IR | −0.0027 |
| IC t | −0.23 |
| bootstrap CI | [−0.0053, +0.0041] — **straddles zero** |
| p_le_0 | 0.596 |
| DSR | **0.0002** |
| FDR-q / BHY-q | 0.596 / 0.893 |

A signal containing **no price information** scores a clean null on every axis. **The harness does
not manufacture significance on this substrate**, so C1's result means what it says — and,
retrospectively, the five Taiwan probes' significant ICs were genuine rank correlations that simply
were not capturable, not artefacts of the measurement.

This is the first probe today that could distinguish those two explanations, which is why the
control was worth a slot.

### 6.2 C1 clears every pre-committed bar

| bar (from §2, committed before the run) | measured | |
|---|---|---|
| realized sign == +1 | +1 | ✓ |
| IC-IR ≥ 0.05 · \|t\| ≥ 3.0 | 0.1005 · 8.63 | ✓ |
| **bootstrap CI excludes zero** | **[+0.0046, +0.0584]**, p_le_0 0.0104 | ✓ |
| **frictionless > 0 AND net@standard > 0** | **+0.121 / +0.081** | ✓ |
| **decile spread shares the IC sign** | +0.00006 … +0.00135, positive at all 5 horizons | ✓ |
| control must FAIL | C2 DSR 0.0002 | ✓ |

IC ladder: 1d +0.048 · 5d +0.067 · 10d +0.075 · **21d +0.101** · 63d +0.064 — a coherent hump
peaking at the pre-registered primary horizon. DSR 1.0000, FDR-q 0.021, BHY-q 0.031, **HLZ True**.
Cost wall **0.040** (no transaction tax, unlike Taiwan). Recent-2y IC-IR **+0.391** — stronger in the
live window, not inverted.

**This is the first signal in six probes today to clear the capture bar.**

### 6.3 Why it is still NOT an alpha

**Net Sharpe at standard cost is +0.081, against a max drawdown of −0.374.**

That is a real effect of negligible economic size. For scale, the validated house edge — cross-asset
TSMOM — is net SR **0.60** on a book with far shallower drawdowns. A 0.08 Sharpe is not a deployable
strategy; it is a measurement that the effect exists.

The harness independently declines it, on its own gates rather than mine: verdict **LOGGED**, not
PROMISING, because `min_subperiod_ic_ir = −0.024 < 0` (it inverts in one of four sampled
subperiods) and CPCV p05 = −0.12 with 73% of 15 paths positive.

So the honest verdict is: **cross-sectional country-equity momentum is REAL on 30 years of free
data, and too small to trade.** Reporting it as a discovered alpha would be exactly the overclaim
this pre-registration's bars were written to prevent.

### 6.4 Ledger entry (durable)

- **Cross-sectional country-equity momentum: CONFIRMED REAL, NOT DEPLOYABLE.** IC-IR 0.101 with a
  CI excluding zero and positive net Sharpe at both cost models — but net SR 0.081 and DD −0.374.
  Do **not** re-probe for a bigger version by changing the lookback, the skip, or adding a vol
  scale (§4 stop rule). If it is ever revisited, the live question is whether the recent-2y
  strengthening (+0.391 vs 0.101 full-sample) is regime or noise — which needs forward data, not
  another backtest.
- **The negative control is now a validated instrument on this harness** and should be standard in
  future probe batches. It cost one slot and converted "is this IC real?" from an argument into a
  measurement. Five earlier probes today lacked it.

## 7. Follow-up — is 0.081 the SIGNAL or the BOOK? (answered: the signal)

C1's net SR 0.081 came from the funnel's naive equal-weighted decile book. This project's own TSMOM
edge reaches 0.60 only via per-asset vol scaling, so the fair question was whether a competent
construction extracts more. §4's stop rule bars re-probing **"if C1 fails"** — C1 did not fail, it
cleared every pre-committed bar — so this is a portfolio-construction question on a fixed,
already-measured signal, not a re-specification. **One** construction was tried: the house's own
validated `xsec_momentum_falsification.xsmom_weights` (rank 12-1, long top third / short bottom
third, vol-scaled). No sweep.

`scripts/research/country_momentum_book.py` → `results/country_momentum/country_momentum_book.json`:

| construction | turnover/yr | Sharpe @5bp | ann vol | max DD |
|---|---|---|---|---|
| funnel naive decile | 5.07 | **+0.081** | — | −37.4% |
| house vol-scaled `xsmom_weights` | **47.40** | **+0.003** | 39.3% | **−99.7%** |

Frictionless 0.064 · 2bp 0.039 · 5bp 0.003 · 10bp −0.057. At a common 10% vol: **+0.03%/yr**.

**The better-engineered book is WORSE** — 9x the turnover and a −99.7% drawdown. So the answer is
unambiguous: **the small Sharpe is the signal, not the book.** C1 is closed as non-deployable from
both directions, and no further construction will be tried (that would be the sweep this note
explicitly avoided).

Ledger, final: **cross-sectional country-equity momentum — REAL (IC-IR 0.101, CI excludes zero,
negative control clean) and NOT DEPLOYABLE under either the naive or the house-validated book.**

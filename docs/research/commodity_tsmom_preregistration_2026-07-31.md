# PRE-REGISTRATION — COMMODITY-breadth time-series momentum as a diversifying sleeve (T3)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the sleeve was built or any Sharpe was seen.**

## 0. Why this is the right test, given T1 and T2

T1 and T2 bracketed the problem and produced a *specification*, not just two failures:

| sleeve | standalone SR | ρ to existing book | outcome |
|---|---|---|---|
| T1 country-ETF TSMOM | **0.409** ✓ | **+0.693** ✗ | return without independence |
| T2 crypto TSMOM | **−0.067** ✗ | **−0.063** ✓ | independence without return |

A diversifying sleeve needs **both**. So the target is a universe that is (a) time-series amenable —
the style that works on this stack — (b) genuinely additive to the existing 18-instrument book, and
(c) has a strong prior for actually earning a trend premium.

**Commodities are the best-documented home of trend-following** (Hurst-Ooi-Pedersen, *A Century of
Evidence on Trend-Following*; managed-futures/CTA returns are historically commodity-heavy). The
existing book carries only **5 commodity instruments out of 18**, so commodity breadth is the most
under-represented axis in what is already held.

**This is not the closed commodity-carry cell.** `project_commodity_carry_sleeve_nogo_s553` found
commodity **CARRY** dilutive. Carry and trend are different signals with different literatures; the
carry NO-GO says nothing about trend. And `breadth_expansion`'s NO-GO widened the **same pooled
book** 18→32 — it did not test a **separate commodity sleeve combined at the book level**.

## 1. Universe (LOCKED) — deliberately EXCLUDES what is already held

The existing book's commodity leg is **GLD, SLV, DBC, USO, DBA**. All five are **excluded** so this
measures *incremental* commodity breadth rather than re-running instruments already owned — which
would manufacture exactly the ρ +0.693 problem that killed T1.

Candidates (free, yfinance, ≥2,000 daily bars measured before writing this):
`DBB, DBE, DBP, GSG, UNG, USL, UGA, PALL, PPLT, BNO, CORN, CANE, SOYB, WEAT, NIB, FTGC, COMT`
— 17 instruments, histories 2,963-5,037 bars (2006-2026).

Tickers are fixed here; none is added or dropped after seeing a result.

## 2. What is measured (LOCKED)

House machinery **unchanged** and identical to T1/T2 — `tsmom_signal` (trend = mean of
sign(trailing return) over lookbacks 63/126/252, skip 5), `vol_scaled_weights`, `backtest`,
`portfolio_frontier.risk_parity`. **No new statistic, no tuned parameter.**

- **T3 standalone** net Sharpe at 2bp / 5bp / 10bp.
- **Correlation** to the existing cross-asset TSMOM sleeve (`build_momentum_net`).
- **Combined** risk-parity book {cross-asset TSMOM, T3}.

**Expected sign: +1** (trend continuation). Negative standalone Sharpe is a FAIL.

## 3. Pre-committed pass conditions — identical bars to T1/T2

1. **standalone net SR ≥ 0.30** at 5bp;
2. **correlation to cross-asset TSMOM ≤ 0.60**;
3. **combined net SR > 0.66** — must beat the existing 0.60 sleeve by a real margin; a combined
   0.60-0.65 is **NOT** a pass.

All three must hold. This is the third consecutive sleeve judged on the same unchanged bars, so the
comparison across T1/T2/T3 is apples-to-apples.

## 4. Honest priors, before the run

**For:** commodity trend is the single most replicated TSMOM cell; the universe is genuinely
additive by construction (the 5 held names are excluded); commodities are fundamentally driven by
supply/inventory rather than discount rates, so a lower correlation to an equity/rates-heavy book is
structurally plausible; T1 showed trend sleeves are barely cost-sensitive.

**Against:** several candidates are **broad-index near-duplicates** of the excluded DBC (GSG, COMT,
FTGC), so the effective breadth is smaller than 17; single-commodity ETFs like UNG and USO-family
products carry **severe contango/roll decay** that can swamp a trend signal; and 2006-2026 includes
the post-2011 commodity bear where trend did poorly. A negative would most likely come from roll
decay, not from absence of trend.

## 5. Stop rule

One hypothesis, one construction. **No T4.** If it fails, commodity-breadth trend on free ETF data
is recorded closed and is not re-probed with different lookbacks, a vol target, roll-adjusted
proxies, or a hand-picked sub-basket.

## 6. Promotion ceiling

Not survivorship-free (delisted commodity ETFs absent). A pass earns **forward-incubate + Tier-2
before any capital**, and does not by itself authorise changing TAILWIND.

## 7. Results — **NO-GO**, but the closest of the three sleeves

Run 2026-07-31, `results/commodity_tsmom/commodity_tsmom_sleeve.json`. 17 tickers, 5,037 days
(2006-07-21→2026-07-30), 226 monthly rebalances, turnover 32.2/yr, 4,995 overlapping days.

| pre-committed bar (§3) | measured | |
|---|---|---|
| standalone net SR ≥ 0.30 @5bp | **0.356** | ✓ **PASS** |
| correlation to cross-asset TSMOM ≤ 0.60 | **+0.485** | ✓ **PASS** |
| combined risk-parity SR > 0.66 | **0.568** | ✗ **FAIL** |

Standalone by cost: frictionless 0.381 · 2bp 0.371 · 5bp 0.356 · 10bp 0.330 — barely cost-sensitive,
like T1. On the overlap: existing sleeve 0.609, T3 0.370, ρ +0.485, combined 0.568 (DD −19.3%).

**T3 is the first sleeve of the session to clear BOTH the return and the correlation bars** — the two
conditions T1 and T2 each failed one of. And it still does not help.

### The failure is not an artifact of equal weighting — checked analytically

The obvious objection is that `pf.risk_parity` equal-weights risk, so perhaps a smarter allocation
would rescue it. It would not. For two sleeves the closed forms are:

- equal-risk: `(s1+s2)/√(2+2ρ)` = (0.609+0.370)/√2.970 = **0.568** — reproduces the measured value
  exactly, confirming the combine did what it should;
- max-Sharpe (the theoretical ceiling): `√((s1²−2ρ·s1·s2+s2²)/(1−ρ²))` = **0.615**.

**Even optimally weighted the book gains +0.006 over the 0.609 it already earns.** And the general
bar: at ρ = 0.485 a sleeve must carry standalone SR > **0.441** to improve an equal-risk combine at
all. T3 carries 0.370 — a real trend premium, but not enough to overcome its correlation.

So the NO-GO is structural, not a weighting choice.

### Ledger (durable)

- **Commodity-breadth TSMOM (17 ETFs excluding the 5 already held, 2006-2026): REAL but
  INSUFFICIENT — NO-GO as a diversifying sleeve.** Standalone 0.356 at 5bp with ρ 0.485; adds
  +0.006 at best. Per §5 there is no T4 and no re-probe with different lookbacks, a vol target,
  roll-adjusted proxies, or a hand-picked sub-basket.
- **The sleeve-admission rule, now closed-form.** Stop asking "is this signal real?" and ask
  "does it beat the bar its own correlation sets?" For an equal-risk combine against an existing
  sleeve of Sharpe `s1`, a candidate must satisfy **`s2 > s1·(√(2+2ρ) − 1)`**. Against the 0.60
  book that is SR > 0.36 at ρ=0.3, > 0.44 at ρ=0.485, > 0.55 at ρ=0.7. **Compute this before
  running anything** — it would have predicted all three of today's sleeve failures from their
  correlations alone.

### The three sleeves, side by side

| | standalone | ρ | combined | verdict |
|---|---|---|---|---|
| T1 country TSMOM | 0.409 | +0.693 | 0.498 | return, no independence |
| T2 crypto TSMOM | −0.067 | −0.063 | 0.425 | independence, no return |
| **T3 commodity TSMOM** | **0.356** | **+0.485** | **0.568** | **both — still short of the bar** |

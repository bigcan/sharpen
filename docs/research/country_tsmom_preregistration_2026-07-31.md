# PRE-REGISTRATION — TIME-SERIES momentum on country ETFs as a DIVERSIFYING SLEEVE (T1)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the sleeve was built or any Sharpe was seen.**

## 0. The blind spot this corrects

All eight probes today were **cross-sectional** — rank names against each other, measured by rank-IC
through the signals scorecard. Every one failed on IC, on sign, or on capture.

But **the only edge this project has ever validated is TIME-SERIES**: cross-asset TSMOM, each asset
against its own trend, net SR **0.60**, PBO 0.0009. I have been repeatedly testing the one style
that has never worked here, using a harness that can only measure that style.

The related NO-GO does not cover this. `breadth_expansion` tested widening the **same pooled book**
from 18 to 32 instruments and found it *worse* (0.563 < 0.601). It did **not** test building a
**separate sleeve on a different universe and combining at the book level** — which is exactly how
TAILWIND already combines momentum with BAB, and how the validated 0.60 book is constructed.

So: TSMOM, the style that works, on a universe it has never been run on — 30 US-listed country
equity ETFs, 7,641 daily bars (1996-2026), free.

## 1. What is measured (LOCKED)

Using the house's own validated machinery unchanged — `xsec_momentum_falsification.tsmom_signal`
(trend score = mean of sign(trailing return) over lookbacks 63/126/252, skip 5),
`vol_scaled_weights`, `backtest`, and `portfolio_frontier.risk_parity` for the combine. **No new
statistic is implemented and no parameter is tuned**; the constants are the ones already locked for
the 0.60 book.

- **T1 standalone:** country-ETF TSMOM net Sharpe at 2bp/5bp/10bp.
- **Correlation** of the T1 daily net series to the existing cross-asset TSMOM sleeve
  (`portfolio_frontier.build_momentum_net`).
- **Combined book:** risk-parity combine of {existing cross-asset TSMOM, T1}, net Sharpe.

**Expected sign: +1** (trend continuation). A negative standalone Sharpe is a FAIL.

## 2. Pre-committed pass conditions

This is a *diversification* claim, so the bar is about what it adds, not what it is:

1. **T1 standalone net SR ≥ 0.30** at 5bp. Below that it cannot carry weight in a book already
   holding a 0.60 sleeve.
2. **Correlation to existing cross-asset TSMOM ≤ 0.60.** Country equity ETFs are one asset class and
   the existing book already holds SPY/QQQ/IWM/EFA/EEM, so a high correlation is the *expected*
   failure mode and is committed as a bar in advance, not discovered afterwards.
3. **Combined net SR > 0.66** — i.e. the combine must beat the existing 0.60 sleeve by ≥0.06, a
   margin, not a rounding difference. A combined Sharpe of 0.60-0.65 is **not** a pass; it means the
   sleeve added nothing and the result is that country TSMOM is redundant.

All three must hold. Any one failing is a NO-GO for the sleeve.

## 3. Why this could fail, stated in advance

The honest prior is mixed. Country equity ETFs are **one asset class**, while the 0.60 book's
strength comes from cross-asset diversification (equity/rates/commodity/fx). A 30-country
equity-only trend sleeve is plausibly one large equity-beta bet wearing a trend costume, in which
case condition 2 fails and there is no diversification to harvest. That is the most likely negative
outcome and it is written here before the run.

## 4. Multiplicity and stop rule

One hypothesis, one construction (the house's, unchanged). **No T2** — if it fails, country TSMOM is
recorded redundant/insufficient and is not re-probed with different lookbacks, a different skip, a
vol target, or a sub-selected country list.

## 5. Promotion ceiling

Not survivorship-free (delisted country ETFs absent). A pass earns **forward-incubate + a Tier-2
audit before any capital** — the same ceiling as every other sleeve. It does **not** authorise
re-sizing TAILWIND on its own.

## 6. Results

*(Empty at commit time on purpose — verifiable from git history.)*

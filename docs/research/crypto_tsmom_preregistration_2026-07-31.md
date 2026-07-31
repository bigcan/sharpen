# PRE-REGISTRATION — CRYPTO time-series momentum as a diversifying sleeve (T2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the sleeve was built or any Sharpe was seen.**

## 0. Scope boundary — this is NOT a re-probe of the barred crypto cross-section

The K1/K2 pre-registration barred re-probing **"the crypto cross-section"** with different lookbacks,
vol scaling, or funding conditioning. **That bar is respected and this does not touch it.** K1/K2
ranked coins *against each other* (cross-sectional rank-IC). T2 is a **time-series** sleeve — each
coin against its **own** trend — which is a different construction, a different statistic, and a
different literature. No cross-sectional signal is computed here.

I am flagging the boundary explicitly rather than quietly stepping over it, because "a different
construction" is exactly the excuse a stop rule exists to refuse. The distinction that makes it
legitimate: the barred object was *the crypto cross-section is empty*; T2 asks a question that
finding cannot answer.

## 1. Why this specific test

T1 (country-ETF TSMOM) just established two things:

1. **Time-series is the style that works on this stack.** Eight cross-sectional probes produced
   nothing ≥0.30; the first time-series probe produced 0.409 standalone.
2. **T1 failed on CORRELATION alone** (ρ +0.693 to the existing cross-asset TSMOM), because country
   ETFs are equity and the existing 18-instrument book is already 5/18 equity. Its standalone
   Sharpe cleared the bar comfortably.

So the diagnosis is precise: the style is right, the *universe* was too close to what is already
held. The fix is a universe with genuinely low correlation to a cross-asset ETF book. **Crypto is the
lowest-correlation liquid asset class available on free data**, and trend-following is its most
widely deployed systematic strategy.

Data: `data/crypto_cache/silver_ohlcv.parquet`, 10 perps, hourly → daily, 2022-01→2026-04
(~1,570 days). **Zero fetch.**

## 2. What is measured (LOCKED)

House machinery **unchanged** — `xsec_momentum_falsification.tsmom_signal` (trend = mean of
sign(trailing return) over lookbacks 63/126/252, skip 5), `vol_scaled_weights`, `backtest`, and
`portfolio_frontier.risk_parity` for the combine. **No new statistic, no tuned parameter.** Identical
treatment to T1 so the two are directly comparable.

- **T2 standalone** net Sharpe at 5bp / 10bp / 20bp (crypto taker costs; wider grid than ETFs).
- **Correlation** of T2's daily net series to the existing cross-asset TSMOM sleeve.
- **Combined** risk-parity book {cross-asset TSMOM, T2} net Sharpe.

**Expected sign: +1** (trend continuation). A negative standalone Sharpe is a FAIL.

## 3. Pre-committed pass conditions — identical bars to T1

1. **standalone net SR ≥ 0.30** at 10bp (the middle crypto cost, stricter than T1's 5bp);
2. **correlation to cross-asset TSMOM ≤ 0.60**;
3. **combined net SR > 0.66** — must beat the existing 0.60 sleeve by a real margin. A combined
   0.60-0.65 is **NOT** a pass; it means the sleeve added nothing.

All three must hold.

## 4. Honest priors, written before the run

**For:** crypto's correlation to traditional assets is structurally low, which is exactly the axis T1
failed on; trend-following is the best-documented systematic crypto strategy; costs are ~5-20bp with
no transaction tax, and T1 showed trend sleeves are barely cost-sensitive.

**Against, and these are serious:** only **4.3 years** — roughly one crypto cycle — so a positive
result is heavily exposed to having ridden the 2023-24 bull; **N=10 with BTC/ETH driving most
common variance**, so this is closer to a levered crypto-beta timing bet than a diversified trend
book; and the panel is **not survivorship-free** in a market where tokens die. A pass here is
therefore a *candidate*, not a conclusion, and would need forward incubation before any capital —
stated now, not after.

## 5. Stop rule

One hypothesis, one construction. **No T3.** If it fails, time-series momentum on free crypto data is
recorded closed, and is not re-probed with different lookbacks, a vol target, a funding overlay, or a
BTC/ETH-only subset.

## 6. Promotion ceiling

Not survivorship-free. A pass earns **forward-incubate + Tier-2 before any capital**, and does not
by itself authorise changing TAILWIND.

## 7. Results — **NO-GO** (correlation axis confirmed; no return to contribute)

Run 2026-07-31, `results/crypto_xsec/crypto_tsmom_sleeve.json`. 10 coins, 1,579 days, **42 monthly
rebalances** after warmup, turnover 8.52/yr, 1,083 overlapping days with the cross-asset sleeve.

| pre-committed bar (§3) | measured | |
|---|---|---|
| standalone net SR ≥ 0.30 @10bp | **−0.067** | ✗ **FAIL** |
| correlation to cross-asset TSMOM ≤ 0.60 | **−0.063** | ✓ **PASS** |
| combined risk-parity SR > 0.66 | **0.425** | ✗ **FAIL** |

Standalone by cost: frictionless −0.054 · 5bp −0.060 · 10bp −0.067 · 20bp −0.080. Ann vol 44.4%,
max DD −74.8%. On the overlap: cross-asset TSMOM 0.833, T2 −0.252, combined 0.425 (DD −10.7%).

### What this establishes

**The diagnosis from T1 was correct, and it was not enough.** T1 failed purely on correlation
(ρ +0.693); T2 was chosen to fix exactly that axis, and it did — **ρ −0.063, essentially
uncorrelated**, the best diversification property of any sleeve tested today. But a sleeve needs
*both* low correlation and positive return, and T2 has none of the second: crypto trend-following
**loses money** across 2022-2026 at every cost level, including frictionless.

An uncorrelated sleeve with a negative Sharpe does not diversify a book, it dilutes it — the
combined 0.425 is well below the 0.601 the cross-asset sleeve earns alone.

**Weight this negative as §4 committed in advance:** 42 monthly rebalances over ~3.5 post-warmup
years is a thin decision count, the window is roughly one crypto cycle, BTC/ETH dominate the common
variance, and the panel is not survivorship-free. This is scouting-grade evidence that this
*parameterisation* of crypto trend does not work on this window — not a general claim that crypto
trend-following is dead.

### Ledger (durable)

- **Crypto time-series momentum (house TSMOM parameterisation, 10 perps, 2022-2026): NO-GO.**
  Negative standalone Sharpe at every cost level. Per §5 there is no T3 and no re-probe with
  different lookbacks, a vol target, a funding overlay, or a BTC/ETH-only subset.
- **The two-condition lesson, now measured from both sides.** T1: high return (0.409), high
  correlation (0.693) ⇒ dilutes. T2: ideal correlation (−0.063), no return (−0.067) ⇒ dilutes. A
  diversifying sleeve needs both, and this session found neither combination. Any future sleeve
  proposal should state **both** bars before measuring, as T1 and T2 did.

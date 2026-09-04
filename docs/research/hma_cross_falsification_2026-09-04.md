# HMA-crossover strategy (CodeTrading / "CoinQuant community") — end-to-end falsification

**Date:** 2026-09-04 · **Script:** `scripts/research/hma_cross_falsification.py` ·
**Artifacts:** `results/hma_cross/crypto_4h.json`, `results/hma_cross/etf_daily.json`
**Source:** <https://youtu.be/b6W3GsDZ_Kg> — "I Cloned This Strategy From a Trading Community and Tested It on 20+ Assets"

## Verdict

**NO-GO as an alpha.** On the substrate with statistical power — 18 ETFs × 18.65 years — the published
rules produce a median per-asset alpha of **−0.005 %/yr (t = −0.01)**, lose to buy-and-hold
(Sharpe **0.323** vs **0.510**), lose to the validated TSMOM core (**0.576**), and **fail their own timing
null at p = 0.58**: randomly re-timing the identical set of trades earns *more* than the actual entry
rules (null median total return 19.84 % vs the real 17.65 %).

On 20 crypto assets × 3.9 years the book does look good (Sharpe 0.779 vs 0.238 buy-and-hold) and does
beat its timing null (p = 0.0045) — but that sample contains a −99.5 % year, the entire advantage is
crash-avoidance, and the result fails multiplicity control (DSR **0.19**, needs ≥ 0.95) with a bootstrap
Sharpe CI of **[−0.45, +1.93]**.

**What the strategy actually is:** a de-risking overlay. It holds beta ~13 % of the time with a measured
beta of ~0.13, cuts drawdown hard, and adds no return per unit of capital risked. That is a
risk-management wrapper, not an edge — and it is the same conclusion as
`project_tailwind_sizing_null_2026_08_26` ("de-levering in costume") reached on a different overlay.

## The rules as tested

Taken verbatim from the video description; nothing was tuned.

| | |
|---|---|
| Entry (long only) | HMA(16) crosses above HMA(65) **AND** RSI(14) > 52 **AND** close > linear-regression curve on the next timeframe up |
| Exit | HMA(16) crosses below HMA(65) |
| Stops / leverage | none |

The one parameter the video never states is the linear-regression length; TradingView's default of **100
coarse bars** was used and then swept over 50–300 (see Sensitivity).

## Substrates

| | crypto 4H | ETF daily |
|---|---|---|
| Panel | `data/crypto_cache/silver_spot_ohlcv.parquet`, 20 spot assets, 1H → 4H | `results/tailwind_v1/ohlcv_daily.parquet`, the 18-ETF TAILWIND panel (manifest `PASS`) |
| Window | 2022-04-19 → 2026-03-16 (3.91 y) | 2007-12-05 → 2026-07-31 (18.65 y) |
| Coarse regime TF | daily | weekly |
| Cost charged | 10 bps taker per side | 2 bps per side |
| Bars/yr | 2190 | 252 |

Timestamps in the crypto panel are bar-**open** times (verified: `open[t] == close[t-1]`), contiguous with
zero gaps.

## Result 1 — ETF daily, 18.65 years (the powered test)

Equal-weight book, net 2 bps, all books scored on identical bars:

| book | Sharpe | PF | ann ret % | MaxDD % | exposure | α ann % | α t |
|---|---|---|---|---|---|---|---|
| **HMA published rules** | **0.323** | 1.074 | 0.91 | −5.98 | 0.194 | 0.34 | 0.56 |
| HMA, no RSI filter | 0.256 | 1.055 | 0.83 | −7.75 | 0.241 | 0.12 | 0.17 |
| HMA, no regime filter | 0.552 | 1.117 | 1.95 | −6.07 | 0.292 | 1.09 | 1.51 |
| HMA, bare crossover | 0.265 | 1.055 | 1.65 | −18.28 | 0.513 | −0.52 | −0.50 |
| **TSMOM (shipped core)** | **0.576** | 1.108 | 43.89 | −93.85 | 11.82 | 39.37 | **2.24** |
| **equal-weight buy & hold** | **0.510** | 1.098 | 5.14 | −30.77 | 1.00 | — | — |

Per asset: **5/18** beat their own buy-and-hold Sharpe, **4/18** beat buy-and-hold held at the strategy's
own average exposure. Median alpha **−0.005 %/yr**, median t **−0.01**, median beta **0.136**.

**Timing null: p = 0.582.** Circularly shifting each asset's position series by an independent random
offset — preserving exposure, trade count and holding-period distribution exactly, destroying only the
alignment to price — produces a median Sharpe of 0.356 against the real book's 0.324. The entry
conditions carry no information about *when* to be long. 0/18 assets reach p ≤ 0.05; BH-FDR survivors: 0.

Regime by regime, the HMA book underperforms buy-and-hold in **3 of 4** eras (2010-15, 2016-20, 2021-26);
it edges ahead only across 2006-09.

Note the ordering: **dropping the regime filter improves the book** (0.552 vs 0.323). The daily-trend gate
the video presents as the point of the system is a net negative on this panel.

## Result 2 — crypto 4H, 3.9 years (the video's own substrate)

| book | Sharpe | PF | ann ret % | vol % | MaxDD % | exposure |
|---|---|---|---|---|---|---|
| **HMA published rules** | **0.779** | 1.083 | 11.93 | 15.32 | −18.39 | 0.129 |
| equal-weight buy & hold | 0.238 | 1.015 | 15.35 | 64.55 | −75.25 | 1.00 |

Per asset (full window): **15/20 profitable**, 12/20 beat buy-and-hold Sharpe, 15/20 beat matched-exposure
buy-and-hold return, median alpha +8.89 %/yr at median **t = 0.455**, median beta 0.133, median exposure
**0.130**. Trade counts 60–95 per asset at 30–46 % win rate — close to the video's "125 trades, 43 %".

The video's "~90 % of assets profitable" reproduces as **75 %** at 10 bps over this window (60 % over the
shorter TSMOM-comparable window).

**Where the performance comes from — regime split:**

| regime | HMA Sharpe / ann ret | buy & hold Sharpe / ann ret |
|---|---|---|
| 2022 bear | **+0.786 / +8.98 %** | −1.419 / **−99.5 %** |
| 2023-24 recovery | +1.281 / +22.34 % | **+1.567 / +93.65 %** |
| 2025-26 | −0.267 / −3.59 % | −0.712 / −48.91 % |

The strategy wins the crashes and loses the rally by a factor of four. That is the video's own thesis,
and it is confirmed — it is also the definition of a risk overlay rather than an alpha.

**Where it survives and where it fails:**

- Timing null **passes**: real Sharpe 0.781 vs null median −0.17, **p = 0.0045**. On this sample the entry
  timing is informative.
- Multiplicity **fails**: DSR **0.192** against a required ≥ 0.95; the deflated benchmark SR\* is **1.30
  annualized** versus an observed 0.78, over 80 counted trials (20 assets × 4 rule variants).
- Per-asset significance **fails**: 4/20 assets at p ≤ 0.05 against 1.0 expected by chance; under BH-FDR
  (q = 0.10) exactly **one** survives on total return (DOGE) and **zero** on Sharpe.
- Bootstrap Sharpe CI95 **[−0.45, +1.93]**, p(SR < 0.5) = 0.32.

The TSMOM comparator scores Sharpe 0.012 on crypto, but that is a **substrate mismatch, not a defence of
the HMA rules**: TSMOM's constants are locked to an 18-ETF cross-asset panel and its per-asset 10 % vol
target produces an 87 %-vol book on 20 highly-correlated crypto names. It should not be read as
"HMA beats TSMOM".

## Sensitivity to the unstated parameter

Equal-weight book, across linear-regression lengths (the one value the video does not publish):

| linreg len | ETF Sharpe | ETF null p | crypto Sharpe | crypto null p |
|---|---|---|---|---|
| 50 | 0.356 | 0.565 | 0.592 | 0.023 |
| 100 | 0.323 | 0.572 | 0.779 | 0.005 |
| 150 | 0.396 | 0.728 | 0.756 | 0.013 |
| 200 | 0.287 | 0.858 | 0.624 | 0.033 |
| 300 | 0.161 | 0.955 | 0.536 | 0.093 |

ETF buy-and-hold over the same span: Sharpe 0.575. The ETF conclusion holds at **every** setting — the
book never beats buy-and-hold and never beats its own re-timed null. The crypto result degrades
monotonically as the filter lengthens.

## Verification performed

| gate | result |
|---|---|
| WMA / HMA / linear-regression curve vs independent slow reference | exact, max diff ~1e-13 |
| RSI vs hand-rolled Wilder (SMA-seeded) | exact, max diff 2.8e-14 |
| `bar_backtest` vs shipped `xsec_momentum_falsification.backtest` | **bit-for-bit**: gross 0.0, cost 0.0, turnover 0.0 |
| `tsmom_weights` vs shipped `tsmom_signal` + `vol_scaled_weights` | **0.0** max abs weight diff |
| LEAK-2 future-bar sweep | 0.0 |
| LEAK-2 current-bar crash (past unchanged **and** current bar responds) | 0.0 / response 1.0 (crypto), 6.0 (ETF) |
| Coarse-bar map asserted against the distinct own-bar value | PASS |
| PF-XCHECK (close vs (H+L)/2 mid) | 17.08 % divergence, < 30 % threshold |
| OHLC validation (high < max(o,c) or low > min(o,c)) | 0 violations |
| ruff | clean |

Two defects were found and fixed during construction, both of the kind this project keeps re-learning:

1. The first `assert_causal` returned **`current_bar_response: 0.0`** — it had picked a bar where the book
   was flat, so the current-bar leg could not have fired whatever the code did. A check that cannot fail
   is not a check (`feedback_mutation_check_false_survivor_crlf`). It now selects a bar where the book is
   long and *raises* if the position does not move.
2. The vectorized TSMOM comparator initially diverged from the shipped core by a full leverage cap (2.0)
   because it used a plain mean over look-backs where the shipped code uses `np.nanmean` — silently
   flattening the book for years of early history. Caught only because parity was asserted rather than
   assumed.

## Caveats

- The crypto panel holds 20 **still-listed** majors; coins that died over 2022-26 are absent. That
  survivorship flatters buy-and-hold, so the HMA-vs-B&H gap is if anything understated — but the absolute
  crypto figures are not a tradeable-universe result.
- 3.91 years of crypto with a single −99.5 % year dominating is not enough to separate "trend filter with
  edge" from "trend filter that avoided one crash".
- Both panels are close-to-close with a one-bar execution lag; no funding, borrow, or partial-fill model.
  Long-only spot needs none of those, so this is not a material omission here.

## What follows

Nothing to wire. The rules are a slower, long-only, worse-instrumented member of the trend family whose
one validated representative (TSMOM, net SR ~0.60) already sits in `configs/tailwind_v1.yaml` and beat
these rules head-to-head on the panel with 18.65 years behind it.

The one transferable observation: the crypto regime table is a clean demonstration that **exposure
reduction and alpha are different things and get confused by any metric measured against zero**. The
video's "90 % of assets profitable" is true, reproduces at 75 % under costs, and means nothing — a book
long 13 % of the time at *random* is profitable on most of these assets too.

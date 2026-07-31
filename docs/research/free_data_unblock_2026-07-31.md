# The free-data constraint is BROKEN — one viable cell found (EURUSD, 3h holding)

**Date:** 2026-07-31 (S553-cont-146). Supersedes the "purchase, not a search" conclusion in
`discovery_data_spec_2026-07-31.md`.

## What I got wrong, twice

I concluded discovery was data-blocked and asserted the fix was **buying** data — without testing
whether free data could meet the spec. It can. My own standing rule is to re-test a blocker before
obeying it, and I did not apply it to my own conclusion.

## The source

**Dukascopy**, free and unauthenticated, verified live:
`https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5`

EURUSD ticks in one mid-session weekday hour: **2003** 843 · 2005 958 · 2010 1,779 · 2015 8,319 ·
2020 8,630 · 2025 5,229 · **2026** 6,302. Coverage also confirmed for USDJPY, GBPUSD,
`USA500IDXUSD` (S&P), `USATECHIDXUSD` (Nasdaq), `DEUIDXEUR` (DAX), `XAUUSD`, `LIGHTCMDUSD`.
**~23 years**, roughly twice the derived requirement. Fetcher: `scripts/data/fetch_dukascopy.py`
(resumable, weekend-skipping; 540 hour-files → 528 bars in 114 s at 12 workers).

## Measured inputs, not assumed ones

| instrument | spread (median) | ann. vol | source of vol |
|---|---|---|---|
| **EURUSD** | **0.256 bp** | **12.03%** | 6,226 bars = full year 2015 |
| USA500IDXUSD (S&P CFD) | 1.79 bp | 14.7% | 3-day sample |
| DEUIDXEUR (DAX CFD) | 1.92 bp | 28.7% | 3-day sample |

**FX is the reason this works.** EURUSD's spread is ~7× tighter than the index CFDs, and cost drag
scales linearly with it. The S&P and DAX CFDs fail at **every** holding period at any depth
available — 1.8-1.9 bp is simply too wide for hourly-scale rebalancing.

## The joint solve, with measured inputs

`N_eff = holdout/(2H−1)` · `drag = 0.70 · (bars_yr/H) · bps / vol` · ceiling 0.50 · 23 y × 6,226
bars/yr, holdout 35,799. MDE from the measured corrected-contract curve.

| H (hours) | drag | N_eff | MDE | verdict |
|---|---|---|---|---|
| 1 | 0.93 | 35,799 | off grid | — |
| 2 | 0.46 | 11,933 | 0.354 | drag **>** MDE → no |
| **3** | **0.31** | **7,159** | **0.448** | ✅ **VIABLE** |
| 4 | 0.23 | 5,114 | 0.508 | MDE > ceiling → no |
| 6+ | ≤0.15 | ≤3,254 | ≥0.689 | too shallow |

**Exactly one viable cell: EURUSD at 3-hour holding.** MDE 0.448 ≤ 0.50 **and** drag 0.31 < 0.448.

### The vol estimate was load-bearing and my first one was wrong

A 3-day sample gave EURUSD ann. vol **17.4%**; the full 2015 year gives **12.03%**. At 17.4%, H=2
looked viable; at the true 12.03% it is not (drag 0.46 > MDE 0.354) and H=3 is the only cell. **A
3-day vol estimate nearly produced a false "viable" claim** — the same class of error as reading a
quantile off a degenerate null (RC-11) or an overlay gain off one subperiod (V1).

## What this does and does not mean

**Does:** for the first time this session there is a (substrate, holding) configuration on **free**
data where the funnel could *detect* a marginal edge that survives costs. The 14-probe arc's blocker
— power and economics never coinciding — is broken for this one cell.

**Does not:** it is **not an alpha**. "Viable" means detectable, not present. Finding an edge there
still requires a pre-registered probe or a mining run, and it must clear the same bars every probe
today faced.

**Margin is narrow.** MDE 0.448 against a 0.50 ceiling, drag 0.31. It is sensitive to the 0.70
turnover assumption and to vol; a book turning over more, or a calmer FX regime, closes it. Treat
H=3 as the centre of a thin band, not a comfortable pass.

## Cost to realise

Full 23-year EURUSD hourly download ≈ 138,000 weekday hour-files ≈ **4 h at 12 workers, ~2.2 h at
24**. Resumable. Verified working end-to-end on 2015 (6,226 bars).

## Two silent decode bugs found (neither raised an exception)

1. **ask/bid are scaled int32, not float32** — decoding as floats gave ~1e-45 denormals *and* a
   plausible-looking 0.27 bp "spread" computed from garbage. Caught only by checking a decoded price
   against a known level (EURUSD ≈ 1.09 in 2015).
2. **Index CFDs need divisor 1e3, metals 1e2** — S&P decoded as 2,114,349 instead of 2,114.3.
   Spread *ratios* are divisor-invariant, so the bug hid from the spread check specifically.

Both reinforce the session's recurring lesson: a pipeline that runs without erroring is not a
pipeline that is correct. **Sanity-check a decoded price against a known level for every new
instrument family.**

## Next step

Pre-register a probe on EURUSD 3-hour bars over the full free history. That is now a legitimate
experiment rather than a foregone conclusion.

# PRE-REGISTRATION — EURUSD 3-hour short-horizon REVERSAL (G1) + control (G2)

**Written:** 2026-08-01, **before the signal was computed or any Sharpe seen.**

## 0. Why this is legitimate and not a sign-flip of F1

F1 (EURUSD 3h TSMOM, sign +1) failed yesterday: gross Sharpe −0.020, net −0.092, CI straddling
zero. Its stop rule bars re-probing **trend** — "no other lookbacks, bar sizes, session filters, or
vol conditioners" — and that bar is respected. Its own §6.3 states the result "closes the *trend*
hypothesis, not the cell."

**This is NOT F1 with the sign reversed.** That would be sign-flip opportunism on data already seen,
which every pre-registration in this campaign forbids. The distinction is concrete:

| | F1 (failed) | G1 (this) |
|---|---|---|
| lookback | **63 / 126 / 252 bars** (≈8-32 trading days) | **1 bar** (the previous 3 hours) |
| construction | mean of `sign(trailing return)` over three long windows | z-score of the last bar's return vs its trailing distribution |
| mechanism | trend continuation | liquidity provision / order-flow reversal |
| literature | monthly-horizon FX momentum | sub-daily FX mean reversion |

Different horizon regime, different statistic, different mechanism, different literature. F1's own
§3 prior named this one explicitly, **before** F1 ran: *"at sub-daily horizons the literature leans
toward mean reversion and order-flow effects."* Testing the alternative that a failed hypothesis's
own pre-written prior pointed to is the intended use of a falsification campaign.

## 1. The substrate (unchanged, already built)

EURUSD 3-hour bars from free Dukascopy ticks — **47,433 bars, 2004-01-01 → 2026-07-31**, verified
uniform at ~6,225-6,290 hourly bars/yr after the retry fix. This remains the only cell measured this
campaign where power and economics coincide: N_eff ≈7,050, MDE ≈**0.45** ≤ 0.50 ceiling, cost drag
**0.31** < MDE at one-bar (3-hour) holding.

## 2. The signals (LOCKED)

### G1 — one-bar reversal · sign **−1**
- **Signal (causal):** `z[t] = r[t] / σ[t]` where `r[t]` is the completed 3-hour bar return and
  `σ[t]` is its trailing 63-bar standard deviation, **both known at the close of bar t**.
- **Position:** `−tanh(z[t])`, vol-targeted on trailing realised vol, leverage capped at 3. `tanh`
  bounds the position so a single outlier bar cannot dominate; it is fixed here, not tuned.
- **Applied to the return from t to t+1** — one-bar (3-hour) holding, exactly the viable cell.
- **Expected sign: −1** (fade the last move). A positive Sharpe is a FAIL — it would mean
  continuation, which F1 already falsified at long lookbacks and which cannot be re-labelled here.

### G2 — NEGATIVE CONTROL
Deterministic pseudo-random position, no price information.
**Scored on GROSS (zero-cost) Sharpe** — applying yesterday's correction. F1's condition 5 was
mis-specified: it scored the control on *net* Sharpe, where a high-turnover null loses ~6%/yr by
arithmetic and its "significance" says nothing about the harness. A control must be scored on the
same cost basis at which it is used as a null.

## 3. Pre-committed pass conditions — ALL must hold

1. **net Sharpe ≥ 0.30** at the measured 0.256 bp spread;
2. **block-bootstrap CI (200-bar blocks) excludes zero**;
3. **positive in ≥3 of 4 subperiods**;
4. **still positive at 0.5 bp** (≈2× measured spread);
5. **G2 GROSS Sharpe CI includes zero** (a clean null).

## 4. Honest priors, before the run

**For:** sub-daily FX mean reversion has a real microstructure basis (inventory effects, liquidity
provision against uninformed flow), and it is the mechanism F1's own prior named. The cell's
arithmetic permits detection.

**Against, and serious:** EURUSD is the most liquid instrument on earth and any reversal at 3-hour
scale is precisely what high-frequency market makers are paid to arbitrage away — a persistent
edge surviving 2004-2026 is a strong claim. One-bar holding means **maximum** turnover for this
cell, so the cost budget is at its tightest. And this campaign has now produced fifteen negatives;
the base rate is the strongest prior available. **A negative is again the more likely outcome.**

## 5. Multiplicity

**Declared cumulatively at 2 on this cell** (F1 trend + G1 reversal). The cell is new, so the
Taiwan/country/crypto counts do not carry over.

## 6. Stop rule

One hypothesis, one control, one parameterisation. **No G2-variant.** If G1 fails, the EURUSD 3-hour
cell is recorded as tested for **both** major directional mechanisms — continuation and reversal —
and closed for directional signals. It is not re-probed with other z-windows, other bounding
functions, session filters, or spread conditioners.

## 7. Results

*(Empty at commit time on purpose — verifiable from git history.)*

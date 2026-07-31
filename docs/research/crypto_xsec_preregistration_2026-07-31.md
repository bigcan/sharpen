# PRE-REGISTRATION — Crypto cross-sectional probes (K1 reversal, K2 momentum) + control (K3)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE any signal was computed or any IC was seen.**
**Substrate:** `data/crypto_cache/silver_ohlcv.parquet` — 10 perps, hourly, 2022-01→2026-04, on disk.
**Zero fetch.**

## 0. Why crypto, and why it is genuinely untested

Seven probes today have failed. Crypto is the one substrate left with **structural** reasons to
carry edge and no prior linear test:

- **The closed crypto cells are RL failures, not signal tests.** Sync-1H's pilots blew up with
  A2C/SAC/PPO agents (return −62.8%, Sharpe −20.4); funding-arb, options-VRP and GMGP1-BTC were
  likewise strategy/agent closures. **Whether the crypto cross-section carries linear structure has
  never been measured** through this funnel.
- **Costs are ~5bp, not Taiwan's ~210bp.** Every Taiwan probe today died on capture. R2/S1 died even
  frictionlessly, but R1-style liquidity-provision mechanisms are exactly the kind that a 0.30%
  transaction tax kills and a 5bp venue does not.
- **Retail-dominated and 24/7** — the same structural argument that motivated the Taiwan probes,
  without the tax.
- My own eligibility table (`crucible_substrate_eligibility.py`) flagged crypto perp hourly as the
  **only powered cell** on disk (N_eff 9,250, MDE 0.468 < 0.50 ceiling).

**Honest limitation, stated up front:** the panel is **N=10 names over ~4.3 years**. That is a thin
cross-section and a short history — far weaker than the Taiwan (612 × 21y) or country-ETF (30 × 30y)
panels. A negative here is therefore weaker evidence than today's other negatives, and a positive
would need forward confirmation before meaning much. This is a scouting probe, and it is labelled
as one **before** the result rather than after.

## 1. The three signals (LOCKED)

Daily bars (hourly resampled to daily closes). `horizons = (1,5,10,21,63)`, `primary_horizon = 5`
— **shorter than the Taiwan/country probes on purpose**: crypto's documented cross-sectional effects
live at days-to-weeks, and with only ~1,570 days a 21d primary would leave too few independent
observations. Committed now, not chosen after seeing the ladder.
`neutralization = ("winsor","zscore")` — no size control: with N=10 a size regression removes most
of the cross-sectional variation and would be degenerate.

### K1 — Short-term reversal · `crypto_xs_reversal` · sign **−1**
- **Signal (causal):** 7-day past return, `close[t]/close[t−7] − 1`, bars ≤ t.
- **Sign: −1** (long past losers). **Mechanism:** liquidity provision to uninformed retail flow —
  the same mechanism as Taiwan R1, now on a venue where costs cannot kill it.

### K2 — Medium-horizon momentum · `crypto_xs_momentum` · sign **+1**
- **Signal (causal):** 30-day past return, `close[t]/close[t−30] − 1`, bars ≤ t.
- **Sign: +1** (long past winners). **Mechanism:** documented crypto cross-sectional momentum.
- K1 and K2 carry **opposite signs at different horizons on purpose** — these are two distinct
  documented effects, not a hedge. If BOTH pass, that is internally coherent (reversal at a week,
  continuation at a month). If both fail, the crypto cross-section is recorded empty.

### K3 — NEGATIVE CONTROL · `crypto_null_control` · sign **+1**
- Deterministic pseudo-random score, no price information. **Expected: IC ≈ 0 and FAIL.**
- Validated as an instrument on today's country-ETF run (scored DSR 0.0002 on a clean null). If K3
  scores "significant", the run is **invalidated** and K1/K2 are discarded regardless of how they
  look.

## 2. Pre-committed pass conditions (same bars as the country run)

- realized sign == committed sign;
- IC-IR ≥ 0.05, |IC t| ≥ 3.0;
- **block-bootstrap CI must exclude zero** (not merely a large t — overlapping returns inflate t);
- **frictionless Sharpe > 0 AND net Sharpe at the standard cost model > 0**;
- **decile spread must share the sign of the IC**;
- K3 must FAIL.

**And one economic bar, committed now:** given TSMOM already delivers net SR 0.60, a crypto sleeve is
only interesting at **net SR ≥ 0.30** — below that it cannot move a portfolio that already holds a
0.60 book, and calling it a discovery would repeat today's C1 overclaim risk (real, 0.081, useless).

## 3. Multiplicity

Declared **3** for this batch, new substrate (crypto perps). Taiwan's 8 and the country ETFs' 2 do
not carry over — different data.

## 4. Stop rule

Complete at 2 hypotheses + 1 control. **No K4.** If both fail, the crypto cross-section is recorded
empty on free data and is not re-probed with different lookbacks, vol scaling, or funding-rate
conditioning.

## 5. Promotion ceiling

Not survivorship-free (the 10 names are current survivors; delisted/failed perps are absent — a
**material** bias in crypto, where failure is common). Results are UPPER BOUNDS. A pass earns
forward-incubate only.

## 6. Results

*(Empty at commit time on purpose — verifiable from git history.)*

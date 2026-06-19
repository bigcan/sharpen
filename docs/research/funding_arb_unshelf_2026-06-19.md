# Funding-Arb Un-Shelf Probe v2 — Fresh Regime + Cross-Sectional Dispersion

> **Status:** RESEARCH VERDICT (2026-06-19, Session 553-cont-55). Operator: *"revisit funding-arb, full investigation — can we un-shelf it and make it profitable?"*
> **Verdict:** **NO as a standalone strategy — verdict UNCHANGED and STRENGTHENED.** Fresh data confirms the funding regime has **not** recovered through June 2026, and the one previously-untested angle (cross-sectional funding *dispersion*) is now **also falsified** by a clean cost/borrow decomposition.
> **Plan + cheap CPU falsification only — no training/deploy/RL authorized.**
> Builds on `docs/research/funding_arb_unshelf_2026-06-06.md` (v1: directional-carry verdict). Env causal cleanliness re-confirmed there (~96%, no leak); not re-litigated here.

---

## 1. Why revisit, and what was actually new to test

The v1 verdict (2026-06-06) had **two** explicit escape hatches:
1. **Binding data caveat** — the silver cache ended 2026-04-28, so the *current* funding regime was invisible. v1's lone "what would change this verdict" was a **funding-regime recovery (>8%/yr level)**.
2. **Untested strategy axis** — v1 only tested **directional** carry (bet on the funding *level*: long-only standard arb + level-thresholded two-sided). It never tested **cross-sectional dispersion** (relative value: trade the funding *spread* across assets), which is the project's own validated paradigm (TSMOM linear core, net Sharpe 0.60) applied to funding and is regime-robust to the aggregate level.

This probe resolves both. Data cache was **still frozen at 2026-04-28** (not refreshed in the 13 days since v1).

Artifacts: `scripts/research/funding_arb_xsec_dispersion_probe.py`, `results/funding_arb_xsec_dispersion/results.json`, fresh OKX pulls `C:\tmp\okx_funding_fresh.parquet` / `okx_perp_ohlcv_fresh.parquet`.

## 2. Fresh data: the funding regime has NOT recovered

Pulled fresh funding + 4H perp OHLCV from **OKX** (Bybit/Binance geo-blocked here; funding is tightly arbitraged across venues so OKX is a faithful regime proxy). Funding history reaches back to **2026-03-16**, overlapping the silver cache through Apr 28 for cross-validation and extending fresh to **2026-06-19**.

Cross-validation in the overlap (Mar–Apr 2026): silver level −0.97/+1.40%/yr vs OKX −0.24/+2.62%/yr; dispersion silver 9.4/6.9 vs OKX 7.9/7.1 — same ballpark, same signs. OKX validated as a proxy.

| Month (2026) | Funding **LEVEL** (ann %) — what *directional* carry needs | Cross-sectional **DISPERSION** (ann %) — what *RV* needs |
|---|---|---|
| Mar (OKX) | **−0.24** | 7.86 |
| Apr (OKX) | **+2.62** | 7.11 |
| **May (OKX, fresh)** | **+3.69** | 6.69 |
| **Jun (OKX, fresh)** | **+1.00** | 7.54 |

**The funding *level* is still dead** — 1–4%/yr (below the 7.3%/yr borrow + ~20bp round-trip), exactly the 2026 collapse v1 documented (0.39%/yr). No leverage-bull recovery materialized. The v1 reactivation trigger (>8%/yr) is **not** met. → **Directional carry remains uneconomic. v1 confirmed.**

Note the *dispersion* (~7%/yr) is consistently larger than the *level* (~1–4%/yr) — which is exactly why the cross-sectional angle was worth a dedicated test.

## 3. Cross-sectional dispersion: real gross edge, uneconomic to harvest

Rank-based, dollar-neutral book: short-perp the top-k highest-funding names, long-perp the bottom-k lowest, re-ranked each settlement on a causal funding-EMA. Cost/sign model verbatim from the v1 falsification. Two constructions:

| Book | Full sample (2022→2026-04) | Fresh OKX (2026) | Verdict |
|---|---|---|---|
| **basis_neutral** (delta-neutral legs; pays borrow on reverse leg) | ann **−11 to −12%/yr**, PF 0.09–0.12, 0/8 WF windows | ann **−37 to −43%/yr** | **FALSIFIED** |
| **perp_only** (long-low/short-high perp; no borrow, takes price risk) | ann **−1.7 to +0.9%/yr**, Sharpe **0.04**, maxDD **−43 to −55%**, yearly swings −47%→+24% | ann **−7 to −8%/yr** | **FALSIFIED (gamble)** |

### Decomposition (k=3, silver) — the decisive economics

| Component | Annualized | Meaning |
|---|---|---|
| **Gross funding spread harvested (realized t+1)** | **+10.53%/yr** | The dispersion edge is **real and correctly signed** — funding ranking persists. |
| Borrow drag (reverse leg, half the book) | ≈ −2.9%/yr (book), 7.3%/yr per short-spot unit | Cost of being market-neutral via the long-perp/short-spot leg. |
| **basis_neutral ceiling, funding − borrow, ZERO txn cost** | **≈ +1.3%/yr** | Below T-bills even idealized. |
| Turnover cost (rank churn, 10bp/leg × ~1095 settle/yr) | ≈ **−12.7%/yr** | Daily re-ranking churns the book → this is what drives the −11%/yr net. |

- **basis_neutral**: the funding spread (+10.5%/yr gross) is **too small** relative to the cost of harvesting it market-neutrally. Net of borrow alone the ceiling is **+1.3%/yr** (below T-bills); net of realistic rank-churn turnover it is **−11%/yr**. The classic *real-gross-edge / killed-by-costs* cost-gap pattern. Reducing turnover can't rescue it — the zero-cost ceiling is already sub-T-bill, and holding longer lets the ranking go stale (decays the +10.5%).
- **perp_only**: avoids borrow but the P&L is **price noise, not funding** — per-settlement price-spread vol **54.8%/yr** vs funding-spread vol **0.4%/yr** (137×). Mean price spread looks +27%/yr but at 54.8% vol → Sharpe 0.04, with −50% drawdown years (2023, 2026 short-squeeze regimes). It is a relative-price gamble that leans short-high-funding, not an arbitrage.

## 4. The complete logic tree — all three harvest axes fail

| Harvest axis | Borrow? | Result | Why |
|---|---|---|---|
| **LEVEL** — long-only standard arb (richest names) | No | Dead | Level collapsed to 1–4%/yr (v1 + fresh through Jun-2026), < borrow+cost |
| **SPREAD, market-neutral** — basis_neutral dispersion | Yes | Falsified | Gross +10.5%/yr; net of borrow ceiling +1.3%/yr (< T-bills); net of churn −11%/yr |
| **SPREAD, perp-only** — long-low/short-high perp | No | Falsified | 99.99% price noise, Sharpe 0.04, −50% tail years — a gamble, not an arb |

There is no fourth borrow-free way to harvest dispersion: market-neutrality *requires* the reverse (short-spot, borrow-paying) leg, and the borrow + churn exceed the spread.

## 5. Verdict by path (unchanged from v1, now with dispersion ruled out)

| Path | Verdict | Rationale |
|---|---|---|
| Standalone (prop-firm or always-on sleeve) | **NO** | Level dead; dispersion falsified; both axes closed |
| Cross-sectional funding RV | **NO (new)** | Real gross edge (+10.5%/yr) but uneconomic net of borrow + churn |
| Regime-timed dormant sleeve | **MARGINAL (option, not strategy)** | Correctly sits in cash now; reactivates only on a funding-bull |
| Funding-carry factor in `MultiAssetAllocatorEnv` | **deferred** | v1's recommended slot exists (`carry_ary`, ADR-6, ships 0) but funding-carry was **never wired**. No reason to wire now — level is dead. Wire only on regime recovery |

## 6. Recommendation

**Keep funding-arb shelved. Do not revive the DSAC pipeline, do not build an RL allocator overlay, do not wire a funding-carry factor now.** The cheap, mandated, pre-RL linear falsification returns a clear NO across *every* harvest axis. The signal is clean and the gross dispersion edge is real — but the current regime pays neither a harvestable *level* nor a *net* spread.

**One conditional reactivation trigger (monitor, don't build):** sustained aggregate funding **> 8%/yr** (leverage-bull). At that point the dormant directional sleeve and the `carry_ary` slot become live candidates. Watch the monthly LEVEL row of the regime table as a cheap monitor; re-run this probe when it crosses.

## 7. Caveats

- OKX funding history caps at ~3 months, so the fresh window is Mar→Jun 2026; the longer trajectory relies on the silver cache. Cross-validation in the overlap confirms OKX as a faithful proxy.
- perp_only price PnL uses OKX 4H closes resampled to settlement bars; spot proxied by perp close (basis ≈ 0), identical convention to v1 — would only *lower* numbers.
- Borrow (7.3%/yr) and costs (1bp spot + 5bp perp + 2bp slip/leg) are taken verbatim from `configs/funding_arb_dsac_l1_multiseed.yaml`. Sensitivity: even halving borrow leaves basis_neutral net negative once churn is included.

## 8. Reproduce

```bash
python scripts/research/funding_arb_xsec_dispersion_probe.py
# -> results/funding_arb_xsec_dispersion/results.json
# Fresh data: C:\tmp\fetch_okx_funding.py (OKX, public, no auth)
```

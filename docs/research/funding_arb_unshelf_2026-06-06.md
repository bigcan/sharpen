# Funding-Arb Un-Shelf Probe — Real Edge, Decayed Regime

> **Status:** RESEARCH VERDICT (2026-06-06, Session 553-cont-34+). Operator asked: *"can we un-shelf funding-arb and make it work?"*
> **Verdict:** **NO as a standalone strategy** (the regime that powers it is currently dormant); **YES as an uncorrelated carry *factor* in the cross-sectional allocator** already under construction.
> **Plan only — no training/deploy/code change authorized by this document.**
> **Distinction that matters:** unlike sg1-btc / gmgp1-btc, funding-arb's failure is **honest regime decay, not a look-ahead leak.** The signal and the code are clean.

---

## 1. Trigger & scope

Funding-Arb DSAC (QR-SAC + CVaR, 10-asset delta-neutral perp-funding carry) was **HALTED at Stage-3 WF on 2026-04-29 (S508)** — 4/8 windows profitable, FAIL G1+G2, "regime-fragile." The shelving memory proposed a ~2–4 week research path (funding-rate-quantile gate / regime-conditioned retrain). This probe tests whether that path is viable **before** spending the cycle, using the project's own validated methodology: a cheap, causal, net-of-cost **linear falsification** of the signal, in front of any RL re-investment.

Artifacts: `scripts/research/funding_arb_carry_falsification.py`, `results/funding_arb_carry_falsification/results.json`.

## 2. Evidence

### 2.1 The env is causally CLEAN (the key distinction from the falsified directional strategies)

Adversarial leak trace of `sharpen/crypto/envs/funding_arb_env.py` + `crypto/data/crypto_array_builder.py` + `crypto/features/funding_arb_features.py` across all six LEAK-2 vectors — **no material look-ahead found (~96% confidence):**

| Vector | Verdict | Evidence |
|---|---|---|
| Funding-rate timing | CLEAN | Settlement-gated 00/08/16 UTC (`_build_funding_mask`); 8h rate ffill'd to 1h with boundary truncation; earns last-settled rate only |
| Obs vs reward asymmetry | CLEAN | Obs carries last-known funding; reward credits realized settlement — correct (observe current, earn next) |
| Basis / mark price | CLEAN | Spot reindexed to perp ts, same-bar basis |
| EMA-Z normalization (LEAK-1) | CLEAN | Window-local renorm `_renormalize_within_window` (`crypto_array_builder.py:320`) |
| Resample / OI publication lag | CLEAN | OI lagged 1 bar (`PUBLICATION_LAGS`); funding lag 0 (published at settlement) |
| Reward shaping | CLEAN | Plain next-step PnL; no hindsight/future-price reference |

⇒ The PF the strategy reported was **never a leak artifact.** The funding-carry premium is a real, documented risk premium (carry factor, AMP 2013), correctly implemented.

### 2.2 The WF failure = regime decay + policy churn (not absent edge)

From `results/funding_arb_dsac_wf/wf_report_20260429_095051.json`, the losing windows fail for **economic** reasons:

- **Window 9 (Q1-2025):** −2.13%, `total_funding_earned = −30` (funding flipped **negative**) while `cumulative_fees = 1779` — **8× the ~200 normal**. The DSAC policy *churned* harvesting carry that had turned against it. (Delta exposure tiny 0.0015 → not a hedge breakdown.)
- **Windows 3 / 21:** thin carry eaten by fees (funding/cost ratio 1.11 / 0.42).

A disciplined **static** harvester loses far less in the same windows (W9: −0.74% vs the policy's −2.13%) → confirms policy overtrading — but **still can't make the recent windows positive.**

### 2.3 The carry premium is real but regime-gated on *elevated* funding — which has decayed

Cross-asset mean annualized funding by year, and a causal net-of-cost static carry book (spot 1bp + perp 5bp + 2bps slip, 7.3%/yr borrow on reverse legs, max_gross 0.80):

| Year | Mean funding | Static carry book net |
|---|---|---|
| 2023 | 7.96%/yr | **+5.88%** (Sh 18) |
| 2024 | 12.42%/yr (peak) | **+10.51%** (Sh 24) |
| 2025 | 4.16%/yr | −0.59% |
| **2026** | **0.39%/yr** (58% of settles >0 ≈ coin-flip) | **−3.15%** (Sh −17) |

The strategy was HPO'd/trained on the **2023–2024 funding bull.** In the current regime, net of borrow + ~20bp round-trip cost, **there is no harvestable carry.**

### 2.4 Regime gating does NOT fix it — falsifies the shelving memo's #1 suggested fix

Per-asset funding-EMA threshold gating made every variant **worse** (it buys the funding top right before mean-reversion and churns at the threshold boundary). The only "positive" per-asset gated book sits **87% in cash** earning ~0.6%/yr — not a strategy.

A **portfolio-level** regime switch (deploy the whole book only when aggregate trailing funding is rich) behaves correctly but parks in cash exactly now. Best case in the deploy-forward regime (2025–26): **+0.58%/yr, Sharpe 2.51 — below T-bills.**

### 2.5 It IS genuinely uncorrelated

Static carry book corr to BTC = **0.098**; max DD < 4% (delta-neutral). A real diversifier — the problem is purely that the absolute return is too thin in the current regime to stand alone.

## 3. Verdict by path

| Path | Verdict | Rationale |
|---|---|---|
| Standalone Velotrade prop-firm | **NO** | Recent-regime net edge ≈ 0; gating can't lift it; Velotrade already pivoted to GMGP1-BTC / SG-1-BTC (S508) |
| Always-on diversification sleeve | **NO (now)** | Bleeds −1.3%/yr in 2025–26 low-funding regime |
| Regime-timed dormant sleeve | **MARGINAL** | Correctly sits in cash now; only earns on the next leverage-bull funding spike. A dormant *option*, not an active strategy |
| **Carry factor in `MultiAssetAllocatorEnv`** | **✅ BEST** | Carry (AMP 2013) is the documented *sibling* of the TSMOM linear core just validated (net Sharpe 0.60). Add a clean `funding_carry` signal column; let the allocator's vol-target / regime-weighting size it — pays when funding is rich, ~0 when thin |

## 4. Recommendation

**Do not revive the DSAC standalone pipeline.** The lesson of the ~500-session campaign repeats with a twist: here the *signal is sound and the code is clean*, but the *risk premium is regime-conditional and currently dormant.* No RL or per-asset gate manufactures carry that isn't being paid.

**Fold the funding-carry signal into the allocator build** (`e639bc55` Phase 2, `multi_asset_allocator_env.py`). It already carries a clean, tested `trend_conviction` column; add `funding_carry` from this verified-clean pipeline as a second uncorrelated factor. This harvests the real premium where regime-sizing is built in, rather than bolting a regime gate onto a standalone book — and it directly serves the north-star (native uncorrelated sleeve generator).

## 5. What would change this verdict

- A sustained return of elevated cross-asset funding (>8%/yr regime, leverage-bull) — at which point the dormant regime-timed sleeve reactivates. Watch aggregate funding-EMA as a regime monitor.
- The probe used perp `close` as the spot proxy (silver cache has no separate spot series), so basis-risk P&L is modeled ≈0. A true spot/perp basis series would add second-order basis risk — would only *lower* the numbers, not raise them. Verdict is robust to this.

## 6. Reproduce

```bash
python scripts/research/funding_arb_carry_falsification.py
# -> results/funding_arb_carry_falsification/results.json
```
Data: `data/crypto_cache/silver_funding.parquet` + `silver_ohlcv.parquet` (10 assets, 2022-01 → 2026-04-28). For a recent-OOS refresh, extend the cache past 2026-04-28 (currently ~6 weeks stale).

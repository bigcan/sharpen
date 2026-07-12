# ETF-vs-Underlying(-Stock) Arbitrage — Stage-0 Falsification Pre-Registration

> **Status:** SPEC (pre-registered 2026-07-12, Session S553-cont-128) — **gates frozen BEFORE any result.**
> **Design philosophy:** falsify-before-optimize (R1 / FFD / ETF-pairs / futures-basis pattern). Stage-0 CPU probe (~$0 GPU).
> **Operator stance (this book):** trading fees **deprioritized** — judged on GROSS structure + statistical validity; net + a per-leg **harvestability decomposition** reported for the record.
> **Mechanism:** the classic ETF arbitrage — ETF price vs its underlying constituents (AP creation/redemption keeps `0050` ≈ its basket). Third distinct mechanism in the sweep after ETF-ETF same-index spread ([[project_etf_pairs_arb_nogo_s553]]) and stock-index-futures basis ([[project_futures_basis_arb_planned_s553]]).
> **Why this one could differ:** unlike the two prior tests, **both legs close at 13:30** (same Taiwan equity session/auction) ⇒ **no timing-mark artifact**, and both are **directly-tradeable liquid instruments** ⇒ **no cash-index staleness**. So a real reversion here would be genuinely harvestable — the two failure modes that killed the earlier books don't apply. The competing prior: `0050` and `2330` are among the most-traded Taiwan names ⇒ the spread may already be arbitraged flat (small gross), or MM-cost-gated like the ETF-ETF spread.

---

## 0. Instruments, data, scope

`0050` is ~**50% TSMC (`2330`)** by weight, so "ETF vs underlying" reduces to a tractable 2-liquid-leg test rather than a 50-stock basket (no official `0050` weight file held).

| Pair | Legs | Source | Overlap |
|------|------|--------|---------|
| **P1 (primary)** | `0050` (ETF) vs `2330` (TSMC, ~50% wt) | panel `ohlcv_daily.parquet` + `taiwan_universe/taiwan_daily.parquet` | 2017-01 → 2026-06, ~2,308 daily |
| P2 (secondary) | `0050` vs **equal-weight top-10 holdings** basket (`2330,2317,2454,2308,2412,2882,2881,2303,3711,2891`) | same | ~2017-2026 |

P2 is an approximate basket (no official weights ⇒ equal-weight proxy) and is exploratory — **cannot flip the verdict**.

**Data-quality (Gate D):** both legs `close > 0`, no NaN, no leading flat run on the used window.
**NOT in scope:** true intraday premium/discount-to-iNAV (needs creation/redemption baskets + iNAV — data-blocked); official cap weights; any Crucible grammar change (built only on a clean pass).

---

## 1. Construction (locked)

- **Spread:** `s_t = ln(P_A,t) − ln(P_B,t)` (A=`0050`; B=`2330` or the EW top-10 log-basket `mean_i ln P_i`). β=1; the rolling z-window demeans the slow relative drift.
- **Signal:** causal rolling z `z_t = (s_t − mean_W s)/std_W s`, **W ∈ {20,40,60}** (frozen), bars ≤ t.
- **Position:** `pos_t = −clip(z_t, ±2)/2` (short the spread when `0050` is rich vs the underlying).
- **Return:** `r^A_{t+1}, r^B_{t+1}` close-to-close (forward, via next-day price); `spread_ret = r^A − r^B`. **Primary lag0** (both legs close 13:30 → synchronized → same-day is clean, unlike the futures basis); **lag+1 reported** as robustness.
- **Costs (record only):** one-way 10 bps on `0050` leg + 20 bps on the stock leg (Taiwan stock 0.3% sell tax > ETF 0.1%); 2× for the harsh check.

---

## 2. Metrics & Gates (FROZEN — gross/validity per operator stance)

**Primary family** = P1 (`0050/2330`) × `W ∈ {20,40,60}`. Reported statistic = **median-W GROSS Sharpe (lag0)**.

**GO requires ALL:**
1. **G1 — structure:** median-W gross Sharpe ≥ **0.50**.
2. **G2 — significance:** block-bootstrap (block 5d, 10k) p05 gross > 0.
3. **G3 — OOS:** last 30% gross > 0.

**Harvestability decomposition (REPORTED, decides the follow-on):** split `pos·(r^A − r^B)` into **ETF leg `pos·r^A`** and **stock leg `pos·(−r^B)`**. Both legs are liquid/tradeable here, so — unlike the futures basis — an edge in *either* is harvestable; a NEGATIVE per-leg Sharpe on both with positive total = pure diversification/vol-timing (fragile). Also report **net @ (10+20)bps** and **2×**.

**Tripwires (must PASS or INVALID):** TW-1 causality-gap `gross_lag0 − gross_samebar>0.30`; TW-2 permutation-null p<0.05; TW-3 Gate-D data quality; TW-4 oracle ≫ causal.

---

## 3. Predictions

- **H1:** `0050`/`2330` shows real, significant reversion (G1–G3), with edge in ≥1 liquid leg ⇒ a genuinely tradeable ETF-vs-underlying signal (the two prior failure modes absent) → Stage-1 (cost realism + basket generalization).
- **H0 (prior):** the pair is already arbitraged flat (gross < 0.50) or MM-cost-gated like the ETF-ETF spread ⇒ NO-GO; book closes for a few CPU-minutes.

---

## 4. VERDICT (run 2026-07-12, S553-cont-128) — NO-GO (efficient, no gross edge)

`scripts/research/etf_underlying_arb_probe.py`, seed via futures-probe helpers.

- **P1 `0050/2330` (TSMC, primary), n=2,303 daily (2017-2026):** median-W gross Sharpe **−0.17** (all W negative), G2 bootstrap p05 −0.48 + permutation p=0.485 (insignificant), G3 OOS −0.46. Leg decomposition both negative (ETF −0.09, stock −0.22). **All gates fail → NO-GO.** Harness tripwires PASS (TW-1 causality gap +0.97, TW-4 oracle +3.61 ≫ causal) — the probe is valid; the signal is simply null (slight continuation, not reversion).
- **P2 `0050` vs EW top-10 basket (secondary, exploratory), n=1,974:** gross +0.41 but permutation p=0.089 (**not** significant), net ~+0.19, mixed under lag+1. A weak whisper that cannot flip the verdict.

**Interpretation:** unlike the two prior arb books, this failed **not** on cost or staleness but on **efficiency** — both legs are liquid, directly tradeable, and synchronized (13:30 auction), so no artifact is available to blame; `0050` tracks its underlying (esp. TSMC) so tightly (the AP creation/redemption relationship is the most continuously arbitraged in markets) that there is **no gross reversion to harvest**. Cleanest possible NO-GO. True intraday premium/discount-to-iNAV arb stays data-blocked. No Crucible grammar change; gates_hash untouched.

**Sweep complete (3 mechanisms, 3 failure modes):** ETF↔ETF same-index = cost/MM-gated (real gross); futures basis = staleness-unharvestable (real gross); ETF↔underlying = efficient/no-edge. Taiwan arbitrage on free daily data is exhausted; any revival needs MM-side execution, a tradeable non-stale spot proxy, or intraday iNAV/creation-redemption data — a data/execution problem, not an idea problem.

---

## 5. FOLLOW-UP (2026-07-12): proper cap-weighted basket NAV + intraday — both confirm NO-GO

After the operator asked about iNAV sourcing, two refinements (scout code in scratchpad; verdicts here):

**5a. Intraday variant** (FinMind 5-sec TAIEX + 1-min `0050`, 40-day window; TX 1-min): the daily 13:30 re-test is REDUNDANT (5-sec TAIEX @13:30:00 == the daily close already used). The intraday 1-min basis reversion shows huge gross SR (+15 to +25 ann) but is a **futures-lead-cash lead-lag microstructure NO-GO** — edge entirely in the lagging leg (TX-leg ≈0), turnover ~70-100×/day, annihilated by any intraday cost + µs-HFT-arbitraged.

**5b. Proper cap-weighted basket NAV** (FinMind `TaiwanStockMarketValue` weights over the 45-stock Taiwan-50 basket; daily premium/discount reversion): the full 45-basket **passed every standard gate** (gross +0.59, bootstrap p05 +0.55, permutation p=0.0005, OOS +0.92) — a near-false-positive. The **liquidity-sensitivity test unmasked it**: top-10-liquid basket gross **−0.14** (no edge, matches `0050/2330`=−0.17), top-20 +0.49, all-45 +0.59 — the edge GROWS monotonically with the illiquid mid-cap tail (basket-leg +0.10→+0.73→+0.87). **Constituent-staleness signature**, net-NEGATIVE at every size. NOT real alpha, NOT harvestable.

**Methodology lesson:** standard significance gates (permutation-null, bootstrap CI, OOS) do NOT distinguish real alpha from illiquidity/staleness — a reconstructed-basket premium/discount can be strongly "significant" and OOS-stable yet be pure stale-close catch-up. Liquidity-sensitivity (restrict to liquid names) + net-of-cost + per-leg decomposition are required to catch it.

**Confirms the sweep:** the liquid/tradeable part of every Taiwan index-ETF-futures relationship is efficient; the only "edge" anywhere is the illiquid/lagging/stale leg, which is not retail-harvestable at any resolution. iNAV proper remains data-blocked (FinMind has no NAV feed; DIY reconstruction from stale closes just re-manufactures the artifact).

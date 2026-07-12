# ETF Pairs-Spread Arbitrage — Stage-0 Falsification Pre-Registration

> **Status:** SPEC (pre-registered 2026-07-12, Session S553-cont-128) — **gates locked BEFORE any result is computed.**
> **Design philosophy:** falsify-before-optimize (R1-illiquidity / FFD-channel pattern). A Stage-0 CPU probe (~$0 GPU) must clear a locked bar before anything larger (a `pairs` candidate-type in the Crucible grammar) is justified.
> **Motivation:** Crucible's proposal grammar currently expresses only `cross_sectional` (rank/momentum) and `overlay` (flow tilt) archetypes — it has **no pairs/spread/cointegration/NAV candidate**, so ETF arbitrage has never been mined. This probe asks the cheapest possible falsification question first: *does a same-index Taiwan-ETF log-spread even have daily mean-reverting, cost-surviving structure?*
> **Prior (the null we expect to confirm):** `0050` and `006208` track the **same** index (FTSE TWSE Taiwan 50), heavily arbitraged by Taiwan brokers/market-makers. Expectation: the spread is already tight; any daily edge is an MM/latency edge (earn-the-spread), not a daily-bar taker edge — same lesson as the Taiwan intraday-RL NO-GO (`project_taiwan_intraday_rl_canary_nogo_s553`, "MM-side not taker-side") and the liquid-equity X-sec closure. Net Sharpe ≤ 0 after realistic costs is the predicted outcome.

---

## 0. Scope, assets, what is NOT under test

**Pairs (frozen).** Both legs must be genuinely trading (NaN-gated common window; the panel back-pads pre-inception rows with NaN, not flat prints — verified 2026-07-12):

| Role | Pair | Economic link | Common real window | ~bars |
|------|------|---------------|--------------------|-------|
| **Primary** | `0050` / `006208` | **Same index** (FTSE TWSE Taiwan 50) → β≈1 | 2012-07-18 → 2026-07-06 | ~3,450 |
| Secondary | `0056` / `00878` | Both hi-dividend *theme*, **different** underlying indices | 2020-07-21 → 2026-07-06 | ~1,470 |

Secondary is exploratory: it **cannot flip the verdict**, only corroborate.

**Out of scope / NOT tested here:**
- True **premium/discount (NAV) arbitrage** — needs intraday iNAV + creation/redemption baskets we do not hold; **data-blocked**, logged as UNTESTABLE, not attempted.
- Intraday / MM-side execution — this is a *daily close-to-close taker* probe only.
- Any Crucible grammar change — the `pairs` candidate-type is built **only if** this Stage-0 passes.

**Data-quality precondition (Gate D).** Each pair's used window must have: no NaN on either leg, no leading flat run (TW-4), close-to-close returns finite. A leg failing Gate D → pair logged UNTESTABLE, never silently dropped.

---

## 1. Construction (locked)

- **Spread:** `s_t = log(P_a,t) − β · log(P_b,t)` on close.
  - **Primary β = 1** (same index → economically unit hedge; zero fitted params ⇒ no leak surface).
  - Secondary robustness: `β` from trailing-252d causal OLS (past-only, LEAK-2 safe). Exploratory.
- **Signal (mean-reversion):** causal rolling z-score `z_t = (s_t − mean_{W}(s)) / std_{W}(s)` over trailing window **W ∈ {20, 40, 60}** (pre-registered set; no OOS tuning). Windows use bars ≤ t only.
- **Position:** `pos_t = −clip(z_t, −Zc, +Zc) / Zc`, `Zc = 2` (linear-capped reversion; short the spread when rich). Leg A holds `+pos_t`, leg B holds `−β·pos_t` (dollar-neutral).
- **PnL (causal):** `pos_t` formed on close_t earns spread log-return `r_{t+1} = r_a,{t+1} − β·r_b,{t+1}`. No same-bar fills.
- **Costs:** charged on turnover of **both legs**: `cost_t = c · (|Δpos^A_t| + |Δpos^B_t|)`.
  - **Primary one-way `c = 10 bps`** (matches the repo Taiwan `base_sleeves` `cost_bps=0.0010` convention; covers brokerage + half-spread + amortized 0.1% ETF sell tax).
  - Robustness: `2× → 20 bps`.

---

## 2. Metrics & Gates (FROZEN)

**Primary family** = pair P1 (`0050/006208`) × `W ∈ {20,40,60}` × `β=1` = **3 cells**. Reported primary statistic = **median net Sharpe across the 3 W** (kills W cherry-pick). P2 and rolling-β are secondary.

**Metrics per cell:** annualized Sharpe (×√252) gross & net; net profit factor; avg daily leg-turnover; max drawdown.

**GO requires ALL of the following on P1:**
1. **G1 — Sharpe floor:** median-over-W **net** Sharpe ≥ **0.50** (the project's honest deployable-Sharpe floor).
2. **G2 — Cost robustness:** net Sharpe > 0 at **2× cost** (median-W).
3. **G3 — Significance:** circular block-bootstrap (block = 5 trading days, 10,000 draws) **5th-percentile net Sharpe > 0** on the median-W cell.
4. **G4 — OOS:** last **30%** as holdout; net Sharpe_OOS > 0 (median-W).

Fail any ⇒ **NO-GO**, book closed for ETF *pairs-spread* arb on free daily data, no grammar change.

**Tripwires (must PASS or the whole probe is INVALID — reported, never overridden):**
- **TW-1 leak-shift:** a same-bar variant (signal `z_t` → return `r_t`) must produce a **materially higher** Sharpe than the causal (`z_t` → `r_{t+1}`) version. If causal ≈ same-bar, the wiring is peeking — abort.
- **TW-2 shuffled null:** permute the spread-return series (fixed-seed RNG) → net-Sharpe bootstrap CI must **bracket 0**.
- **TW-3 cost monotonicity:** `netSharpe(0) ≥ netSharpe(1×) ≥ netSharpe(2×)`.
- **TW-4 padding guard:** assert used window is NaN-free and has no leading flat run on either leg.

---

## 3. Predictions

- **H1 (would overturn the prior):** the `0050/006208` log-spread mean-reverts; G1–G4 all clear; a `pairs` candidate-type is then justified for Crucible.
- **H0 (expected):** gross edge is tiny and net edge ≤ 0 after 10 bps × 2 legs — the same-index spread is already arbitraged; fails G1 (and likely G2/G3). Closes the book for a few CPU-minutes.

**Follow-on (separate book, after this):** *stock-index-futures basis arbitrage* (TAIEX cash vs TX future) — a distinct mechanism with its own pre-registration; not covered here.

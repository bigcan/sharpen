# TAILWIND-v1 — Breadth expansion test (Fable review item 2)

**Date:** 2026-07-01 · **Tests:** Fable review item 2 ("widen 18→~35-40 free-data instruments under
the frozen signal = the honest route to raise DSR"). · **Script:**
`scripts/research/breadth_expansion.py` · **Artifact:** `results/tailwind_v1/breadth_expansion.json` ·
**Follows:** the R1 DSR/PBO result (`tailwind_v1_R1_dsr_pbo_2026-07-01.md`, honest DSR 0.896 < 0.95).

## Question

Does applying the **frozen** TSMOM signal (63/126/252, skip 5, vol_win 63, tgt 0.10, cap 2.0 —
UNCHANGED) to a wider universe raise the honest momentum Sharpe enough to push the combined
momentum+BAB book over DSR ≥ 0.95? This is the one lever that *could* clear DSR without new selection
bias (adding hedges can't lift a return-DSR; adding searched sleeves worsens multiplicity).

Widened universe: **32 liquid cross-asset ETFs** (all present, full 2006→2026 window):
equity 10 (SPY QQQ IWM EFA EEM EWJ FXI EWZ VGK VNQ), rates 8 (TLT IEF LQD SHY TIP HYG EMB AGG),
commodity 7 (GLD SLV DBC USO DBA UNG DBB), fx 7 (UUP FXE FXY FXB FXA FXC FXF).

## Answer: **No — breadth does not help. NO-GO for the breadth path.**

| Metric | 18-ETF (R1) | 32-ETF (wide) | Δ |
|--------|-------------|---------------|---|
| momentum pooled net Sharpe | 0.601 | **0.563** | **−0.038** |
| BAB net Sharpe | 0.412 | 0.437 | +0.025 |
| corr(momentum, BAB) | −0.043 | −0.112 | more negative |
| combined Sharpe | 0.732 | 0.751 | +0.019 |
| combined OOS-2018 | 0.807 | 0.848 | +0.041 |
| **combined DSR (raw basis)** | 0.974 (N=24) | **0.920 (N=23)** | **still < 0.95** |
| PBO (CSCV) | 0.0009 | **0.0** | not overfit |

**Breadth *lowered* the momentum Sharpe** (0.563 < 0.601), confirming the direction of Fable's own
"untouched 32-ETF = 0.389 vs curated 18-ETF" precedent: the 18-ETF universe was **mildly favorably
selected**, and a broader, more honest universe does not carry a stronger trend edge. The combined book
is marginally more robust (OOS 0.848, corr −0.11), but the **combined DSR still fails (0.920 < 0.95)** —
the wider momentum grid's higher trial dispersion raises the deflation benchmark (SR*), offsetting the
extra diversification.

## Conclusion

The free-data cross-asset **TSMOM + BAB book does not clear DSR ≥ 0.95 on either 18 or 32 instruments**.
The edge is genuine — PBO ≈ 0 (not selection-overfit), combined Sharpe ~0.73–0.75, positive every
subperiod, cost-survivable, sleeves uncorrelated — but it sits just under the self-imposed 0.95
deflated-Sharpe capital bar (DSR ~0.90–0.92 honest). **Breadth is not the missing lever**, so items R1
and this test together close the two cheapest honest routes to lifting the return-DSR.

**What this does and does not mean:**
- **Own-capital (DSR≥0.95 gate):** the book remains correctly BLOCKED; there is no free-data lever left
  that lifts it over 0.95. A genuine *second return premium* needs data we don't have (the standing
  conclusion). Do not keep searching free-data ETF corners for it.
- **Prop-firm path (the near-term goal):** DSR≥0.95 is the *own-capital* gate, NOT the prop objective.
  The prop challenge objective is P(hit target before breaching DD/daily-limit) — a path-dependent,
  finite-horizon problem (Fable review item 1). A real ~0.73-Sharpe uncorrelated book with PBO≈0 is a
  perfectly serviceable prop-challenge vehicle; the next lever is the **challenge Monte-Carlo simulator**
  (review item 1, highest-ROI prop item), not more DSR chasing.

**Recommendation:** stop trying to clear DSR≥0.95 on free-data ETFs (both cheap routes now closed);
redirect to the prop-firm challenge simulator (review item 1) where this book's real, uncorrelated,
non-overfit edge is fit-for-purpose.

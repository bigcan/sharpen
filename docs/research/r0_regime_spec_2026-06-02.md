# R0-regime — Regime A/B Separation Test (gmgp1-gold de-leaked signal)

> **Status:** SPEC + RUN (Session 553-cont-30, 2026-06-02). CPU-only analysis, no training/deploy.
> **Parent decision:** `docs/research/redesign_pivot_2026-06-02.md` §Update cont-29 — lever LOCKED = *regime-gated price signal*.
> **Falsifies-or-supports:** the chosen lever. R0-regime is the cheap, causal falsification gate the redesign protocol requires *before* any HPO/training.

---

## 1. The question (decisive)

The de-leaked gmgp1-gold multiscale signal has a **real-but-decaying** edge: WF ens_pf_weighted = **1.83 / 1.61 / 0.99 / 1.04** across 4 monthly folds. The fold timeline is a clean temporal split:

| Fold | Test window (GC front, 15m) | ens_pf_weighted | Verdict |
|------|------------------------------|-----------------|---------|
| 0 | 2025-07-31 → 2025-08-29 | 1.83 (+23%) | **WIN** (regime A) |
| 1 | 2025-08-29 → 2025-09-30 | 1.61 (+23%) | **WIN** (regime A) |
| 2 | 2025-09-30 → 2025-10-31 | 0.99 (−0.7%) | **BREAKEVEN** (regime B) |
| 3 | 2025-10-31 → 2025-11-28 | 1.04 (+1.3%) | **BREAKEVEN** (regime B) |

**Lever hypothesis:** the edge is *active* in some regime A and *dead* in regime B. If a **causal, leading** regime variable separates A from B, we gate the existing de-leaked policy (trade in A, flat in B) and recover deployable risk-adjusted return. If **no** causal variable separates them beyond the single macro Aug→Oct shift, regime-gating is **falsified** for gold and the lever fails.

**Decisive answer required on:** *does regime A/B exist as a causally-detectable, gateable thing — or is the good→bad transition only visible in hindsight?*

## 2. Why this is the right cheap test

- Reuses existing artifacts only: `results/gmgp1_gold_ensemble_wf_x2deleak/fold_0{0..3}/*_trajectory.parquet` (validated: Δportfolio_value reproduces verdict `pf_bar` to ~1%) + `results/prism_research/prism_features_gc_2025.parquet` + `data/processed/gc_2025_15min_front.parquet`.
- No GPU, no retrain. Per the redesign protocol's *falsify-before-optimize* rule: a hypothesis must clear a cheap causal A/B before any HPO.

## 3. Candidate regime variables

**Data reality check (from inspection):**
- Prism **Chronos** is DEAD in this file — `chronos_spread` all-zeros, `confidence` constant 0.7, all `*_prob` columns are the default template. **Chronos cannot be tested.** Only the GAHMM *hard labels* are real.
- Prism GAHMM: `price_regime` ∈ {0=bear,1=neutral,2=bull} (counts 13/102/198 in 2025 — bull-dominated), `vol_regime` ∈ {0=low,1=normal,2=high} (103/103/107, balanced), `composite_code`. **Daily** granularity, **backward-looking** (labels regime after observing the day's close).

| # | Family | Variable | Granularity | Causal handling |
|---|--------|----------|-------------|-----------------|
| C1 | Prism GAHMM | `price_regime`, `vol_regime`, `composite_code` | daily | **lag 1 day** (a label computed at close of day d−1 is the most recent thing known at the open of day d). Contemporaneous (no lag) kept ONLY as a leaky upper bound. |
| C2 | Causal price — trend | trailing K-bar / K-day return sign & magnitude; distance from trailing MA | 15m + daily | trailing window ending at t−1; `.shift(1)` |
| C3 | Causal price — vol | trailing realized vol; ATR percentile vs trailing distribution | 15m + daily | trailing window ending at t−1; `.shift(1)` |
| C4 | Causal time | session bucket (Asia/London/NY), hour-of-day | 15m | from timestamp only — trivially causal |

Chronos (C0) is reported as **untestable (no data)**, not as a negative result.

**Leak guard (LEAK-2 / CAUS-01..05):** every regime variable at bar/day t must be a pure function of data stamped ≤ t−1. The script asserts this with a tripwire (shift-by-one reconstruction); any variable that fails the assert is rejected, not silently used.

## 4. Tests

Primary policy = `ens_pf_weighted` (the verdict's selected rule). Robustness re-run on `solo_456` and `ens_mean`.

PnL series = within-fold `Δportfolio_value` (validated). PF = Σ(+ΔPV)/|Σ(−ΔPV)|. Daily panel keyed by (fold, calendar_date) to avoid crossing the fold-reset boundary.

- **T1 — Period characterization.** Distribution of each regime variable in WIN (folds 0-1) vs BREAKEVEN (folds 2-3). A variable that doesn't even shift between the two *known* regimes cannot be the gate (necessary, not sufficient).
- **T2 — Daily-PnL separation (causal, pooled over all 4 folds).**
  - Categorical (price_regime, vol_regime, session): mean daily PnL + PF per state; Kruskal–Wallis across states.
  - Continuous (trend, vol, atr_pct): Spearman ρ with daily PnL; AUC predicting an up-day; median-split Mann–Whitney U.
- **T3 — Gate simulation (pre-registered, a-priori rules — no tuning).** For each fixed rule (e.g. trade only if price_regime=bull; vol_regime∈{low,normal}; causal trend>0; atr_pct<median; session∈{London,NY}): gated PnL = pnl where regime-A else 0 (flat = 0, zero-cost — *optimistic* for the gate). Report gated PF / return / time-in-market, per fold and pooled, vs ungated.
- **T4 — OOS-calibrated gate (anti-hindsight, decisive).** For the best continuous feature, choose the threshold that maximizes gated Sharpe/PF **on folds 0-1 only**, freeze it, apply to folds 2-3. Does it lift held-out folds 2-3 PF meaningfully above ungated (0.99/1.04) while keeping 0-1 profitable? This separates generalizable regime information from one-event curve-fitting.
- **T5 — Leaky upper bound vs causal gap.** Recompute the gates with *contemporaneous* Prism (no lag) and with the threshold tuned on the *full* data → best-possible-hindsight gate. The gap (hindsight − causal/OOS) quantifies how much apparent benefit is illusory. Large gap = the gate is hindsight, not signal.

## 5. GO / NO-GO decision rule

- **GO (regime A/B real & gateable):** ≥1 **causal** variable gives significant daily-PnL separation (T2, p<0.05, correct sign) **AND** its OOS-calibrated gate (T4) lifts held-out folds 2-3 net PF clearly above ungated (target ≥ ~1.2) while folds 0-1 stay profitable, **AND** the hindsight gap (T5) is modest. → Name the gate variable; proceed to A/B the gate on the de-leaked policy.
- **NO-GO (lever falsified):** no causal variable separates beyond the single macro shift; gates that "work" only do so by hindsight-removing Oct-Nov; OOS gate fails to generalize; large hindsight gap. → Regime-gating lever falsified for gold → report to operator; pick next lever.
- **AMBIGUOUS:** borderline; document the strongest causal variable and the specific follow-up that would resolve it.

**Caveat (stated up front):** with one macro good→bad transition (n≈1 regime *event*; n≈107 trading days), any variable that merely flips around 2025-10-01 will look explanatory in-sample. The OOS gate (T4) and the within-period finer-grain separation (T2 pooled) are what make the answer decisive rather than a hindsight narrative. The flat=zero-cost gate model is optimistic for the gate, so a NO-GO under it is conservative (the gate failed even with its best-case accounting).

## 6. Outputs

`results/r0_regime/`: `verdict.json` (GO/NO-GO + per-test stats), `t2_daily_separation.csv`, `t2b_within_period.csv`, `t3_gate_sim.csv`, `t4_oos_gate.csv`, `t6_intraday_gate.csv`, `summary.md`. Script: `scripts/research/r0_regime_separation.py`.

---

## 7. VERDICT (run 2026-06-02) — **NO-GO**, robust across all 3 policy rules

Full writeup: `results/r0_regime/summary.md`. Decision was computed programmatically.

**No causal regime variable — Prism GAHMM (price/vol), causal daily price features
(mom/dist/vol/vol-pct), or intraday session/vol — separates gold's WIN folds from
BREAKEVEN folds finely enough to gate.** Four confirmations:
1. **No within-period separation** (T2b): every MWU p ≥ 0.61 inside both periods.
2. The lone pooled hit (`vol_20`, p=0.031) is a **calendar confound** — vol and PnL both drift with the fold index (corr +0.23 / −0.43); fails Bonferroni; null within each period.
3. **No OOS-calibrated gate is deployable** (T4/T6, bar-PF): best = `prism_vol_calm` 1.107 (76% flat) and `session_NY` 1.186 — both < the 1.20 bar; rest ≈ ungated 1.007.
4. **Look-ahead doesn't help** (T5): leaky-best 1.51 ≈ causal-best 1.51 ≈ ungated 1.32 — nothing to detect at daily-regime granularity even *with* hindsight.

Chronos was **untestable** (dead data). Causality tripwire passed.

**Consequence:** the locked lever (#2 regime-gated price signal) is **falsified for gold**
across every price-only regime granularity available. The Aug→Oct decay is the directional
signal decaying, not a gateable regime. Next is a **lever choice** (non-price/macro regime axis,
regime-*conditioned* RL, or cross-sectional family), not more tuning. R0-regime killed the
cheapest lever cheaply, pre-GPU.

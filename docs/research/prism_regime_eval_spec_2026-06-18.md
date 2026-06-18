# PRISM Regime-Detector Eval — How GOOD / How USEFUL (price + vol regimes)

> **Status:** SPEC + PRE-REGISTRATION (Session 553-cont-52, 2026-06-18). CPU-only readout, no training/deploy.
> **Parent campaign:** PRISM falsification. cont-49/50/51 killed the Chronos *daily forecast* head (no IC edge, fine-tune doesn't help, confident-tail doesn't help). This eval turns the lens on the **GAHMM regime detectors** — the other half of PRISM — that cont-49 only ever pointed at *direction*.
> **Falsifies-or-supports:** the usefulness of PRISM's price-regime and vol-regime detectors at daily granularity. This is the cheap, causal, pre-registered gate the redesign protocol requires before any regime-conditioned build.

---

## 0. Why this is not a re-run

Two prior results bound the question, so this eval is deliberately scoped to the **gap** between them:

| Prior | What it tested | Verdict |
|------|----------------|---------|
| **cont-49** regime pillar (`prism_predictive_eval.py`) | GAHMM regime as a **directional** signal: `IC(price_bull−price_bear, fwd_return)` + bull-gated long, BTC/gold/EURUSD, clean Path-A | every GAHMM IC **inside shuffle-null**; bull-gated long net PF ≤ 1.0 |
| **R0-regime** (cont-30, `docs/research/r0_regime_spec_2026-06-02.md`) | GAHMM (+ causal price) **gating the gold directional strategy** to recover its decayed edge | **NO-GO**; T5 leaky≈causal "nothing to detect at daily-regime granularity" — **gold only, directional only** |

So **"does the regime predict next-day direction?"** is already answered **NO**. What neither touched:

1. **`vol_regime` as a volatility *forecaster* for *risk-scaling*** — pointed at forward *realized vol*, not return. Vol clusters and is forecastable (Moreira–Muir 2017 vol-managed portfolios), so this is the detector's natural home and its single most-likely-to-pay-off use. **Never run.**
2. Whether either detector **beats a cheap baseline built from its own inputs** (the vol HMM literally consumes a 25-bar realized-vol → vol-of-vol pair; if it can't out-forecast a 25-day rolling RV, the ML machinery earns nothing).
3. Price-regime directional usefulness on **BTC/EURUSD** (R0 was gold-only) — low prior, included to formally close it on all three.

**Honest prior:** price-regime is *probably* dead (cont-49 + R0 both lean that way); the vol-regime-for-risk-scaling branch is the one with a real chance. The design spends its energy there.

---

## 1. The two detectors, their natural use, and the baseline each must beat

GAHMM = two independent 3-state HMMs (`SAFFS/core/hmm/ensemble.py`), exposed as **soft posteriors** by the Path-A provider (`SAFFS/exports/finrl_feature_provider.py`) and persisted to `results/prism_research/prism_features_<asset>_daily.parquet` by `precompute_prism_pathA.py`.

| Detector | Fields (soft, sum→1) + hard label | Actual inputs (SAFFS) | Natural use | **Cheap baseline it must beat** |
|---|---|---|---|---|
| **Vol regime** | `vol_low/normal/high_prob`; `vol_regime ∈ {0,1,2}` | 25-bar realized vol + vol-of-vol | **position sizing / risk-scaling** | **25-bar trailing realized vol** (rolling RMS of log-returns) |
| **Price regime** | `price_bear/neutral/bull_prob`; `price_regime ∈ {0,1,2}` | logret, RSI-14, Garman-Klass vol, realized skew | directional gate | **20-bar momentum sign** |

Continuous scores used in the eval:
- `vol_score(t) = vol_high_prob(t) − vol_low_prob(t)` (higher ⇒ expects higher vol; scale-free)
- `price_score(t) = price_bull_prob(t) − price_bear_prob(t)` (higher ⇒ expects up; scale-free)

---

## 2. Causality / leak surface — Stage 0 gate (non-negotiable)

Two independent code-reads of the Path-A GAHMM **disagreed**: one read the per-refit `window_df = gahmm_df.iloc[:bar_idx+1]` as a "full-window" leak; the other cited the byte-identical truncated-vs-full tripwire. Per project doctrine (*trust the tripwire, not the code-read*; LEAK-2 / "distrust degraded tool output"):

- **Stage 0 = run `pytest tests/prism_research/test_pathA_walk_forward.py`** and confirm the GAHMM regime columns are byte-identical whether the provider sees the full frame or a frame truncated at `t`. No regime byte is trusted until this is green. (The refit on `iloc[:bar_idx+1]` is an **expanding causal** window — past+present only — which is *why* the tripwire passes; Path-B REST is the leaky `[now−730d, now]` refit and is **not** used here.)
- **New tripwire** (`tests/prism_research/test_regime_eval_causality.py`): asserts the eval's own forward-target and trailing-signal builders are causal — the forward realized-vol target at `t` is invariant to any change in `close_{≤t}` and the trailing-vol signal at `t` is invariant to any change in `close_{>t}`, with a negative control that catches an injected leak.

All features come from the in-process Path-A parquets; **Path-B REST is never touched.**

---

## 3. Tests

All metrics reuse the **exact** functions the Chronos eval used (`_rank_ic`, `_shuffle_null_ic`, `_trade_sim` from `prism_predictive_eval.py`) — identical bar, no bespoke scoring. Assets: **BTC, gold, EURUSD**, daily. Horizons `k ∈ {1, 5}` days. One-way cost from the gates file (default 5 bps).

### Stage 1 — How GOOD (validity; cheap, kills early)

**A. Vol-regime → forward realized vol (the key new test).** Target `RV_k(t) = sqrt(Σ_{i=1..k} r_{t+i}²)` (k=1 ⇒ `|r_{t+1}|`), strictly forward.
- `IC(vol_score, RV_k)` vs shuffle-null95.
- Head-to-head baseline `IC(TRV_25, RV_k)` where `TRV_25(t) = sqrt(mean(r_{t-24..t}²))` (causal). **Marginal value = `IC_gahmm − IC_baseline`.** The HMM must *beat* the rolling RMS, not tie it.
- Kruskal–Wallis of `RV_k` across the 3 hard `vol_regime` states + monotonicity check (mean RV ordered low < normal < high?).

**B. Price-regime → forward return (confirmatory; cont-49 ≈ dead).** `IC(price_score, fwd_ret)` vs null; baseline `IC(mom_20, fwd_ret)`; bull-gated long net PF. Expected inside null — re-confirmed on all three, clean Path-A.

**C. Label sanity (degeneracy guard).** Per hard label: state fractions and mean dwell (run-length). A near-constant label (R0 found gold price-regime was bull-dominated 13/102/198) has a meaningless IC *by construction* — flagged, not silently scored. Transition matrix reported.

### Stage 2 — How USEFUL (tradable; only for Stage-1 survivors)

**Vol-regime risk-scaling (PRIMARY success bar).** Deliberately dumb base position = **constant long (+1)** so we test the *scaler*, not a signal. Pre-registered, monotone regime→size map (de-risk in high vol): `{low: 1.5, normal: 1.0, high: 0.5}`. Three position streams, scored net-of-cost by `_trade_sim`:
1. **unscaled** (pos ≡ 1.0)
2. **GAHMM-scaled** (`size_map[vol_regime_t]`)
3. **baseline-scaled** (bucket `TRV_25` into terciles via **train-frozen** breakpoints, same size map)

Metrics: net **Sharpe** (primary — scale-free, so leverage-neutral), **MaxDD**, total return.
- Uplift vs unscaled = `Sharpe(GAHMM) − Sharpe(unscaled)`.
- **Marginal value vs baseline** = `Sharpe(GAHMM) − Sharpe(baseline)` (≤ 0 ⇒ HMM adds nothing over rolling-RV vol-targeting).

**Price-regime gate (secondary).** Long-only when bull vs ungated vs 20d-momentum gate; net PF. Low prior; documents completeness.

### OOS protocol

Mirror cont-51: forward targets computed on the **full** series first (so the last in-window bar keeps its realized forward value), then restrict. Train = bars `≤ train_end` (default **2021-11-30**); test = bars after. Baseline tercile breakpoints **fit on train, frozen, applied to test**. The regime→size map is a fixed pre-registered constant (no fitting). Every metric reported **full-sample AND OOS-test**; the verdict keys on **OOS**.

---

## 4. Pre-registered GO / NO-GO (thresholds in `configs/prism_regime_eval.gates.yaml`, never hardcoded)

- **VOL-REGIME USEFUL** iff, on **OOS**, for **≥ 2 of 3** assets: `|IC(vol_score, RV)| > null95` **AND** `IC_gahmm − IC_baseline > 0` **AND** (`Sharpe_uplift ≥ +0.10` **OR** `DD_reduction ≥ 15%`) net of cost **AND** `Sharpe(GAHMM) − Sharpe(baseline) > 0`.
- **PRICE-REGIME USEFUL** iff, on **OOS**, for **≥ 2 of 3**: `|IC| > null95` **AND** gated net PF > **1.1** **AND** beats the 20d-momentum baseline.
- **DEGENERACY VOID** for any asset whose hard label fails the sanity guard (state fraction < 5% or mean dwell < 2 bars) — its IC is reported but excluded from the "useful" tally.
- Otherwise → **"PRISM regime detectors falsified for daily use; document, do not deploy."**

**Caveat stated up front:** Sharpe is the primary risk-scaling metric precisely because it is scale-free — a vol scaler that merely de-levers cuts both vol and return, so DD alone would flatter it. A NO-GO under Sharpe is conservative. The `IC_gahmm − IC_baseline` and `Sharpe(GAHMM) − Sharpe(baseline)` deltas are the decisive "does the ML add anything" terms; the unconditional IC/Sharpe could pass while the marginal-over-baseline fails — in which case the *detector is real but redundant*, which is reported as a distinct outcome from "useful".

---

## 5. Outputs

`results/prism_research/regime_eval/`: `verdict.json` (per-asset stats + GO/NO-GO, full & OOS), `regime_eval_summary.csv`, `regime_riskscale_summary.csv`.
Scripts: `scripts/prism_research/prism_regime_eval.py`. Tripwire: `tests/prism_research/test_regime_eval_causality.py`.
Gates: `configs/prism_regime_eval.gates.yaml`.

---

## 6. VERDICT (run 2026-06-18) — **FALSIFIED** (0/3 vol, 0/3 price), but with a sharp nuance

Stage-0 gate green: `test_pathA_walk_forward.py` ran against the live SAFFS provider and **passed** — GAHMM regime columns are byte-identical truncated-vs-full at runtime (no leak). New target/signal tripwires (8/8) pass. Full stats: `results/prism_research/regime_eval/verdict.json`.

**Price regime — DEAD (confirms cont-49 + extends R0 beyond gold).** Directional IC *inside* the shuffle-null on all three (BTC −0.019<0.044, gold −0.013<0.057, EURUSD −0.008<0.055); bull-gated net PF 0.95 / 1.13 / 0.94. Gold scrapes PF>1.1 but its IC is noise, so it fails. **0/3.** The "regime predicts direction" thesis is now closed on BTC/EURUSD too, not just gold.

**Vol regime — REAL but REDUNDANT (the decisive finding).**
- *It is a genuine vol detector*: `IC(vol_score, fwd realized vol)` beats its shuffle-null on **all 3 assets × both horizons** (k1: 0.088/0.175/0.188 vs null ~0.05; k5 stronger), Kruskal p ≤ 6e-3 everywhere, mean fwd-vol monotone low<normal<high on gold/EURUSD. Vol clusters and the HMM sees it.
- *But it is strictly dominated by a 25-day rolling realized vol*: **ΔIC < 0 on all 3 assets × both horizons** (k1 −0.089/−0.034/−0.040; k5 −0.182/−0.067/−0.065). The HMM's own raw input, left continuous, forecasts vol **better** than the 3-state posterior. Discretization + Student-t/stickiness **lose** information — the ML earns nothing.
- *As a risk-scaler it is a non-robust near-miss*: OOS Sharpe uplift vs unscaled = **BTC −0.01, gold +0.28, EURUSD −0.04** — helps only gold. It beats the (tercile-discretized) rolling-RV *sizer* on all 3 (vs-base +0.04/+0.36/+0.33), but that edge exists only because the baseline sizer was handicapped by the same discretization; against a *continuous* rolling-RV vol-target the gold/EURUSD wins likely evaporate. The naive 1.5×-in-low-vol leg **backfires on BTC** (OOS max-DD −78% → −87%): crypto crashes ignite from calm, so levering up in low vol *adds* tail risk.

**Pre-registered gate → 0/3 vol pass.** The conjunctive bar (beat null **AND** beat the rolling-RV forecast **AND** improve risk-adjusted return **AND** beat the rolling-RV scaler) fails on the ΔIC<0 term for every asset. **Disclosed knife-edge:** dropping the ΔIC requirement and scoring risk-scaling alone, gold (uplift +0.28) and EURUSD (DD-reduction +0.18) would pass 2/3. The verdict therefore hinges on the pre-registered, well-motivated requirement that the expensive GAHMM *beat a trivial rolling std* — and it decisively does not.

**Consequence.** This is the **third PRISM head to fall** to the same pattern (Chronos forecast cont-49, Chronos confidence-tail cont-51, GAHMM regimes here): each is real-but-redundant or absent, matched/beaten by a trivial baseline. The vol regime carries genuine information but **nothing a 25-day rolling vol doesn't already give you, cheaper**; do not wire GAHMM regimes as a feature/sizer at daily granularity. If vol-managed sizing is wanted, use the **continuous rolling realized vol** directly (and not on BTC, where inverse-vol sizing backfires). Revival of GAHMM regimes would need a granularity (intraday) or use (regime-conditioned RL obs) not tested here, and clears the Tier-2 stakes gate. The PRISM falsification (S413+, `prism.enabled: false`) **stands and strengthens**.

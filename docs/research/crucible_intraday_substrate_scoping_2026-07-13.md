# Scoping — Crucible Intraday (FinMind) Substrate

**Date:** 2026-07-13 · **Session:** S553-cont-129 · **Status:** SCOPE / pre-Architect · **Author decision required**

> This is a **scoping** memo (feasibility + staged plan + decision gate), not an architecture. Per the skill
> chain (Research → GO → Architect → implement), a full `.claude/skills/architect` design doc follows only if
> the Stage-0 gate below passes.

---

## 1. Why this exists

The LLM proposer is live and works (cont-129: 3 forced Taiwan ticks, 37 mechanism-diverse overlays, blindness
moat intact). But **every daily substrate is statistically power-bound**: the E1/E2 calibration
(`project_crucible_calibration_e1e2_s553`) shows the funnel needs realized marginal **ΔSR ≈ 1.40** to detect
anything on the deepest daily panel (T=4044, holdout=1011), versus a realistic true alpha of **0.3–0.5**. The
cont-129 run confirmed this a third time live: pushing the best raw OOS ΔSR from 0.53 → 0.66 changed nothing —
all candidates deflate to **DSR 0.000 / HLZ-t ≤ 1.67**, 0 PROMISING.

The calibration named exactly two levers: **higher-FREQUENCY data**, or a deliberately less-deflated gate.
Extending calendar history was already closed as a dead end (would need ~130 years). This memo scopes the
**higher-frequency** lever — the only one that makes the LLM proposer's output *testable at power*.

## 2. The data is available (FinMind Sponsor tier, verified cont-128)

`reference_finmind_datasets_capability` — verified live on our token:

| Dataset | Content | Depth | Bars/day |
|---|---|---|---|
| `TaiwanVariousIndicators5Seconds` | 5-sec TAIEX cash index | **2019-06 →** | ~3,241 |
| `TaiwanStockKBar` | minute bars (ETFs + stocks) | 2019+ | ~266 (`0050`) |
| `TaiwanStockPriceTick` | tick-by-tick deals, µs stamps | 2019+ | ~41k (`0050`) |
| `TaiwanFuturesTick` / `TaiwanStockEvery5SecondsIndex` | futures ticks / 5-sec sector indices | 2019+ | — |

Fetchers already exist: `scripts/data/fetch_taiwan_intraday_finmind.py`. **No NAV/iNAV** (that stays data-blocked —
irrelevant here). Raw 5-sec TAIEX 2019→now ≈ **~5.1M bars**.

## 3. Power: solved on paper, NOT on the daily calibration

Naive application of the calibration's own law `mde(h) = 1.40·√(1011/h)`:

- To clear the 0.50 ceiling: holdout ≥ 1011·(1.40/0.50)² ≈ **7,926 bars** (T ≈ 31,700).
- 5-sec data clears that in **~2.5 trading days**; the full 2019+ panel's holdout (~1.27M bars at frac 0.25) is
  ~160× over that threshold (naive implied MDE ≈ **0.04**).

**Do not believe the 0.04.** The daily law assumes ~independent bars. 5-sec bars are heavily autocorrelated
(microstructure), so the *effective* sample size `N_eff = N / (1 + 2Σρ_k)` is orders of magnitude smaller than
the raw count, and the Sharpe-estimator variance is inflated. **The daily E1/E2 sweep + the 0.50 ΔSR ceiling
cannot be transplanted to intraday frequency** — both are daily-calibrated artifacts. Whether intraday genuinely
buys power is precisely the unknown the Stage-0 probe must measure, not assume.

## 4. Build surface (if GO)

| # | Component | Notes / risk |
|---|---|---|
| 1 | **Intraday data pull + clean** | 5-sec TAIEX + minute/tick ETF for the universe, 2019→now, through `scripts/clean_ohlcv.py` (DATA-CLEAN invariant, `.bak`). Fetcher exists; bounded but multi-hour under FinMind rate limits. |
| 2 | **Intraday panel loader** | New `taiwan_intraday_panel_loader.py` (analog of `taiwan_panel_loader.py`). Must handle the intraday time axis + **session boundaries** — LEAK-1 EMA-Z reset now fires at every session open, not just split boundaries; overnight gaps must not leak across days. |
| 3 | **Substrate registration** | Add a `taiwan_intraday` branch to `_build_substrate` / `prepare()` (`orchestrator/substrate.py`), byte-shape-parallel to the `taiwan` branch. Daily T86/TAIFEX alt-data slots forward-fill onto intraday bars with a **strict PIT lag** (an EOD-published flow figure is only available from the *next* session open — not intrabar). |
| 4 | **Funnel-gate re-calibration** ⚠️ | **The critical de-risking step.** Re-run `scripts/research/crucible_calibration.py` on *intraday-realistic* synthetic substrates (with microstructure autocorrelation) to produce a new `calibration_mde_sweep.json` and an intraday `plausible_delta_sr_max`. Without this the power guard's MDE stamp is meaningless intraday. `gates_hash` stays frozen (`519158fa1450`) — a new `taiwan_intraday_signal_eval.gates.yaml` + `crucible_power.gates.yaml` variant is a MINOR version bump per CRU-1, must not touch existing verdicts. |
| 5 | **Intraday cost model** | Taiwan: 0.1425% commission ×2 + transaction tax (0.3% stock / 0.1% ETF, sells) + spread. The daily base-sleeve 10-bps convention does NOT apply. Every intraday candidate must be netted here before any GO. |
| 6 | **LEAK-2 intraday tripwires** | Intraday amplifies look-ahead: partial in-progress bar, resample `label`/`closed` conventions, overnight carry, multi-scale coarse-bar mapping. Each needs a **negative** test that fails on reintroduced leakage (the X2 lesson). |

## 5. Risks — high prior of "real ≠ tradeable"

The entire Taiwan sweep points one way at intraday frequency:
- **Cost-gating dominates.** ETF pairs-spread (cont-128): gross SR +1.4–2.1 but net +0.24 < 0.50 = MM-side only.
  Futures-basis (cont-128): gross +1.37 but the tradeable leg was +0.26, edge lived in the untradeable stale
  cash leg. Gamma/TX intraday-momentum (cont-92): cost-killed at 1 bp.
- **Microstructure reversal.** The Taiwan intraday-RL canary (`project_taiwan_intraday_rl_canary_nogo_s553`)
  found a bid-ask *reversal* — intraday "edges" that are bounce off the spread, not alpha.
- So a statistically-detectable intraday signal is the *start*, not the finish: it must survive component 5's
  cost model and a leg/liquidity decomposition before it means anything.

## 6. Recommendation — stage it to fail cheap

**Do not build the full substrate first.** Run a pre-registered **Stage-0 feasibility gate** (days, ~CPU-only,
no Crucible grammar change), analogous to the arb-sweep Stage-0 probes:

1. **Pull a limited slice** — 12 months of 5-sec TAIEX + 2–3 liquid ETFs (`0050`, `0056`).
2. **Measure the real intraday MDE** — re-run the calibration harness on an intraday-autocorrelation synthetic
   substrate. *This is the make-or-break number:* does intraday frequency actually push MDE below a
   sensibly-redefined ceiling, or does `N_eff` collapse eat the gain?
3. **Net one known mechanism vs realistic intraday costs** — e.g. the cont-129 `gold_etf_dealer_net`
   defensive-rotation signal, or a plain TAIEX intraday reversion — through component-5 costs + a leg/liquidity
   decomposition.

**GO to the full Architect build** only if BOTH: (a) intraday MDE clears the redefined ceiling with realistic
autocorrelation, AND (b) ≥1 mechanism survives costs gross. Else **NO-GO, cheaply** — and the standing
conclusion becomes "daily is power-bound and intraday is cost/microstructure-bound," which would retire the
Crucible-Taiwan discovery thread rather than expand it.

## 7. Decision

- [x] **GO Stage-0** — pre-register + run the feasibility probe above (operator decision, 2026-07-13, cont-129).
      Pre-registration: `docs/research/crucible_intraday_stage0_preregistration_2026-07-13.md`.
- [ ] **Defer** — park until the daily TAIFEX accumulation store matures (unblocks the un-scoreable OI overlays)
      or a less-deflated-gate experiment is preferred instead.
- [ ] **NO-GO** — accept daily power-bound as the terminal Crucible-Taiwan finding; redirect effort.

**Related:** `project_crucible_calibration_e1e2_s553` (the power bottleneck), `project_crucible_llm_proposer_via_cli_s553`
(cont-129 run), `reference_finmind_datasets_capability` (data), `project_futures_basis_arb_planned_s553` /
`project_taiwan_intraday_rl_canary_nogo_s553` (the cost/microstructure priors).

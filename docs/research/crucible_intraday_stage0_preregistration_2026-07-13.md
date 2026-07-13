# PRE-REGISTRATION — Crucible Intraday Substrate, Stage-0 Feasibility Gate

**Date:** 2026-07-13 · **Session:** S553-cont-129 · **Seed:** 20260713 · **Status:** FROZEN before any result
**Parent scope:** `docs/research/crucible_intraday_substrate_scoping_2026-07-13.md` (GO Stage-0, operator, 2026-07-13)

> Falsification-first. All gates + tripwires below are frozen BEFORE running. No verdict is read until
> the tripwire triad for that part is fully green. Same discipline as the cont-128 arb-sweep probes
> (`etf_pairs_arb_preregistration`, `futures_basis_arb_preregistration`).

---

## 0. Question

Daily-Taiwan Crucible discovery is **power-bound** (E1/E2 calibration; cont-129 confirmed 3× live —
best raw ΔSR 0.66 → DSR 0.000). Stage-0 asks the two cheapest questions that decide whether an
**intraday** substrate is worth building:

- **Part A (POWER):** After the microstructure **autocorrelation haircut**, does an *achievable* intraday
  panel actually detect a realistic marginal edge — i.e. does the implied MDE fall below ceiling at
  realistic frequency, or does effective-N collapse eat the raw-count gain?
- **Part B (TRADEABILITY):** Does ≥1 concrete intraday mechanism show real gross structure that survives
  a realistic Taiwan intraday cost model in a *liquid* leg (not the futures-basis "stale leg" trap)?

**Stage-0 GO ⟺ A-PASS AND B-PASS.** Either failing → NO-GO, cheaply, and daily-power-bound + intraday-
cost/microstructure-bound becomes the terminal Crucible-Taiwan finding.

---

## 1. Part A — intraday power (make-or-break)

Bar arithmetic (frozen): Taiwan cash session 09:00–13:30 = 4.5 h ⇒ **3,240 five-sec bars/trading day**;
**~794k bars/yr**; FinMind 5-sec depth 2019-06→now ≈ **~4.86M bars (~6 yr)**. Daily MDE anchor (E1/E2):
1.40 at holdout N_eff 1011 ⇒ MDE constant **C = 1.40·√1011 ≈ 44.5** (MDE = C/√N_eff_holdout).

### A0. IID upper bound (analytical, no run needed)
The harness generates **IID** synthetic returns and already validated `mde_falls_with_t` ≈ `1/√T_holdout`
on the daily grid. The IID intraday MDE therefore crosses 0.50 at T ≈ 4044·(1.40/0.50)² ≈ **31,705 bars
(~9.8 trading days)**. OPTIMISTIC (independent bars); **not** a gate — only says "raw count isn't the
constraint."

### A1. Autocorrelation-haircut recalibration (THE gate)
Extend the harness with a **pre-registered** persistence knob on the synthetic strategy PnL (NOT the
funnel — CRU-1 `gates_hash 519158fa1450` frozen; touches only synthetic test-data generators + a new
`configs/crucible_calibration_intraday.gates.yaml`). Model:

- The planted base-sleeve return series carries **AR(1) persistence** `φ = 1 − 1/H` for holding period
  `H` bars (H=1 ⇒ φ=0 ⇒ the current IID harness exactly — backward-compat tripwire TA-1). The Sharpe-
  estimator variance inflates by **VIF = (1+φ)/(1−φ) = 2H−1**, so **N_eff = N/(2H−1)** and
  **MDE ∝ √((2H−1)/N)** — Math-verified.
- Report MDE over a `(T_raw, H)` grid, fit the law, extrapolate to the full ~4.86M panel.
- **Frozen grids:**
  - Run grid `T_raw ∈ {50_000, 100_000, 200_000, 400_000}` (compute-tractable); full 2019+ panel
    (~4.86M) reported by extrapolation of the fitted `√((2H−1)/T)` law.
  - `H ∈ {1, 12, 60, 300}` bars = {IID check, 1 min, 5 min, 25 min} holding.
  - `holdout_frac = 0.25`; `power_target = 0.80`; beta grid extended DOWN to
    `{0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.030}` to bracket the (lower) transition at large T.

**Analytic frontier to be confirmed** (min panel to reach MDE ≤ 0.50 at each H; the numerical 5-condition
gate is the real test):

| Holding H | Min panel for MDE ≤ 0.50 | Feasible (≤4.86M)? |
|---|---|---|
| 1 min (H=12) | ~729k bars (~0.9 yr) | ✅ comfortably |
| 5 min (H=60) | ~3.77M bars (~4.75 yr) | ✅ only with ~full panel |
| 25 min (H=300) | ~19M bars (~24 yr) | ❌ infeasible |

- **A1-PASS ⟺** the numerically-fitted frontier confirms MDE ≤ 0.50 is reachable **within the available
  ~4.86M-bar (2019+) panel at a tradeable holding period H ≤ 60 (≤ 5 min)**. Deliverable = the
  `(min-panel-years, H)` frontier itself — it tells Part B which holding-period regime to even test.
  (H=300 is a reported stress column, expected FAIL, not gated.)

**A1 TRIPWIRES (all must pass before reading the MDE):**
- **TA-1 IID reduction:** at H=1 the extended harness reproduces the shipped daily MDE curve to within MC
  noise (the AR knob is a clean generalization, not a new artifact).
- **TA-2 monotone-in-H:** at fixed T_raw, MDE non-decreasing in H (more persistence ⇒ fewer effective
  samples ⇒ harder).
- **TA-3 N_eff law:** fitted MDE(T_raw, H) tracks `√((2H−1)/T_raw)` within slack — the haircut is the AR
  variance inflation, not a coincidence.

### Part A method note
The AR extension is a **synthetic-substrate** change only. Skill chain on the code change:
**Math** (verify `φ=1−1/H`, VIF `(1+φ)/(1−φ)=2H−1`, `N_eff=N/(2H−1)`) → **Audit** (harness code + config +
backward-compat TA-1 test). No funnel gate byte changes; `gates_hash 519158fa1450` untouched.

---

## 2. Part B — cost-netted mechanism (tradeability screen)

### B0. Data (limited slice)
FinMind Sponsor tier (`reference_finmind_datasets_capability`), fetched via
`scripts/data/fetch_taiwan_intraday_finmind.py`, cleaned through `scripts/clean_ohlcv.py` (DATA-CLEAN,
`.bak`):
- **12 months** (frozen: 2024-07-01 → 2025-06-30, fully out of the current daily panel's tail) of:
  `TaiwanVariousIndicators5Seconds` (5-sec TAIEX) + `TaiwanStockKBar` minute bars for **`0050`, `0056`**.

### B1. Mechanism (ONE, pre-registered)
Primary: **intraday TAIEX mean-reversion** on 5-sec bars — z-reversion of TAIEX log-price vs a trailing
intraday EMA, `W ∈ {60, 300, 900}` bars (5 / 25 / 75 min), positions reset at session open (no overnight
carry). Executed on `0050` (the liquid index-tracking ETF) as the tradeable leg.
Secondary (if primary degenerate): the cont-129 **`gold_etf_dealer_net`** daily-flow → next-session
intraday-timing overlay.

### B2. Cost model (frozen, Taiwan intraday)
Per round-trip on an ETF day-trade: brokerage **0.06% × 2** (electronic discounted) + securities
transaction tax **0.15%** (ETF day-trade halved rate) + spread **1 tick (~0.05%)** ≈ **≈ 32 bps
round-trip**. Futures-leg variant (if used): ~1 bp/side + 0.002% tax. Numbers pre-registered; verified
against a live broker schedule before any GO is acted on.

### B3. Gates (frozen)
- **G1 gross structure real:** gross Sharpe ≥ 0.50 AND tripwire-confirmed genuine (not spread bounce).
- **G2 net-of-cost:** net Sharpe (after B2) **> 0** at the frozen cost; reported at 2× cost for robustness.
- **G3 liquid-leg:** per-leg / liquidity decomposition (futures-basis lesson) shows the edge lives in the
  **tradeable** `0050` leg, not an untradeable/stale index leg.
- **B-PASS ⟺ G1 AND G2 AND G3.**

**B TRIPWIRES (all green before reading gates):**
- **TB-1 causality-signature:** same-bar execution variant differs from the t→t+1 signal as mechanically
  expected (the cont-128 rewrite — reversion signals go mechanically negative same-bar).
- **TB-2 permutation-null:** break the position↔return pairing (NOT order-shuffle — the cont-128 fixed
  bug); perm-p < 0.01.
- **TB-3 cost-monotonicity:** net Sharpe strictly decreasing in assumed cost (no sign flips from a coding
  error).
- **TB-4 session-boundary leak:** a negative test that fails if any feature/position at bar `t` uses a
  bar from a later session (LEAK-2 intraday: no overnight carry, no partial-bar look-ahead).

---

## 3. Decision

| Outcome | Meaning | Next |
|---|---|---|
| **A-PASS ∧ B-PASS** | Intraday buys power AND ≥1 mechanism is tradeable | → full Architect build (scoping §4) |
| **A-FAIL** | Autocorrelation haircut kills the power gain | NO-GO: daily power-bound + intraday N_eff-bound; retire thread |
| **A-PASS ∧ B-FAIL** | Power exists but mechanisms are cost/microstructure-bound | NO-GO for now; revisit only with an MM-side execution or a genuinely tradeable non-stale signal |

**Execution order:** Part A first (cheap, CPU-only, decisive, no data pull) — if A-FAIL, stop before the
FinMind pull. Part B only if A-PASS.

**Related:** `project_crucible_calibration_e1e2_s553`, `project_crucible_llm_proposer_via_cli_s553` (cont-129),
`reference_finmind_datasets_capability`, `project_futures_basis_arb_planned_s553` (leg-decomposition lesson),
`project_taiwan_intraday_rl_canary_nogo_s553` (bid-ask reversal prior).

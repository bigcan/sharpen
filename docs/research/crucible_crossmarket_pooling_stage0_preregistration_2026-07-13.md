# PRE-REGISTRATION — Crucible Cross-Market Pooling, Stage-0 Feasibility Gate

**Date:** 2026-07-13 · **Session:** S553-cont-130 · **Seed:** 20260713 · **Status:** FROZEN before any result
**Parent:** `project_crucible_calibration_e1e2_s553` (the power bottleneck), `project_crucible_llm_proposer_via_cli_s553`
(the `gold_etf_dealer_net` candidate). Same discipline as the cont-128 arb-sweep and cont-129 intraday Stage-0:
gates + tripwires frozen BEFORE running; no verdict read until the tripwire triad is green.

---

## 0. Question

The cont-129 review named **cross-market meta-analysis** as the one power lever the E1/E2 calibration never
considered: pool the *same* economic mechanism across independent markets to raise effective sample size
without more history (extend-history: DEAD ~130 yr) or more frequency (intraday: CLOSED, MDE 0.86 @ 5-min).
The candidate is the cont-129 best raw overlay, `twse_inst:gold_etf_dealer_net` (dealer net flow in the
Taiwan gold-futures ETF `00635U`; best raw OOS ΔSR 0.66 → DSR 0.000). Its US analog is directly available:
**CFTC COT gold** (`088691`, `comm_net` / `noncomm_net`) + a US gold price leg (`gc.f` / GLD).

**Stage-0 asks the single cheapest make-or-break question before any data pull:**

- **Part A (POWER):** Given the funnel's *measured* per-market MDE (1.403 at holdout 1011, the deepest real
  daily substrate — E1/E2 anchor), **how many independent markets `k` must a fixed/random-effects pool
  combine to drag the pooled MDE below the 0.50 ceiling — and do that many gold-positioning markets exist?**
- **Part B (REPLICATION):** *Only if A-PASS* — pull US COT gold + US gold price, run the identical simple
  transform through the real `combination_fitness` gate, and pool with the Taiwan estimate.

**Stage-0 GO ⟺ A-PASS AND B-PASS.** Either failing → NO-GO, cheaply. **Execution order: Part A first
(analytic + MC, CPU-only, no network). If A-FAIL, stop before the US COT / price pull.**

---

## 1. Part A — pooled-power feasibility (make-or-break)

### A0. The candidate is an operator overfit (recorded, not gated)
The cont-129 winner is `decay_linear³(stddev(stddev(delta(gold_etf_dealer_net, 60), 3), 30), 30)` with an
**empty economic rationale** — a triple-smoothed vol-of-vol of a 60-day flow change. The nonlinear operator
stack *is* the overfit; there is no clean "positioning → return" signal to transplant verbatim. So Part B (if
reached) tests the **economic invariant** with ONE pre-registered simple transform (frozen in §2), never the
nested stack. This is context; the gate is A1.

### A1. Pooling variance law (THE gate) — Math-verified inline
Model each market `i ∈ 1..k` as giving an estimator `ΔŜR_i` of a **common** mechanism effect, each with
per-market standard error `σ` (equal across markets by construction — same transform, comparable panel
length) and equicorrelated pairwise PnL correlation `ρ = corr(pnl_i, pnl_j)`. The equal-weight pooled
estimator `ΔŜR_pool = (1/k) Σ ΔŜR_i` has variance (standard equicorrelated-mean result):

```
Var(ΔŜR_pool) = σ² · (1 + (k−1)ρ) / k
```

Because MDE at fixed target power (0.80) is a constant multiple of the SE, **MDE_pool = m · √((1+(k−1)ρ)/k)**,
where **m = 1.403** is the funnel's per-market MDE anchor (E1/E2, holdout 1011). Inverting for the markets
needed to reach ceiling **c = 0.50**, with **f = (c/m)² = (0.50/1.403)² = 0.1270**:

```
k_needed(ρ) = (1 − ρ) / (f − ρ),   valid only when ρ < f
correlation floor:  MDE_pool → m·√ρ  as k→∞   ⇒  if ρ ≥ f = 0.127, pooling NEVER reaches 0.50
```

- **A1-PASS ⟺** `k_needed(ρ*) ≤ k_available` at the pre-registered realistic `ρ*` band, where
  `k_available` = the count of **independent markets with the same gold institutional-positioning
  mechanism and accessible free data**.
- **Frozen inputs:** `m = 1.403` (from `results/crucible_calibration/calibration_mde_sweep.json`,
  holdout 1011); `c = 0.50` (the E2 realistic-alpha ceiling, `project_crucible_calibration_e1e2_s553`);
  `ρ*` reported across the band **{0.0 (optimistic, independent), 0.05, 0.10, 0.127 (floor)}**.
- **Frozen `k_available` enumeration** (the same-gold-mechanism markets with accessible connectors):
  US CFTC COT gold (`088691`) = 1; Taiwan T86 `00635U` dealer net = 1. **k_available = 2** (base case);
  a generous upper bound admitting related metals/venues as "independent" = **4**. Both reported.
- **Optimistic sensitivity (recorded):** a single *pre-registered* per-market test drops multiplicity, so
  its per-market MDE `m_single < 1.403`. Report the same frontier at `m_single ∈ {1.10, 1.20}` to show
  whether ANY realistic `(k, ρ, m)` corner reaches ceiling. Not the gate (the gate uses the measured
  m=1.403); a decision aid so the NO-GO — if it comes — is airtight against the "but single-test is
  cheaper" objection.

**A1 TRIPWIRES (all green before reading `k_needed`):**
- **TC-1 closed-form ↔ Monte-Carlo:** simulate `k` equicorrelated unit-variance estimators (Cholesky of the
  equicorrelation matrix) over ≥200k draws; empirical `Var(mean)` matches `(1+(k−1)ρ)/k` within MC noise
  (<1%) across the `(k, ρ)` grid. Fails if the pooling law is mis-stated.
- **TC-2 anchor reproduction:** the harness's per-market MDE at holdout 1011, read through the shipped
  `interp_mde` on the real sweep JSON, reproduces **1.403** exactly (same anchor-validation as the intraday
  probe). Fails if `m` is not the funnel's own number.
- **TC-3 degenerate limits:** `k=1 ⇒ MDE_pool = m` (pooling one market changes nothing);
  `ρ=1 ⇒ MDE_pool = m` for all `k` (perfectly correlated markets add zero information). Both must hold
  exactly, else the formula is wrong at the boundaries.

### Part A method note
Analytic + MC only. No funnel gate byte changes; `gates_hash 519158fa1450` untouched. Tooling:
`scripts/research/crucible_crossmarket_power.py`, config `configs/crucible_calibration_crossmarket.gates.yaml`
(validates, changes no verdict — not part of the frozen funnel hash, per CRU-1).

---

## 2. Part B — live replication (only if A-PASS)

### B0. Data (limited slice)
- US gold price: Stooq `gc.f` (or GLD), 2015→now, via the existing `StooqConnector`.
- US gold positioning: CFTC COT `088691` `comm_net` + `noncomm_net`, via `CftcCotConnector` (business-day
  release lag already PIT-correct, CR-4).
- Taiwan: existing `00635U` dealer-net store (2015+) + Taiwan gold-ETF price already in the panel.

### B1. Mechanism (ONE, pre-registered simple transform — NOT the overfit stack)
`signal_i(t) = zscore_252( Δ_20 net_positioning_i(t) )` — the 252-day z-score of the 20-day change in
institutional net positioning, sign-mapped to time the local gold sleeve return `r_i(t→t+1)`. Identical in
both markets (Taiwan `dealer_net`; US `comm_net`, with `noncomm_net` as the frozen robustness alternate).
No nested operators; one transform, pre-registered.

### B2. Gates (frozen)
- **G1 per-market:** each market's ΔSR estimated with SE on truly held-out data (single pre-registered test,
  plain t-hurdle — NOT the 37-way miner multiplicity).
- **G2 pooled:** fixed-effect (and DerSimonian–Laird random-effect) pool of the `k` markets clears the
  Part-A-derived detectability bar AND pooled t ≥ single-test 80%-power hurdle.
- **G3 heterogeneity honesty:** report Cochran's Q / I² — with `k=2–3` the between-market variance `τ²` is
  barely estimable; a pooled "hit" driven by one market is disclosed, not hidden.
- **B-PASS ⟺ G1 AND G2 AND G3.**

**B TRIPWIRES:** measured cross-market PnL `ρ̂` must fall inside the Part-A `ρ*` band used to clear A1 (else
A1's power claim was based on the wrong `ρ`); permutation-null on the pooled statistic (break the
signal↔return pairing) `p < 0.01`; COT/T86 PIT-lag negative test (no positioning value read before its
release stamp — LEAK-2).

---

## 3. Decision

| Outcome | Meaning | Next |
|---|---|---|
| **A-PASS ∧ B-PASS** | Enough independent markets AND the invariant replicates & pools over ceiling | Promote to a pre-registered forward-incubation sleeve (lockbox) |
| **A-FAIL** | Too few independent gold-positioning markets to close 1.40→0.50 by pooling | NO-GO: third power lever closed; only the not-recommended less-deflated gate remains → **TSMOM → paper** |
| **A-PASS ∧ B-FAIL** | Power exists but the invariant doesn't replicate | NO-GO: the cont-129 winner was market-specific noise (consistent with its empty rationale) |

**Execution order:** Part A first (analytic + MC, no network). If A-FAIL, stop before the US data pull.

**Related:** `project_crucible_calibration_e1e2_s553`, `project_crucible_llm_proposer_via_cli_s553`,
`project_commodity_carry_sleeve_nogo_s553` (the broaden-to-many-commodities escape hatch is a *different*,
already-NO-GO mechanism), `crucible_intraday_stage0_preregistration_2026-07-13.md` (the sibling lever).

# Crucible Independent-Audit Remediation — Tier B (fixed) + Tier C (deferred)

**Date:** 2026-07-15 · **Session:** S553-cont-132 · **Upstream:** the independent expert audit
`docs/research/crucible_independent_audit_report_2026-07-14.md` (verdict: *(C) compounded on (B); (A)
unclaimable; stop-mining SURVIVES*), whose Tier-A record-honesty corrections shipped in commit
`eb882160`.

The audit graded its recommendations in three tiers. This document records the disposition of the
remaining two.

| Tier | What | Disposition |
|------|------|-------------|
| **A** | Record-honesty corrections (ledger relabel, retract "0.66 OOS", surface TA-3, register Taiwan gate hash, flag inert killed-families) | ✅ DONE (`eb882160`) |
| **B** | Anti-conservative (false-positive-direction) defect fixes — F14 | ✅ FIXED this session (`crucible-v3.0`) |
| **C** | Structural gate-seal defects — F1 / F2 / F3 / F5 | ⛔ REGISTERED as known + **deliberately NOT patched**, in favor of TSMOM→paper |

Both tiers rest on the same standing decision (`project_randd_direction_memo_s553`,
`project_tailwind_wired_paper_s553`): the mass-mining discovery instrument is **retired**; effort goes
to the TSMOM→paper path. Tier B is *optionality insurance* — it hardens the eval infrastructure that a
future pre-registered probe (audit §5) or a salvaged component would reuse; Tier C is the rework the
audit explicitly recommends **against** doing.

---

## Tier B — the four anti-conservative fixes (F14)

All four are **false-positive-direction** defects: they made a hypothetical survivor look *better* than
reality. Every fix is **monotone-stricter**, so on the shipped funnel it can only make the already-
0-PROMISING record more-0. They change **scoring CODE + `FitnessConfig` defaults only** — no gates-YAML
byte — so the frozen funnel `gates_hash` (`519158fa1450`) and the Taiwan hash (`22a18172be1a`) are
**UNCHANGED** (CRU-1). Because they change the verdict *function* (stricter on real base books), the
system version is a **MAJOR** bump: `crucible-v2.9 → crucible-v3.0`.

### F14-1 + F14-2 — overlay tilt cost (`evolve._overlay_returns`)

Old: `cand[t] = m[t-1]·base_net[t] − cost_bps·|Δm[t]|`. Two defects, one root — the base book was
treated as a unit-gross cost-free multiplier target:

- **gross under-charge (~11×):** the overlay's own rescaling turnover `|Δm|` was charged at unit gross
  while the tilted book runs gross `G ≈ 11` (a vol-scaled directional book). A survivor's net Sharpe
  was overstated ~10× in cost terms.
- **short-tilt rebate:** since `base_net = b_gross − c_base`, for a short tilt (m<0)
  `m·base_net = m·b_gross + |m|·c_base` — the base book's embedded cost was *credited* (+2.0 bp/day
  measured) instead of paid.

New: `cand[t] = m[t-1]·b_gross[t] − |m[t-1]|·c_base[t] − cost_bps·|Δm[t]|·G[t]`. Charges the embedded
cost as always-paid `|m|·c_base` and the overlay turnover at the true gross `G`. Reduces **bit-for-bit**
to the old formula when `(b_gross=net, c_base=0, G=1)` — the exact decomposition of a synthetic /
planted / proxy return-stream sleeve, so the synthetic / calibration / reproduce path is unchanged.

**Full re-plumbing** (operator-chosen over a self-contained conservative bound): `base_sleeves.py`
exposes a per-sleeve `SleeveComponents` (gross / cost / gross-exposure) via `return_components=True`
(additive, default-off, `net` bit-identical); it threads
`PreparedSubstrate.base_components → run_hypothesis_loop → evolve → _overlay_returns`, and likewise
through `cohort_eval.evaluate_cohort` and `lockbox/incubation.forward_evidence`. On the unit-gross
fallback (`base_components=None`) the correction is an exact no-op.

### F14-3 — dsr_aug deflates against AR(1)-effective N (`fitness.combination_fitness`)

Old passed raw `n_obs = bclean.size` to `deflated_sharpe_ratio`; the `marginal_t` leg already used
`_ar1_effective_n`. Daily marks under a multi-day hold are positively autocorrelated, so raw N
understates the Sharpe's SE → over-states dsr → easier to pass. New passes
`n_obs = _ar1_effective_n(bclean)`, consistent with `marginal_t`. **Gate-monotone**: a passing
candidate (dsr≥0.90>0.5 ⇒ SR>SR*) gets a *lower* dsr under the haircut, so the pass-set can only
shrink.

### F14-4 — degenerate-vol candidate cull (`fitness.combination_fitness`)

The shared inverse-vol combiner weights a stream ∝ 1/σ, so a near-zero-vol candidate hijacks the
convex weights (aug book ≈ candidate), inflating the uplift-leg null tail (audit: null ΔSR up to 3.3).
New leg: cull a candidate whose realized per-period vol is below `degenerate_vol_frac` (default 0.10)
of the smallest base sleeve's vol. The threshold lives in `FitnessConfig` (a correctness floor, **not**
a tunable decision gate) so the frozen gates-YAML hash is untouched. Monotone-stricter (only removes
potential false-positives). The shared `envs/allocator_factory.dynamic_sleeve_alphas` was **not**
touched — the cull is crucible-side, so the live TSMOM/paper path that reuses the combiner is
unaffected.

### CRU-1 preservation — empirical evidence

The four fixes are stricter, but the decisive check is empirical, not the pointwise-monotone argument
(the overlay term is only *aggregate*-stricter: `G` can dip below 1 in high-vol bars). Re-querying every
recorded **real** ledger:

| real ledger | rows | max dsr | max marginal_t | passed dsr≥0.90 | passed t≥3.0 | PROMISING |
|-------------|------|---------|----------------|-----------------|--------------|-----------|
| `crucible_orchestrator/real` (cross_asset) | 18 | **0.674** | −1.341 | 0 | 0 | 0 |
| `taiwan_manual/real` | 52 | 0.049 | **2.116** | 0 | 0 | 0 |
| `taiwan_llm_diversity_probe/real` | 92 | 0.000 | 1.673 | 0 | 0 | 0 |
| `taiwan_v2/real` | 241 | 0.000 | 1.496 | 0 | 0 | 0 |

The **largest dsr ever recorded (0.674)** sits 0.226 below the 0.90 gate; the **largest marginal_t ever
(2.116)** sits 0.884 below the 3.0 gate; **no candidate ever cleared either leg** under the *looser*
pre-fix scoring. Stricter scoring only widens those gaps → 0-PROMISING is preserved with large margin.
CRU-1 holds.

### Deferred within Tier B — E1 realistic-null re-calibration (F14 item 5)

The audit's fifth F14 item — re-run the E1 false-positive calibration on a **realistic** null (fat
tails / vol clustering / common factor) because today's FPR protection partly rides on the mis-specified
legs — is **DEFERRED**, not done. It is a **prerequisite for any future discovery-grade use** of the
funnel (audit §5), not for the fixes themselves. The current E1 null (`crucible_calibration._planted_*`)
is IID Gaussian. If the pre-registered probe pathway (§5) is ever built, this re-run is a hard
gate before trusting E1-GREEN. Registered here so it is not silently skipped.

---

## Tier C — structural gate seals: REGISTERED, NOT PATCHED

The audit's proximate cause of the 0-PROMISING record — layer (C) — is a set of **structural gate
seals** that reject candidates independent of quality and sample size. The audit's own recommendation
(§5, §6.1) is **not** to patch them in place but to **retire the mass-mining contract** and, if
discovery ever continues, build a *parallel* pre-registered probe pathway. We follow that: the seals are
documented here as known and deliberately un-remediated, with in-code pointers so nothing silently
re-relies on them.

| ID | Seal | Why not patched |
|----|------|-----------------|
| **F1** | `marginal_t` is a substitution-residual mean-test — `marg = w_c·(r_c − b_base)` under the convex combiner, so a genuine variance-reducing diversifier gets `E[marg]<0` and `t→−∞` as N→∞. Sign-inverted; scale-dependent; contradicts its own docstring + the GP4-03 intent that created it. | Fixing it is a **verdict-function redesign** of the binding significance leg — exactly the mass-miner rework the audit says to abandon. **Do NOT execute roadmap NEXT-2** (promote `marginal_t` to the single binding leg) as written — it inherits F1. |
| **F2** | `dsr_aug` is a book-level bar on a ≈0-Sharpe TRAIN base book (Taiwan TSMOM train SR −0.007), so all 160 Taiwan candidates score dsr≤0.049 vs 0.90 — sealed at the train pre-filter, reflecting the base book's decade, not the candidate. | Un-sealing requires re-architecting the deflation target (candidate-relative, not book-level) — a funnel redesign, out of scope for a retired instrument. |
| **F3** | The certified pool-based **holdout gate has never executed** (0 in production, 0 in E1); every recorded metric is a train-split value. | This is a *record-honesty* fact (Tier-A relabelled the `delta_sr_oos` column). Making the holdout gate actually bind is part of the §5 probe redesign, not a patch. |
| **F5** | The overlay path collapses every alt-data formula to one broadcast scalar timing the whole base book — per-asset flow, cross-sectional flow, conditionals, events, seasonality, >120-bar horizons are architecturally unposable. "37 diverse overlays" ≈ 5–10 effective hypotheses. | An expressiveness fix (per-asset overlay routing + conditional operators) is the highest-leverage *platform* change **but** only worth it after the §5 pathway exists, and it re-opens E1 (audit §6 item 3). |

**In-code guardrails added this session** (so a future session cannot silently re-rely on a seal):

- `fitness.py::combination_fitness` — the `marginal_t` computation carries a Tier-C NOTE pointing at
  F1 and the NEXT-2 prohibition.
- A **tripwire test** (`tests/signals/test_tier_c_seals_registered.py`) pins F1's sign-inversion
  (a genuine diversifier gets negative `marginal_t`) and F2's train-seal (a ≈0-Sharpe base book bars
  dsr) as *known* behaviors — if a future edit "fixes" one, the test fails loudly, forcing a conscious
  version + gates decision rather than an accidental verdict-semantics change.

### If discovery ever resumes (audit §5, for the record)

Not more mass-mining ticks and not NEXT-2. Instead: low-N, pre-registered, mechanism-driven
single-hypothesis probes on the full sample, scored by one marginal-effect statistic on the ΔSR/CPCV
distribution at t≥2.33 with a **binding** LORD++ account, on a **parallel gates file** (CRU-1 MINOR, the
lockbox/cohort precedent). Prerequisites: the Tier-B F14 fixes (done) **and** the deferred E1
realistic-null re-run. Even then MDE80 ≈ 0.7–0.9 on these substrates — the power wall is ultimately a
**data** problem (`project_crucible_calibration_e1e2_s553`,
`project_small_operator_strategy_reframe_s553`), not a statistics fix.

---

## Verification summary

- **Math skill:** PASS WITH NOTES — all three formulas correct; overlay strictness is aggregate (not
  pointwise: `G` can dip <1), so CRU-1 rests on the empirical re-score above.
- **Frozen `gates_hash`:** `519158fa1450` (US) / `22a18172be1a` (Taiwan) UNCHANGED (`test_version.py`).
- **New tripwires:** `tests/signals/test_base_sleeves_components.py` (5),
  `tests/signals/test_tier_b_anticonservative_f14.py` (14), plus the Tier-C seal registry test.
- **Suite:** full `tests/signals` + `tests/crucible` green except two **pre-existing** failures
  (`test_reproduce_cohort`, `test_reproduce_p5`) that fail on clean HEAD — the power guard refuses the
  underpowered T=320 synthetic panel and the tests don't pass `--force-underpowered` (tracked
  separately; not introduced here).

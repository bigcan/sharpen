# Crucible Selection-Aware Monte-Carlo Null — the AUTHORITATIVE cohort gate

**Status:** DESIGN (no code changes). Author: session 553-cont-108 → -109.
**Doc 2 of 3** in the weak-signal-ensemble design set:
- **Doc 1** — [`crucible_weak_signal_ensemble_spec.md`](crucible_weak_signal_ensemble_spec.md): the cohort
  evaluator + the **cohort deflation statistic** `SR*_cohort` (novel statistic #1) + correlation control + combiner.
- **Doc 2 (this)** — the **selection-aware Monte-Carlo null** (novel statistic #2): the binding gate a cohort
  must clear to be recorded PROMISING.
- **Doc 3** — [`crucible_diverse_proposer_spec.md`](crucible_diverse_proposer_spec.md): the upstream blocker
  (the pool has no diversity to admit from). *Measure data breadth first.*

> **Why this is its own doc.** The analytic `SR*_cohort` (Doc 1) is a *cheap pre-filter* — an analytic floor
> that assumes selection-by-own-Sharpe + static equal-risk weights. The **exact** null under greedy
> `delta_sr_oos` admission + the dynamic combiner cannot be written in closed form; it must be estimated by
> this **full-pipeline Monte-Carlo**. This MC is the authoritative verdict; the formula is only the thing that
> keeps the MC off the ~99% of cohorts that never had a chance. Both were adversarially math-verified and both
> have their naive forms explicitly **blocked** — that verification is the substance of this doc, not an aside.

**Nothing here is implemented.** This is design + verification, ready for a build decision.

---

## 1. What null this tests

> **H₀:** the cohort book has no genuine edge — its Sharpe is entirely attributable to **selection**
> (choosing which `m` of `N` candidates to combine) plus **combination** (√m variance reduction), given the
> candidates' realized volatility, autocorrelation, and *cross-candidate* covariance.

Rejecting H₀ (small `p`) means the observed book Sharpe is larger than select-and-combine luck can manufacture
from zero-edge streams with the same second-moment structure. This is precisely the quantity the **naive
`n_trials=N` deflated-Sharpe gate failed to bound** (Doc 1, Part 1 audit block): DSR's `SR*` is the expected
*maximum of N single-trial Sharpes* — the null for "I searched N configs, I report the best ONE." A cohort is
the Sharpe of the *best select-AND-combine of the top-m of N* — a different random variable the max-of-one bar
cannot bound, and the gap **widens** as the pool grows (`SR*` grows ~√(2 ln N), the noise book grows ~√m).

This MC is the **panel-data analogue of White's (2000) Reality Check / Hansen's (2005) SPA**, adapted from
"best single track of N" to "best select-and-combine of m-of-N."

---

## 2. ⛔ MATH AUDIT (Fable-5 + Math skill, cont-108) — architecture GO, three BLOCKERs, all fixed

A dedicated MC-validation pass built a simplified-but-structurally-identical pipeline (harness `mc_null_verify.py`,
second math pass — **must be re-created and ported into the test suite at build time; the prior-session scratchpad
copy is not preserved**). **The architecture is correct** — demean → joint stationary block bootstrap → re-run the
full selection+combination → add-one one-sided p is the right selection-aware Reality Check, and it calibrates
(size 1.0–3.5% at α=5%) **once three defects are fixed.** Each defect was empirically demonstrated with a failing
number, so each has a permanent regression tripwire (§6).

### BLOCKER 1 — resample the BASE sleeves too, jointly

Holding `b_base` fixed while resampling only the candidates (the original draft) **destroys candidate↔base
covariance and time-alignment** → **size 8.3% > α** (anti-conservative, and it worsens as base loading β grows).

**Fix:** include the base sleeve columns in the *same joint block resample* (same block indices); demean **only**
the candidate columns (H₀ concerns candidates; the base keeps its real drift). `T_stat` inside a replicate uses
*that replicate's own* base book `b_base⁽ᵇ⁾`, rebuilt from its resampled base columns.

### BLOCKER 2 — the replicate time axis must be the ORIGINAL monotone `timestamps`, applied positionally

The combiner (`dynamic_sleeve_alphas`) is time-order-dependent: trailing vol, trailing corr, and month-end
rotation via `_monthly_held`. If a replicate carries its *source* (scrambled) timestamps, `_monthly_held` flags
month-ends at arbitrary pasted positions and produces **silent garbage rotation — no error raised** → invalid null.

**Fix:** row `k` of every replicate uses `timestamps[k]` — the original monotone axis — NOT the source timestamp
of the pasted block. The time axis (trailing windows, the month-end grid, warmup) is part of the *test procedure*
and is held fixed; only returns are randomized. Trailing windows then legitimately span block seams (standard
block-bootstrap behavior). The combiner is **re-run every replicate, never frozen** — freezing conditions on the
observed configuration (cousin of the frozen-weights anti-pattern; MC: **32% false-pass**).

### BLOCKER 3 — the MC statistic is ΔSR, NOT the marginal-stream Sharpe

*(This REVERSES the first audit's MEDIUM "gate the marginal stream" — for the MC gate specifically.)* With the
convex combiner (Σα = 1), `E[marginal] = Σ w_j(μ_j − μ_base)` — the marginal mean measures member drift *in
excess of the base's own drift*. So a cohort of genuine weak diversifiers with `μ_j ≤ μ_base` that raises book
Sharpe purely via **variance reduction** has marginal mean ≤ 0. At the spec's own power target (m = 8
independent, s = 0.05) the marginal-Sharpe MC passes only **6–9% of *true* cohorts** — the authoritative gate is
nearly blind to the very thesis it exists to detect.

**Fix:** `T_stat = SR_pp(combine(base ∪ cohort)) − SR_pp(b_base)` — **within-replicate ΔSR**. Power → **98%**,
size 3.5% (validated). Luck-de-risking is not a loophole: null replicates enjoy the same de-risk-by-noise
channel, so the ΔSR null quantile prices it in. Keep the **marginal-HLZ-t as a SEPARATE funnel gate** (a policy
against paying for pure de-risking), not as the MC's `T_stat`.

### Confirmed correct (no change)

- The **demean-then-resample** H₀ imposition coincides *exactly* with White's recentering (std/corr/vol/
  trailing-corr are location-invariant, so demeaning shifts only the mean — the right choice, not an approximation).
- **Re-running admission inside each replicate** is the correct Reality-Check / Romano–Wolf extension (no
  residual conditioning; conservative under the least-favorable null).
- The block length is **not** circular (admission/combining don't depend on ℓ).
- The **add-one one-sided p-value**, `≥` tie-handling, and SE arithmetic all check.
- The bootstrap carries dependence natively, so the DSR-vs-HLZ autocorrelation inconsistency (Doc 1, LOW) does
  **not** affect this gate.

### Lower-severity fixes (carried, below the fold)

- **(HIGH / cost)** the cost model is load-bearing — faithful `delta_sr_oos`-ranked admission needs a
  per-candidate combiner + CPCV *inside every replicate* → **O(B·N·(combiner+CPCV))**, hours not minutes. Either
  budget for it (nightly/on-survivor gate, never a per-genome inner loop) or **pre-register a cheaper ranking
  statistic used identically in observed and null pipelines** (§5).
- **(MEDIUM / ℓ)** pick ℓ by Politis–White (2004) auto-block-length on the *input columns* (median across
  columns), pinned; add an ℓ ∈ {5, 21, 63} sensitivity sweep.
- **(MEDIUM / pool)** pin the pool = *all* scored candidates incl. LOGGED (any upstream performance-correlated
  pre-filter is unmodeled multiplicity → anti-conservative); define `T_b := −∞` when a replicate admits
  `< min_cohort_size`.
- **(LOW)** frozen holdout weights = the last month-end α vector (no re-estimation); unit-pin the holdout floor
  (annualized vs per-period); pin a holdout-reuse policy (repeated reads accumulate their own multiplicity); add
  a combiner-weight-variance reproduction check.

---

## 3. Construction

### 3.1 Inputs (all precomputed once, reused across reps)

- `R` — the `(T × N)` panel of candidate **net return streams** = the pool = **ALL scored candidates incl.
  LOGGED** (any upstream performance-correlated pre-filter is unmodeled multiplicity → anti-conservative). Same
  streams the funnel computed via `_candidate_returns` / `_overlay_returns`.
- `B_base` — the base **sleeve columns** `(T × S)` (tsmom, rates_carry, …). Resampled *jointly with* the
  candidates (same block indices) — NOT held fixed (BLOCKER 1). Each replicate rebuilds its own
  `b_base⁽ᵇ⁾ = combine(B_base⁽ᵇ⁾)`.
- `admit(·)` — the greedy de-correlated admission (Doc 1, #2a): rank by in-sample `delta_sr_oos`, admit while
  `|corr|` to admitted ≤ `max_pairwise_corr`, cap `max_cohort_size`.
- `combine(·)` — the C1 combiner (Doc 1, #3): `dynamic_sleeve_alphas`, `redundancy_strength = λ_r`.
- `T_stat` — **within-replicate ΔSR** `SR_pp(combine(base ∪ cohort)) − SR_pp(b_base)` (BLOCKER 3).

### 3.2 Imposing H₀ + resampling (the crux — get this exactly right)

Four requirements; the scheme satisfies all four:

1. **Zero drift on candidates only (impose H₀).** Demean each CANDIDATE stream over the full sample:
   `R̃[:,j] = R[:,j] − mean(R[:,j])`. Coincides exactly with White's recentering. **Do NOT demean the base
   sleeve columns** — H₀ is about the *candidates*; the base keeps its real drift so `b_base` in each replicate
   is the true book (BLOCKER 1/3 interaction).
2. **Resample base + candidates JOINTLY.** Stack `[B_base ‖ R̃]` `(T × (S+N))` and resample **contiguous
   time-blocks of the whole stacked panel with the SAME block indices** — a stationary block bootstrap (Politis
   & Romano 1994, mean block length ℓ). Joint row-wise resampling keeps candidates *and the base* time-aligned,
   so panel covariance + candidate↔base covariance + each stream's within-block autocorrelation survive. **Do
   NOT** rotate/shift streams independently (destroys cross-correlation).
3. **The replicate carries the ORIGINAL monotone `timestamps` positionally** (BLOCKER 2). Row `k` of every
   replicate uses `timestamps[k]`. The time axis is held fixed; only returns are randomized. Set ℓ by
   Politis–White (2004) auto-block-length on the *input columns* (median across columns), pinned in gates;
   fallback ℓ = 21. (Tuning ℓ on the post-selection marginal stream is snooping-adjacent and targets the wrong
   object — the null's dependence is driven by the inputs.)
4. **Re-run selection AND the combiner INSIDE every replicate** (the single most important line). For each rep
   `b`: draw the joint bootstrap panel; rebuild `b_base⁽ᵇ⁾ = combine(B_base⁽ᵇ⁾)`; `cohort_b = admit(R̃⁽ᵇ⁾)`;
   `book_b = combine(base⁽ᵇ⁾ ∪ cohort_b)`; `T_b = SR_pp(book_b) − SR_pp(b_base⁽ᵇ⁾)`. Because admission AND
   weights re-run on the resampled noise, `T_b` **includes the selection benefit** — netting out the
   cherry-picking the analytic max-of-one bar missed. The null-side functional `T_stat∘combine∘admit` must be
   **identical** to the observed-side one (including the `delta_sr_oos` ranking — see §5 cost). Define
   `T_b := −∞` when a replicate admits `< min_cohort_size`.

> **Anti-pattern (reproduces the original flaw):** fixing the observed `m` members and only bootstrapping their
> returns, OR freezing the combiner weights. Either conditions on the observed configuration → anti-conservative
> null (**MC: 32% false-pass**). Selection AND combine must re-run every rep. This is a pinned tripwire (§6).

### 3.3 Statistic, p-value, gate

- **Observed:** `T_obs = SR_pp(combine(base ∪ admit(R))) − SR_pp(combine(B_base))` on the **real
  (non-demeaned, non-resampled)** panel with its true `timestamps`.
- **Null:** `{T_b}` for `b = 1..B`.
- **p-value:** `p = (1 + #{b : T_b ≥ T_obs}) / (B + 1)` (the `+1` is the standard finite-B bias guard).
- **Gate:** `passes_mc = (p ≤ α_cohort)`. `α_cohort` lives in `configs/signal_eval.gates.yaml :: generation`
  (propose `0.05`; a *new* pre-registered gate, not a reused one — never hardcoded, per CLAUDE.md).
- **B:** `B ≥ 1000` for a binding gate (p-value SE at α = 0.05 is `√(α(1−α)/B) ≈ 0.007` at B = 1000 vs 0.015 at
  B = 200 — 200 is fine for the dev/pre-filter loop, 1000 for the recorded verdict). Pin `B` in gates.

---

## 4. Two independent guards (MC is necessary, holdout is complementary)

The MC (§3) is run **on the CPCV/selection span** and controls *selection bias statistically*. The **embargoed
holdout** is a separate, orthogonal guard controlling *out-of-sample persistence*: freeze the observed cohort's
members **and** combiner weights from the full pipeline, apply them to the embargoed holdout tail, and require the
**ΔSR** there `≥` a floor (unit-pinned: state annualized vs per-period, and match `min_book_uplift`'s units).
"Frozen weights" = the last month-end α vector from the full pipeline, applied with no re-estimation.

A cohort must clear **BOTH** `p ≤ α_cohort` **and** the holdout floor. They fail differently — MC catches "lucky
selection on the training span," holdout catches "selection that didn't transfer" — so requiring both is not
redundant (different data, different failure modes; the conjunction can only shrink size). Pin a holdout-reuse
policy: repeated cohort attempts reading the same holdout accumulate their own multiplicity.

---

## 5. Cost & evaluation order (don't run MC on doomed cohorts)

**Corrected cost (audit HIGH):** faithful admission re-ranks by `delta_sr_oos`, which requires a per-candidate
`_combined_book` + CPCV *inside every replicate* → **O(B·N·(combiner+CPCV))**, realistically **hours**, not
minutes. Two ways out (pick one, **pre-register it**):

- **(a)** budget for the full cost — fine for an on-survivor / nightly gate, never a per-genome inner loop; or
- **(b)** pre-register a *cheaper ranking statistic* for admission (e.g. rank by standalone per-period Sharpe
  instead of `delta_sr_oos`) and use it **identically** in the observed and null pipelines. The p-value is valid
  as long as null-`f` ≡ observed-`f`, whatever `f` is.

**Evaluation order:** **(i)** analytic `SR*_cohort` pre-filter (Doc 1) → reject cheaply; **(ii)** MC null on
survivors → binding; **(iii)** holdout confirmation. The analytic floor keeps the MC off the ~99% of cohorts that
never had a chance — the main lever that makes (a) affordable.

---

## 6. Determinism (reproducibility invariant — no `Math.random`)

Seed the bootstrap RNG deterministically: `seed = int_hash(gates_hash ‖ pool_content_hash ‖ run_id)`. Block
indices and lengths derive from that seed. `crucible reproduce` must yield a **byte-identical p-value**. This
mirrors the existing `rng_seed: 7` determinism ADR (ADR-C3-6).

---

## 7. Calibration + power tests (MATH + Audit gated — acceptance criteria, not hopes)

Re-create the audited harness (second math pass) and port it into the test suite; its validated targets **are**
the acceptance criteria.

| Test | What it fires on | Required result |
|---|---|---|
| **Calibration (size)** | pool of `N` pure-noise streams, ≥200 seeded meta-reps | false-pass rate `≈ α_cohort` — measured **1.0–3.5%** at α = 5% for the corrected design |
| **BLOCKER-1 regression** | same, but base held fixed | must reproduce **8.3%** (keep the regression caught) |
| **Tripwire: m=9 / N=100** | pure-noise cell (naive DSR: 100% false-pass) | must reject at rate `≈ α_cohort` — permanent tripwire |
| **Power (anti-vacuity)** | inject `m=8` genuinely-uncorrelated weak streams (`s ≈ 0.05`) into a noise pool | MC must PASS at high rate — **ΔSR → 98%**; marginal-Sharpe → **6–9%** (BLOCKER-3 failure). Must use ΔSR. |
| **Faithfulness** | joint block bootstrap of the real panel | reproduces mean pairwise corr, lag-1 autocorr, **and combiner-weight variance** within tolerance |
| **ℓ-sensitivity** | size at ℓ ∈ {5, 21, 63} | bounded drift (≤ ~4.7% at 20× misspecification on AR(0.25)); pin the sweep |
| **Anti-pattern regression** | frozen-members + frozen-weights variants | must reproduce **~32%** false-pass so nobody "optimizes" back into them |

---

## 8. Where it plugs in / build guardrails

- Consumes the `CohortEvidence` apparatus of Doc 1 (evaluator, `admit`, combiner). It is **not** standalone code:
  the MC re-runs `admit∘combine`, so Doc 1 must exist first.
- Runs **after** the analytic `SR*_cohort` pre-filter (Doc 1 fix 1) and **before** the embargoed holdout (§4).
- **No threshold is loosened.** `α_cohort` is a *new* pre-registered gate; the reused funnel thresholds (DSR 0.90
  / HLZ-t 3.0 / uplift 0.10) apply to the √N-larger cohort quantity, not to each candidate.
- This is a **new critical-path statistic** → whole change is Audit-skill + Math-skill gated at build.
- **Any cohort that clears this gate is PROMISING, not GO.** Tier-2 deep lifecycle audit + survivorship-free
  re-validation remain non-negotiable before any capital (CLAUDE.md).

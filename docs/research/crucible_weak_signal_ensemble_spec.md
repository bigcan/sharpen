# Crucible Weak-Signal-Ensemble Redesign — cohort evaluator + the cohort deflation statistic

**Status:** DESIGN (no code changes). Author: session 553-cont-108 → -109.
**Doc 1 of 3** in the weak-signal-ensemble design set:
- **Doc 1 (this)** — the cohort evaluator + correlation control + combiner + the **cohort deflation statistic**
  `SR*_cohort` (novel statistic #1, naive `n_trials=N` DSR **blocked**).
- **Doc 2** — [`crucible_mc_null_spec.md`](crucible_mc_null_spec.md): the **selection-aware Monte-Carlo null**
  (novel statistic #2, marginal-stream / base-fixed / time-scrambled forms **blocked**) — the *binding* gate.
- **Doc 3** — [`crucible_diverse_proposer_spec.md`](crucible_diverse_proposer_spec.md): the upstream blocker (no
  diversity to admit from). *Measure data breadth first.*

**Motivation:** Operator hypothesis — "we don't need one strong signal; we can ensemble many
weak uncorrelated signals." This doc (a) measures how far the cont-107 overlay batch actually
landed below the gates, and (b) specs the three architectural changes needed to make the
ensemble thesis *reachable* by the funnel. Conclusion up front: the thesis is not blocked by
threshold values — it is blocked by **the unit of evaluation** (one candidate vs. a fixed base
book) and by **two disabled/absent mechanisms** (cross-candidate correlation control; a
correlation-aware combiner). The two statistics that make the corrected evaluator sound (the analytic
`SR*_cohort` here, the MC null in Doc 2) were each adversarially math-verified with their **naive forms
blocked** — see the audit blocks in Part 1 and Doc 2. **Nothing here is implemented.**

---

## Part 0 — cont-107 gate-distance analysis (the data)

Source: `results/crucible_orchestrator_overlay/real/trial_ledger.db`. The ledger holds two
record types: **20 pre-registered** (12 `altdata` overlays + 8 `101alpha` cross-sectional) and
**20 scored/LOGGED** (10 overlay + 10 cross-sectional; `family` is not re-stamped on the scored
row, hence `NULL`). All scored candidates are **LOGGED (0 PROMISING)**.

Binding generation gates ([fitness.py:310](../../finrl_pro_ds/signals/generation/fitness.py)):
`ΔSR_marginal ≥ 0.10` ∧ `DSR_aug ≥ 0.90` ∧ `marginal-HLZ t ≥ 3.0` ∧ `corr-to-base ≤ 0.70` ∧ not-fragile.

| metric | gate | batch max | batch median | batch min |
|---|---|---|---|---|
| `dsr` (DSR of augmented **book**) | ≥ 0.90 | 0.802 | 0.507 | 0.002 |
| `delta_sr_oos` (mean CPCV ΔSR) | ≥ 0.10 | +0.595 | +0.453 | +0.368 |
| `marginal_hlz_t` (t of `b_aug−b_base`) | ≥ 3.0 | **−1.037** | **−1.286** | **−1.393** |

### Three findings that reframe "are the gates too strict?"

1. **The binding gate is failed on the *wrong side of zero*, not by a hair.** Every candidate's
   `marginal_hlz_t` is **negative** (~−1.0 to −1.4). The gate wants ≥ +3.0. These are not
   "weak positive signals that just miss" — their significance-weighted marginal contribution is
   *negative*. Ensembling things whose marginal contribution is negative does not produce a
   positive book; it compounds the drag. **Lowering the t-gate would admit these, and they would
   hurt the book.**

2. **`delta_sr_oos > 0` while `marginal_hlz_t < 0` is a red flag, not a near-miss.** The mean of
   per-path Sharpe *differences* is positive (+0.45) while the marginal *return stream*
   `b_aug − b_base` has a negative Sharpe. The two measures disagree because Sharpe is a nonlinear
   ratio: an overlay that mostly *de-risks* the book can raise path-Sharpe while its incremental
   return stream has a slightly negative mean. This is exactly the pattern the multi-gate
   conjunction exists to catch (cf. the BAB "crash-hedge not alpha" and commodity-carry "dilutive"
   findings in memory). The positive `delta_sr_oos` here is **not** statistically robust.

3. **`DSR ≈ 0.80` is misleading as a "how close is the signal" gauge.** That DSR is the deflated
   Sharpe of the *augmented book* — dominated by the base sleeves (tsmom + rates_carry), **not** a
   measure of the candidate. A candidate contributing *nothing* still inherits ~the base book's
   DSR. So "top DSR 0.802" does **not** mean "89% of the way to a good signal." Candidate strength
   lives in `delta_sr_oos` / `marginal_hlz_t`, and there it is negative.

### Why THIS batch cannot be rescued by ensembling — and why that's not a fair test

The 20 formulas are **near-duplicate variants of two ideas**:
- overlays: all `ts_min(correlation(rank(adv20), delta(<altdata>)))` with the alt-data term swapped;
- cross-sectional: all `stddev(decay_linear(volume, …))` permutations.

They are therefore **highly mutually correlated**. The ensemble multiplier for signals with mean
pairwise correlation ρ is `√(N / (1 + (N−1)ρ)) → 1/√ρ`, independent of N. A pool of 10 variants of
one idea has ρ near 1 → ensemble benefit ≈ 0. So this batch is a *degenerate* ensemble input: one
idea in ten costumes, with negative marginal significance. It neither supports nor refutes the
thesis — it shows the current **fixed-seed-bank proposer cannot produce the diverse pool the
thesis requires**. (No LLM proposer exists yet; see memory `crucible_altdata_bridge_overlay_experiment`.)

**Implication for the spec:** a cohort evaluator (#1) is worthless without (a) genuinely diverse
candidates and (b) cross-candidate de-correlation (#2). All three changes below are
interdependent; ship them together or not at all.

---

## Part 1 — Cohort evaluator (`CohortEvidence`)

> ### ⛔ MATH AUDIT (Fable-5 + Math skill, cont-108) — the naive deflation is BLOCKED
>
> A Monte-Carlo verification against the *audited* `deflated_sharpe_ratio` itself falsified the
> original step-4 construction below. **Do not implement steps 4–5 as first drafted.**
>
> **The flaw:** DSR's `SR*` is the expected **maximum of N single-trial Sharpes** — the correct
> null for "I searched N configs, I report the best ONE." The cohort statistic is a *different
> random variable*: the Sharpe of an equal-risk **combination of the top-m of N**. For i.i.d.
> noise, greedy selection admits ≈ the top-m (noise pairwise corr ~N(0,1/T), so the corr filter
> rejects almost nothing), and the equal-risk book Sharpe ≈ `√(m/(1+(m−1)ρ̄)) · Ā(m,N)` — which
> **exceeds the max-of-*one* bar `A(1,N)≈√(2 ln N)` by ~√m.** The lucky draws average to a
> still-high value while their noise diversifies away, and `SR*` never sees the subset-selection
> step. Worse, `SR*` grows only ~`√(2 ln N)` (log-rooted) while the noise book grows ~`√m`, so
> **a bigger pool makes the false verdict *stronger*, not safer.**
>
> **MC evidence** (audited fn, honest `n_trials=N`, pure noise, greedy admit + equal-weight combine):
>
> | N pool | m cohort | mean DSR | P(DSR ≥ 0.90) |
> |---|---|---|---|
> | 100 | 1 (audited single path) | 0.48 | **1%** ✔ calibrated |
> | 100 | 4 | 0.93 | **77%** ✗ |
> | 100 | 9 | 0.99 | **100%** ✗ |
> | 500 | 12 (`max_cohort_size`) | 1.000 | **100%** ✗ |
>
> **Two BLOCKER fixes (both required):**
>
> 1. **Replace step-4's SR\*** with the **top-m order-statistic benchmark** (reduces to the audited
>    BLdP form at m=1, MC-validated <2%):
>    ```
>    SR*_cohort = σ_trials · √( m / (1 + (m−1)·ρ̄) ) · (1/m) Σ_{k=1..m} Φ⁻¹( 1 − (k − 3/8)/(N + 1/4) )
>    DSR_cohort = Φ[ (SR_book − SR*_cohort) · √(n_obs−1) / √(1 − g1·SR_book + (g2+2)/4·SR_book²) ]
>    ```
>    N = `n_candidates_seen`, m = `n_members`, ρ̄ = `mean_pairwise_corr`, σ_trials = std of the
>    per-period trial pool. This is an analytic **floor** (assumes selection-by-own-Sharpe + static
>    equal-risk weights); the *exact* null under greedy-by-`delta_sr_oos` + the dynamic combiner
>    must be estimated by a **full-pipeline Monte-Carlo** (joint stationary block bootstrap of the
>    base+candidate panel, demean candidates only, rerun admission+combiner+gate — see Doc 2
>    (`crucible_mc_null_spec.md`) for the audited construction; the ΔSR statistic, joint base resampling, and fixed time axis are all
>    load-bearing). The MC null is the authoritative gate; the formula is the cheap pre-filter.
> 2. **Make the untouched holdout the *binding* gate.** The evaluator is placed *before* holdout
>    validation, so — as first drafted — admission ranking (`delta_sr_oos` desc) AND the gated
>    DSR/HLZ-t are computed on the *same* CPCV span the members were selected on (double-use-of-data
>    leak). Fix: select the cohort on the CPCV span; read the binding DSR/HLZ-t on data never used
>    for admission. In-CPCV numbers become advisory only.
>
> **Also (lower severity):** (MEDIUM) gate on the **marginal stream** `b_cohort − b_base`'s own
> significance, not the aug-book DSR — the latter is base-book-dominated (Part 0 finding 3 applies
> to the cohort gate itself; when the trial pool clusters near the base book, σ_trials→0 ⇒ SR\*→0 ⇒
> the DSR degenerates to an undeflated PSR of a base-dominated book). (LOW) the DSR z uses raw
> `√(n_obs−1)` while the HLZ-t uses AR(1) effective-N — inconsistent autocorrelation treatment,
> inherited from the single-candidate path.
>
> **Clean parts (audit-confirmed):** the √m ensemble arithmetic and the `1/√ρ` correlation ceiling
> (Part 0 / the Math note) are algebraically correct; SR\* is strictly monotone in N; the
> per-period (`periods_per_year=1`) convention is applied consistently; the reuse of the audited
> `deflated_sharpe_ratio` and the frozen thresholds is sound. The *slogan* "gate the destination,
> deflate for the journey" is right in spirit — the original draft simply undercounted the journey
> (it omitted the *choose-which-m-to-combine* step). With the order-statistic SR\* + holdout gate,
> the construction becomes sound.

**Goal:** evaluate whether a *set* of individually-sub-threshold candidates, combined into one
book, clears a book-level significance bar — with multiplicity deflated against the *search*, not
against each candidate in isolation.

### Where it plugs in
[evolve.py:278](../../finrl_pro_ds/signals/generation/evolve.py) currently does:
```python
train_passers = [c for c in ranked if c.result is not None and c.result.passes_gate]
```
Add a parallel path that consumes the **LOGGED-but-real** tail — candidates that fail the
per-candidate gate but are individually non-degenerate — and assembles a cohort book. This runs
*after* per-candidate scoring, *before* holdout validation, and emits a new artifact
(`CohortEvidence`) that the orchestrator records alongside the per-candidate cards.

### New module: `finrl_pro_ds/signals/generation/cohort.py`

```python
@dataclass(frozen=True, slots=True)
class CohortConfig:
    max_cohort_size: int          # cap N (deflation + combiner cost); e.g. 12
    min_cohort_size: int          # below this, no cohort verdict (e.g. 3)
    max_pairwise_corr: float      # admission: drop candidates with |corr| > this to an
                                  #   already-admitted member (greedy de-dup; ties to #2)
    promising_dsr: float          # book-level DSR floor (REUSE 0.90 — do NOT loosen)
    cohort_hlz_t_min: float       # book-level marginal-HLZ t floor (REUSE 3.0)
    min_book_uplift: float        # ΔSR of cohort book vs base book (REUSE 0.10, but now
                                  #   applied to the COHORT, not each member)

@dataclass(frozen=True, slots=True)
class CohortEvidence:
    members: tuple[str, ...]      # spec_hashes admitted (after de-dup)
    n_members: int
    n_candidates_seen: int        # pool size before admission (for the deflation N)
    delta_sr_oos: float           # mean CPCV ΔSR of (base ∪ cohort) vs base
    dsr_cohort_book: float        # deflated per-period Sharpe of the augmented cohort book
    cohort_hlz_t: float           # t of the cohort's marginal stream b_cohort − b_base
    mean_pairwise_corr: float     # realized diversification of the admitted set
    passes_gate: bool
```

### Algorithm
1. **Candidate pool.** Take all scored candidates with a finite, non-degenerate return stream
   (include LOGGED ones — that is the whole point). Exclude only hard-infeasible / leak-culled.
2. **Greedy de-correlated admission** (ties into #2). Rank the pool by per-candidate
   `delta_sr_oos` desc. Walk the list; admit a candidate iff its `|corr|` to every
   already-admitted member ≤ `max_pairwise_corr`. Stop at `max_cohort_size`. This yields a
   *diverse* subset, not the top-N-by-score (which would be near-duplicates, per Part 0).
3. **Build the cohort book.** Extend the base sleeve set with *all admitted candidates at once*:
   `aug = {**base, cand_1: r_1, …, cand_m: r_m}`, then run the C1 combiner
   (`_combined_book(aug, ts, cfg)`; see #3 for the combiner it should use). Marginal stream =
   `b_cohort − b_base`.
4. **Deflate at the cohort level.** ⛔ **SUPERSEDED by the MATH AUDIT block above.** The original
   draft (`n_trials = n_candidates_seen` into the unmodified BLdP `SR*`) is falsified — it admits
   noise at 77–100% for m ≥ 4. Use the **order-statistic `SR*_cohort`** (audit block, fix 1) as the
   cheap pre-filter and a **full-pipeline Monte-Carlo null** as the binding gate. `trial_sharpes`
   remains the cross-search dispersion pool (GP4-02); `σ_trials→0` degeneracy is documented there.
5. **Gate.** ⛔ **SUPERSEDED — see audit block.** `passes_gate` must read on the **untouched holdout**
   (fix 2, no double-use of the selection span) and on the **marginal stream** `b_cohort − b_base`
   (MEDIUM), not the base-dominated aug-book DSR. Thresholds are still reused unchanged; the
   relaxation is structural, but the *statistic* the threshold is applied to is the corrected one.

### Math note (audit-CONFIRMED arithmetic; audit-CORRECTED deflation)
For m uncorrelated sleeves each with per-period Sharpe s, the equal-risk cohort book Sharpe is
`s·√m` (√m arithmetic and the `1/√ρ` ceiling are algebraically correct — audit Q2). A member with
s = 0.03 (t≈1.4 over ~2000 bars, individually LOGGED) is invisible to the per-candidate gate, but 9
uncorrelated such members give a book Sharpe ≈ 0.09 → cohort t ≈ 4.0 > 3.0. **The trap the audit
caught:** this same √m amplification applies to *cherry-picked noise*, and the max-of-one `SR*`
cannot bound a top-m *combination* — so the deflation must use the order-statistic `SR*_cohort`
(which subtracts the selection benefit) or an MC null, NOT the naive `n_trials=N`. **Gate the
destination (book), deflate for the WHOLE journey (search + the choose-which-m-to-combine step).**

### Failure modes to test (negative tests)
- **Noise cohort (RE-SPECIFIED per audit — the naive form is vacuous AND falsely asserted).** Do
  NOT test "combine m *given* noise streams" (that trivially passes — SR_book ~ N(0,1/T)). The test
  MUST generate a **pool of N ≫ m** noise candidates, run the **full greedy admission + combiner +
  gate**, and assert a false-pass **rate ≤ α** over ≥100 seeded replications. Pin the **m=9 / N=100
  cell as a known-must-fail tripwire** for the naive `n_trials=N` design (MC: 100% false-pass) — it
  must PASS (i.e. correctly reject) only under the corrected order-statistic `SR*_cohort` / MC null.
- **Correlated cohort:** m copies of one signal → admission (#2) must reduce it to 1 member; if
  de-dup is bypassed, `dsr_cohort_book` must NOT show √m inflation (the combiner in #3 must not
  reward concentration).
- **cont-107 replay:** feed the 20 scored candidates → cohort verdict must remain FAIL (their
  marginal streams are negative; a positive cohort verdict on this input is a bug).
- **Single-candidate reduction (calibration anchor):** at m=1 the corrected `SR*_cohort` must equal
  the audited BLdP `SR*` (MC-validated <2%), and pure-noise false-pass at the 0.90 gate must be ≈α
  (1% observed) — guards against the correction breaking the existing single-candidate path.

---

## Part 2 — Cross-candidate correlation control

**Gap today:** the only correlation guard is `_base_span_corr` in
[fitness.py:177](../../finrl_pro_ds/signals/generation/fitness.py) — candidate vs. the *base
sleeves*. Nothing measures candidate-vs-candidate correlation, so the search can (and in cont-107
did) surface a pool of near-duplicates. Without this, #1's cohort is a mirage: N correlated
members give book Sharpe ≈ s·√(N/(1+(N−1)ρ)), not s·√N, and the deflation would flag it as
overfit anyway.

### Two mechanisms (both needed)

**2a. Admission de-dup (greedy, in #1).** Already specified in Part 1 step 2:
`max_pairwise_corr` gate on the correlation of realized return streams. Implementation: a helper
`pairwise_corr_matrix(returns: Mapping[str, np.ndarray]) -> np.ndarray` on commonly-finite bars,
reused by both #1 and the report. This is the cheap, deterministic guard.

**2b. Novelty pressure in the proposer/search.** Prevent the search from *spending its budget* on
variants of an already-scored idea. Add to the generation loop a **novelty penalty**: a candidate
whose max `|corr|` to any *already-scored genome* exceeds a threshold has its fitness down-weighted
(not hard-rejected — a rediscovery via a different formula is still weak evidence, but it should not
crowd the pool). Interface: extend `FitnessConfig` with
`novelty_corr_cap: float` and `lambda_novelty: float`, and thread the scored-genome return pool
into `combination_fitness` (it already receives `trial_sharpe_pool`; add an optional
`scored_return_pool`). Fitness becomes
`F − λ_novelty · max(0, max_corr_to_pool − novelty_corr_cap)`.

This is the mechanism that would have stopped cont-107 from returning 10 `stddev(decay_linear(
volume,…))` twins. It is a *search-efficiency* fix; 2a is the *correctness* fix. Ship 2a with #1;
2b can follow once an LLM proposer produces enough genuinely distinct candidates to matter.

### Tests
- de-dup: a pool of k identical streams admits exactly 1.
- de-dup: two streams at exactly `max_pairwise_corr` — boundary is inclusive/exclusive per spec
  (choose exclusive: `>` rejects), asserted by a tripwire.
- novelty: a genome identical to a scored one gets fitness strictly below its no-penalty value.

---

## Part 3 — Correlation-aware combiner

**Gap today:** the shipped combiner is convex inverse-vol risk-parity with
`redundancy_strength = 0.0` ([allocator_factory.py:561](../../finrl_pro_ds/envs/allocator_factory.py)).
It weights by 1/σ only — it does **not** account for correlation, so it cannot harvest the
diversification the ensemble thesis promises, and it *over-weights* a cluster of correlated
members (each gets its own 1/σ slice, concentrating the book on their shared factor).

**Good news:** the machinery already exists and is *off*, not missing.
`dynamic_sleeve_alphas` already computes `redund_s = exp(−λ_r · ρ̄_s)` from
`_trailing_mean_abs_corr` ([allocator_factory.py:458](../../finrl_pro_ds/envs/allocator_factory.py)),
where `ρ̄_s` is sleeve s's trailing mean `|corr|` to the other sleeves. With `λ_r = 0` this
degrades to `redund ≡ 1` (byte-identical to the live inverse-vol book — the back-compat identity
that protects the paper executor). The dispatch seam `combiner_alphas` already routes on a
`sleeve_combiner` config block with a `mode` enum.

### The change (config-only for existing sleeves; opt-in for the cohort)
1. **Do NOT change the live paper book.** Leave `sleeve_combiner` absent / `mode: inverse_vol`
   for `tailwind_v1` and all live configs. The redundancy down-weight must stay off there until it
   has its own Tier-2 (it changes realized weights).
2. **The cohort book (#1) uses `mode: dynamic` with `redundancy_strength > 0`.** This is the only
   place many-weak-members meet, and it is exactly where correlation-blind weighting does the most
   damage. Calibrate `λ_r` on synthetic cohorts (below), pinned in `signal_eval.gates.yaml ::
   generation` (never hardcoded).
3. **Consider a max-diversification / MV upgrade later.** The `exp(−λ_r·ρ̄)` down-weight is a
   heuristic (penalize average correlation), not the optimal combine (min-variance /
   max-diversification would use the full covariance). For the first cut the heuristic is enough
   and reuses audited code; a covariance-based `mode: max_div` is a follow-on ADR if the cohort
   path proves out.

### Calibration + tests (MATH-gated — combine weights feed realized returns)
- **Diversification recovery:** m truly-independent equal-Sharpe streams → combiner should
  approach equal-risk weights and the book Sharpe should approach `s·√m` (within CPCV noise). If it
  under-delivers, λ_r or the vol estimate is miscalibrated.
- **Concentration penalty:** m streams where m−1 are copies of one → `redund` must drive the
  cluster's *total* weight toward that of a single member (no √m inflation from duplicates).
- **λ_r = 0 identity:** cohort combiner with λ_r=0 must be byte-identical to inverse-vol (protects
  the reuse claim and the live book).

---

## Appendix A — MOVED to Doc 2 (`crucible_mc_null_spec.md`)

The full-pipeline Monte-Carlo null — the **authoritative** cohort gate — is now a standalone peer document:
[`crucible_mc_null_spec.md`](crucible_mc_null_spec.md). It was promoted out of this appendix because it is the
*binding* verdict (the analytic `SR*_cohort` in Part 1 is only a cheap pre-filter), and because it is the second
of the two novel statistics — each deserves its own reviewable artifact at the build decision.

What lives in Doc 2 (do not duplicate here): the H₀ definition; the second math audit (three BLOCKERs — joint
base resampling, fixed monotone time axis, ΔSR statistic — all fixed and each with a pinned regression
tripwire); the demean → joint stationary block bootstrap → re-run admission+combiner → add-one one-sided
p-value construction; the two-guard MC+holdout design; the O(B·N·(combiner+CPCV)) cost analysis; determinism;
and the calibration/power acceptance table. **Evaluation order stays:** (i) analytic `SR*_cohort` pre-filter
(Part 1) → (ii) MC null (Doc 2) → (iii) embargoed holdout.

---

## Sequencing & guardrails

1. **#2a + #1 together** (a cohort evaluator without de-dup is unsafe). The binding gate is the
   **MC null of Doc 2 (`crucible_mc_null_spec.md`)** (analytic `SR*_cohort` pre-filter → MC → holdout), NOT the naive DSR that
   the math audit blocked. The MC construction is **math-verified GO** with the three second-audit
   BLOCKER fixes applied (joint base resampling, fixed monotone time axis, ΔSR statistic); implement
   to Doc 2 §7's validated targets and Audit-skill the whole change (new critical-path statistic).
2. **#3** enabled *only* on the cohort book, λ_r calibrated on synthetics, live book untouched.
3. **#2b** last, once a diverse proposer exists (fixed seed bank makes it moot today).
4. **No threshold is loosened.** DSR 0.90 / HLZ t 3.0 / uplift 0.10 are reused verbatim; the
   relaxation is that they now apply to a √N-larger cohort quantity. This keeps the frozen funnel
   `gates_hash` semantics defensible and does not reopen the false-discovery door that the realness
   gates exist to hold shut.
5. **Any cohort that clears the gate is still PROMISING, not GO** — Tier-2 deep lifecycle audit +
   survivorship-free re-validation remain non-negotiable before any capital (CLAUDE.md).

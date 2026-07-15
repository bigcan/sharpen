# Independent Expert Audit — Crucible Alpha-Mining Platform: Final Report

**Date:** 2026-07-14 · **Auditor:** independent (external mandate per
`docs/research/crucible_independent_audit_brief_2026-07-13.md`) · **Scope:** the complete `0 PROMISING`
record, the 5-leg PROMISING gate, the E1/E2 calibration, the three power-lever closures, the proposers/DSL,
the cost model, and the ledgers.

**Method:** 12 independent audit passes (9 parallel deep finders + 3 adversarial skeptics assigned to
*refute* the verdict-deciding findings), with every pivotal claim re-verified directly by the lead auditor
against the ledgers and by re-running experiments through the **shipped** code (nothing re-implemented).
Reproduction scripts are cited per finding; ledger queries are one-liners.

---

## 1. Verdict

**The record is explained by (C) compounded on (B); (A) is unclaimable. The internal recommendation to stop
mining survives — but the internal account of *why* is materially wrong.**

Precisely, as a quantified mixture:

- **(C) — proximate cause of the record.** On every substrate ever mined, at least one gate leg is
  structurally sealed against the candidate class being mined, independent of candidate quality and of
  sample size. The single decisive experiment (brief §6.1, run through the shipped two-gate funnel on the
  **real** Taiwan substrate, re-verified by the lead auditor): a **planted perfect weekly-hold timing oracle
  — holdout annualized Sharpe 1.80 — fails 0/5 at every skill level from p=0.52 to p=1.00**. Only a *daily
  perfect-foresight* oracle (SR 12.6) passes. A machine that rejects perfect weekly foresight cannot return
  information about whether realistic edges exist; `0 PROMISING` was determined by the machine, not the
  market.
- **(B) — the residual floor underneath.** After removing every machine defect, an irreducible power wall
  remains: even an idealized single pre-registered t≥2 test on the full 16-year panel has MDE80 ≈ **0.71**
  annualized ΔSR — above the realistic 0.3–0.5 target. No honest discovery-grade contract detects 0.3–0.5
  edges on free daily data of this length. **The stop-mining decision is therefore correct** even though the
  calibration that motivated it substantially measured the machine's own defects (§4).
- **(A) — unsupported, except in a near-vacuous bounded form.** Matched-null searches (shipped `evolve` +
  gate, real budget pop 200 × gens 40, base book rescaled to the measured real train Sharpe) produce
  hall-of-fame ΔSR **0.80–0.94** and marginal_t up to **2.93 under pure noise**. The real record's maxima
  (ΔSR 0.99 / 0.71 / 0.66; t ≤ 2.12) sit **at or below the noise ceiling**. The record is exactly what
  no-signal looks like *through this instrument* — which supports only: "no edge with true train-book ΔSR
  ≥ ~1.2 exists in the narrow expressible-and-proposed class" (and **no bound at all** for weekly-or-slower
  overlays, where the statistic saturates at a perfect-oracle ceiling the noise search already reaches).
  Nothing about 0.3–0.8 edges, nothing about the inexpressible classes (§3.4), nothing about the market.

**If (B), what would have to be true of real alpha for the internal narrative to hold?** Real alpha would
have to arrive in units of ΔSR ≥ 1.4–3.5 (deployed operating point, §4) for this machine to see it. The
project's own history shows edges of that gross size *were* found by pre-registered single-hypothesis probes
(arb sweeps: gross SR 1.4–2.1, detected at power ≈ 1.0) and died on costs — consistent with this audit:
the power wall is a property of the *mass-miner contract*, not of hypothesis testing on this data per se.

---

## 2. The mechanism: why 0 PROMISING was certain a priori

The PROMISING gate is a 5-leg AND (`fitness.py::combination_fitness`, thresholds
`configs/*signal_eval.gates.yaml::generation`) that must pass **twice** — on train CPCV, then re-scored on
the ~25% embargoed holdout (`evolve.py:282,296-318`). Three layers each independently guarantee (or nearly
guarantee) zero survivors:

### 2.1 Structural seals (candidate-independent, sample-size-independent) — the (C) core

**S-1. `marginal_t` tests the wrong quantity — and every substrate's record shows it.**
`marg = b_aug − b_base` where both books come from the convex sum-to-1 inverse-vol combiner
(`allocator_factory.py:568-603`). Verified as an *identity* on the shipped code path:
`marg == w_c·(r_c − b_base)` (max deviation 1.7e-18). Hence `E[marg] = w̄_c·(μ_cand − μ_base_book)` — a
**substitution residual**: the leg demands the candidate *out-mean the book it joins*, at t≥3. A genuine
diversifier that raises book Sharpe through variance reduction has `E[marg] < 0` and **marginal_t → −∞ as
N → ∞** — zero asymptotic power, at any frequency, on any substrate with a positive-mean base book.
- Ledger (verified): all 10 cross_asset rows ΔSR ∈ [+0.368, +0.424] with t ∈ [−1.399, −1.341]; all 20
  overlay-bridge rows ΔSR ∈ [+0.368, +0.595] with t ∈ [−1.393, −1.037]. Positive-uplift candidates,
  uniformly negative t.
- The statistic is also **candidate-scale-dependent** (same signal: t = −1.67 at 0.25× leverage, +1.81 at
  8×) — incoherent even as a deliberate "must-add-mean" criterion.
- It contradicts its own docstring (`fitness.py:284-287`: "an honest diversifier … now clears the hurdle")
  **and** the Tier-2 finding that created it (GP4-03,
  `docs/research/C3_alpha_generation_base_sleeves_deep_lifecycle_audit_2026-06-30.md:164`: hurdle on "the
  marginal ΔSharpe … confirm a genuine diversifier can clear it"). No document records a conscious
  mean-vs-Sharpe trade-off. **Adversarially confirmed** (skeptic pass, all refutation angles failed).
- On Taiwan the sign flips positive only because the base book's mean ≈ 0 — and the bar is then arithmetic:
  t≥3 on ~1011 holdout bars requires the marginal stream to carry **annualized Sharpe ≈ 1.50** (2.2 with
  ρ₁=0.2 after dilution); the perfect weekly oracle reached **2.92 on train — still below 3.0**.

**S-2. `dsr_aug` is a book-level bar that measures the base book's era, not the candidate — and it seals
Taiwan at the train pre-filter.** The leg deflates the *whole augmented book's* Sharpe. The Taiwan base book
(TSMOM-only, `base_sleeves.py:562-578`) measures **annualized Sharpe ≈ −0.007 on the train split**
(2010→2022; −0.147 on the cached-panel variant) vs **+1.364 on the holdout** — so on train no bounded tilt
of it can reach the required book Sharpe (~1.3–2.0 annualized incl. the fallback-pool inflation, §2.2), and
verdicts are a function of the base book's decade, not candidate quality. Ledger (verified): **all 160
Taiwan candidates scored dsr ≤ 0.0494** (taiwan_v2 max 3.9e-05, LLM probe max 6.6e-06) against a 0.90 gate,
while every one passed the uplift leg. Refinement of the prior internal audit: the *holdout* dsr leg is
reachable (the base book alone clears it there) — the seal lives specifically at the train pre-filter, which
nothing ever passed. **96.6% of all FDR-charged hypotheses (225/233) ran on this sealed substrate.**

**S-3. The expressible hypothesis space excludes the classes where daily edges are documented.** The overlay
path — the *only* route for alt-data — collapses every formula to one broadcast scalar timing the whole base
book (`evolve.py:80-135`): per-asset flow prediction (the literature's actual Taiwan daily edge: foreign-flow
persistence in the very ETFs whose per-asset flows the connectors ingest), cross-sectional flow, regime-gated
conditionals (not growable: `grammar.py:79-81` `_VALUE_OPS` has no ternary/comparison/div), event-window and
seasonality signals (no calendar terminals), and >120-bar horizons are all **architecturally unposable**. The
"37 mechanism-diverse overlays" are 6 smoothing archetypes over 17 correlated slots of one economic genus —
effective independent hypotheses ≈ **5–10** (11/30 logged probe genomes are *exact-ΔSR duplicates* under the
affine-invariant tanh-z transform). "Power-bound, not idea-bound" was not demonstrated; the diversity was
cosmetic.

### 2.2 Mis-specified statistics at the only gate that ever ran — (C), impact-bounded

**S-4. The deciding gate is the *uncertified* train pre-filter with a fallback DSR; the certified "binding"
holdout gate has never executed — anywhere.** Train scoring (`evolve.py:248`) passes no
`trial_sharpe_pool`, so `fitness.py:271-274` builds SR\* from the book's **own 15 CPCV path Sharpes**
(within-book time-variability, sd ≈ 0.023 pp) multiplied by the best-of-N order statistic for the **search
count** — a category error inflating SR\* ≈ 2.5× vs an honest cross-genome pool (sd ≈ 0.009 pp). No real
candidate ever passed train, and E1 shows `holdout_evals = 0` across 2349 genomes — the pool-based gate the
Tier-2 audits certified (GP4-02) is dead code in practice, and E2, the intraday Stage-0, and the cross-market
Stage-0 all calibrated the fallback configuration.
**Adversarial correction (important):** the artifact was *historically redundant* — across all 190 scored
real rows, **zero failed the DSR leg alone** (every DSR-failure also failed marginal_t), and at T=4044 the
MDE≈1.40 anchor is set by the honest t≥3 marginal_t floor (fallback-DSR passes 8/8 at β=0.004 where
marginal_t fails). The artifact inflates short-T MDEs ~25% (T=756: ~2.9 honest vs 3.63 measured) and the
Taiwan seal is co-produced by it, but removing it would not have changed a single recorded verdict.

**S-5. Deflation-N pathologies.** `gen_n_eff` counts functional duplicates as independent trials (affine
invariance + commuting filters; 25–37% exact-duplicate rate in logged rows; one real run charged
**N = 4706**), and the fallback pool contains the candidate's own regime dispersion, producing a
**self-defeating deflation**: E2's dsr pass rate drops 0.975 → 0.80 as planted edge strength rises
0.025 → 0.04 (reproduced exactly; mechanism: SR\* grows with the signal's own cross-path dispersion — the
funnel penalizes the strongest regime-timed edges *more*). One-shot dedup burns each hypothesis forever at
its lowest-power moment (`hypothesis.py:112-114`) — while in practice the operator reset ledgers per
directory, re-charging 32+23 hypotheses (the FDR "account" has no cross-directory identity).

### 2.3 The honest wall — (B), real but roughly half the height claimed

At the deployed operating point (holdout re-pass at t≥3 with full file-drawer N), power at true ΔSR 0.5 is
≈ **2%**, and ≈ 42% even at ΔSR 1.4; deployed MDE80 ≈ **1.9–3.5** depending on substrate length — not the
1.40 headline (§4). But even the *ideal* contract (one pre-registered hypothesis, full panel, t≥2, no
deflation) yields MDE80 ≈ 0.71 at T=4044 / 1.42 at T=1011. Detecting 0.3–0.5 needs ~22–40 years at t≥2.33,
or different data. This is the true (B)-residual, and it is why the stop-decision survives every correction
in this report.

---

## 3. Ranked findings

Severity ranks the finding's weight in the A/B/C verdict; every finding is high-confidence unless noted.
Categories: **[BUG]** correctness (code ≠ documented intent), **[DESIGN]** intended but consequential,
**[CAL]** calibration methodology, **[DATA]** genuine world/data limit, **[PROC]** process/record integrity.

| # | Sev | Cat | Finding (evidence anchor · repro) |
|---|-----|-----|------------------------------------|
| F1 | S1 | BUG/DESIGN | `marginal_t` is a substitution-residual mean-test — sign-inverted for diversifiers, scale-dependent, contradicts GP4-03 intent and its own docstring; seals cross_asset (30/30 recorded candidates: positive ΔSR, negative t) and bars even a perfect weekly oracle (t 2.92 < 3.0). `fitness.py:289-294` · ledger queries + `scratchpad/planted_sweep.py`. Skeptic: **CONFIRMED**. |
| F2 | S1 | DESIGN | Taiwan is sealed at the train pre-filter: book-level `dsr_aug` on a ≈0-Sharpe train base book (TSMOM-only) → all 160 Taiwan candidates dsr ≤ 0.049 vs 0.90; 96.6% of all charged hypotheses ran on this substrate; verdicts reflect the base book's era, not candidates. `base_sleeves.py:562-578`, `evolve.py:282` · `planted_sweep.py` (train base SR −0.007, oracle 0/5). |
| F3 | S1 | PROC | The certified pool-based holdout gate has **never executed** (0 production, 0 in E1: `holdout_evals=0`); all recorded metrics are train-split CPCV values (ledger column `delta_sr_oos` is misnamed); the headline "best OOS ΔSR 0.66" is a train-split GP mutation with NULL rationale — not an LLM hypothesis — and sits at the matched-noise ceiling. `evolve.py:282,296-318`, `loop.py:192` · ledger + `calibration_both.json`. |
| F4 | S1 | CAL | The record is uninformative and (A) unclaimable: matched-null searches at the real budget yield HoF ΔSR 0.80–0.94, t ≤ 2.93 (pure noise) ≥ real record maxima; joint gate power ≈ 7e-4 at true ΔSR 0.5; E[discoveries \| all 233 charged were true 0.5-edges] ≈ 0.16–0.4. Bounded-(A) only at ΔSR ≥ ~1.2 in the narrow expressible class. `scratchpad/phase2/null_grid_sim.py`. Skeptic: **CONFIRMED**. |
| F5 | S1 | DESIGN | Overlay-only alt-data routing cannot express the documented edge classes (per-asset flow, cross-sectional flow, conditionals, events, seasonality, >120-bar); "37 diverse overlays" ≈ 5–10 effective hypotheses; "power-bound not idea-bound" not established. `evolve.py:106,131`, `grammar.py:79-81`, `altdata_bridge.py:122-124`. |
| F6 | S1 | BUG | Fallback-DSR category error at the only deciding gate (within-book path dispersion × search-count N; SR\* ×2.5; E2/intraday/cross-market all calibrated this config). **Impact bounded by skeptic:** 0/190 real rows failed DSR alone; T=4044 MDE traces to marginal_t; short-T MDEs inflated ~25%. `evolve.py:248`, `fitness.py:271-274`. Skeptic: **PARTIALLY-REFUTED (mechanism confirmed, decisiveness refuted)**. |
| F7 | S1 | DATA | The residual honest power wall: ideal single-prereg t≥2 MDE80 ≈ 0.71 (T=4044) / 1.42 (T=1011); deployed contract ≈ 1.9–3.5; ΔSR 0.3–0.5 undetectable under any discovery-grade contract on this data. The stop-mining decision is correct. `scratchpad/forward_power.py`. |
| F8 | S2 | CAL | The 1.40 MDE anchor is not the deployed operating point (full-panel, gen_n_eff=50, fallback pool, IID plant with zero representational attenuation, embeds ~0.27 ΔSR of synthetic churn cost, ±30% grid coarseness). Every mismatch points the same direction: deployed is *worse* — E2-RED is a fortiori sound, but all three lever-closure docs consumed 1.40 as the deployed number. |
| F9 | S2 | CAL | The "MDE flattens to N^-0.21" intraday closure is floor-plus-√N in disguise: MDE(N) ≈ 0.39 + c/√N, the 0.39 floor being the marginal_t dilution drag (a designed-in constant, not physics). Also the pre-registered TA-3 tripwire **failed on one of two validation cells** (ratio 0.59, `A1_verdict=INCONCLUSIVE`) and the result doc cites only the passing cell; failure direction implies 5-min MDE could be ~0.51 — *at* the ceiling, not clearly above. The practical NO-GO likely survives (the marginal_t floor independently blocks), but not with the advertised rigor. `results/crucible_intraday_power/intraday_power.json`. |
| F10 | S2 | DESIGN | Double-pass + double multiplicity at holdout (unselected OOS statistic deflated by full file-drawer N + pool + FDR charge): latent (never executed) but caps power at ~2% for ΔSR 0.5 if ever reached. `evolve.py:278,308-310`. |
| F11 | S2 | CAL | Self-defeating deflation (F6's pool contains the candidate's own regime dispersion): E2 dsr non-monotonicity 0.975→0.80 is systematic, penalizing the strongest regime-timed edges most. `scratchpad/e2_dsr_decompose.py`. |
| F12 | S2 | BUG | `gen_n_eff` counts functional duplicates (affine-invariant tilt; 25–37% duplicate rate; one run charged N=4706) → systematic over-deflation; one-shot dedup burns hypotheses at lowest power, while per-directory resets simultaneously voided the FDR account's identity (32+23 re-charges). |
| F13 | S2 | PROC | Governance layer is theater on the current record: `killed_families` structurally inert (zero writers of any killing verdict — the "~20 NO-GOs" moat never existed; verified by grep + all-ledger group-by); LORD++ is accounting-only (`fdr.py:26-37` admits nothing consumes α_t); lockbox criterion is a coin-flip (LR 1.23 null-vs-true-0.5); real-mode verdicts are unreproducible (`crucible_reproduce.py` synthetic-only). None suppressed discoveries; all overstate the platform's guarantees. |
| F14 | S2 | BUG* | **Anti-conservative (false-positive-direction) defects — fix before ANY gate relaxation:** overlay tilt cost charged at unit gross while the tilted book runs gross ≈ 11× (a future survivor's net Sharpe would be overstated ~10× in cost terms); short tilts receive a spurious rebate of embedded base-book cost (+2.0bp/day measured); `dsr_aug` z uses raw n_obs with no AR(1) haircut (inconsistent with marginal_t); the uplift statistic has a degenerate null tail (inverse-vol combiner hijack by near-zero-vol candidates: null ΔSR up to 3.3); E1's null is structurally easy (IID Gaussian, no fat tails/vol clustering/common factor) and has zero coverage of the holdout code path. |
| F15 | S2 | DATA/PROC | Record-integrity gaps: the whole history is 233 charged hypotheses from 10 mined ticks in 11 days (2026-07-02→07-13) — not "months"; cross_asset never charged a single alt-data overlay and only ever mined a 504-bar panel (never re-mined post-fix); taifex_oi slots had 4 days of history when mined; Taiwan overlays are forced flat over ~43% of the train split (panel 2010+ vs T86 2015+), diluting every statistic; flow overlays carry a gratuitous extra 1-day lag on top of release stamping (discarding the strongest response day). |
| F16 | S3 | PROC | CRU-1 bookkeeping: all Taiwan runs carry gates_hash `22a18172be1a` (`configs/taiwan_signal_eval.gates.yaml`, thresholds identical to the frozen US file) — legitimate, but no frozen reference hash is registered for it, so CRU-1 compliance is unverifiable for 95% of the record. Also: one LLM-probe tick_ts collision clobbered a manifest (repro contract broken for 66 rows). |
| F17 | S3 | DESIGN | Cost model: internally consistent (base and candidate both net, same bps/clock — "cost bug suppresses discoveries" is **refuted**), but the flat 5×-harsh 10bp removes 0.11–0.39 SR from cross-sectional candidates (a compounding haircut, ~7× too small to explain the record) and is ~7–14× harsh for TAIFEX futures legs. Positive verification: PIT joins, COT/T86/EDGAR release stamping, and the CRU-2 moat are **clean** — no false-positive-direction leak found anywhere in the data path. |

**What is NOT wrong (verified clean):** the primitive statistics (BLdP expected-max vs Monte Carlo, Mertens
bracket, AR(1) N_eff formula, per-period z convention incl. the anti-saturation choice, CPCV purge/embargo
geometry, NaN fail-closed gate assembly), PIT correctness of every connector traced, cost symmetry between
base and candidate books, and the CRU-2 anti-oracle moat (agent view = hash/type/family only; the LLM prompt
provably carries no scores). E1's GREEN (0/2349, upper-95% per-candidate FPR 0.13%) is real as an upper
bound — with the caveat that much of that protection is supplied by the same mis-specified legs, so any fix
must re-run E1 on a realistic null.

---

## 4. Is the deflation correctly calibrated? (deliverable 4)

**No — it is simultaneously over-conservative and mis-aimed, and its headline calibration number does not
describe the deployed gate.**

- **The five ANDed legs are near-independent rejectors, not redundant re-tests.** At planted β=0.008 the
  joint pass rate (0.05) ≈ the product of the two significance legs (0.25 × 0.225) — the AND *compounds*.
  The conjunction costs ≈ 0.44 ΔSR of MDE beyond a clean single test at identical FPR, and the
  double-pass adds more (power 0.026 at ΔSR 0.5 even with both legs relaxed to t≥2.33).
- **Two legs test the wrong quantity** (F1: mean-substitution instead of Sharpe-contribution; F2: base-book
  era instead of candidate). These are not "conservatism" — they are orthogonal to candidate quality, so
  their false-negative rate does not decrease with edge size or sample size. That is what makes the record
  a machine artifact rather than a power shortfall.
- **The deflation N is dishonest in both directions:** inflated by functional duplicates and GP churn
  (F12), yet reset per-directory whenever a new out-dir was used, and charged a second time at the holdout
  on a statistic that selection never touched (F10) — plus a third accounting-only LORD++ charge.
- **E1 GREEN + E2 RED → "honest-but-underpowered" is only half-honest.** E1 is sound as an upper bound. E2's
  RED is directionally right and even understated (deployed MDE 1.9–3.5, not 1.40) — but its *mechanism
  attribution* was wrong: the wall it measured is substantially the designed-in marginal_t floor plus (at
  short T) the fallback-pool artifact, i.e., properties of the gate, not of the data. The three power-lever
  closures reached defensible stop-decisions partly for the wrong reason, anchored on a number that is not
  the deployed operating point, and (intraday) over a pre-registered tripwire failure that the result doc
  did not surface.

---

## 5. The single highest-leverage change (deliverable 3)

**Stop using the per-hypothesis-deflated GP mass-miner as the discovery instrument. If any discovery program
continues, run low-N, pre-registered, mechanism-driven single-hypothesis probes on the full sample, scored by
one marginal-effect statistic on the ΔSR/CPCV distribution at t ≥ 2.33 with a binding (not
accounting-only) LORD++ account.**

- **Power:** at true ΔSR 0.5, per-candidate power rises from ≈ 0.00 (shipped) to ≈ 0.37 full-panel
  (T=4044); at 0.8 → ≈ 0.81. This is precisely the instrument that already worked on this project (the arb
  sweeps detected gross SR 1.4–2.1 at power ≈ 1.0 and failed only on costs).
- **E1 stays green:** one-sided t≥2.33 = 1% per-candidate FPR by construction; retained guards (uplift,
  fragility, collinearity) cut the joint null rate to ≈ 0.2%; the per-tick ceiling is held by making the
  already-existing LORD++ account binding at promotion (it currently gates nothing).
- **CRU-1 cost:** implemented as a **parallel pathway with its own gates file** (the lockbox/cohort ADR
  precedent) it is **MINOR** — the frozen funnel and its verdicts are untouched. Replacing the funnel gate
  in `signal_eval.gates.yaml` would be a verdict-changing **MAJOR**.
- **Prerequisites (non-negotiable):** fix the F14 anti-conservative defects first (overlay cost gross
  mismatch, short-tilt rebate, dsr N_eff inconsistency, uplift null tail), and re-run E1 on a realistic
  null (fat tails, vol clustering, common factor) because today's FPR protection partly rides on the very
  legs being replaced. Do **not** execute the internal roadmap's NEXT-2 (promote `marginal_t` to the single
  binding leg) as written — it would inherit F1.
- **Honesty clause:** even this contract has MDE80 ≈ 0.7–0.9 on these substrates. It makes the instrument
  honest; it does not make 0.3–0.5 edges detectable on 11–16 years of free daily data. The power problem is
  ultimately a **data problem** (breadth, frequency-with-capacity, niche datasets), which is a strategy
  decision, not a statistics fix.

**If the platform is retired instead (the internal recommendation):** the audit endorses retirement of the
*mass-mining contract*, with salvage of the genuinely good components — the PIT-clean connectors, the
as-of-join + canonical checker, the pre-registration discipline, the ledger, and the (repaired) probe
harness — for pre-registered probes in service of the TSMOM→paper path.

## 6. Future directions, ranked (deliverable 5)

1. **Redeploy effort to the TSMOM→paper path now.** Correct on this audit's own numbers, independent of any
   Crucible fix.
2. **If discovery continues: the pre-registered probe pathway of §5** on the existing rails (MINOR).
3. **Target different data, not better statistics:** per-asset/cross-sectional breadth (the N-axis the
   current substrate wastes — per-asset T86 flows already ingested but unposable), capacity-constrained
   niches per the small-operator reframe. An expressiveness fix (per-asset overlay routing, conditional
   operators) is the highest-leverage *platform* change for Taiwan specifically — but only worth it after
   (2), and it re-opens E1.
4. **Bayesian shrinkage as a reporting layer** (hierarchical prior over mechanism families): converts
   "0 PROMISING" into calibrated effect-size posteriors for stop/continue decisions. Cheap; adds no
   detection power (same √T wall) but ends the uninformative-zero problem.
5. **Cohort/ensemble enable:** honest math, MINOR flip, but currently pointless — the analytic floor reuses
   the defective legs and short-circuits before the lenient MC null, and the supply of decorrelated genuine
   mechanisms is K ≈ 1–2 (needs ≥ 3). Same √k wall that closed cross-market pooling.
6. **Sequential re-testing / dedup epochs:** fix one-shot burn for ledger honesty, not power (a 0.5 edge
   needs ~22–40 more years to pass even t≥2.33).
7. **Correct the record** (independent of any decision): mark ledger `delta_sr_oos` as train-split; retract
   or re-caption the "best OOS ΔSR 0.66" claim and the "37 mechanism-diverse" characterization (cont-129);
   surface the intraday TA-3 tripwire failure in the Stage-0 result doc; register the Taiwan gates-file
   hash under CRU-1; fix the killed-families writer or delete the claim.

---

## 7. Reproduction index

All scripts under the session scratchpad (`…\scratchpad\`); every ledger claim is a one-line sqlite query
against `results/crucible_orchestrator*/**/trial_ledger.db`.

| Claim | Repro |
|---|---|
| Perfect weekly oracle 0/5; train base SR −0.007 | `planted_sweep.py` (re-run by lead auditor, output verified) |
| Sign-inverted marginal_t on cross_asset + overlay-bridge | `SELECT delta_sr_oos, marginal_hlz_t FROM trial_ledger WHERE verdict='LOGGED'` on `real/` and `_overlay/real/` |
| Taiwan dsr seal (max 0.049 / 3.9e-05 / 6.6e-06) | same query on the three taiwan ledgers |
| marg = w_c(r_c − b_base) identity; scale-dependence | skeptic scratch `phase2/` scripts |
| Fallback-pool decomposition; 0/190 dsr-only failures | `pool_dispersion_probe.py`, `e2_dsr_decompose.py`, ledger group-by |
| Matched-null HoF ΔSR 0.80–0.94, t ≤ 2.93 | `phase2/null_grid_sim.py`, `null_max_sim.py` |
| MDE floor law 0.39 + c/√N; per-leg binding by T | `perleg_floor_diag.py`, `offset_mechanism.py` |
| Deployed holdout MDE 3.0–3.5 at T=1011 | `perleg_floor_diag.py` T=1011 rows |
| Duplicate rate / affine invariance | `overlay_degeneracy_check.py` + ledger exact-ΔSR group-by |
| Cost consistency; overlay gross mismatch; tilt rebate | `cost_audit.py`, `base_cost_audit.py`, `ls_vol_audit.py` |
| Killed-families zero writers; FDR accounting-only; lockboxes empty | grep + ledger/lockbox counts (verified by lead auditor) |
| Intraday TA-3 tripwire failure | `results/crucible_intraday_power/intraday_power.json` `tripwires.TA3_detail`, `A1_verdict` |

---

*Audit method note: 9 finder passes + 3 refute-mandated skeptic passes; the lead auditor independently
re-verified every S1 empirical claim (ledger queries, code reads, and a full re-run of the planted-oracle
experiment) before accepting it. One finder (analytic-power) was lost to an infrastructure error; its
mandate was covered by the gate-math finder and the lead auditor's own derivations.*

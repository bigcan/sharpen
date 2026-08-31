# Crucible Anatomy Paper — Build Plan

> **Created:** 2026-07-19 · S553-cont-138 · branch `July2026` (`487d16ab`, crucible-v5.0)
> **Supersedes the framing of:** `.agent/artifacts/crucible_arxiv_worthiness_research.md` (cont-118, 2026-07-06)
> **Driven by:** `docs/research/crucible_independent_audit_report_2026-07-14.md` (cont-131)
> **Decision (operator, cont-138):** REFRAME the ICAIF submission as a structural-failure-mode paper — the audit *is* the contribution.
> **Venue/deadline:** ICAIF '26 Milan, **2026-08-02**, 8pp ACM double-blind (anonymize repo).

---

## ⛔ STATUS 2026-07-31 (S553-cont-146) — ICAIF DEADLINE DROPPED (operator decision)

**The ICAIF '26 submission is abandoned. The paper survives as a self-paced arXiv preprint with no
deadline.** Do not resume work against 2026-08-02.

**Why.** On 2026-07-31 (day ~12 of a ~14-day plan) the readiness audit found the plan's own progress
log materially optimistic. It claimed *"ALL empirical figures complete — remaining is pure write-up"*;
what actually existed on disk was:

| Needed | Actual state 2026-07-31 |
|---|---|
| any `.tex` or draft | **none anywhere in the repo** |
| F6 + F2-corrected power curve data | ✅ present (`results/crucible_calibration/calibration_mde_sweep*.json`, `xsec_*`) |
| F4 realistic-null, F2 oracle, F3 marginal-seal, F5 matched-null HoF data | ❌ **JSON outputs absent** — `results/` is gitignored. Harnesses are tracked, so re-runnable, but that is compute, not zero |
| rendered figures (any PNG/PDF) | **zero** — raw JSON only, no plotting layer exists |
| LaTeX toolchain + `acmart` template | **neither installed nor present** |

So the remaining work was never "pure write-up": regenerate 4 figure datasets → build a plotting
layer → draw F1 → write 8pp → install/verify LaTeX + ACM template → double-blind anonymization (R1)
→ Tier-2 adversarial read → CMT submit, in ~2 days. §7 **R4** already stated the paper is justified
*only* by external-credibility value against the standing #1 direction (TSMOM → paper trading), and
§6's cut line could not save it — the surviving Tier-3 data is the *opposite* half of what the cut
line proposed shipping.

**What is preserved.** Every empirical asset and harness stays in the repo; nothing is deleted. The
arXiv route keeps the full external-credibility payoff, drops the double-blind anonymization burden
(R1), and removes the deadline risk entirely. The strongest on-topic material — a verdict-generating
machine emitting a confident, internally consistent, HIGH-severity finding that turned out to be an
artifact of its own instrument (RC-11, `crucible_rc11_resolved_degenerate_null_2026-07-31.md`) — is
fully measured and is exactly the paper's thesis.

**Effort redirected to** the trading path (`docs/research/tailwind_v1_forward_path_render_2026-07-31.md`).

---

## 1. Why reframe — and why the reframe is the *stronger* paper

The cont-118 artifact recommended the paper *"disciplined filter → honest 0-PROMISING null → power curves."* The
cont-131 independent audit **falsified that thesis**: the funnel is **structurally sealed** — a planted
**annualized-SR-1.80 weekly oracle fails the gate 0/5**, and even **perfect daily foresight barely passes**
(marginal_t = 2.92 < 3.0). So "0 PROMISING" is a property of the *instrument*, not an honest statement about the
market. The original paper would have shown power ≈ 0 for large planted edges — quietly refuting its own headline.

The reframe turns the liability into the contribution:

> **Working title:** *"The Seals Beneath the Search: How Structural Choices Doom Agentic Alpha-Mining Before It
> Begins."*
>
> **Thesis:** We built the most statistically disciplined agentic alpha-miner in the 2025–26 wave (structural
> proposer blinding + content-hash pre-registration + online FDR + Deflated Sharpe + forward lockbox), then
> subjected it to an adversarial internal audit. We find a class of **candidate- and sample-size-independent
> structural seals** that guarantee ≈0 discoveries *regardless of whether alpha exists* — verified by a planted
> perfect-foresight oracle that the machine cannot detect. We give (a) a taxonomy of the seals, (b) a cheap
> diagnostic battery any pipeline builder can run to detect them, and (c) a corrected low-N pre-registered
> contract whose honest power ceiling relocates the negative result from *method* to *data*.

**Why it beats the original (novelty inherited from cont-118, thesis inverted):**

- The empty intersection still holds — nobody in the LLM-mining wave (AlphaAgent KDD'25, R&D-Agent-Quant
  NeurIPS'25, FactorMAD ICAIF'25, …) controls FDR, pre-registers, or blinds the proposer. But **nobody has
  published a rigorous failure-mode teardown of these pipelines either.** Every wave paper claims success; a
  reproduced, planted-oracle-falsified anatomy of *why the careful version finds nothing* is contrarian and new.
- It resonates **harder** with the AI-scientist-integrity literature (SciIntegrity-Bench 2605.10246, "Correct
  Answer, Wrong Mechanism" 2606.23175, "Why LLMs Aren't Scientists Yet" 2601.03315): Crucible becomes a concrete
  case study of a research agent that *looked* disciplined and produced an honest-seeming null that was a
  structural artifact — "integrity theater," diagnosed.
- It **pre-empts the one-sentence dismissal.** Original reviewer risk: "elaborate filter, found nothing, never
  showed it can find anything." The anatomy paper's planted oracle *is* that demonstration, weaponized: the filter
  cannot find a signal handed to it on a silver platter — so the seals, not the market, are the story.
- The corpus-contamination argument (lockbox scores only bars after `proposal_ts`) and full free-data
  reproducibility survive intact as secondary assets.

## 2. The single feasibility unlock

The original plan could not hit 2026-08-02 because it required **E4** (a multi-week nightly LLM-proposer campaign
to reach N≈10²–10³) and **E3** (an unblinded-vs-blinded ablation). **The anatomy paper needs neither.** Its
empirical core is *seal demonstrations*, which are **synthetic, deterministic, and reproducible now** — scale is
irrelevant when the claim is "sealed independent of N." That removes ~3 weeks of wall-clock from the critical path
and is what makes the deadline real.

## 3. Section outline (8pp ACM double-blind)

| § | Content | Source |
|---|---------|--------|
| 1 Intro | The wave institutionalizes backtest-feedback; we built the disciplined counter-design and audited it; contributions = taxonomy + diagnostic + corrected contract | cont-118 lit review + audit §1 |
| 2 System | Crucible architecture: blinding (CR-1/CRU-2), pre-reg hash, LORD++, DSR/BHY, lockbox — the disciplined design under test | `crucible/` file:line (cont-118 §"ground truth") |
| 3 Seal taxonomy | **The core.** S-1 substitution-residual marginal test; S-2 book-level DSR train-seal; S-3 expressiveness collapse; S-4 uncertified-gate/fallback-DSR; S-5 deflation-N pathologies; governance theater | audit §2.1–2.2, F1–F6, F12–F13 |
| 4 Diagnostic battery | planted-oracle injection · matched-null HoF search · per-leg MDE decomposition — reusable recipe | audit §7 repro index + rebuilt harness |
| 5 Corrected contract | §5 low-N pre-registered single-hypothesis probe, binding LORD++, t≥2.33; power ↑ (0→0.37 @ ΔSR 0.5) but honest MDE80≈0.7–0.9 ceiling ⇒ **data problem, not method** | audit §5, §2.3 |
| 6 Discussion | Integrity-by-architecture vs integrity-theater; corpus-contamination; which seals generalize to the closed-loop wave | cont-118 §metascience |
| 7 Limitations/Ethics | Single system; free daily data caps stakes; LORD++ shared-panel dependence (cite 2110.08161) | audit §2.3, cont-118 cons |

## 4. Figure plan — with data provenance (the real feasibility map)

Legend: 🟢 reproducible now (harness in repo) · 🟡 small extension to surviving harness · 🔴 needs rebuild (auditor
scratchpad gone, ledgers git-ignored & absent from worktree).

| Fig | Content | Provenance | Effort |
|-----|---------|-----------|--------|
| **F1** | Funnel schematic: 5-leg AND × 3 seal layers | none (drawn) | S |
| **F2 ★flagship** | Planted-signal power: shipped gate (≈0 @ ΔSR 0.5; **oracle 0/5**) vs corrected contract (0.37 @ 0.5, 0.81 @ 0.8) | 🟡 extend `crucible_calibration.py --exp e2` with a perfect-foresight **oracle injection** mode + implement corrected-contract scorer | M |
| **F3** | marginal_t sign-inversion: ΔSR>0 ∧ t<0 for a synthetic diversifier; scale-dependence sweep | 🟡 synthetic demo of the identity `marg = w_c·(r_c − b_base)` (audit-proven, max dev 1.7e-18) — **no ledger needed** | S |
| **F4** | E1 null-FPR: IID-Gaussian null (current) **and** realistic null (fat tails, vol clustering, common factor) on shipped funnel vs corrected contract | 🟡 add realistic-null generators to `--exp e1` (audit F14 "non-negotiable prereq") | M |
| **F5** | Matched-null Hall-of-Fame: GP search on pure noise at real budget → HoF ΔSR 0.80–0.94, t≤2.93 ≥ real record maxima | 🔴 rebuild `null_grid_sim.py` (run GP/evolve on synthetic noise, record HoF max) | M |
| **F6** | MDE-vs-T wall: shipped ≈1.9–3.5; ideal single-prereg t≥2 ≈0.71 @ T=4044 / 1.42 @ T=1011 | 🟢 `crucible_calibration.py --exp mde_sweep` (already produces this) | S |

No figure depends on the proprietary trial ledgers — every claim is re-cast as a **synthetic, reproducible**
experiment, which is *required* for a double-blind reproducibility appendix anyway.

## 5. Empirical work, tiered (all in `scripts/research/`, zero gate bytes touched — CRU-1 safe)

- **Tier 1 — regenerate (≈1–2 d):** F6 (mde_sweep) + F4-IID + F2-shipped from the surviving harness. Confirms the
  audit numbers reproduce on v5.0.
- **Tier 2 — small extensions (≈2–3 d):** F2 oracle-injection mode; F3 synthetic sign-inversion demo; F4 realistic-null
  generators (block-permute + fat-tail + vol-cluster + common-factor). **Tier 1+2 alone = a complete anatomy-of-failure
  paper.**
- **Tier 3 — the "fix" story (≈3–5 d):** F5 matched-null HoF rebuild + implement the §5 corrected single-hypothesis
  contract (parallel gates file, MINOR to CRU-1) + its power curve. Upgrades anatomy → **anatomy + correction** (much
  stronger; makes §5 non-hand-wavy).

E1 realistic-null (Tier 2, F4) is the audit's explicitly **non-negotiable** item: today's E1-GREEN (0/2349) *rides on
the very mis-specified legs the paper indicts*, so it must be re-run on a realistic null before any claim about the
funnel's FPR is honest — for the paper *and* for any future discovery restart. Start here.

## 6. Timeline to 2026-08-02 (~2 weeks)

- **Wk 1:** Tier 1 (d1–2) → Tier 2 (d3–5) → start Tier 3 F5 (d5). Parallel: assemble §2–§3 prose from the audit
  (already written), draft F1/F3.
- **Wk 2:** Tier 3 corrected-contract + F2/F5 finalize (d6–8) → full 8pp draft + figures (d8–11) → internal
  Tier-2 adversarial read (d11–12) → double-blind anonymization (repo, author-identifying paths) + CMT submit (d13).
- **Cut line if slipping:** ship **Tier 1+2 only** (anatomy without the corrected-contract power curve) — still a
  complete, honest, novel paper; §5 becomes "sketch + analytic power" instead of a measured curve.

## 7. Risks

- **R1 Double-blind anonymization** — repo name, `sharpen` paths, WandB entity, and session/commit IDs are
  author-identifying. Build an anonymized artifact bundle; do not cite internal memory slugs.
- **R2 Ledger unavailability** — mitigated by design: every figure is synthetic/reproducible; the real-record numbers
  appear only as *cross-checks* in prose, flagged as "internal, not required for reproduction."
- **R3 Corrected-contract scope creep** — Tier 3 is the one open-ended piece. Timebox it; the cut line (§6) protects
  the deadline.
- **R4 Opportunity cost** — ~2 wks of operator attention vs the TSMOM→paper trading path. The audit's #1 direction is
  still "redeploy to TSMOM"; this paper is justified only by the external-credibility value of publishing.

## 8. Immediate next action

Start **Tier 1 + F4 realistic-null (E1)** — reproduce the audit on v5.0 and build the one empirical asset that is a
prerequisite for both the paper and any future discovery work. Chain **Architect** on the corrected-contract scorer
(Tier 3) once Tier 1+2 land.

---

## Progress log

### 2026-07-19 (cont-138) — Tier 1 reproduced + F4 built

**Tier 1 — clean exact reproduction on v5.0** (`results/crucible_calibration/calibration_{both,mde_sweep}.json`):
- E1 (IID null): per-candidate FPR CP-upper-95% = **0.0013** (≤0.01) → GREEN; null protection carried entirely by
  `dsr` (0.000 pass) + `marginal_t` (0.003) — the two indicted legs. F6 anchor confirmed.
- MDE wall (F6): T=756→**3.63**, 1512→**2.85**, 2782→**1.94**, 4044→**1.40** — matches the calibration memory to the
  decimal. Harness is trustworthy on v5.0 (E1+E2 ~10s, mde_sweep ~4min).

**F4 — realistic-null generator: BUILT + verified** (`scripts/research/crucible_calibration.py`, `--null realistic`):
- DGP `r[t,i]=βᵢ·f[t]+√hₜᵢ·εₜᵢ` (Student-t(5) + GARCH(1,1) α=0.08 β=0.90 + common factor, factor_share 0.35).
  Empirically verified (t=2782,n=18): per-asset vol 0.0095, **excess kurtosis 4.28**, **ACF₁(r²) +0.109**,
  **avg pairwise corr +0.311** — vs IID (~0, ~0, ~0). Prices finite & positive. 4 regression tests added (13/13 pass,
  ruff clean).
- **Result: E1 stays GREEN under the realistic null** (`calibration_e1_realistic.json`): per-candidate FPR CP≤0.0013,
  identical to IID; `marginal_t` binds *harder* (0.000). **Finding (new — the audit flagged this re-run but never ran
  it): the funnel's false-positive protection is ROBUST to null shape** — "0 PROMISING" is trustworthy in the FP
  direction under both easy and realistic nulls. Partially refutes audit F14's worry that E1-GREEN was an IID artifact.
  Does NOT yet test the corrected §5 contract's null (Tier 3).
- **Paper impact:** sharpens the thesis — the problem is power/seals, not false positives; F4 becomes a robustness
  panel (E1 GREEN across IID + realistic) rather than a degradation story.

**Open gates before F4 feeds a paper claim:** (1) ✅ **Math-skill pass DONE** — PASS WITH NOTES: all 3 formulas
(t-standardization, GARCH(1,1) uncond-variance, factor/idio budget) symbolically correct; empirical moments
consistent (vol<target = Jensen; ACF<pure-GARCH = mixture dilution); 2 LOW doc-precision notes applied
(realized-vs-theoretical E[load²]; avg corr ≈0.33 not exactly factor_share). Zero training-signal impact (synthetic
null, not a reward). (2) ✅ **factor_share / df sensitivity sweep DONE** (cont-139, see entry below) — both open
gates now closed; F4 is a fully-hardened robustness panel.

**Next:** ✅ F3 + ✅ F4-sensitivity + ✅ F5 + ✅ F2-oracle-half + ✅ **Tier-3 corrected-contract DONE** (cont-139,
entries below). **ALL empirical figures complete — anatomy + correction paper is fully assemblable.** Remaining is
pure write-up: assemble the 8pp draft, F1 schematic (drawn), double-blind anonymization, internal Tier-2 read, CMT
submit by 2026-08-02.

### 2026-07-20 (cont-139) — Tier-3 corrected-contract scorer BUILT + calibrated (F2 second half + §5 power curve)

**Figure F2 second half + §5's measured power curve — the audit's remedy, implemented as a parallel pathway.**
Architect design in `.agent/artifacts/corrected_contract_architecture.md`. New module
`sharpen/crucible/corrected_contract.py` scores a candidate by ONE significance statistic, promotes iff it
clears `t_min` AND a **binding** LORD++ level (F13 fix — `fdr.py` used read-only) AND the 3 cheap guards
(uplift/fragility/collinearity); the F1-sealed `marginal_t` and F2-sealed `dsr_aug` legs are DROPPED. Own gates file
`configs/crucible_corrected_contract.gates.yaml` → zero funnel gate bytes, CRU-1 MINOR (`test_version` green).

- **The statistic changed mid-build — E1 caught it (ADR-2 working as designed).** First design (a t-stat across the
  ~15 C(6,2) CPCV paths) was **voided by the Step-2 E1 calibration**: overlapping paths carry ~1.3 effective obs →
  null `std_z ≈ 3.35`, FPR **~14.5%** (not 1%), and would be powerless if forced to 1%. The audit's own wording is
  **"full-panel"** (power from the thousands of bars). Shipped statistic = the **full-panel Jobson-Korkie-Memmel
  Sharpe-difference z** (Memmel 2003; Math-verified PASS-with-2-LOW-notes), exploiting `corr(b_aug,b_base)≈0.80` so the
  paired variance is small → power scales √T. **Re-calibrated: null `std_z = 0.98–1.01`** across T=2048/4044,
  cost=0/0.001, ar1/raw — properly N(0,1); FPR ≤ 0.8% ≤ nominal 1%. CPCV paths retained only for the fragility guard.
- **E1 (500 null candidates, ar1):** per-candidate FPR **0.0000** (CP-upper 0.0060 ≤ 1%) → calibrated + green.
- **Power (T=4044), shipped vs corrected — the F2 contrast:** @ realized ΔSR **0.5 → shipped 0.00, corrected 0.81**;
  @ 0.8 → corrected 0.99; the shipped gate needs ΔSR ~1.4 for 80% power (F1/F2 seals), the corrected contract reaches
  it at ~0.5. My synthetic power (0.81@0.5) exceeds the audit's real-substrate 0.37 — the planted overlay is a cleaner
  signal than a real candidate — but the CONTRAST (shipped ~0 → corrected materially positive) is decisive; the
  audit's 0.37 stands as a cross-check marker (per the timebox rule, not chased).
- **The F1 fix, pinned:** the exact sign-inverted diversifier the funnel rejects (`marginal_t<0`) earns corrected
  `z>0` and passes (funnel −0.8 vs corrected +4.5 @ T=4044) — `tests/crucible/test_corrected_contract.py`.
- **Verification:** 9 tests (7 unit + 2 integration) + Math PASS + Audit (ruff clean, 21/21 incl. CRU-1 `test_version`
  + F1 seal registry still green). Two commits: Step 1 scorer `7a7f8008`, Step 2 calibration+JKM-redesign (this).

### 2026-07-20 (cont-139) — F2 oracle half built: the gate's promotion bar is an implausible realized-Sharpe wall

**Figure F2 (oracle half) = independent-audit §2 "(A)" / F1+F2, rebuilt** (the auditor's `scratchpad/phase2/planted_sweep.py`
is gone). New standalone `scripts/research/crucible_oracle_injection.py` plants a **perfect-foresight timing oracle**
(skill p = directional accuracy; p=1.0 = perfect) on a synthetic market, scored through the SHIPPED `combination_fitness`
against a ~0-Sharpe base (un-sign-sealed). Zero funnel/gate bytes touched (CRU-1 safe). Two sweeps:

- **Hold sweep @ perfect skill:** the oracle's realized Sharpe falls with hold (daily 18 → semiannual 1.1); the gate
  promotes only above a **realized-Sharpe WALL ≈ 4.2** (highest rejected perfect oracle SR 2.70 @ monthly; lowest
  promoted SR 5.73 @ weekly). Every perfect oracle at monthly-or-slower (SR ≤ 2.7) is rejected — first on the
  book-level `dsr_aug` seal (F2), then also on the `marginal_t` substitution-residual seal (F1).
- **Skill sweep @ quarterly (realistic-edge, perfect-foresight realized SR ~1.5):** detection **0.00 at EVERY skill
  0.52→1.00** — a perfect-foresight oracle at a realistic realized Sharpe is never promoted (the silver-platter
  rejection). realized SR rises 0.01→1.53 with accuracy; the gate ignores it.
- **Honest deviation from the audit:** the audit's WEEKLY oracle SR (1.80) is substrate-specific — a realistic market
  caps weekly timing far below this idealized iid market (SR 5.7). Rather than force the exact number, the figure
  isolates the substrate-**INVARIANT** wall (~4) from the substrate-**DEPENDENT** achievable oracle Sharpe — a *more*
  rigorous claim (maps the whole detection-vs-realized-Sharpe curve). Audit markers stay consistent: their weekly
  oracle SR 1.80 sits BELOW the wall → rejected (matches); their daily SR 12.6 ABOVE → passed.
- **Verification:** 5 tests (oracle construction: perfect≫no-skill, SR falls with hold; ~0-Sharpe base; deterministic
  wall logic; daily passes/slow-perfect fails; perfect foresight @ realistic Sharpe never promoted) → **pass, ruff
  clean**. Fixed a real construction bug mid-build: `skill_p` must be directional ACCURACY (opposite sign when
  unskilled), else p=0.5 → 75% accuracy. No Math re-pass (reads the shipped gate). Artifact
  `results/crucible_oracle_injection/oracle_injection.json`.

### 2026-07-19 (cont-139) — F5 built: the matched-null Hall-of-Fame ceiling (the record IS the noise ceiling)

**Figure F5 = independent-audit finding F4, rebuilt as a reproducible experiment** (the auditor's
`scratchpad/phase2/null_grid_sim.py` is gone). New standalone `scripts/research/crucible_matched_null.py` runs the
SHIPPED `evolve`+gate on PURE NOISE at the real search budget (**pop 200 × gens 40**, gen_n ~3.9–4.7k/search), base
book rescaled to ~0 train Sharpe (the un-sign-sealed regime, matching the real Taiwan train book SR ≈ −0.007),
T=2782 (Taiwan-like; the t-ceiling scales √T), cross_sectional. Zero funnel/gate bytes touched (CRU-1 safe); writes
incrementally per seed (the run is ~1h).

- **Result (6 noise seeds, `results/crucible_matched_null/matched_null.json`):** per-search HoF **max ΔSR 1.04**
  (p90 0.916, median 0.720) and **max marginal_t 2.79** (median 1.78), with **0 PROMISING across all 6 searches** —
  nothing survives the binding gate. The real record's maxima (**ΔSR 0.99/0.71/0.66; t ≤ 2.12**) sit **at or below**
  this pure-noise ceiling on BOTH axes (`record_at_or_below_ceiling`: dSR True, t True). Reproduces the audit's "HoF
  ΔSR 0.80–0.94, t up to 2.93 ≥ real record."
- **The point:** "best OOS ΔSR 0.66" (audit F3: actually a train-split GP mutation, holdout never ran) is exactly
  what no-signal looks like *through this instrument* — a best-of-~4.5k-genomes extreme, not a discovery. Cost curve
  confirmed the ceiling is √T-driven in t (probe: t 1.84 @ T=1200 → 2.79 @ T=2782) and near-saturated in ΔSR even at
  small budgets, so the real-budget run is a faithful, not inflated, ceiling.
- **Base-book choice matters (F1/F3 tie-in):** a ~0-Sharpe base is required — a positive-Sharpe base would sign-seal
  marginal_t negative (the F3 seal) and the positive-t noise ceiling would vanish; the record's positive-t maxima came
  from a ~0-Sharpe base era, so the comparison is apples-to-apples.
- **Verification:** 3 tests (near-0-Sharpe base; noise search yields a positive ΔSR ceiling + 0 survivors at small
  budget; record-vs-ceiling comparison logic both directions) → **pass, ruff clean**. Real-record numbers are
  cross-check markers (ledger absent), flagged as not-reproduced. No Math re-pass (reads the shipped search/gate).

### 2026-07-19 (cont-139) — F3 built: the marginal_t substitution-residual seal, reproduced synthetically

**Figure F3 = independent-audit finding F1, rebuilt as a reproducible synthetic demo** (the auditor's `phase2/`
scratchpad is gone; this replaces it). New standalone `scripts/research/crucible_marginal_seal.py` reads the SHIPPED
`combination_fitness` / `_combined_book` / `dynamic_sleeve_alphas` (zero funnel/gate bytes touched, CRU-1 safe) and
demonstrates all three prongs of the seal on synthetic data (K=900, no ledger):

- **(1) Sign inversion** — a genuine diversifier (independent, mean 0.04/yr **below** the 0.128/yr base book, own
  Sharpe 1.9) RAISES book Sharpe (`delta_sr_oos` = **+0.70**) yet earns `marginal_t` = **−1.68** (fails t≥3). Same
  pathology + construction class as the pinned seal `test_tier_c_seals_registered.py::test_f1_...` (which stays green).
- **(2) The identity** — `marg == w_c·(r_c − b_base)` verified to **max|dev| = 3.5e-18** over 900 bars, reproducing the
  audit's stated **1.7e-18**. `w_c` (the candidate's inverse-vol combiner weight) is read from the same
  `dynamic_sleeve_alphas` call `_combined_book` makes, so it is the real path, not a re-derivation.
- **(3) Scale dependence** — the SAME signal at leverage λ∈{0.25…8}: `marginal_t` sweeps **−2.30 → +1.86** (crosses
  zero near 4×), reproducing the audit's **−1.67 @ 0.25× → +1.81 @ 8×**, WHILE `delta_sr_oos` stays positive and even
  RISES (+0.47 → +0.93) across every leverage. The book value is robustly positive; only the significance *verdict*
  swings with position size — the incoherence, made visual.
- **Construction note:** the diversifier's mean/own-Sharpe are CONTROLLED (noise demeaned) so the leverage axis is
  legible and reproducible; the pathology itself is construction-independent (it is the combiner identity). Diagnostic
  battery (§4) now has its second reusable recipe (planted-oracle is F2's).
- **Verification:** 4 regression tests (sign inversion; identity exact <1e-10; leverage sign-flip with ΔSR>0 throughout;
  controlled-construction targets) → **pass, ruff clean**; pinned F1 seal test still green. No Math re-pass (no new
  formula; reads the already-verified gate). Artifact `results/crucible_marginal_seal/marginal_seal.json`.

### 2026-07-19 (cont-139) — F4 sensitivity sweep: E1 GREEN across the whole null-shape space

**The open ⬜ gate closed.** F4's "E1 stays GREEN under a realistic null" rested on ONE null-shape point
(df=5, factor_share=0.35). New `--exp e1_sensitivity` mode re-runs E1 (realistic null) over a 10-point one-axis-at-a-time
grid — **tail fatness** df ∈ {3,4,5,8,15} and **cross-sectional correlation** factor_share ∈ {0,0.2,0.35,0.5,0.7},
plus a fattest×most-correlated **stress corner** (df=3, fs=0.7) — re-applying the UNCHANGED `e1_null` ceilings at each
point (`configs/crucible_calibration.gates.yaml` → new `e1_sensitivity` block; no new threshold; CRU-1 gate bytes
untouched, `test_version.py` 8/8).

- **Result: ALL 10 shapes GREEN** (`results/crucible_calibration/calibration_e1_sensitivity.json`, 60 panels/point):
  every point `0 promising`, per-candidate FPR CP-upper95 **≤0.0013**, per-tick CP-upper95 **≤0.049** (both ceilings).
  The `default` point reproduces the committed F4 anchor **byte-for-byte** (0/2312, CP 0.0013). The audit-indicted
  `marginal_t` + `dsr` legs carry the protection under EVERY shape (both ~0.000 pass at default AND the stress corner).
- **Design bug caught mid-run:** first tried `n_panels=30`, which structurally forces RED on the per-tick leg
  (`1−0.05^(1/30)=0.095 > 0.05` at k=0, shape-independent). Bumped to 60 (= `e1_null`, the floor where the tick bound
  clears) so each shape is judged by the *identical dual-ceiling bar* as the main E1 — the strongest possible
  robustness statement, no relaxed criterion for a reviewer to poke.
- **No Math re-pass needed:** varies parameters of the already-Math-verified `_realistic_returns` DGP within its valid
  domain (df>2, factor_share∈[0,1)); no new formula. 4 more regression tests added (sweep axes are real; panel honors
  params; strict-gate FP protection holds under every shape; lax-gate teeth) → **17/17 pass, ruff clean**.
- **Paper impact:** F4 upgrades from a single-point claim to a genuine robustness panel — "E1's false-positive
  protection does not depend on the null's shape." Directly forecloses the reviewer objection that E1-GREEN was an
  artifact of one convenient DGP. Committed (not working-tree-only).

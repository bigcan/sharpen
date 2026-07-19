# Crucible Anatomy Paper — Build Plan

> **Created:** 2026-07-19 · S553-cont-138 · branch `July2026` (`487d16ab`, crucible-v5.0)
> **Supersedes the framing of:** `.agent/artifacts/crucible_arxiv_worthiness_research.md` (cont-118, 2026-07-06)
> **Driven by:** `docs/research/crucible_independent_audit_report_2026-07-14.md` (cont-131)
> **Decision (operator, cont-138):** REFRAME the ICAIF submission as a structural-failure-mode paper — the audit *is* the contribution.
> **Venue/deadline:** ICAIF '26 Milan, **2026-08-02**, 8pp ACM double-blind (anonymize repo).

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

- **R1 Double-blind anonymization** — repo name, `finrl_pro_ds` paths, WandB entity, and session/commit IDs are
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
null, not a reward). (2) ⬜ sweep factor_share / df sensitivity (currently single default point). **Not committed** —
working tree only.

**Next:** F3 (marginal_t sign-inversion synthetic demo — no ledger needed) + F2 (oracle-injection mode) + F5 (matched-null
HoF rebuild), then Tier 3 corrected-contract scorer (chain Architect).

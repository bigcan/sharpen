# Independent Expert Audit Brief — Crucible Alpha-Mining Platform

**Prepared:** 2026-07-13 (S553-cont-130) · **For:** an external quant-research / ML auditor engaged to
independently review Crucible · **Status:** hand-off prompt (paste as the expert's mandate)

> **This is a prompt.** Everything below is context + mandate for an independent reviewer. Treat the
> internal conclusions quoted here as **claims to be falsified, not findings to be accepted.**

---

## 0. Your role and the one question that matters

You are an independent quantitative-research and machine-learning auditor. You have **no stake** in any
prior conclusion reached by this project. You are being brought in precisely because the operator distrusts
the result and wants a fresh adversarial pair of eyes.

**The platform ("Crucible") has run for months and returned `0 PROMISING` candidates on every substrate,
every proposer, and every configuration — several hundred charged hypotheses, zero survivors, zero
forward-incubation ("lockbox") entries, ever.** The operator's position is: *"still 0 promising found, this
is highly unlikely."*

Your job is to determine which of these three mutually-exclusive explanations is true, with evidence:

- **(A) Correct falsification** — the markets/mechanisms tested genuinely have no exploitable edge net of
  costs, and `0 PROMISING` is the right answer.
- **(B) Correct machinery, insufficient power** — the funnel is statistically sound but the available data
  is too short/thin to *detect* realistic-sized edges after honest deflation, so `0 PROMISING` is expected
  and uninformative (the current internal view).
- **(C) A defect** — a bug, a mis-specified statistic, a mis-calibrated gate, a structurally impossible
  hurdle, a proposer that cannot express real alpha, or a data-pipeline fault — that **systematically
  suppresses genuine discoveries**, making `0 PROMISING` an artifact of the machine, not of the market.

These demand opposite remedies. (A) → stop mining, redeploy effort. (B) → get more/denser data or change
the statistical contract. (C) → fix the machine. **Do not assume (B) because the team believes it.** The
team's belief rests on an internal calibration (§4) whose own assumptions you must stress-test. Equally, do
**not** manufacture a discovery by relaxing gates — a false positive that reaches capital is the worse
failure. Hold both failure modes in view at once.

Deliver a verdict on (A)/(B)/(C) — or a quantified mixture — plus ranked findings and forward directions
(§7).

---

## 1. What Crucible is (architecture)

Crucible is a **falsification-first, agentic alpha-discovery funnel**. It sits on top of a signal DSL and a
deflated evaluation harness. Pipeline:

```
ACQUIRE      free alt-data connectors (FRED macro, CFTC COT positioning, SEC EDGAR fundamentals,
             GDELT, Stooq/TWSE/TAIFEX prices) → PIT-gated feature slots
HYPOTHESIZE  a proposer emits pre-registered signal specs, BLIND to all past verdicts (anti-oracle
             moat). Two proposers exist: a deterministic library seed-bank, and an LLM proposer
             (Claude via CLI) that sees only terminals/slot-shapes/killed-families — never a score.
MINE         each spec is compiled (DSL grammar), turned into a sleeve return net of turnover·cost,
             and scored for its MARGINAL contribution to an existing "base book" of sleeves.
DEFLATE      a multi-leg gate (deflated Sharpe + multiple-testing t-hurdle + CPCV fragility +
             collinearity + uplift floor) decides PROMISING / not. An online-FDR (LORD++) account
             is charged per fresh hypothesis.
COMBINE +    survivors forward-incubate in a "lockbox" over real calendar time before any human
LOCKBOX      Tier-2 audit or capital. No survivor has ever reached it.
```

Two hard invariants constrain any fix you propose:
- **CRU-1:** the funnel's gate definition is frozen by a `gates_hash` (`519158fa1450`). New
  connectors/capabilities are allowed but must not change existing verdicts. (You may *recommend* changing
  the gate; just flag it as a verdict-changing MAJOR change.)
- **CRU-2 (anti-oracle moat):** the proposer may read only a dedup/killed-family view — never verdicts,
  DSR, or holdout data. This is the guard against the discovery loop overfitting to its own test set. Do
  not propose widening it.

---

## 2. The exact record you must explain

- **Substrates mined:** `cross_asset` (US, ~4044 daily bars ≈ 16y, TSMOM base book + FRED/COT/EDGAR
  overlays) and `taiwan` / `taiwan_v2` (~2782–4044 daily bars, TWSE-T86 institutional-flow + TAIFEX-OI
  overlays). Both returned 0 PROMISING.
- **Proposers:** the deterministic library **and** an LLM proposer that produced 37 distinct
  mechanism-diverse overlays across a forced 3-tick probe — best raw out-of-sample marginal ΔSR reached
  **0.66**, but every candidate deflated to **DSR ≈ 0.000 / marginal-t ≤ 1.67** → still 0 PROMISING.
- **Three "power levers" were each investigated and closed** (pre-registered probes, see §5 artifacts):
  1. *Extend history* — closing the detection gap needs ≈130 years of daily data. Dead.
  2. *Higher frequency* — an intraday (5-sec/minute, ~4.86M-bar) substrate was modelled; the funnel's
     minimum-detectable-effect (MDE) was measured to **flatten to ~N^-0.21** (not the assumed 1/√N) above
     N_eff≈1000, giving MDE 0.86 at a 5-min holding — still above the 0.50 target. Closed.
  3. *Cross-market pooling* — pooling the same mechanism across markets needs ~8 independent markets to
     reach target; only ~2 exist for the tested mechanism (gold positioning). Closed.

**Your first duty is to decide whether that record is a property of the world or of the code.**

---

## 3. The gate you must scrutinize first (prime suspect for explanation C)

The PROMISING decision is a **single conjunction of five independent hurdles**, every one of which must
pass (`sharpen/signals/generation/fitness.py::combination_fitness`, thresholds from
`configs/*.gates.yaml::generation`; current values shown):

```
PROMISING  ⟺   delta_mean   ≥ 0.10        (mean CPCV-path marginal book ΔSR — the "uplift floor")
          AND  dsr_aug      ≥ 0.90        (DEFLATED Sharpe of the AUGMENTED BOOK, deflated by the
                                           generation search's effective trial count gen_n_eff)
          AND  marginal_t   ≥ 3.0         (t-stat of the MARGINAL contribution stream b_aug − b_base,
                                           using an AR(1) EFFECTIVE-N, not √N)
          AND  max_base_corr ≤ 0.70       (candidate not collinear with the span of base sleeves)
          AND  delta_median ≥ 0  AND  frac_positive_paths ≥ 0.50   (CPCV fragility: central path not a
                                                                    loss AND majority of paths positive)
```

Specific concerns for you to test — each is a plausible route to *systematic false negatives*:

1. **`dsr_aug` is a BOOK-level bar, not a candidate-level bar.** It deflates the Sharpe of the *whole
   augmented book* (base sleeves + candidate). On the `taiwan` substrate the base book is TSMOM-only and
   may have low or negative Sharpe. **Question:** is `dsr_aug ≥ 0.90` reachable *at all* on that substrate,
   for *any* candidate, given the base book's own Sharpe? If not, the gate is sealed independent of
   candidate quality. (A prior internal audit asserted exactly this — verify or refute it.)
2. **Five ANDed hurdles → compounding false-negative rate.** Each leg individually is stringent (a
   marginal-t ≥ 3.0 is a ~0.1% one-sided bar; DSR ≥ 0.90; uplift ≥ 0.10; majority-positive paths). Even if
   each has modest individual power against a true ΔSR≈0.5 edge, the **conjunction's joint power can be near
   zero**. Quantify the joint false-negative rate against planted alpha of known size. Is the conjunction
   statistically coherent, or is it double/triple-counting the same evidence (e.g. DSR and marginal-t both
   effectively re-testing significance)?
3. **Deflation N.** `gen_n_eff` (the file-drawer trial count) drives the DSR benchmark `SR*`. Is it the
   right N? A GP search that generates-and-discards thousands of genomes charges a large N; but the
   *dedup* is one-shot formula-hash, so a hypothesis is burned once at its lowest-power moment and never
   re-tested as data accrues. Does the deflation N match the true multiple-comparison exposure, or
   over-penalize?
4. **CPCV geometry.** `n_groups=6, k_test=2, embargo=21, purge_horizon=1` on ~2800–4000 bars yields short
   test paths. Does the fragility leg (`frac_positive ≥ 0.50`) reject genuinely time-varying edges as
   "fragile"? A prior internal repair already removed a p05-veto leg for being CPCV-geometry noise — check
   whether the replacement has the same disease.
5. **Cost-in-the-return.** Candidate returns are netted of turnover·bps *before* scoring. Is the cost model
   realistic, too harsh, or applied inconsistently across substrates? A too-harsh cost turns real gross
   edges into sub-threshold net edges (the entire Taiwan arb sweep died this way).

---

## 4. The internal diagnosis you must adversarially test (do not accept it)

The team's current defense of `0 PROMISING` is a calibration harness (`scripts/research/crucible_calibration.py`,
results in `results/crucible_calibration/`) that drives the **real** shipped gate on synthetic substrates:

- **E1 (null / false-positive rate): claimed GREEN** — 0 PROMISING across 2349 pure-noise genomes
  (Clopper-Pearson upper-95% FPR ≈ 0.0013). Claim: the funnel does not leak false positives.
- **E2 (planted power): claimed RED** — the minimum realized marginal ΔSR detectable at 80% power falls
  ~1/√T but is **≈1.40 even on the longest real substrate**, i.e. ~3× a realistic alpha of 0.3–0.5. Claim:
  every `0 PROMISING` is "honest-but-underpowered," i.e. explanation (B).

**This calibration is the linchpin of the entire (B) argument — so break it if you can.** Questions:
- Does the **synthetic planted alpha** used in E2 faithfully resemble *real* alpha — its autocorrelation,
  fat tails, regime-dependence, correlation to the base book? If the synthetic edge is "easier" or "harder"
  than real alpha, the MDE estimate is biased and the (B) conclusion is unsafe.
- E1 tests the *false-positive* direction. **Is there an equivalent measured bound on the false-NEGATIVE
  direction** for realistically-structured alpha, and does it isolate *which* of the five gate legs is
  binding? (If one leg — say `dsr_aug` book-bar — is doing all the rejecting, that reframes the whole
  story from "underpowered" (B) to "mis-designed" (C).)
- The MDE was found to flatten to ~N^-0.21 intraday. **Is that a true property of the estimator, or an
  artifact** of the autocorrelation-haircut model used to derive it? Which gate leg drives the flattening?
  (The internal note flags this as "measured, not derived" — pin it down.)
- A prior internal Tier-2 **design audit** (`docs/research/crucible_design_audit_2026-07-07.md`, 104
  findings) already concluded `0 PROMISING` is a "machine artifact, not market truth," citing 5-leg gate
  stacking, the `dsr_aug` book-bar, one-shot dedup, and a "coin-flip" lockbox criterion. **Read it, then
  independently confirm or refute its top findings** — do not inherit them. It is an internal document and
  may be wrong or may have been superseded.

---

## 5. Artifacts to read (repo-relative paths)

**Design & specs**
- `docs/research/crucible_agentic_discovery_spec.md` — the design spec (funnel, gate rationale, moat)
- `docs/research/crucible_weak_signal_ensemble_spec.md`, `crucible_mc_null_spec.md`,
  `crucible_diverse_proposer_spec.md` — the cohort / ensemble extension (built, ships DISABLED)

**Prior internal analyses (claims to verify, NOT ground truth)**
- `docs/research/crucible_design_audit_2026-07-07.md` — 104-finding design audit ("machine artifact")
- E1/E2 calibration: `scripts/research/crucible_calibration.py` + `results/crucible_calibration/`
- Power-lever closures: `docs/research/crucible_intraday_stage0_{preregistration,partA_result}_2026-07-13.md`,
  `docs/research/crucible_crossmarket_pooling_stage0_{preregistration,result}_2026-07-13.md`

**Code — the machine itself**
- Gate/fitness: `sharpen/signals/generation/fitness.py` (the 5-leg gate — §3)
- Deflated Sharpe: `sharpen/crypto/eval/statistics.py::deflated_sharpe_ratio`
- Evaluation harness / CPCV: `sharpen/signals/eval_harness.py`, `sharpen/signals/gates.py`
- DSL grammar & search: `sharpen/signals/generation/{grammar.py,dsl_signal.py,evolve.py}`,
  `sharpen/signals/library/` (operators, alpha101)
- Cohort/ensemble (disabled): `sharpen/signals/generation/{cohort.py,cohort_eval.py,cohort_mc.py}`
- Orchestration, proposer, moat, connectors: `sharpen/crucible/` (`agentic/llm_proposer.py`,
  `orchestrator/substrate.py`, `data/*`)
- Gate thresholds: `configs/crucible_{cohort,lockbox,power}.gates.yaml`,
  `configs/signal_eval.gates.yaml` (the frozen funnel — `gates_hash 519158fa1450`)

**Ledger / state (ground-truth what actually ran)**
- `results/crucible_orchestrator/**/*.db` (trial ledger, lockbox, governance, FDR account)

---

## 6. Concrete tests worth running (illustrative, not exhaustive — design your own)

1. **Planted-alpha recovery on a REAL substrate.** Inject a synthetic sleeve with *known* marginal ΔSR ∈
   {0.3, 0.5, 0.8, 1.2, 1.6} and realistic structure (autocorr, fat tails, base-book correlation) into the
   actual `taiwan` / `cross_asset` base book. At what true ΔSR does the *shipped* gate start returning
   PROMISING, and does that threshold match the E2 calibration's 1.40 claim? A large mismatch impeaches the
   calibration.
2. **Per-leg ablation.** Turn off each of the 5 gate legs in turn and re-score the historical hypothesis
   set. Which single leg is responsible for the most rejections? If one leg (e.g. `dsr_aug`) rejects
   ~everything, that localizes the problem and distinguishes (B) from (C).
3. **`dsr_aug` reachability.** Holding the candidate fixed at a strong planted edge, sweep the base book's
   own Sharpe. Show the `dsr_aug ≥ 0.90` frontier — is it satisfiable on the Taiwan base book at all?
4. **Joint false-negative curve.** For planted ΔSR = 0.5, estimate the *conjunction's* power (all 5 legs)
   vs. each leg alone. Quantify how much power the ANDing destroys.
5. **Deflation-N sensitivity.** Re-run with `gen_n_eff` set to the true count of *distinct, pre-registered*
   hypotheses (not the GP generate-and-discard count). Does the verdict change? This tests whether the
   file-drawer penalty is honest or punitive.
6. **Proposer expressiveness.** Independently ask: can the DSL + the proposer even *represent* the kinds of
   edges that exist in these data (cross-sectional, event-driven, non-linear, conditional)? A machine that
   cannot express real alpha will return 0 forever regardless of statistics — a distinct flavor of (C).
7. **Pipeline correctness spot-checks.** Trace one hypothesis end-to-end: PIT alignment of the alt-data
   slot, cost netting, CPCV purge/embargo seams, the FDR charge, dedup. Look for off-by-one leakage or
   silent NaN-dropping that biases Sharpe.

---

## 7. Deliverables

1. **Verdict** on (A) / (B) / (C) — or a quantified mixture — with the evidence that decides it. If (C),
   name the defect(s) and show the mechanism. If (B), state what would have to be true of real alpha for it
   to hold, and whether you could confirm it.
2. **Ranked findings** (severity × confidence), each with a concrete failure scenario and a minimal repro.
   Separate *correctness bugs* from *design/calibration choices* from *out-of-scope data limits*.
3. **The single highest-leverage change** that would most increase the funnel's power **without** raising
   its false-positive rate (E1 must stay green) — and its cost in the CRU-1 `gates_hash` (is it MINOR or a
   verdict-changing MAJOR?).
4. **Is the deflation correctly calibrated?** Specifically: is the gate over-conservative (systematic false
   negatives), correctly conservative, or hiding a leak that its conservatism masks? Address the "5 ANDed
   legs" and "book-level DSR" concerns directly.
5. **Future directions.** Is a per-hypothesis-deflated GP miner even the right instrument for *short daily*
   substrates? Consider and rank alternatives: pre-registered single-hypothesis probes (which on this
   project detected large gross arb signals easily and failed only on costs, i.e. did *not* hit the power
   wall); Bayesian/shrinkage posteriors over mechanism effect instead of a frequentist pass/fail; enabling
   the built-but-disabled weak-signal cohort/ensemble path; targeting different data (higher-value or
   capacity-constrained niches) rather than better statistics on free daily data; or retiring the platform
   in favor of the project's one proven edge (a linear cross-asset TSMOM sleeve at net Sharpe ~0.60).

## 8. Guardrails

- **Falsification-first.** Your goal is truth, not a discovery. Do not recommend relaxing gates merely to
  produce a PROMISING; a false positive reaching capital is the worse error. If the honest answer is "the
  machine is right and there is nothing here," say so plainly.
- **Respect CRU-2.** Any fix that lets the proposer or search see verdicts/holdout is off-limits — it would
  overfit the discovery loop to its own test set.
- **Distinguish "real" from "tradeable."** On this project, several signals were statistically real but
  died on realistic costs / liquidity / staleness. Keep detection power and net-of-cost tradeability as
  separate axes in your verdict.
- **Show your work.** Every quantitative claim should be reproducible from the cited artifacts or a script
  you provide.

---

*Context note for the reader of this brief: the internal team's current recommendation is to stop mining
and redeploy to the deployable TSMOM→paper path. This audit exists to check whether that recommendation is
premature — i.e., whether a fixable defect (C) is masking real edges that the machine should have found.*

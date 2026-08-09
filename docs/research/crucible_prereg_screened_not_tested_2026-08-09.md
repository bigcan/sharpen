# The us_equity mine never ran its binding test — and the power it was gated on is measured at a threshold production doesn't use

Session 553-cont-153, 2026-08-09. Follows
[the first powered mine](../../randd_log.md) (S553-cont-152) and its `us_equity` GO.

Two findings. The first invalidates the *stated basis* of the `us_equity` verdict (the verdict itself
survives re-testing). The second says the substrate's "adequately powered" claim was made against a
threshold the production path does not use, and that mining it more makes it strictly worse.

---

## Finding 1 — a pre-registered spec was being SCREENED on train, not TESTED on the holdout

### What the record said

`mined=True fdr_tests=8 promising=0`, recorded as "a clean negative on a substrate that could
actually have detected something — the first time that sentence has been true here."

### What actually happened

The tick's own log line reads `prereg_only: 104 of 104 train-survivors are evolved offspring`. Under
`evolve`'s corrected path that arithmetic is decisive:

```python
train_passers = [c for c in ranked if _passes_cheap_prefilter(c.result, corrected_cfg)]   # -> 104
train_passers = [c for c in train_passers if c.formula in prereg]                         # -> 0
```

Zero of the eight pre-registered seeds cleared the **train** cheap pre-filter, so the holdout loop
iterated over an empty list and `corrected_contract_fitness` — the single decision the substrate's
entire power argument is about — **never executed on one pre-registered hypothesis**. The LORD++
account was charged for eight tests anyway (`fdr_charged_total = 0.0399`).

`promising=0` was therefore true by vacuity, not by evidence, and produced a log indistinguishable
from a run where the test did execute.

This is the same shape as the defect `crucible-v6.0` was built to remove — the 2026-07-29 audit found
the holdout gate had never executed in production across 403 lifetime trials, because train re-applied
the final gate. It reappeared through a different door: an **economic-size** screen (uplift ≥ 0.10 ΔSR)
rather than a significance one, with the identical consequence.

### Measured, per seed

`scripts/research/crucible_prereg_prefilter_forensics.py` →
`results/crucible_prereg_forensics/us_equity_prereg_forensics.json`. Panel T=4930, train 3183,
holdout 1726 bars (6.85 calendar years).

**R1 — the train cheap pre-filter (the gate that actually decided the run):**

| hash | train ΔSR | median | frac+ | base corr | pre-filter |
|---|---|---|---|---|---|
| `1207bdca1aa9` | −0.247 | −0.174 | 0.27 | 0.05 | False |
| `342f77a90c05` | −1.028 | −1.027 | 0.00 | 0.04 | False |
| `4bfd4a485462` | −0.502 | −0.522 | 0.00 | 0.05 | False |
| `5b68aadbe74a` | −0.419 | −0.460 | 0.20 | 0.05 | False |
| `7b0dba063910` | −0.547 | −0.571 | 0.00 | 0.03 | False |
| `94283b6f464f` | −0.190 | −0.228 | 0.13 | 0.06 | False |
| `94d1bc4ee8b0` | −0.752 | −0.735 | 0.00 | 0.05 | False |
| `d8e5a1c1b3c6` | −0.436 | −0.468 | 0.07 | 0.02 | False |

0/8. The binding leg is `uplift` on every one — the base book is the validated ETF core at **SR
+0.519**, so a candidate must *add* 0.10 ΔSR to a book that already wins. (Contrast the intraday
substrate, where a base book bleeding 30.6%/yr let zero-alpha nulls clear the same leg 15.3% of the
time by dilution. Both failure modes are the base book's Sharpe acting through the uplift gate; only
the sign differs.)

**R2 — the corrected holdout gate, run on all eight anyway (diagnostic only):**

| hash | holdout ΔSR | corrected t | p | legs t/lord/upl/frag/coll | pass |
|---|---|---|---|---|---|
| `94283b6f464f` | +0.109 | 0.44 | 0.329 | n/n/Y/Y/Y | False |
| `1207bdca1aa9` | −0.173 | −0.69 | 0.756 | n/n/n/n/Y | False |
| `4bfd4a485462` | −0.209 | −0.82 | 0.793 | n/n/n/n/Y | False |
| `7b0dba063910` | −0.281 | −1.13 | 0.870 | n/n/n/n/Y | False |
| `5b68aadbe74a` | −0.372 | −1.45 | 0.927 | n/n/n/n/Y | False |
| `94d1bc4ee8b0` | −0.399 | −1.54 | 0.939 | n/n/n/n/Y | False |
| `342f77a90c05` | −0.403 | −1.55 | 0.940 | n/n/n/n/Y | False |
| `d8e5a1c1b3c6` | −0.415 | −1.45 | 0.926 | n/n/n/n/Y | False |

**0/8, and not marginally**: best `corrected_t` = 0.44 against `t_min` 2.33; seven of eight have a
negative holdout ΔSR. So the pre-filter cost nothing *on this record* — the recorded `promising=0`
stands. What changes is that it is now a result rather than a vacuity.

### The fix — `crucible-v12.0`

Under `eligibility.offspring_policy: prereg_only` (the default) the cheap train pre-filter no longer
applies to pre-registered seeds: the eligible set **is** the pre-registration, and every spec that
produced a `FitnessResult` reaches the binding holdout gate.

Three reasons it is the right direction, not merely the convenient one:

1. **It is the corrected contract's own stated design** — "the significance decision is taken exactly
   once, on the holdout."
2. **The screen selected in the wrong direction.** Keeping only the pre-registrations that already
   look good *in-sample*, then testing those out-of-sample, is precisely what pre-registration exists
   to prevent.
3. **No guard is lost.** `uplift` / `fragility` / `collinearity` are all re-applied on the holdout
   inside `corrected_contract_fitness`, where they judge out-of-sample evidence instead of in-sample
   fit. Compute is negligible: the eligible set is the tick's handful of specs, not the search.

`offspring_policy: all` is untouched — with an unbounded GP population feeding it, the pre-filter is a
compute bound rather than a screen on pre-registrations.

Two side effects, both wanted:

* **`rejection_class` stops being NULL** (the open defect carried from cont-152). Seeds now reach the
  holdout, fail it, and get classified. Blast radius checked before shipping: `decisive_mde_multiple:
  1.0` × `uplift_min` 0.10 ⇒ DECISIVE requires implied MDE ≤ 0.10, and `us_equity` sits at 1.312, so
  **every** rejection classifies UNDERPOWERED and no family is killed — exactly what the search-memory
  gates file predicts in its own header. U4 becomes live (park + re-admit) with no risk of
  manufacturing NO-GOs out of low power.
* **The denominator now travels with the verdict.** `GenerationReport.n_holdout_tested` →
  `TickRecord.n_holdout_tested` (nullable; NULL on pre-v12 rows means UNKNOWN, never 0), plus a loud
  orchestrator warning when it is zero. Two consecutive sessions lost real time to a `promising=0`
  whose denominator was unreadable — the cont-152 dedup livelock, then this pre-filter.

CRU-1 verdict-preservation is **claimed and verified**, not assumed: the eight seeds are the only
pre-registrations this change would have routed differently, and R2 above re-scored all eight through
the shipped holdout gate. No gate byte moves; the three sealed moats are unchanged.

---

## Finding 2 — the power guard calibrates on a FRESH LORD++ account; production runs on a depleted one

### The mechanism

The substrate-power stamp comes from the MDE curve in `crucible_calibration.py`, which scores every
planted panel at `fresh_lord_level(corrected)` and says so:

> "The LORD++ level is a FRESH account's first level ... a power curve is one hypothesis at a time, so
> the stream has not yet decayed."

Production is not one hypothesis at a time. The orchestrator charges one LORD++ test per pre-registered
spec, the account **persists across ticks**, and `_tick_lord_level` hands `evolve` the **tightest**
level of the batch. On `us_equity`, one 8-spec tick has already moved the next level from the fresh
**0.021874** (z 2.02 — so `t_min` 2.33 binds) to **0.000650** (z 3.21 — so the LORD++ leg binds and
`t_min` is slack). A 40-spec second round would run at **4.5e-05** (z 3.91).

The mismatch is one-directional: over a barren stream levels only decay. It therefore **grows
monotonically with every test the substrate has ever run**, and is invisible in the stamp.

### Measured

`scripts/research/crucible_lord_depletion_power.py` → `results/crucible_lord_depletion/`. Same planted
machinery, same scorer, `us_equity` holdout size (1726 bars); the LORD++ level is the only thing that
varies. 40 seeds × 6 plant strengths per level.

| LORD++ level | p-threshold | MDE ΔSR @ power 0.80 | vs fresh |
|---|---|---|---|
| fresh — what the guard calibrates at | 2.19e-02 | **0.750** | 1.00× |
| live after 8 tests, next single | 6.50e-04 | **0.840** | **1.12×** |
| live after 8, batch of 8 | 2.59e-04 | **1.021** | **1.36×** |
| live after 8, batch of 40 | 4.47e-05 | **1.071** | **1.43×** |

The per-leg tally isolates the mechanism. At a fixed plant (realized ΔSR +0.551) the `t` leg passes
0.53 at *every* level — it is level-independent by construction — while the `lord` leg collapses
**0.72 → 0.15 → 0.07 → 0.03**. The LORD++ p-gate, not `t_min`, is what binds on a depleted account.

Pre-registered read was "within ~5% ⇒ harmless documentation fix". It is 12% worse for the next single
test and **36–43% worse for a realistic batch**. The guard is anti-conservative by construction.

### Re-measured on the CROSS-SECTIONAL curve — the one `us_equity` is actually gated by

⚠ The measurement above is on the calibration's **overlay** planted fixture; `us_equity` mines
**cross_sectional**, and `crucible-v7.0` exists precisely because those curves differ. So it was
re-measured on the cross-sectional path (`--path cross_sectional`), reproducing the shipped
`calibration_xsec_mde_sweep_corrected.json` sweep's own parameters (holdout_frac 0.25, hold 21,
`ls_min_names` 6, its beta grid, 24 seeds — **not** `us_equity`'s), re-deriving its rows at each level,
and then reading the implied MDE back through the guard's **own accessor** (`_pooled_points` +
`interp_mde` at holdout 1726). Only the two grid depths that anchor the interpolation are re-measured,
since `interp_mde` cannot use any others.

Two efficiencies make this exact rather than approximate: the LORD++ level enters
`corrected_contract_fitness` **only** through `lord_pass`, so one scoring pass serves every level and
the levels are compared on identical draws; and `non_lord_pass` is taken as `passes_corrected` scored
at `lord_level=1.0` rather than re-ANDing leg flags by hand, so a leg this script never names (the
cont-151 `exposure_pass`) cannot be silently dropped. Verified: re-running with the hand-ANDed version
gives identical numbers, i.e. the exposure leg was inert here — but by construction now, not by luck.

**Harness validation.** On the shipped beta grid the fresh level reproduces the recorded stamp
**exactly**: 1.3121 against the tick's `implied_mde_delta_sr = 1.3120527546583958`.

| LORD++ level | shipped grid | ratio | refined grid (16 betas) | ratio | overlay ratio |
|---|---|---|---|---|---|
| fresh (what the guard calibrates at) | **1.312** | 1.00× | **1.029** | 1.00× | 1.00× |
| live after 8, next single | 1.464 | 1.12× | 1.367 | **1.33×** | 1.12× |
| live after 8, batch of 8 | 1.464 | 1.12× | 1.367 | **1.33×** | 1.36× |
| live after 8, batch of 40 | 1.464 | 1.12× | 1.431 | **1.39×** | 1.43× |

**The effect replicates on the cross-sectional path.** At matched resolution the ratio is 1.33–1.39×
against the overlay's 1.36–1.43×. The transport caveat is discharged: this is not an overlay artifact.

The shipped grid's flat 1.12× is **quantization**, not a flat effect. `_pooled_points` takes the worst
MDE across `n` at each depth, and `first_point_mde` reports the realized ΔSR at the first grid beta
reaching 0.80 power; the steps are wide (n=100 at holdout 1011 jumps 1.398 → 2.001 between adjacent
betas), so all three live levels landed on the same step. The individual rows do move — n=25 at holdout
2016 goes 1.011 → 1.312 — the pooled maximum just doesn't.

### ⚠ Correction to this report's own earlier claim

The first version of this section said the live 8-spec MDE was **≈1.78 ⇒ REFUSE**. That is **wrong and
is retracted.** It multiplied the *overlay* ratio by the *shipped-grid* stamp — two different
estimators — and refined-grid absolutes run systematically lower than shipped-grid ones because a
finer grid finds the power crossing earlier. Like-for-like:

* **On the guard's own grid**: fresh 1.312 → live **1.464**, against the ceiling **1.457**. It crosses,
  by **0.5%** — a margin far inside one quantization step of the estimator that produced it.
* **On the refined grid**: fresh 1.029 → live **1.367–1.431**, both **below** 1.457.

So **the depletion effect does not clearly close `us_equity`**, and the earlier "REFUSE" overstated
it. What is established is narrower and still worth acting on: the fresh→live gap (+0.34 ΔSR refined)
is **comparable to the entire margin between the stamp and the ceiling**, so at this substrate the
guard's verdict is set as much by the LORD++ convention and the calibration grid's resolution as by
the substrate itself. A gate whose answer flips on its own estimator's step size is not measuring what
it claims to three digits.

Also retracted: "spend the budget in small high-prior batches, since a wide tick is disproportionately
expensive." On the cross-sectional path single and 8-spec batches are **identical** (1.367) and 40
specs costs only ~5% more. That advice was an overlay artifact too.

### Not fixed here, deliberately

Making the power guard read the live account is a gate-semantics change, and on the guard's own grid
it flips `us_equity` to REFUSE. That is an operator decision under the CLAUDE.md gate rule, not a call
to make inside a measurement session. What ships is the measurement, the tooling, and this statement.

Two options when it is taken up:

1. **Stamp power at the live level.** Correct on principle — the guard would measure the test it
   actually runs. On the shipped grid it refuses `us_equity` (1.464 > 1.457); on a refined grid it
   does not (1.367 < 1.457). So this should be taken **together with** re-running the calibration
   sweep on a finer beta grid, or the gate's answer is set by its own step size.
2. **Reset/partition the LORD++ account per campaign.** Defensible only with an explicit argument
   about what family-wise error is being controlled across which hypotheses — otherwise it is
   multiplicity laundering.

**On the second `us_equity` round** (the plan carried into this session): power is *not* the blocker
it looked like one measurement ago — the refined-grid live MDE 1.367 sits under the 1.457 ceiling, and
batch size barely matters (single and 8-spec are identical; 40 specs costs ~5% more). The real
constraints on that round are the ones already on the record: the default `LibrarySeedProposer` bank
**is** the 8 seeds already scored, so new hypotheses mean a new pre-registered bank (92 unused WQ101
formulas); and every candidate must still be reported at 1/2/5 bp, because the 2 bp cost is a stated
assumption, not a measurement.

---

## Artifacts

| item | path |
|---|---|
| prereg pre-filter / holdout forensics | `scripts/research/crucible_prereg_prefilter_forensics.py` |
| — output | `results/crucible_prereg_forensics/us_equity_prereg_forensics.json` |
| LORD++ depletion power curves | `scripts/research/crucible_lord_depletion_power.py` |
| — overlay output | `results/crucible_lord_depletion/lord_depletion_power.json` |
| — cross_sectional, shipped grid (reproduces the stamp) | `results/crucible_lord_depletion/lord_depletion_power_xsec.json` |
| — cross_sectional, refined grid | `results/crucible_lord_depletion/lord_depletion_power_xsec_fine.json` |
| the fix | `finrl_pro_ds/signals/generation/evolve.py` (train pre-filter branch) |
| reporting | `substrate.py` / `orchestrator.py` `n_holdout_tested` (+ migration) |
| version rationale | `finrl_pro_ds/crucible/version.py` (`crucible-v12.0`) |
| regression tests | `tests/crucible/test_contract_corrected_evolve.py` (3), `test_orchestrator.py` (1) |

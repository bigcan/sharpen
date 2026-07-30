# RC-11 RESOLVED — it was my null, not the gate (2026-07-31)

**Verdict: RC-11 as filed is WITHDRAWN.** The uplift leg is not substrate-dependent, the Taiwan overlay
path is not un-calibrated, and no threshold needs changing. The finding was an artifact of the null I
built to test it. What replaces it is a smaller but real defect in the shared calibration substrates
(NULL-DEGEN-01, below), which reaches further than RC-11 did.

Harness: `scripts/research/crucible_uplift_null.py` (new `--randomize-regime`, `--base-sharpe`,
`--cost-bps`). Reports in `results/crucible_uplift_null/`.

## What RC-11 claimed

From the 2026-07-30 U7 measurement: the null distribution of `CorrectedResult.delta_sr` differed by
~200× in its 95th percentile between substrates — cross_asset overlay q95 −0.004 versus **Taiwan overlay
q95 +0.831, centred at +0.45, with 91.5% of pure-noise overlays clearing `uplift_min = 0.10`**. Filed
HIGH, mechanism unresolved, with the note that it might be a *seal* like F2's `dsr_aug`.

## The actual cause: the null had an effective sample size of ~4

`cal._noise_panel` and `cal._realistic_noise_panel` both build the timing slot as

```python
regime = np.sin(2.0 * np.pi * np.arange(t) / 80.0) + 0.2 * rng.standard_normal(t)
```

— a **fixed period, fixed phase** sinusoid, identical on every seed but for a small noise term. The four
overlay null seeds (`macro:regime`, `delta(...,20)`, `decay_linear(...,10)`, `delay(...,5)`) are
deterministic transforms of that one signal, so each produces nearly the same tilt on every "independent"
panel, and hence nearly the same ΔSR against a fixed base book.

Measured on the original Taiwan draws (600 = 150 panels × 4 formulas):

| formula | mean ΔSR | sd across 150 panels |
|---|---|---|
| f0 | +0.453 | 0.083 |
| f1 | +0.747 | 0.089 |
| f2 | +0.465 | 0.032 |
| f3 | +0.127 | 0.077 |

**Between-formula variance of the means is 11.8× the mean within-formula variance.** The draws are four
tight clusters, not 600 samples. A q95 read off that is not a null quantile — it is "the second best of
four fixed sine patterns", and the +0.831 is one chance alignment between that single sinusoid and
Taiwan's holdout return path, replicated 150 times.

## The fix, and the control

`--randomize-regime` gives each null panel its own timing-slot shape (random period 20–400 d, random
phase, random cycle/random-walk/white-noise mix). The variance ratio inverts from 11.76 to **0.21** — the
draws now vary with the panel, which is what a null is supposed to do.

| overlay null | fixed sine q95 | %pass 0.10 | randomized q95 | %pass 0.10 |
|---|---|---|---|---|
| cross_asset | −0.004 | 0.00% | −0.140 | 0.33% |
| **taiwan** | **+0.831** | **91.5%** | **−0.007** | **2.33%** |

The ~200× substrate dependence collapses to a ~2 pp difference, both substrates land negative, and
Taiwan's pass rate falls from 91.5% to 2.33% — in line with cross_asset. **`uplift_min = 0.10` admits
~2% of properly-constructed null overlays, not 37.6%.**

## Two secondary channels, both measured and both minor

**Turnover/cost — real but not dominant.** Randomized slots produce choppier tilts: annualized tilt
turnover 99.4 vs 34.3. Re-running at `--cost-bps 0`:

| | mean ΔSR, real cost | mean ΔSR, zero cost |
|---|---|---|
| fixed sine | +0.448 | +0.656 |
| randomized | −0.995 | −0.436 |

Cost accounts for ~0.56 of the ~1.44 gap; **~0.88 survives at zero cost**, so the dominant channel is the
tilt's *shape/timescale*, not its cost. A slow smooth tilt is a genuine market-timing bet whose sign is
set by phase alignment; a fast tilt dilutes the book (variance up, mean ≈ 0) and is systematically
negative.

**Base-book Sharpe — FALSIFIED as the driver, and it runs the other way.** The three natural cells
ordered suggestively in base SR (+0.045 → +0.496 → +1.364 against q95 −0.004 → +0.061 → +0.831), which is
what made "base Sharpe" the leading hypothesis. Holding substrate, panel, sleeves and tilt fixed and
shifting only the base book's mean to hit a target Sharpe:

| base SR | 0.0 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| overlay null mean ΔSR | +0.492 | +0.390 | +0.285 | +0.178 |

Monotonically **decreasing**, ≈ −0.21 ΔSR per unit of base Sharpe — the opposite sign, and an order of
magnitude too small to produce the observed gap. The cross-cell correlation was confounded with
substrate identity (i.e. with which phase of the one sinusoid happened to fit that substrate's returns).
Note the +0.49 at base SR 0.0: the effect does not need a good base book at all, because a location shift
does not change the return *path* the sine aligns with.

## NULL-DEGEN-01 (NEW, MEDIUM-HIGH) — the degenerate slot is shared, and E1 consumes it

The same fixed sinusoid is built in **three** places:

| file | line | consumer |
|---|---|---|
| `scripts/research/crucible_calibration.py` | 137 | `_noise_panel` → E1 (IID null) |
| `scripts/research/crucible_calibration.py` | 215 | `_realistic_noise_panel` → E1 (realistic null), this harness |
| `scripts/research/crucible_orchestrator.py` | 164 | `_synthetic_panel` → the synthetic mining substrate |

`run_e1` scores `_OVERLAY_NULL_SEEDS` on `panel_gen` (line 467/472), so **E1's per-candidate null FPR
inherits the same degeneracy on its overlay half**: its Clopper-Pearson bound is computed over
`genomes_total` as if those were independent Bernoulli trials, and on the overlay side they are not. That
bound is therefore tighter than the evidence supports. This matters because the recorded E1 result is
already close to its ceiling — the corrected-contract gates file records per-tick FPR 0.025 with
**CP-upper95 0.0487 against a 0.05 ceiling, ~2% headroom, "one more false positive would have flipped
it."**

**Scope discipline — what is NOT claimed.** I have not re-run E1, so I am *not* claiming E1 is actually
RED. The claim is narrower and checkable: the independence assumption behind its confidence bound is
violated on the overlay half, so the stated headroom is overstated by an unknown amount. The check is to
re-run E1 with randomized slots and compare.

Separately, the **planted** panel (`_planted_panel:265`, feeding E2 and the MDE sweeps) uses the same
sinusoid, but there it is the signal *by design* — the base book is built to depend on its lag, and
measuring detection of a known signal is the point. The caveat there is generalization, not degeneracy:
the MDE curves say "MDE for a slow-sinusoidal timing signal", and a choppier real alpha could have a
different MDE. That is a scope limit on the power curves used in the readiness analysis, not an
invalidation of them.

## Corrections this forces to the 2026-07-30 record

* RC-11 **withdrawn** in the audit findings table, the U7 report, `version.py`, the corrected-contract
  gates comment, the readiness doc and three memory files.
* The U7 headline stands unchanged: `uplift_min = 0.10` remains calibrated and confirmed. The
  cross-sectional measurements (which use real DSL genomes on real OHLCV noise, not the slot) were never
  affected — only the overlay half of the null was.
* The readiness doc's blocker 3 ("the Taiwan overlay path is un-calibrated") is **removed**.

## Lesson worth keeping

The failure mode was not a bad statistic — it was **a null that could not vary**. Every guard in this
project checks the *result* of a measurement against a threshold; nothing checked whether the
measurement's own draws were independent. The between/within variance ratio is a two-line diagnostic that
would have caught this immediately, and it is now the first thing this harness reports. A null whose
effective n is 4 will happily produce a confident-looking 95th percentile, a tight bootstrap CI, and a
HIGH-severity finding — all of them wrong together, and all of them internally consistent.

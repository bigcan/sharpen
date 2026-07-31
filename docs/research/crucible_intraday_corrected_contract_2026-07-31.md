# Crucible intraday power under the CORRECTED contract — re-opening a stale NO-GO

**Date:** 2026-07-31 (S553-cont-146) · **Branch:** `July2026`
**Artifact:** `results/crucible_intraday_power/intraday_power_corrected.json`
**Harness:** `scripts/research/crucible_intraday_power.py --contract corrected` (new flag)
**Grids:** `configs/crucible_calibration_intraday_corrected.gates.yaml` (new; the shipped-contract
pre-registration is left byte-stable)

## Why re-open a measured NO-GO

`crucible_intraday_stage0_partA_result_2026-07-13.md` closed the higher-frequency lever: MDE 0.86 at
5-min holding vs a 0.50 ceiling, "effectively closes the 'higher-frequency data' lever". That verdict
was measured on the **shipped** contract (commit `10406cc8`, which predates `v6.0`/`v7.0`/`v8.0`; the
corrected contract only became DEFAULT at `v8.0`).

Its two stated blockers were (1) MDE flattening to ~`N^-0.21`, and (2) — per the cont-131 audit, and
quoted verbatim in `interp_mde`'s own docstring — a **data-independent `marginal_t` floor (~0.4)**:

> "the cont-131 audit pinned a data-INDEPENDENT `marginal_t` floor (~0.4), so the true MDE does not
> decay to zero at all — ANY power law → 0 is asymptotically fail-open"

**`marginal_t` is one of the two legs the corrected contract structurally DROPS**
(`corrected_contract.py:5-7`: `marginal_t` is the F1 substitution-residual seal, `dsr_aug` the F2
book-level seal). So the single load-bearing reason the guard believes 0.50 is unreachable **at any
N** is a leg that does not exist in the contract now shipping by default. That makes the verdict's
premises stale, which is a reason to re-measure — not a reason to assume the answer flips.

## What was built

`--contract {shipped,corrected}` on the intraday harness, mirroring `crucible_calibration`'s
`_e2_power_curve` exactly (same window convention, same `fresh_lord_level`, same holdout slice), plus
a `_CorrectedLegTally` recording per-leg pass rates — because "MDE is flat" is uninterpretable without
knowing *which* leg binds, and three of the corrected contract's five legs (`uplift`, `fragility`,
`collinearity`) are N-independent.

Two verification gates before trusting any number:

- **Regression.** The shipped arm re-runs **bit-for-bit** against the July artifact (MDE 3.0499 /
  3.1920 / 1.1806 / 3.1910 on the quick grid, identical to 12+ significant figures). The wiring did
  not disturb the frozen path.
- **Cross-harness consistency.** At H=1 the AR generator reduces to the shipped IID generator
  bit-for-bit (TA-1), so matched depths should agree with the independent daily corrected sweep:

  | holdout | daily `calibration_mde_sweep_corrected.json` | @beta | this harness | @beta |
  |---|---|---|---|---|
  | ~4030 | 0.554 | 0.003 | 0.562 | 0.003 |
  | ~8060 | 0.393 | **0.0025** | 0.569 | 0.003 |

  Agreement to **1.4%** at holdout 4000 on the same beta. The 8000 gap is fully explained by grid
  quantization — the daily grid carries a `0.0025` rung this one lacked, so the crossing rounds up.
  Not a discrepancy in the statistic.

## Result

Measured MDE-vs-N_eff at H=1, same T-grid, both contracts:

| N_eff | SHIPPED (July, full run) | CORRECTED (this run) |
|---|---|---|
| 500 | 1.977 | 1.149 |
| 1,000 | 2.064 | 1.160 |
| 2,000 | 2.039 | 1.143 |
| 4,000 | 1.229 | 0.562 |
| 8,000 | 1.183 | 0.569 |
| 16,000 | **1.184** | **0.224** |

The shipped curve **plateaus at ~1.18** from N_eff 8,000 onward — more data buys nothing, exactly as
the ~0.4 `marginal_t` floor predicts. The corrected curve **keeps falling**, reaching 0.224. That is
the mechanism made visible: removing the data-independent leg restores the data-dependence of power.
The corrected contract's own N-independent floor is `uplift_min = 0.10` — *below* the 0.50 ceiling
rather than above it, which is the whole difference.

Tripwires all green, including **TA-3 on both validation cells** — `[92000,12]` had FAILED in July
(ratio 0.59, which is why that run's own `A1_verdict` was `INCONCLUSIVE`, not `FAIL`); it now lands
on-curve at ratio 1.06.

## The verdict this does NOT support

The script printed `A1 VERDICT: PASS`. **That PASS is an extrapolation artifact and should not be
cited.** Its two clearing rows (H=1 → MDE 0.00; H=12 → 0.05) are `extrap_high` — read off the top of
the measured span (N_eff max 16,000) by extending a log-log line. That is precisely the fail-open
behaviour `interp_mde` was rewritten to forbid in production, where off-grid returns
`unmeasured_high` → +inf → refuse.

Re-deriving the frontier using **only the measured span**, against the TX bars actually on disk
(`data/taiwan_intraday/`, 2019-01→2026-06, free):

| substrate | holding | N_eff | MDE | verdict |
|---|---|---|---|---|
| TX 1-min (1,859,954 bars) | 5 min | 51,665 | — | `unmeasured_high` → REFUSE |
| TX 1-min | 15 min | 16,034 | — | `unmeasured_high` → REFUSE |
| TX 1-min | 30 min | 7,881 | 0.569 | fails 0.50 |
| TX 1-min | 60 min | 3,907 | 0.576 | fails 0.50 |
| TX 3-min (625,679) | 15 min | 5,394 | 0.565 | fails 0.50 |
| TX 15-min (125,543) | 45 min | 6,277 | 0.567 | fails 0.50 |

**Not one real TX configuration both interpolates and clears.** Everything with enough depth to be
promising sits *above* the measured grid, where the guard correctly refuses.

## Status and next step

This is **not** a NO-GO and **not** yet a GO. It is the forcing function `interp_mde`'s docstring
names:

> "The unblock is to EXTEND the sweep so real substrates INTERPOLATE between measured anchors — this
> branch is the forcing function for that"

Running now: the same harness on `crucible_calibration_intraday_corrected.gates.yaml`, with
`primary_h1_t_grid` extended to T=256,000 (N_eff 64,000) so TX's 5-min and 15-min holdings fall
*inside* the measured span; a beta grid refined through [0.0012, 0.005] (including the `0.0025` rung)
so the staircase step stops straddling the 0.50 ceiling; `n_seeds` 8→12; and a third validation cell
`[368000, 12]` at N_eff 4,000.

That last one matters: **both existing validation cells sit at N_eff 500 and 1,000, inside the flat
low-N region where MDE barely varies** — so TA-3 currently validates the autocorrelation proxy
exactly where nothing happens. The corrected crossing moves at N_eff ≥ 4,000, and the proxy has never
been checked there. Validating only where the curve is flat is how TA-3 became uninformative in the
first place.

The ceiling stays at 0.50. It is not moved to manufacture an ALLOW (refused at `v8.0`, and that
refusal holds).

## Reproduce

```bash
python scripts/research/crucible_intraday_power.py --contract corrected
python scripts/research/crucible_intraday_power.py --contract shipped   # reproduces the July run
```
Funnel `gates_hash` 519158fa1450 untouched; neither intraday grids file is part of it.

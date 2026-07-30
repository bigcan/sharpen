# Crucible — re-discovery readiness (2026-07-30, `crucible-v10.0`)

Written after closing the 2026-07-29 design-audit roadmap. Answers one question: **if we start a
discovery run tomorrow, what happens?**

## Where the roadmap stands

| Audit item | Status | Version |
|---|---|---|
| U1 corrected contract promoted to the decision layer | SHIPPED | `v6.0`, DEFAULT since `v8.0` |
| U2 power guard re-calibrated against the corrected contract | SHIPPED | `v6.0` |
| U3 per-name alt-data reachable cross-sectionally | SHIPPED | `v7.1` + `v8.1` |
| U4 the search has a memory | **SHIPPED** | `v10.0` |
| U5 multiplicity stops depending on submission shape | SHIPPED | `v9.0` |
| U6 Tier-0 causality enforced on genomes | **SHIPPED** | `v10.0` |
| U7 uplift floor calibrated against its own null | **MEASURED — value confirmed at 0.10** | report below |
| U8 the honest ceiling | STANDING CONSTRAINT — not a task | — |
| RC-10 reproducibility contract could not be exercised | FIXED | `--force-underpowered` |

Everything the audit ranked as fixable is fixed. What remains is the thing it named as *not* a fix.

## The machine works

A full synthetic tick, run end-to-end after the bump:

```
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 1 --t 1512 --n 12 \
  --max-proposals 8 --start-ts 2020-01-04T00:00:00 --out <tmp> --force --force-underpowered
```

→ 8 specs pre-registered · 2286 genomes scored · **0 PROMISING** · manifest written at
`crucible_version: crucible-v10.0`, `gates_hash: 519158fa1450` (the frozen CRU-1 moat, unchanged),
`contract: corrected`. Ledger rows carry the U4 `semantic_hash` on all 18 rows. Test baseline:
`pytest tests/crucible tests/signals` green; `ruff` clean.

U4's classifier was **verified end-to-end, not just unit-tested** — driving `run_hypothesis_loop` on a
noise substrate until a pre-registered seed reached the holdout gate and failed it:

| substrate MDE | rejection class | `killed_families()` |
|---|---|---|
| 0.05 (≤ economic floor 0.10) | `DECISIVE` | `['101alpha']` — **fires for the first time in the system's history** |
| 1.40 (Crucible's real MDE) | `UNDERPOWERED` | `[]` |

That table is the whole U4 story: the plumbing is fixed, and at the power we actually have, the
right answer is still "this rejection tells us nothing".

## Three blockers to a real discovery run, in the order they bite

### 1. The corrected-contract MDE curves are absent on this machine (fixable by compute)

`--contract corrected` has been the default since `v8.0`, and the power guard fails closed when it
cannot measure. Present locally: only `calibration_mde_sweep.json` (the **shipped**-contract overlay
curve). Absent: `calibration_mde_sweep_corrected.json` and
`calibration_xsec_mde_sweep_corrected.json`. So a real-mode tick refuses at `unmeasured_empty`
(implied MDE `+inf`) **before the MDE ceiling is ever consulted** — exactly the fresh-clone scenario the
`v8.0` docstring flagged, since `results/` is gitignored.

Consequence beyond the refusal: with an unmeasured stamp, `classify_rejection` fail-safes to *no*
classification, so U4 records nothing either. The readiness tick above shows this — every
`rejection_class` is `NULL`.

Fix (both launched 2026-07-30, long-running):
```bash
python scripts/research/crucible_calibration.py --exp xsec_mde_sweep --contract corrected
python scripts/research/crucible_calibration.py --exp mde_sweep --contract corrected
```

### 2. Even with the curves, every real substrate is still refused (NOT fixable by compute — audit U8)

First measured cells from the sweep now running: MDE **2.15** at holdout 378, **1.85** at holdout 696,
against a ceiling of **0.50**. Consistent with the audit's own numbers (1.28–1.83 across depths). Real
substrate depths — cross_asset holdout 1163, Taiwan 1011 — sit squarely in that range. So:

> Measuring the curves converts "refused because unmeasured" into "refused because underpowered". It
> does not produce a mine.

The lever is **breadth × forward accumulation (T×N)**, not more ticks on the same panels — U8. The
ceiling must not be raised to manufacture an ALLOW; that was explicitly refused at `v8.0` and the same
refusal holds.

### 3. RC-11 — the Taiwan overlay path's economic guard is un-calibrated (NEW, from the U7 measurement)

The uplift leg's null distribution is substrate-dependent by ~200× in its 95th percentile. On the
Taiwan **overlay** path the null is *centred at +0.45* with q95 **+0.831**, so **37.6% of pure-noise
overlay candidates clear `uplift_min = 0.10`** there — versus 2.7% on cross_asset. Base-sleeve count is
falsified as the cause; the mechanism is unresolved and may be a **seal** (a statistic that is simply
wrong on a high-Sharpe base book), in which case no threshold repairs it.

Until RC-11 is resolved, a PROMISING survivor mined through the **Taiwan overlay** path would carry an
economic guard that admits better than one in three of pure noise. The cross_sectional path on both
substrates, and the overlay path on cross_asset, are calibrated and conservative. Full detail:
`docs/research/crucible_u7_uplift_null_calibration_2026-07-30.md`.

## What "ready" honestly means right now

**Ready:** the funnel is correct, causality is enforced rather than asserted, multiplicity no longer
depends on submission shape, rejections are classified with power taken into account, dedup is semantic,
parked hypotheses are re-admissible, the reproduce contract executes, and a tick runs end-to-end with the
frozen moat intact.

**Not ready:** it cannot yet *discover*, and the reason is not a defect anyone can patch. Ranked next
actions:

1. **Finish the two corrected-contract MDE curves** (running) so the power stamp is measurable and U4's
   classifier has a finite MDE to reason about. Cheap, mechanical, unblocks honest tick logs.
2. **Resolve RC-11's mechanism** before the overlay path is allowed to promote anywhere. If the combiner
   is re-levering a correlated near-copy of a high-Sharpe base book, this is a seal of the same shape as
   F2's `dsr_aug`, and finding it now is worth more than any number of ticks.
3. **Attack U8 — breadth.** The only measured route to an ALLOW is deeper/wider substrates (T×N) and
   forward accumulation. Per-name `(T,N)` data is now reachable (`v7.1`/`v8.1`) with TWSE T86 wired at
   32/46 names bridged; N=8 effective is the binding number on the only per-name substrate.
4. Leave the ceiling alone.

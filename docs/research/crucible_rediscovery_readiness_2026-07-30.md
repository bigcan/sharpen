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

## Blockers to a real discovery run, in the order they bite

Verified by running a real-mode tick, not inferred:

```
python scripts/research/crucible_orchestrator.py --mode real --nights 1 --force \
  --start 2007-01-01 --start-ts 2026-07-30T00:00:00 --out <tmp> --no-altdata-slots --max-proposals 6
```
→ real cross_asset panel builds (`T=4652, holdout=1163`) → `UNDERPOWERED: implied MDE inf > ceiling 0.50
(unmeasured_empty)` → `mined=False, 0 PROMISING`.

### 0. `generation.enabled: false` — and `--force` is the ONLY correct way past it

Without `--force` a real-mode tick exits immediately: *"generation.enabled is false in
`configs/signal_eval.gates.yaml` — no-op"* (the GP8-01 operator opt-in). This is the first thing that
stops a run and the least obvious.

**Do not "fix" it by editing the YAML.** `configs/signal_eval.gates.yaml` **is** the frozen CRU-1 moat —
`gates_hash` is a SHA over its raw bytes, so flipping that one flag moves `519158fa1450`, breaks the three
CRU-1 tripwires in `tests/crucible/test_version.py`, and makes every future manifest incomparable to the
entire recorded history. The CLI flag is the intended door; the config flag is a seal.

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

### 2. Every real substrate is still refused — but the curve now shows exactly where the guard opens

The corrected-contract **overlay** sweep **completed** (`calibration_mde_sweep_corrected.json`), and it
is the most useful number in this whole pass:

| T (bars) | holdout bars | measured MDE ΔSR | vs 0.50 ceiling |
|---|---|---|---|
| 756 | 189 | 1.789 | refuse |
| 1512 | 378 | 1.735 | refuse |
| 2782 | 696 | 1.234 | refuse |
| 4044 | **1011** | **1.281** | refuse ← *Taiwan sits here* |
| 6048 | 1512 | 0.789 | refuse |
| 8064 | 2016 | 0.862 | refuse |
| 16128 | 4032 | 0.554 | refuse (just) |
| 32256 | **8064** | **0.393** | **ALLOW** |

So the guard is not unconditionally shut: **it opens at holdout ≳ 8064 bars** on the overlay path, and it
is within ~10% of opening at 4032. That converts U8 from "get more breadth somehow" into a **measured
target**. Both real substrates (cross_asset holdout 1163, Taiwan 1011) sit at MDE ≈ 1.28 — a factor of
~2.6 away.

**8064 holdout bars is ~32 years of daily data, so it is unreachable on a daily panel** — but it is
routine intraday (15-min bars over ~3 years ≈ 25k bars; hourly over ~13 years). **Read that with the
caveat the calibration file itself states:** the sweep's generator is a *daily-scale* DGP with
`periods_per_year=252` baked into the `FitnessConfig`, so a high-`T` row means "MDE at N bars of THIS
per-bar signal-to-noise". Applying it to an intraday substrate additionally assumes intraday per-bar SNR
resembles daily — a modelling assumption this sweep does not measure. Validating that assumption is the
precondition on treating "go intraday" as the U8 answer.

The cross-sectional curve is still measuring; at matched depth it runs equal-or-worse than the overlay one
(the `v7.0` finding), and the stamp takes the WORST across the types a substrate mines — so the ALLOW
depth for a tick mining both types will be at least this deep.

The ceiling must not be raised to manufacture an ALLOW; that was explicitly refused at `v8.0` and the same
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

1. **Finish the cross-sectional corrected curve** (overlay one is done; xsec still running) so the power
   stamp is measurable for a tick mining both types, and U4's classifier has a finite MDE to reason about.
   Cheap, mechanical, unblocks honest tick logs.
2. **Test the depth hypothesis the overlay curve just handed us.** The guard opens at holdout ≳ 8064
   bars, which is intraday-reachable. Before committing to that, validate the per-bar-SNR assumption the
   sweep does not measure — i.e. re-measure the MDE curve on an intraday-scale DGP (or a real intraday
   panel) rather than extrapolating a daily-scale one. This is now the highest-value U8 experiment
   because it is the first one with a measured target instead of a direction.
3. **Resolve RC-11's mechanism** before the overlay path is allowed to promote anywhere — note this is
   the *same path* item 2 would open up, so the two are coupled: going deeper on the overlay path without
   resolving RC-11 would unlock a mine whose economic guard is un-calibrated on at least one substrate.
4. **Breadth (T×N) as the parallel lever.** Per-name `(T,N)` data is reachable (`v7.1`/`v8.1`) with TWSE
   T86 wired at 32/46 names bridged; N=8 effective is the binding number on the only per-name substrate.
   Note the measured caveat: breadth does NOT lower the MDE expressed in ΔSR (the SE of a Sharpe
   *difference* is set by time observations); what it buys is a larger ΔSR for the same per-name signal.
   So breadth moves a real alpha above a fixed ceiling; only depth moves the ceiling down.
5. Leave the ceiling alone.

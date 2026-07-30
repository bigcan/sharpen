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
Before the curves were measured: `UNDERPOWERED: implied MDE inf > ceiling 0.50 (unmeasured_empty)`.
**After** (the state as of this document): real cross_asset panel builds (`T=4652, holdout=1163`) →
`UNDERPOWERED: implied MDE 1.68 ΔSR > ceiling 0.50 (interpolated:cross_sectional)` → `mined=False,
0 PROMISING`, with the tick log's power columns populated (`implied_mde_delta_sr=1.6827`,
`power_interp_mode=interpolated:cross_sectional`). That is the difference blocker 1 made: the refusal is
now a **measured** statement about this substrate rather than "power was never measured here".

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

### 2. Both curves are now measured — and NO depth on the grid clears the ceiling for a normal tick

**Blocker 1 is CLEARED**: both corrected-contract curves now exist
(`calibration_mde_sweep_corrected.json`, `calibration_xsec_mde_sweep_corrected.json`, 24 cells).

Running the **production** stamp (`stamp_substrate_power`, which takes the WORST MDE across the candidate
types a tick mines — the `v7.0` rule) against the 0.50 ceiling:

| substrate | holdout bars | overlay MDE | cross-sec MDE | **worst (what the guard uses)** | verdict |
|---|---|---|---|---|---|
| cross_asset (real) | 1163 | 1.098 | 1.683 | **1.683** | refuse |
| taiwan (real) | 1011 | 1.281 | 1.833 | **1.833** | refuse |
| deep, 16k bars | 4032 | 0.554 | 0.835 | **0.835** | refuse |
| deep, 32k bars | 8064 | **0.393** | 0.644 | **0.644** | refuse |
| deeper than the grid | 16128 | — | — | `unmeasured_high` → +inf | refuse (fail-closed) |

> **Correction to an earlier reading of this data.** On the overlay curve alone, holdout 8064 gives MDE
> 0.393 and looks like an ALLOW. It is not one for a normal tick: the orchestrator mines
> `cross_sectional` **and** `overlay`, the stamp takes the worst across them, and the cross-sectional
> curve is worse at every depth (0.644 at holdout 8064). **No depth on the measured grid opens the guard
> for a both-types tick**, and past the top anchor the stamp is `unmeasured_high` → refuse by design. The
> binding curve is cross-sectional, exactly as `v7.0` predicted.

Two real consequences:

1. **The one door that does open is overlay-only at extreme depth.** A tick restricted to the overlay path
   on a ≥32k-bar substrate would stamp 0.393 and be ALLOWED. That is a genuine opening. *(This originally
   read "and RC-11 gates it, so the two findings are coupled" — RC-11 was WITHDRAWN 2026-07-31 as an
   artifact of a degenerate null, so nothing gates this door on calibration grounds.)*
2. **Breadth does not help here, and the measurement says so twice.** Cross-sectional MDE at holdout 8064
   is 0.420 at N=12 but 0.644 at N=100 — it does not improve with breadth, consistent with the
   calibration file's own note that ΔSR is already risk-adjusted so the SE of a Sharpe *difference* is set
   by the number of TIME observations. Breadth raises a real alpha above a fixed ceiling; **only depth
   lowers the ceiling.**

32k bars is ~128 years of daily data — unreachable — but routine intraday (15-min over ~3y ≈ 25k bars).
**Read that with the caveat the calibration file states itself:** the generator is a *daily-scale* DGP with
`periods_per_year=252` baked into the `FitnessConfig`, so a high-`T` row means "MDE at N bars of THIS
per-bar SNR". Applying it to an intraday substrate assumes intraday per-bar SNR resembles daily — a
modelling assumption the sweep does not measure, and the precondition on treating "go intraday" as the U8
answer.

The ceiling must not be raised to manufacture an ALLOW; that was explicitly refused at `v8.0` and the same
refusal holds.

### 3. ~~RC-11 — the Taiwan overlay path's economic guard is un-calibrated~~ — **WITHDRAWN 2026-07-31**

> **This blocker does not exist.** RC-11 was an artifact of a degenerate null (fixed-phase sinusoidal
> timing slot ⇒ effective n ≈ 4). Under a proper null the Taiwan overlay path admits **2.33%** of noise,
> not 37.6%, in line with cross_asset. `uplift_min = 0.10` is fine and the overlay path is not
> un-calibrated. See `docs/research/crucible_rc11_resolved_degenerate_null_2026-07-31.md`. What replaces
> it is **NULL-DEGEN-01**: the same fixed slot is shared by E1's null panels, so E1's FPR confidence
> bound assumes independence it does not have — worth re-measuring, but it is not a blocker on mining.
> Original text follows for the record.

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

1. ~~Measure the corrected-contract curves~~ — **DONE** (both present, 24 + 8 cells). U4's classifier now
   has a finite MDE to reason about on any measured substrate, and tick logs will state a real refusal
   reason instead of `unmeasured_empty`.
2. **Validate the depth hypothesis before betting on it.** The only measured opening is overlay-only at
   ≥32k bars, which is intraday-reachable — but the curve is a daily-scale DGP. Re-measure the MDE curve
   on an intraday-scale generator (or a real intraday panel) rather than extrapolating. This is the
   highest-value U8 experiment because it is the first with a measured target instead of a direction.
3. ~~Resolve RC-11's mechanism~~ — **DONE 2026-07-31, finding withdrawn.** The overlay path is NOT
   un-calibrated, so it no longer gates (2). Its replacement, NULL-DEGEN-01, is a measurement-quality
   item: re-run E1 with randomized timing slots to get an FPR bound whose independence assumption holds.
4. **Do NOT reach for breadth to fix the ceiling.** Measured twice now: cross-sectional MDE at holdout
   8064 is 0.420 at N=12 and 0.644 at N=100 — breadth does not lower it. Per-name `(T,N)` data is
   reachable (`v7.1`/`v8.1`, TWSE T86 at 32/46 names) and remains worth having, but as a way to raise a
   real alpha's ΔSR above a fixed ceiling, not to move the ceiling.
5. **Extend the sweep grid only if a substrate deeper than 32k bars becomes real** — past the top anchor
   the stamp is `unmeasured_high` → refuse, which is correct fail-closed behaviour, not a bug.
6. Leave the ceiling alone.

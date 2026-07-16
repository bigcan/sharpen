# Crucible: the `min_subperiod_ic_ir` gate — WIRED (crucible-v4.0)

**Date:** 2026-07-16 · **Session:** S553-cont-134 · **Branch:** `July2026`
**Decision:** WIRE the gate (not retire it), after repairing the estimator it reads.
**Version:** `crucible-v3.0` → **`crucible-v4.0`** (MAJOR — changes the verdict FUNCTION)
**Gate bytes changed:** NONE. Frozen moats `519158fa1450` / `22a18172be1a` / `0ccf6dd584f0` all hold.

---

## 1. The finding

`robustness.min_subperiod_ic_ir` was declared in **every** gates YAML
(`signal_eval.gates.yaml:29` — with the comment `# no negative subperiod` — plus
`taiwan_signal_eval.gates.yaml:44` and `taiwan_smallcap_altdata.gates.yaml:48`) and in
`gates.py::_DEFAULTS["robustness"]`, but **no code path ever read it**:

- `Gates.from_dict` never exposed it as a dataclass field — it survived only inside `Gates.raw`.
- `scorecard._finalize` built `promising` from `dsr` / `ic_ir` / `ic_tstat` / `fdr_q` / `hlz_pass`
  plus Tier-0 hygiene, and never touched `Robustness`.
- The only readers were `scorecard._rank_tuple` (a 4th-level rank tiebreaker) and
  `scorecard._card_json` (reporting).

**Consequence:** a signal that INVERTED in a subperiod still scored PROMISING on
dsr/IC/t/FDR alone. The small-cap probe pre-registration
(`taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md` §4, frozen) pre-registers
"**Robustness:** subperiod IC-IR ≥ 0.0" as a THRESHOLD — so the harness did not implement its own
frozen contract. This is the independent audit's "declared seal that never runs" family.

**A second dead key in the same block:** `robustness.recent_oos_years` was likewise never read —
`tier3_robustness` takes `recent_years: int = 2` as a hardcoded default and `evaluate_signal` never
passed it. The YAML value only "worked" by coinciding with the default. Setting it to anything else
was silently ignored. Now plumbed (a verified no-op: all three YAMLs say `2`).

## 2. Decision: WIRE, not RETIRE

Retiring would have required editing the gates YAMLs to document "reported-only", which moves
`gates_hash` — SHA-256 over **raw YAML bytes**, so even a comment moves it (`version.py::gates_hash`,
by design). That would break all three CRU-1 moats, including `0ccf6dd584f0`, the small-cap probe
seal. The pre-registration's own §6 records:

> the gates file (`0ccf6dd584f0`) was not edited after seeing results — **the anti-p-hacking seal held.**

Editing it now — *after* results are known — would retroactively falsify that sentence and produce the
exact optical signature of goal-post moving, in exchange for zero verdict benefit. **Wiring costs no
gate byte at all.** The pre-registration is the frozen contract; making the harness honor it is the
opposite of moving the goalposts. Retiring was rejected on those grounds.

## 3. Repair BEFORE wiring (spec §5 gate-repair-before-freeze)

The gate could not be wired onto the metric as it stood. `tier3_robustness` splits subperiods on raw
ROW index (`np.linspace(0, panel.T, n+1)`), so a panel whose active window is shorter than its date
range hands the leading subperiod a **coverage hole**. On the small-cap panel (bars from 2005,
membership from 2010-02-26) subperiod 1 held **48 valid days across a 1323-row span** — 3.6%
populated, against ~100% for its siblings — and returned a meaningless IC-IR of **1.536**. The
pre-registration §6 filed this and warned the defect can

> spuriously trip the `min_subperiod_ic_ir >= 0.0` gate on any panel whose active window is shorter
> than its date range.

`crucible-v2.0` set the governing precedent: the frozen gates hash "canonizes the FIXED gate, never
the known-broken one". So the repair ships first, in the same MAJOR bump.

**The repair (F2b).** A subperiod must clear BOTH an absolute floor (`_MIN_SUBPERIOD_VALID_DAYS` = 30)
and a populated fraction of its own span (`_MIN_SUBPERIOD_POPULATED_FRAC` = 0.5); otherwise its IC-IR
is NaN and drops out of the min/mean. Per-subperiod valid-day counts are now on the card
(`robustness.subperiod_valid_days`).

**Why the floor is span-aware, not a flat count.** The pathology is the *hole*, not shortness: a
125-day subperiod that is 100% populated is a fine estimate for its size, while 48/1323 is a gap. A
flat absolute floor conflates the two — at 126 days it would NaN *every* subperiod of a
short-but-complete panel (e.g. the `t=500` synthetic books in `tests/signals/`), making robustness
unmeasurable and silently blocking PROMISING on every short substrate. The span-aware rule is an
**exact no-op wherever the active window matches the date range** — every synthetic, calibration and
proxy book — so it touches only the pathology it targets.

**These floors are estimator-validity constants, not gate thresholds.** A starved subperiod is
DROPPED, never failed. They therefore live in code beside `n_subperiods`, not in a gates YAML — which
also keeps the frozen moats intact. (CLAUDE.md's "never hardcode gate thresholds" governs PASS/FAIL
promotion bars; this is a "is this an estimate at all" floor.)

## 4. CRU-1: every recorded verdict is preserved — verified, not assumed

The gate is monotone-STRICTER, so the 0-PROMISING mining record can only stay 0-PROMISING. The only
PROMISING in the entire record is the small-cap P1 probe. Checked against the recorded
`results/taiwan_smallcap_altdata/scorecard.json`:

| probe | subperiod IC-IRs (recorded) | min | recorded verdict | under v4.0 |
|---|---|---|---|---|
| `tw_smallcap_mom_rev` (P1) | 1.536, 0.347, 0.283, **0.109** | **+0.109** | PROMISING | **PROMISING** (passes) |
| `tw_smallcap_holder_conc` (P3) | 1.459, **−0.231**, −0.205, 0.059 | −0.231 | LOGGED | LOGGED |
| `tw_smallcap_margin_crowd` (P2) | 0.140, −0.094, **−0.185**, −0.143 | −0.185 | LOGGED | LOGGED |

P1's binding min (**+0.109**) comes from subperiod **4** — a ~1300-day window — not from the 48-day
hole, so the repair and the gate agree with the recorded verdict. P2/P3 were already LOGGED on
inverted sign + DSR 0.000; the gate only adds a second, independent reason to reject them.

The repair DOES change one **reported** number: P1's `mean_subperiod_ic_ir` (0.569 → 0.246) once the
48-day window is dropped. That makes the harness agree with pre-registration §6's own finding that
"the leading value in each list must be DISCARDED". No verdict moves.

**E1/E2 calibration is untouched.** The calibration harness gates on
`combination_fitness(...).passes_gate` inside `evolve()`, not on `scorecard._finalize`, and
`tier3_robustness` is called only from `scorecard.evaluate_signal`. The E1-GREEN record stands. The
Tier-C F1/F2 seals (`tests/signals/test_tier_c_seals_registered.py`) live in the `combination_fitness`
path and are likewise unaffected.

## 5. Known deviation: the pre-registration's CPCV kill clause — reviewed, NOT wired

Pre-registration §5 lists a kill condition this change does **not** close:

> CPCV subperiod IC-IR < 0 in a majority of folds

**Decision: leave it unwired.** Registered as Tier-C **C1**
(`tests/signals/test_tier_c_seals_registered.py::test_c1_cpcv_fold_distribution_does_not_gate_the_verdict`),
so a future change that makes CPCV gate trips a red and forces a conscious decision. Three reasons:

1. **The named statistic does not exist.** `tier3_5_cpcv` produces an OOS **Sharpe** per path
   (`CPCVResult`: `oos_sharpe_mean/std/p05`, `frac_paths_positive`). There is no per-fold IC-IR
   anywhere in CPCV. The clause welds a Tier-3 concept (subperiod IC-IR) onto a Tier-3.5 object
   (CPCV folds) — a **drafting error in the pre-registration**, not a harness gap. It is
   unimplementable as literally written.
2. **Gating on CPCV would contradict ADR-C2-2.** `tier3_5_cpcv` is documented "Verdict-neutral
   (reported only) ... the value is the OOS *distribution* + standard error vs a single estimate."
   CPCV is designed not to gate; wiring it is a design-boundary change, not a bug fix.
3. **The nearest analog protects nothing.** Wiring `frac_paths_positive >= 0.5` against the recorded
   probes:

   | probe | `frac_paths_positive` | clause verdict | recorded |
   |---|---|---|---|
   | `tw_smallcap_mom_rev` | 1.000 | PASS | PROMISING |
   | `tw_smallcap_holder_conc` | 0.733 | **PASS** | **LOGGED (NO-GO)** |
   | `tw_smallcap_margin_crowd` | 0.000 | KILL | LOGGED (NO-GO) |

   It would **pass a confirmed NO-GO** — weaker than the criteria that actually rejected P3 (inverted
   sign + DSR 0.000), and weaker than the `oos_sharpe_p05 < 0` caveat already in `_finalize`, which
   *does* fire on P3 (p05 −0.247). Recorded impact of wiring it: zero.

Wiring it would therefore be a MAJOR verdict-function change that implements a mis-specified
criterion, contradicts an ADR, and adds no protection. Per the standing rule the pre-registration is
**left unedited**; the deviation is recorded here. If the intent behind §5 is worth rescuing, the
honest route is a *new* pre-registration naming a statistic that exists — not a patch to this one.

## 6. Changes

| file | change |
|---|---|
| `signals/eval_harness.py` | F2b repair: span-aware valid-day floor; `Robustness.subperiod_valid_days` |
| `signals/gates.py` | `Gates.min_subperiod_ic_ir` + `Gates.recent_oos_years` exposed + validated |
| `signals/scorecard.py` | gate folded into `_finalize.promising` + caveats; `recent_oos_years` plumbed; `subperiod_valid_days` in card JSON |
| `crucible/version.py` | `crucible-v3.0` → `crucible-v4.0` + rationale |
| `tests/crucible/test_version.py` | version assertion; **three moat hashes unchanged** |
| `tests/signals/test_robustness_gate_f2b.py` | NEW — repair, wiring, and the CRU-1 regression above |

**Unmeasurable robustness is not a pass.** If no subperiod clears the floor, `min_subperiod_ic_ir` is
NaN and the verdict is LOGGED with an explicit caveat — mirroring the DSR leg's existing
`np.isfinite(dsr)` requirement. Absence of evidence must not read as evidence of robustness.

## 7. The CRU-1 moat was CRLF-fragile — fixed

`gates_hash` is a SHA-256 over RAW bytes, so it was line-ending-sensitive: a stock Windows checkout
(`core.autocrlf=true`) rewrote every gates YAML to CRLF and moved all three frozen hashes
(`519158fa1450` → `d04d7e747b48`). Reproduced on a fresh worktree — three CRU-1 tripwires red for a
reason with nothing to do with the gates.

Two consequences, the first dangerous: the obvious way to "fix" three red hash assertions is to
**re-pin the constants**, which would silently destroy the anti-goal-post-move seal outright. And
because `gates_hash` is stamped into `run_manifest` / `DiscoveryCard` / `CohortCard`, the recorded
provenance differed by platform, so `crucible reproduce` could not verify a run across OSes.

Fixed with `configs/*.gates.yaml text eol=lf` in `.gitattributes` — the same remedy the repo already
applies to container shell scripts for this exact bug class ("Bit us once on 2026-05-01"). The
`eol` attribute overrides `core.autocrlf`, verified end-to-end: with `autocrlf=true` still set, the
files were deleted and re-checked-out, came back LF, and all three moats held. Guarded by
`test_gates_files_are_lf_so_the_moat_is_portable`, which fails with an explicit **"do NOT re-pin the
hashes"** diagnosis rather than leaving three unexplained mismatches.

> **Existing checkouts:** the blobs were always LF, so this records no content change — but a working
> tree that already holds CRLF copies keeps them until refreshed
> (`rm configs/*.gates.yaml && git checkout -- configs/`). The guard test names this remedy.

## 8. Follow-up (not done here)

1. **`git tag crucible-v4.0`** — `run_manifest.crucible_version` must anchor to an immutable commit,
   so the tag has to follow the commit that carries this change.

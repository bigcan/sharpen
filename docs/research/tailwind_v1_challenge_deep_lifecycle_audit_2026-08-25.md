# TAILWIND-v1-challenge — Deep Lifecycle Audit (Tier-2), re-run

**Date:** 2026-08-25 · **Workstream:** `tailwind_v1_challenge` (linear cross-asset TSMOM + BAB,
18 free-data ETFs, monthly rebalance, FTMO/Velotrade prop-challenge path) · **Method:** 6 parallel
finder subagents (P1, P2, P3, P7+P8, P10, P4+P5+P6+P9+P11 — scoped this time to
`configs/tailwind_v1_challenge.yaml`, fixing the 2026-07-01 audit's `args.workstream` binding bug)
+ one skeptic pass (P1) + reconciliation against the existing 2026-08-18 Tier-2 · **Trigger:**
operator request, ahead of any prop-firm challenge attempt.

**Supersedes nothing, reconciles one thing.** This audit does not replace
`docs/research/tailwind_v1_deep_lifecycle_audit_2026-08-18.md` (committed `b5988a88` on branch
`Aug2026`) — it independently re-derives most of the same ground, blind to that document at launch
time, and confirms its verdict holds. Its value is (a) corroboration from a genuinely separate
pass, (b) several new findings that document extends, and (c) a **branch-hygiene finding that is
itself now the most urgent item on the list.**

---

## VERDICT: BLOCK (unchanged)

Nothing in this pass reverses or weakens the 2026-08-18 BLOCK. Six independent subagents, blind to
that document, converged on the same root cause. One severity disagreement was found and is
reported honestly rather than silently resolved (see §7).

Capital gate, restated:

| # | Item | State |
|---|---|---|
| 1 | Forward-path render | ⚠ CLEAR by exactly zero margin on the needless-termination leg (1/20 disjoint windows) |
| 2 | Sizing reconciliation | ✅ measured, mechanism now traced to file:line (this pass, §3) |
| 3 | Tier-2 audit | ❌ **BLOCK, confirmed twice now** |
| 4 | Protocol-v2 wiring | ✅ config-valid, but the controls it validates have no runtime consumer (§4) |
| 5 | Operator go-ahead | ⏳ open |

---

## 0. The finding that has to be fixed first, procedurally

### P-BRANCH · This worktree never received the fixes it was auditing · **S1, process**

Independently discovered by three of the six subagents (P7+P8, P10, P3), not prompted. This
worktree (`claude/open-items-end-goal-83d9ec`, HEAD `269398d8`) branched **before** the 2026-08-17
wiring fix (`300761c8`) and the 2026-08-18 Tier-2 audit (`b5988a88`) landed on `Aug2026`.
`git merge-base --is-ancestor 300761c8 HEAD` → **NOT ANCESTOR**, confirmed independently three
times. Concretely, running `validate_config.py --stage paper-deploy` **in this worktree** returns
the original 6 FAILs, not the WARN state the memory record describes as current.

This is exactly the failure mode `randd_log.md`'s own gap illustrates (§8): three weeks of real
work (cont-162 through the 2026-08-18 audit and the 2026-08-22 executor recompute) happened on a
branch this worktree, and this worktree's memory reads, never saw. **Any further tailwind work in
this worktree must first sync to `Aug2026`** — auditing or editing from here risks re-doing already
-fixed work or, worse, editing the stale pre-fix files and reintroducing closed bugs.

*Not fixed by this audit* — flagging for an explicit operator/merge decision, not silently rebasing
a worktree mid-audit.

---

## 1. Reconciliation with the 2026-08-18 audit's core finding

**P3-01** (certifying numbers computed on a research-basis reconstruction, not the wired executor)
and **P3-01a** (the gross cap binds on 99.13% of rebalances, so the book runs constant-gross, not
vol-targeted) both stand, independently re-derived. This pass adds the mechanism:

### P3-01-mech · Root cause of the static-vs-dynamic behavior gap, traced to file:line · new this pass

The research basis (`scale_causal()`, `tailwind_forward_path_render.py`) is a genuine **trailing
vol-target overlay** — it actively shrinks the book as trailing vol rises. The executor
(`MultiAssetAllocatorEnv._enforce_gross_exposure`) is a **static ceiling** — it only clips when
gross *exceeds* 3.0x, never proactively de-levers. Per-asset sizing is inverse-vol, not
portfolio-level vol-targeted. This is why needless-termination is 0.125 on the executor vs 0.050 on
the research basis: the executor cannot shed risk ahead of a vol regime shift the way the
certifying number assumes it can. **Improvement:** either wire a real portfolio-level trailing-vol
overlay into the executor, or stop citing the research-basis DSR/P(pass)/needless-termination
numbers as the challenge book's expectation.

### P7-03 / P8-01 · The same book-mismatch reaches two more capital-facing artifacts · new this pass

The 2026-08-18 audit's P3-01 named DSR/PBO/P(pass)/the render as computed on the research basis.
This pass found the **same mismatch independently reaches the Stage-4 recent-OOS artifact**
(`tailwind_stage4_recent_oos.py`'s `combined_book_frozen_scalars` hardcodes the own-capital 10%/10%
inverse-vol combine — no gross cap, no execution lag, no slippage). The already-thin OOS verdict
(`PF 1.1335` vs baseline `1.169`, 2 rebalances, `power_sufficient: false`) has never been
recomputed on the executor's actual mechanics either. **Fixing the executor-basis re-run once
closes both this and the original P3-01** — same script, two consumers.

---

## 2. New findings this pass adds (not in the 2026-08-18 document)

| ID | Sev | Finding | Effort |
|---|---|---|---|
| P11-01 | S2 | The project built CSCV/PBO overfitting control for the momentum **signal** grid (PBO=0.0009) but never extended it to the challenge's own **sizing/policy** grid (`vol_multiplier × bank_threshold`, swept by `challenge_simulator.sweep()` and `forward_path_render`'s vol-frontier). The render's razor-thin needless-termination pass (0.0500 vs ≤0.0500) is exactly the kind of result this class of control exists to catch, and a same-session "re-size to 6% vol" recommendation was proposed and retracted after misreading the noisiest cell as the strongest evidence — the garden-of-forking-paths failure mode, observed empirically, not hypothetically. | Next |
| P9-01 | S3 | `run_combiner_selection.py` (in-sample-grid-then-held-out-OOS combiner selection) was built and validated for a different workstream (C3) and has never been run for TAILWIND. The shipped `sleeve_combiner.mode: inverse_vol` is a **default, not a validated choice**, despite the gates file already pointing at a `combiner_beats_static` check. | Next |
| P6-02 | S3 | The P(pass) Monte-Carlo simulator's bootstrap RNG defaults to a single fixed `seed=7`; no test varies it. Whether the "recommended cell" (vol-frontier argmax) is seed-stable is unknown. | Next |
| P2-07 | S2 | The challenge simulator's own declared P(pass) (0.746, 0.711) is itself computed via a full-sample vol-normalization (`normalize_to_vol` on the whole combined series) before bootstrapping — the same look-ahead pattern flagged elsewhere in the codebase. Partially, not fully, mitigated: the render's Check E independently computes a **causal** P(pass) and gates on it, but no test asserts Check E actually fires. | Next |
| P3-04 | S3 | The challenge-specific executor recompute (`tailwind_executor_recompute.py`) uses `TwoSleeveExecutor.sim_oracle()` — explicitly documented as accounting-only / weight-axis-tautological — not the load-bearing `run_independent_recompute()` that actually catches look-ahead bugs. The independent-recompute tests exist but only exercise the own-capital fixture, never the challenge config's specific sizing. | Next |
| P3-02 | S2 | `tailwind_v1_challenge_v2.gates.yaml`'s `p_pass_expected` fields (0.711) are stale leftovers from the pre-resize 1.5x/15% cell; the live config's own `prop_firm.expected_p_pass` has the correct current number (0.746). `tailwind_executor_recompute.py` reads the **stale** field, silently understating the executor's real shortfall (0.615 vs 0.711 reads smaller than 0.615 vs 0.746 actually is) — doesn't flip BLOCK, but corrupts the magnitude anyone reads off that JSON. | Now |
| P10-retrain | S2 | `check_retrain_triggers.py` hardcodes `agent_type="sac"` and requires an RL checkpoint — it is structurally the wrong shape for a frozen linear rule book, independent of the already-known "no `retrain_policy` block declared" gap. Retrain-trigger wiring for TAILWIND needs a design decision (staleness + feature-drift KS, no SAC dependency), not a config patch. | Research |
| P8-06/07 | S3 | `ftmo_compliance_report.py` hardcodes the same two thresholds `tailwind_v1_stage4.gates.yaml` declares (duplicated, not sourced — the exact anti-pattern the project already tests against elsewhere), and has literally never been run against TAILWIND (zero references to tailwind/xsec_momentum/two_sleeve in the file). Separately, Velotrade's stricter `min_trading_days: 5` is never separately checked — only FTMO's looser 4-day rule is evaluated. | Next |
| P8-02 | S3 | The orphaned-gates bug (config pointing at rejected 15%-vol gates) is confirmed fixed on `Aug2026` — but the fixed file's own `provenance:` footer still says "drafted 2026-07-02," "1.5x," and describes the 15%-vol capital gate. A reader who scrolls to the footer for the summary gets stale sizing. | Now |
| P1 (render) | S1 (contested — see §7) / S2 | The forward-path render — one of the explicit capital-gate preconditions — never touches `cross_asset_loader.fetch_and_clean` (the DATA-CLEAN'd path); both sleeves route through a raw, unmanifested `yf.download` cache. | Now |
| P1-08/09/10 | S3 | No dedup/monotonic-index assertion on the OHLCV panel; `clean_ohlcv.py`'s CLI entry point covers only Gold/BTC (ETF coverage is via direct function import, invisible from CLI inspection); the stale-run detector's floor was calibrated for 1-min bars and is untested on daily bars. | Now / Next |

---

## 3. Positives independently reconfirmed (not assumed)

- SHORT-ACCT re-verified directly on `paper_state.py`, the module the challenge executor actually
  runs (no config-conditional branch) — no `notional_debt`, symmetric long/short. Confirms the
  2026-07-01 finding transfers, rather than assuming it does.
- The fill model (`SimFillEngine`) mirrors the env's transaction-cost calc line-for-line and is
  confirmed to be what actually produced the 0.549 executor Sharpe — real friction, not a
  frictionless artifact.
- Gross-cap enforcement and step-ordering ("decide t, earn t→t+1") are causal at the mechanism
  level — the problem is the cap is static, not that it look-aheads.
- The `resolve_gates_path()` de-orphaning fix genuinely works (raises loudly rather than falling
  back to a stale constant) — confirmed by direct read.
- LEAK-1/LEAK-2 signal-layer causality is genuinely clean; `assert_causal`'s current-bar
  perturbation leg (the exact blind spot that let the historical X2 leak survive elsewhere) is
  present and wired at every production call site.
- The data panel itself is empirically clean (0 mid-series NaN, 0 flat-close runs ≥5 bars, extreme
  moves trace to real events) — the defect is the missing gate, not corrupted data.

---

## 4. Live/drift wiring (P10) — reconfirmed as the dominant structural fault

Independently reconfirmed: `ActionDriftTracker` and `CostDriftTracker` are each instantiated at
exactly one call site (`crypto/live/live_engine.py`), an entirely different execution path from
anything that runs TAILWIND. The config's `drift.enabled: true` block validates green with zero
runtime consumers. `kill_file`'s only real reader is container-scoped (`watchdog_docker.py`), and
TAILWIND has no container entry in `docker-compose.yaml`. `paper_soak`'s 21 numeric gates crash on
a missing `performance:` block before ever running. No runner exits non-zero on a blocking verdict
— confirmed to be the actual mechanism by which a stale `RENDER_CLEAR` survived undetected for ~20
hours per the 2026-08-18 record.

---

## 5. Updated "minimum before any capital"

Carrying forward the 2026-08-18 list, with this pass's additions marked **[new]**:

1. Recompute DSR/PBO/P(pass)/render **and the Stage-4 OOS artifact [new: P8-01]** on the
   executor's actual mechanics, using the load-bearing independent-recompute path, not the
   accounting-only oracle **[new: P3-04]**.
2. Decide whether constant-gross is intended; if not, re-derive `max_gross_exposure`/
   `target_vol_asset` so vol-targeting actually binds — now traced to a specific mechanism
   (static ceiling vs. causal overlay) **[new: mechanism]**.
3. Model borrow/financing/cash rate on both mechanisms.
4. A real-data end-to-end executor test with production mechanisms ON.
5. Add a `performance:` block to both challenge gates files, run the paper validation end to end.
6. Non-zero exits on `RENDER_FLAGS`/`overall_status != PASS`.
7. Wire or explicitly retire the drift/kill controls.
8. `lag=-1` BAB leak probe; un-skip the keystone parity test; vary `perturb_frac`; un-pad DSR's
   `T`.
9. Route the six inline-only validator checks through the gates overlay.
10. **[new]** Sync `p_pass_expected` and the provenance footer in `tailwind_v1_challenge_v2.gates.yaml`
    to the current 10%/1.0x sizing (P3-02, P8-02) — cheap, corrects a magnitude-reading error.
11. **[new]** Extend CSCV/PBO-style deflation to the challenge's own sizing/policy grid search
    (P11-01) — the project already owns this control, it's just not applied where it's needed most.
12. **[new]** Run `run_combiner_selection.py` against the tailwind config so the shipped combiner
    is a measured choice, not a default (P9-01).
13. **[new]** Report P(pass) seed-sensitivity across ≥10 RNG seeds (P6-02).
14. **[new, blocking on process]** Sync this worktree/branch to `Aug2026` before any further
    tailwind capital-path work happens from here (§0).
15. **[new]** Point the DATA-CLEAN gate at the render's actual price path, or gate the render on a
    manifest-verified cache (P1, render).

---

## 6. Coverage of this pass

**Pillars run as finders:** P1, P2, P3, P7, P8 (combined), P10, P4/P5/P6/P9/P11 (combined) — all
11 pillars, scoped correctly to `configs/tailwind_v1_challenge.yaml` (fixing the 2026-07-01 audit's
workstream-binding bug, which had swept the whole repo instead).

**Skeptic-adjudicated:** P1 only, for budget reasons — the other five pillars' findings were
instead cross-checked against the existing, independently operator-verified 2026-08-18 document,
which functions as a stronger check than a blind skeptic pass for the ground it already covers.
Genuinely new findings (§2) were not available for that cross-check and should be treated as
finder-only (not yet skeptic-adjudicated) until a future pass.

**Structurally N/A, reconfirmed:** P5, P6 (in the RL sense), P7 (in the fold sense) — TAILWIND is a
frozen linear rule, never trained, no seeds, no folds. This pass found real, non-RL-shaped analogs
for each (the momentum-grid PBO check, the P(pass) simulator's own seed, and the disjoint-window
resampling) rather than reporting a blank N/A.

---

## 7. Disagreement, reported not resolved

**P1 (forward-path render skips DATA-CLEAN):** this pass's finder→skeptic pipeline held severity at
**S1** ("will cause live loss/leakage") after independent verification of the full call chain. The
2026-08-18 document capped the same underlying fact at **S2**, explicitly reasoning that the panel
was checked and found clean (0 NaN, 0 flat runs, real extreme moves) — "what is missing is the
gate, not the correctness." Both readings are evidence-based, not sloppy: S1 is the correct grade
for "no gate exists," S2 is the correct grade for "and nothing has gone wrong yet." Recommend
resolving this the way severity ties are supposed to resolve here: **grade the absence of the gate
at S1** (a gate that would not fire if the data did go bad is the failure mode CLAUDE.md's own
DATA-CLEAN invariant exists to prevent, and the project has been burned by exactly this once
already, S106/gold) — but flag explicitly, as both documents do, that no evidence of actual data
corruption exists today.

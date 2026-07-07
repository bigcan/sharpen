# Crucible Taiwan substrate upgrade — `taiwan_manual` (v1) → `taiwan_v2`

**Date:** 2026-07-06 · **Session:** S553-cont-117 · **Phase:** P1 (of the P0–P3 Taiwan power-restoration plan) · **Crucible:** `crucible-v2.8`

This is a **pre-registration**: it is written and committed *before* the first `taiwan_v2` tick is
scored, so the decision to re-open the overlays cannot be a post-hoc reaction to a better result.

## Why a new substrate directory at all

The original Taiwan out-dir `results/crucible_orchestrator/taiwan_manual/` (live since S553-cont-115)
scored its overlay hypotheses on a **power-starved data surface**. The TWSE T86 institutional-flow
connector had a 400-day lookback cap, so every `twse_inst:*` overlay slot carried only **~267 of the
panel's ~2796 bars (~9.5%)**:

- SE(SR) ≈ 1.0 on ~1.06 yr of data.
- The pre-registered gates (`configs/taiwan_signal_eval.gates.yaml`) require `hlz_t_min = 3.0` on the
  **marginal** contribution plus `promising_dsr = 0.90`. On 267 bars, a flow overlay would need a
  marginal ΔSR ≈ 3 to clear — real institutional-flow signals earn ΔSR ≈ 0.1–0.4.
- **On that substrate, any proposer — however diverse — returns 0 PROMISING with probability ≈ 1**,
  while every tick drains the substrate's LORD++ online-FDR wealth. v1's null was *correct at its
  power*, but uninformative: it never tested the hypotheses at a power where they *could* have passed.

**P0 (committed `6303fd65`)** fixed the data half: a persistent day-keyed T86 accumulation store plus a
one-shot 2015→today backfill lifted the institutional-flow slots to **~2782 bars** (verified: store
holds 4202 days polled, 2790 with data, 2015-01-05 → 2026-07-07; `0050` coverage 2782). SE(SR) drops
from ~1.0 to **~0.30**, making a ΔSR ≈ 0.6–0.9 edge *detectable* instead of impossible. This is a
**materially different data surface** — a new `data_snapshot_hash`.

## Why not just re-run against `taiwan_manual`

The ledger dedup key is the **canonical formula string only** — there is no substrate/data component in
`candidate_hash` ([`hypothesis.py`](../../finrl_pro_ds/crucible/agentic/hypothesis.py) `candidate_hash`).
So the ~24 overlays already scored in `taiwan_manual` at ~1/10th power are **permanently hash-blocked**
there. Re-running against `taiwan_manual` would silently skip exactly the hypotheses we most need to
re-test, scoring only never-seen slots while the already-seen ones stay frozen at their low-power
verdicts. That is not what we want, and quietly re-testing a locked hypothesis on the *same* ledger
would also be a multiple-looks FDR violation (the p-hack Crucible exists to prevent).

A **fresh out-dir** is the clean, honest move:

- fresh `trial_ledger.db` → empty dedup set → **all** ~101 library proposals (the 24 old overlays +
  the ~69 never-scored slots + 8 cross-sectional) are re-opened at full post-backfill power;
- fresh `orchestrator.db` → a **fresh LORD++ FDR account** seeded from the substrate config, correct
  for what is genuinely a *new experiment* on a *new data surface* — not a second look at old data.

This is legitimate precisely because the data surface changed materially. It would be illegitimate as a
way to "get another roll of the dice" on unchanged data; it is not that.

## Report-both discipline (no cherry-picking)

`taiwan_manual/` is **kept and frozen** as the archived v1 record (see its `README_ARCHIVED.md`). Both
substrates are reported together going forward:

| substrate | overlay power | status | verdict-so-far |
|-----------|---------------|--------|----------------|
| `taiwan_manual` (v1) | ~267 bars (~9.5%) | **archived / frozen** | 0 PROMISING (correct-at-its-power, uninformative) |
| `taiwan_v2` | ~2782 bars (~98%) | **active** | *to be scored* |

If `taiwan_v2` also yields 0 PROMISING, that is the honest, informative null we could not get before —
and it is reported as such, alongside v1. We do not discard v1 to make v2 look like the only run.

## What P2 / P3 add on top

- **P2 — exhaust the library batch.** `taiwan_tick.ps1` now passes `--max-proposals 128` (was 32). The
  offline `LibrarySeedProposer` emits ~101 proposals (8 cross-sectional + 3 templates × ~31 feature
  slots); the old cap truncated ~69 overlay slots (semi/gold/wti/esg…) that were **never scored**. 128
  emits the full batch. Deterministic, offline, zero tokens.
- **P3 — LLM proposer available, opt-in.** `taiwan_tick.ps1 -Llm` selects the LLM-backed proposer
  (spec Part A2), which needs `ANTHROPIC_API_KEY` (fails closed without it). Its value over the 3-template
  library is composite/interaction formulas the templates cannot express; the orchestrator now also
  hands every proposer CR-1-legal **data-shape hints** (cross-section width, per-slot bar counts) and a
  per-tick nonce so the LLM can steer toward the deep, powered slots and avoid re-emitting an identical
  batch at temperature 0. None of those hints is a score/verdict (CR-1 preserved).

## Honest ceiling

Even at full power the gates demand a **strong** marginal edge (ΔSR ≈ 0.6–0.9, t ≈ 3 over ~11 yr), and
cross-sectional formulas stay rank-starved on the N=10 Taiwan panel (steer the proposer toward
overlays). **0 PROMISING remains the modal, correct outcome.** The value of this upgrade is that a null
is now *informative* — a fair test the hypotheses could have passed — rather than a foregone conclusion
of low power. No capital action follows a PROMISING survivor without the human-triggered Tier-2 deep
lifecycle audit; this pre-registration changes none of that.

## Invariants preserved

- **CRU-1** — funnel `gates_hash` frozen at `519158fa1450`; no gate byte changed. The offline library
  proposer's output is byte-identical whether or not the new data-shape hints are populated (locked by
  `tests/crucible/test_proposer_datashape.py::test_library_proposer_output_invariant_to_datashape`), so
  no existing verdict or manifest moves.
- **CRU-2 / CR-1** — the proposer still sees only `ledger_agent_view` (dedup keys + killed families)
  plus panel data-shape; no verdict/DSR/holdout is reachable. The new context fields are sourced from
  the panel + tick timestamp, never the ledger.
- **LEAK-2** — the T86 store is causally inert (raw reference-day values; the +1-day release stamp and
  as-of cutoff stay at read time), guarded by the P0 connector parity + as-of tests.

# Crucible Mining Log

The campaign-level record of what has actually been mined. Every run's own artifacts (tick row,
trial ledger, manifest, gates hash) make that *run* reproducible; none of them answer **"what have we
tried, at what settings, at what cost, and what did it rule out?"** This file does, and it is
deliberately independent of any single store.

Two files, two jobs:

| file | authored by | contains |
|---|---|---|
| `crucible_mining_log_facts.md` | `scripts/research/crucible_mining_log.py` | every tick found in every store, deduped, classified. **Never hand-edit.** |
| this file | a human/agent after each campaign | why the campaign was run, what bound it, what it rules out, what would reopen it |

```bash
python scripts/research/crucible_mining_log.py --root C:/FinRL/FinRL-Pro_DS --root C:/tmp --jsonl results/crucible_mining_log/ticks.jsonl
```

Pass every scratch root explicitly. Decisive runs keep landing there (see gap **G3**).

## Why an independent log was needed

1. **The record is scattered.** 50 unique ticks across 21 stores in 9 `results/` directories, other
   worktrees, and scratch dirs. No store holds the campaign.
2. **`promising = 0` is ambiguous.** Of the 23 ticks that mined, **17 never reached the holdout
   gate** — the zero was a screening artifact, not a result. The funnel records this
   (`n_holdout_tested`, since `crucible-v12.0`) but nothing surfaced it, and two separate multi-week
   investigations were spent re-deriving the same misreading.
3. **`crucible_hypothesis_loop.py` writes no tick.** A manual cycle produces a `trial_ledger.db` and
   `loop_summary.json` only, so mining done that way is invisible to `orchestrator.db`.

## Campaign entry schema

Append one entry per campaign — a contiguous set of ticks sharing substrate, version, and question.
Keep it to the eight fields; the numbers belong in the facts file.

```
### Cnn · YYYY-MM-DD · <substrate> · <crucible-vX.Y>
- **Question** — the thing this campaign would have changed our mind about
- **Config** — hold horizon, cost_bps, holdout_frac, gates file, bank size/source
- **Store** — path + canonical|scratch
- **Reached** — mined / scored / holdout-tested / cohort-adjudicated
- **Verdict** — and whether it is a RESULT or VACUOUS (holdout-tested > 0?)
- **Binding constraint** — power, turnover, breadth, supply, or a code defect
- **Cost** — FDR wealth charged
- **Reopens if** — the specific measurable change that would justify re-running
```

---

## Campaign ledger

### C01 · 2026-07-02 · `cross_asset` · `crucible-v2.6`
First real-data mine. 8 pre-registered, 10 scored, 0 promising; three following ticks correctly went
IDLE to conserve wealth. **VACUOUS** — pre-v12, holdout reach unrecorded. Cost 0.0399.
Store `results/crucible_orchestrator/real`.

### C02 · 2026-07-03 · `cross_asset` (overlay) · `crucible-v2.6`
Overlay-shaped bank, 20 pre-registered / 20 scored, 0 promising. **VACUOUS**. Cost 0.0440.
Store `results/crucible_orchestrator_overlay/real`.

### C03–C05 · 2026-07-05 → 07-13 · `taiwan` · `crucible-v2.8` → `v2.9`
Three campaigns on the Taiwan cross-asset panel (`taiwan_manual`, `taiwan_v2`,
`taiwan_llm_diversity_probe`): 9 mining ticks, 225 pre-registered, 160 scored, **0 promising**, cost
0.1398 total. All **VACUOUS** on holdout reach. C05 specifically tested whether an LLM proposer
raises hit rate — it did not; the proposer is not the bottleneck
(`project_crucible_llm_proposer_*`, `MEMORY.md`).
**Terminated by the power guard**: once `panel_T = 4044 / holdout 1011` was measured, MDE ΔSR
**1.40** vs the 0.50 ceiling, and the last three ticks are `SKIP_UNDERPOWERED`.
**Reopens if** a longer or broader Taiwan cross-asset panel brings MDE under 0.50.

### C06 · 2026-07-30 → 08-01 · `cross_asset`, `taiwan` · readiness re-checks
Four ticks, all `SKIP_UNDERPOWERED` (MDE 1.68 / 1.83 / inf). This is the power **units-error**
correction landing: MDEs are ΔSR per 252-**bar** year, so 22.6× more bars buys zero power
(`project_crucible_power_units_error_2026_08_03`). Cost 0.
**These skips are the system working.** Do not read them as failures to mine.

### C07–C08 · 2026-08-02 → 08-04 · `intraday`, `intraday_fx`, `cross_asset` H=1 · `crucible-v11.0`
Four mining ticks, 56 pre-registered, 50 scored, 0 promising, cost 0.1652. All **VACUOUS** on holdout
reach, and all measured underpowered anyway: MDE **1.906** (intraday, T=77 298), **1.960**
(intraday_fx, T=115 985), **1.683** (cross_asset, T=4 652).
**`intraday` / `intraday_fx` / `cross_asset` are CLOSED after 8 attempts.** More bars do not help —
that is exactly what the units error obscured. Reopens only on a genuinely wider cross-section.

### C09 · 2026-08-09 → 08-10 · `us_equity` · `crucible-v12.0`
The first substrate whose MDE (**1.312**, T=4 930, holdout 1 726) was inside reach of a real signal.
Three mining ticks: the first (8 prereg) still **VACUOUS**; ticks 39–40 are the project's **first
TESTED mines — 17 and 9 candidates adjudicated on holdout**, 0 promising. Cost 0.0462.
Found here: unscored preregs deduped themselves forever, so ticks reported `promising=0 status=OK`
while testing nothing (the 5-week livelock — `project_crucible_us_equity_first_powered_mine`).

### C10 · 2026-08-10 · `us_equity` · `crucible-v12.1`
The crossing: `loop.py`'s hard `candidate_type == "overlay"` filter removed, so cross-sectional
candidates finally reach the one gate with measured power. 107 pre-registered, 20 scored, **16
holdout-tested**, 0 promising. Cost 0.0020.

### C11 · 2026-08-11 · `us_equity` · `crucible-v13.0` (cohort-only)
A tick that mines nothing and still spends wealth: the cohort re-adjudicating an existing pool
(`COHORT_ONLY`, 7.06e-06). Four cohort verdicts across C10–C12, all null.

### C12 · 2026-08-11 · `us_equity` at H=21 · `crucible-v13.0` · **sidecar store**
The decisive campaign. Run at `hold_horizon 21` on a **separate** store with `--no-power-guard`, so
it neither spends the production account nor conflates two hypotheses (the ledger dedups on formula,
and the same formula at H=2 and H=21 is not the same hypothesis).
145 pre-registered, **98 holdout-tested** (97 on the `holdout_frac 0.60` variant), 0 promising;
cohort p **0.5534** and **0.2667**, `t_obs` ≈ 0. Cost 0.0482 each.
**Binding constraint — the disjoint window:** powered needs H ≤ 2, where 97/100 WQ101 alphas are
untradeable (68–135/yr turnover vs a 24/yr cap); tradeable needs H ≈ 21, where the power ceiling is
0.626 vs MDE 1.312. No holdout split bridges it — H=21 needs **30.1 holdout-years against a 19.56y
panel**, and the two gates move in *opposite* directions under `holdout_frac`.
⚠ Do not read p 0.5534 → 0.2667 as progress: different samples, not different power levels.
**Reopens if** the panel gets ~2.5× more years, or a higher-IC low-turnover bank than WQ101 appears.
Store: `results/crucible_sidecar/us_equity_h21{,_ho60}/real` (copied out of `C:/tmp` 2026-08-12; the
tmp originals are left in place).

### C13 · 2026-08-11 · `taiwan_smallcap` · `crucible-v13.1` · **WIRED, NOT MINED**
The panel holding the project's only PROMISING got a mining path (`941acbcf`). Breadth measured:
`n_eff` median **38.1 at H=1** (57/100 alphas usable) — better than us_equity's 21.5 — but Taiwan's
0.30% sell-side tax forces H=21 / `cost_bps 0.0021`, where **0/100 alphas clear |IC| 0.004**. Power
guard refuses independently (MDE **1.554** vs 0.50). **Zero ticks. Zero wealth spent.**
Mining needs an explicit `--force-underpowered`, an operator call not yet taken.

---

## Substrate board

| substrate | ticks | mined | TESTED | holdout-tested | PROMISING | wealth | MDE | status |
|---|---|---|---|---|---|---|---|---|
| `cross_asset` | 10 | 4 | 0 | 0 | 0 | 0.1694 | 1.683 | **CLOSED** — underpowered, 8 attempts |
| `intraday` | 1 | 1 | 0 | 0 | 0 | 0.0399 | 1.906 | **CLOSED** — underpowered |
| `intraday_fx` | 1 | 1 | 0 | 0 | 0 | 0.0399 | 1.960 | **CLOSED** — underpowered |
| `taiwan` | 20 | 9 | 0 | 0 | 0 | 0.1398 | 1.403 | **CLOSED** — underpowered |
| `us_equity` | 16 | 6 | 5 | 237 | 0 | 0.1446 | 1.312 | **CLOSED on both axes** — powered and tradeable windows disjoint in H |
| `taiwan_smallcap` | 0 | 0 | 0 | 0 | – | 0 | 1.554 | **WIRED, never mined** — power guard refuses |

Counts are from the facts file; MDE is the production-config value. `us_equity`'s 237 holdout tests
include the two scratch H=21 runs.

## What the record teaches

**L1 — A cheap upstream screen that is stricter than the gate it protects will silently prevent the
gate from ever running.** Four instances: the `dsr_aug`/`marginal_t` pre-filter (0/170), the
`cohort_eval.py:320` hard return on 0 seals, the unscored-prereg dedup livelock, and
`substrate_dirty` keying only on per-candidate novelty so a never-run cohort read as "nothing to do."
*Check:* every screen must be reported alongside the gate's own pass count, never instead of it.

**L2 — Read `n_holdout_tested` before reading any `promising = 0`.** 17 of 23 mining ticks proved
nothing. The facts file now classifies this as `SCREENED_ONLY` / `SCREENED_UNKNOWN`.

**L3 — Power is measured in bars, not calendar time.** 22.6× more bars bought zero power. Any claim
of the form "more data will fix this" must be stated as ΔSR per 252-bar year.

**L4 — Breadth is a property of the substrate *and* the signal, not of the return covariance.** The
WQ101 bank realizes `n_eff` 21.5–81.7 on us_equity and 38.1 on taiwan_smallcap, but 6.8 on
cross_asset and 0.6 on FX — which is why re-mining those is not justified.

**L5 — Two constraints can each be satisfiable and jointly unsatisfiable.** Turnover feasibility and
statistical power both move with H, in opposite directions. Check the *pair* before mining, not each
in turn: `scripts/research/crucible_hold_horizon_tradeoff.py`.

**L6 — Zero is the modal outcome even for a working machine.** E[discoveries] ≈ 0.25 per campaign at
this hit rate; TSMOM (SR 0.60) would be promoted only 5–7% of the time. The 0-alpha record is
evidence about the hypothesis **bank**, not a verdict on the markets
(`project_crucible_zero_alpha_root_cause_2026_08_09`).

## Before mining, in order

1. `python scripts/research/crucible_mining_log.py` — has this substrate/config been mined? At what
   cost? Was the prior zero TESTED or VACUOUS?
2. Measure breadth on the *actual bank* (`crucible_real_alpha_breadth.py --panel <s>`), not the
   covariance participation ratio.
3. Check power and turnover **together** (`crucible_hold_horizon_tradeoff.py`) — L5.
4. Confirm the bank has unscored members for this substrate; an exhausted bank mines nothing while
   reporting `status=OK` (L1).
5. Point the store at `results/`. A run that must not spend the production LORD++ account goes to
   `results/crucible_sidecar/<name>/`, never to a temp dir (G3).
6. After the run: regenerate the facts file, append a campaign entry, and state whether the zero is a
   RESULT or VACUOUS.

## Instrumentation gaps

- **G1 — `rejection_class` was empty in every store that has the column. ✅ CAUSE FOUND + FIXED
  2026-08-12.** The classifier was never disagreeing; it was never reached. `search_memory_cfg` DID
  arrive (`search_memory_gates_hash` is pinned in the manifests), and `us_equity`'s power stamp is a
  finite 1.312 ΔSR — so both of the suspected conditions held. The failure was a **set mismatch**: the
  class was computed while writing the SURFACED rows (`hall_of_fame ∪ promising`), but under
  `offspring_policy: prereg_only` (v12.0) the holdout gate adjudicates PRE-REGISTERED specs, while
  `hall_of_fame` is `ranked[:10]` over every genome the search scored — which offspring win on train
  fitness by construction. Measured in the ledgers: **20/20 surfaced rows are offspring** (`spec_json`
  NULL) and **145/145 pre-registrations** were written by the second ledger loop, which recorded no
  class at all. The two sets never intersected, so 98 adjudicated rejections produced 0
  classifications. Fixed in `loop.py` by classifying on "the holdout gate rejected this
  `(candidate_type, formula)`" in **both** ledger loops; regression test
  `test_every_holdout_adjudicated_rejection_carries_a_class` (it fails 6/7 on the pre-fix code).
  ⚠ **The Flags line will keep firing for the 15 pre-fix stores and cannot be back-filled** — the
  holdout adjudications live only in the in-memory `GenerationReport`, never in the manifest or any
  store. It clears for stores written after the fix.
- **G2 — `crucible_hypothesis_loop.py` writes no tick row**, so manual cycles are invisible to the
  orchestrator record. The extractor labels those stores `manual-cycle` from a bare `trial_ledger.db`
  but cannot recover mined/scored/holdout counts.
- **G3 — Deliberately-separate accounts need a durable home. ✅ CONVENTION SET 2026-08-12.** The two
  H=21 campaigns that closed `us_equity` (C12) were written to `C:/tmp` — the right call for the
  LORD++ account, wrong for the record, since the finding evaporates when tmp is cleared. They now
  live at **`results/crucible_sidecar/<name>/`**, which is the home for any run that must not spend
  the production account. Note `results/` is gitignored: this protects against tmp cleanup, not
  against machine loss. Anything that must survive in git belongs in `docs/research/`.
- **G4 — `--no-power-guard` runs record no power at all.** C12's ticks have NULL `panel_T`,
  `holdout_bars`, and `implied_mde_delta_sr` — the runs that most need their power stated are the
  ones that state none. **This now has a second consequence:** with no MDE stamp, `classify_rejection`
  fail-safes to `None`, so a `--no-power-guard` run still records no `rejection_class` even after the
  G1 fix. That is correct (an unmeasured test cannot be called decisive OR underpowered), but it means
  a sidecar run buys no search memory. When a run needs to proceed on an underpowered substrate but
  should still STAMP its power, use the default `--power-gates` with `--force-underpowered` (which
  overrides the refusal and keeps the stamp) rather than `--no-power-guard` (which drops the stamp).

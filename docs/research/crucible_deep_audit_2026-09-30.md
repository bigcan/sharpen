# Crucible deep audit — 2026-09-30 (`crucible-v15.0`)

Branch `claude/crucible-deep-audit` (from `agentic-trading-lab` @ `20c8e263`). Whole-platform audit, not
diff-scoped: five parallel finders (funnel statistics · GP/DSL engine · core + orchestrator · agentic +
lockbox + governance · data connectors), every claim reproduced with a scratch script, plus a profiled
synthetic tick. Every fix ships with tripwire tests that were run against the pre-fix code and FAIL there
(mutation-checked), and against the fix and PASS.

## What this changes about the record

1. **Pre-registrations were screened, not tested — fifth instance of the project's recurring shape.** The
   GP's search bounds (24 AST nodes; turnover > 2x soft cap) culled pre-registered seeds before fitness.
   40 of the 100 published WQ101 formulas exceed 24 nodes — including curated seed #9, never tested on any
   substrate — and the us_equity H=2 seeds ran 88-168 turnover/yr. The culled specs were ledgered
   SCORED_NOT_SELECTED and **charged a LORD++ test**: the 2026-08-10 us_equity extended-bank tick
   pre-registered 107 and adjudicated 16. The us_equity account holds 152 charged tests for 42 real ones
   (next level 7.0e-06, z≈4.34; compacted to its real tests it would be 3.6e-05, z≈3.97).
   ⚠ **The v12.0 diagnosis was wrong:** the 8 us_equity seeds that motivated it were turnover-culled, not
   uplift-culled; production never tested them. The "0/8, best corrected_t 0.44" evidence came from a
   forensics script that re-implemented scoring without the cull — the verdict stands, the mechanism does not.
2. **The pre-registered direction was not the tested direction.** `expected_sign` was stored and never
   applied; every scoring path trades the formula as written with a one-sided test. 15 of 53 LLM specs in
   `taiwan_v2` (10 of 41 in the diversity probe) declared -1 on an un-negated formula and were scored in the
   mirror image. 5 of the curated 8 were labelled -1 against the WQ101 pre-signed convention.
3. **The binding gate had false-positive channels.** The combiner's all-sleeve equal-weight fallback made an
   unusable candidate turn the "augmented" book into equal-weight(base) vs inverse-vol(base): a constant
   positive stream (cash) scored **z 2.7-15.6** and passed; an all-zero stream passed 0.3-2.3%; v12.0 had
   silently dropped the F14-4 degenerate floor for pre-registrations.
4. **LEAK-2 on live mining paths.** COT release stamps ignored the 2025 shutdown (~16 weekly reports up to
   seven weeks early, inside every current holdout, 18 of 24 us_equity slots), the 2013 and 2018-19
   shutdowns, the 2023 ION outage and ad-hoc federal closures; WALCL on Thursday-holiday weeks; the forward
   label was conditioned on FUTURE universe membership (taiwan_smallcap H=21 dropped 4.81% of labels, mean
   -1.44% vs +1.24% kept, roughly doubling a reversal probe's IC-IR).
5. **Two protections were not live.** The us_equity market-beta leg (configured 0.30) read the previous
   bar's market move against the next bar's book return — it measured beta 0.03 on a 0.54 book and could not
   fire. And POWER-LORD-01 (the power guard at the LIVE LORD++ level) is **not wired** — commit `0c12f38b`
   says so; the auto-memory said otherwise.


## Fixed (by severity)

| # | Severity | Defect | Fix | Tests |
|---|---|---|---|---|
| 1 | CRITICAL | Pre-registrations culled by search bounds, charged, ledgered "scored and lost" | Seeds exempt from search bounds under the corrected contract (never bred); `NOT_TESTED` verdict; LORD++ charged only for holdout decisions; `n_holdout_tested` counts decisions; canonical prereg matching | `test_generation_evolve_v15`, `test_v15_0_fixes` |
| 2 | CRITICAL | `expected_sign` never applied | Author folds -1 into the traded formula; curated signs +1; LLM prompt states the convention | `test_v15_0_sign` |
| 3 | CRITICAL | Degenerate candidates pass the binding gate | `augmented_book` (candidate joins only where usable); degenerate leg; NaN statistic = no decision; never sized on a holdout bar (z = 0 by construction) = no decision; full-timeline scoring (`eval_from`) | `test_v15_0_contract`, `test_generation_evolve_v15` |
| 4 | CRITICAL | COT / WALCL release look-ahead | Pinned actual release dates (CFTC PR 6745-13, 7864-19, 9138-25, 9147-25, Special Announcements; later date where schedules differ); ad-hoc closures in the federal calendar; WALCL holiday rule; FRED period-start series fail closed | `test_v15_0_release_calendar` |
| 5 | CRITICAL | Label selected on future membership | `forward_returns` conditions on formation-time `active` + a valid later price | `test_forward_label_v15` |
| 6 | HIGH | Market-beta leg one bar misaligned | Forward-stamped market series | `test_generation_evolve_v15` |
| 7 | HIGH | Cohort PROMISING (alpha 0.05) credited to the per-candidate LORD++ account charged at ~0.00065 (~41x replenishment) | Credited only when MC p <= the charged level | `test_v15_0_fixes` |
| 8 | HIGH | Whole-pool cohort never ran (nightly batch consumed the trigger); `clear_snapshot` erased the cohort key | Pending cohort runs over the whole pool; only it consumes the trigger; snapshot reset keeps the key | `test_v15_0_fixes` |
| 9 | HIGH | U4 re-admission unscoped, re-credited PROMISING rows every tick, re-admitted un-adjudicated rows forever | Substrate-scoped; settled rows excluded; attempt stamps the MDE | `test_v15_0_fixes` |
| 10 | HIGH | us_equity panel un-buildable (pre-rename pickle) | `sharpen/utils/compat_pickle` | `test_v15_0_compat_pickle` |
| 11 | HIGH | Claude-CLI proposer blind by prompt only (default tools in the shared temp dir) | `--tools ""`, `--strict-mcp-config`, empty per-call cwd | — |
| 12 | HIGH | Declared funnel neutralization controls never applied (9/98 WQ101 cross t=3 on us_equity) | Union of spec + gate steps at the gates' winsor_pct | `test_funnel_v15` |
| 13 | HIGH | T86 store records "not yet published" as a permanent holiday (58 trading days lost) | Recent no-data not persisted; `--repoll-weekday-empty` repair | `test_v15_0_stores` |
| 14 | MEDIUM | TAIFEX save non-atomic; unreadable stores overwritten | Atomic save; corrupt file quarantined | `test_v15_0_stores` |
| 15 | MEDIUM | Proposer capped before dedup (77 extended-bank formulas unreachable) | Library proposer skips registered keys before the cap | `test_v15_0_proposer` |
| 16 | MEDIUM | Statistical duplicates charged separately (39% of one holdout budget) | `statistical_hash` (book-invariant canonical form) as a third dedup key | `test_v15_0_proposer` |
| 17 | MEDIUM | Per-name data only ever pre-registered as overlays | Cross-sectional templates for (T,N) slots; LLM told which slots are per-name | `test_v15_0_proposer` |
| 18 | MEDIUM | Tier-0 tripwire never probed the warm-up; fixed seed; 8 probes | A quarter of probes in [1, T/4); 32 probes seeded per signal in the scorecard | `test_funnel_v15` |
| 19 | MEDIUM | Bootstrap p could be 0 (BH/BHY q = 0 at any m); PSR/MinTRL on raw N | Add-one p; HAC effective N | `test_funnel_v15` |
| 20 | MEDIUM | Scout had no coverage floor | `altdata.min_bar_coverage: 0.50` | `test_v15_0_scout` |
| 21 | MEDIUM | Reproduce did not pin the decision function; FDR alpha/W0 declared but unconsumed; `--nights` stamps collided | Pins added; YAML wired + drift warning; distinct stamps | `test_v15_0_robustness`, `test_v15_0_fixes` |
| 22 | MEDIUM | Governance Tier-2 command could not run (invalid scope, RL pillars) | Survivor pillars C1-C7, valid scope, context | `test_governance` |
| 23 | MEDIUM | GP bred from degenerate-vol genomes: their train uplift is the combiner's EW-fallback artifact (+0.33 on the synthetic substrate); they took 2-4 of the 6 elite slots in 2 of 3 searches | Scored (a pre-registration still reaches the holdout), never bred | `test_generation_evolve_v15` |
| 24 | LOW | A decided pre-registration's ledger label depended on hall-of-fame membership (LOGGED if it surfaced, SCORED_NOT_SELECTED if bred offspring displaced it), so the v15 search skip flipped every label in the benchmark | The verdict states the spec's own test: PROMISING / SCORED_NOT_SELECTED / NOT_TESTED; offspring stay LOGGED | `test_v15_0_fixes` |
| 25 | LOW | Cohort-gate crash left run tests uncharged; LLM non-list payload raised; Jev fallback mislabelled | Cohort failure contained; payload guarded; `+identity-fallback` | `test_v15_0_robustness` |

## Efficiency (verdict-preserving)

* Combiner (`_monthly_held`, per-bar alpha loops) vectorized — bit-identical, pinned against verbatim copies
  of the old loops. It was 756 s of a 1,365 s profiled synthetic tick (400 s in pandas `Period` boxing).
* The offspring GP search is skipped when no offspring can be promoted (corrected + `prereg_only`, the
  shipped default): 8,376 genomes → the pre-registered seeds, every pre-registered decision bit-identical
  (pinned); `--offspring-search on` restores it.
* Base book cached per split; one full-panel evaluation per genome shared by the causality probe and the
  return stream; `_candidate_returns` vectorized (bit-identical).
* **Measured end-to-end** (one synthetic orchestrator tick, identical CLI args, same machine, run
  sequentially; unrelated processes held 5 of 16 cores throughout): v14 baseline (`20c8e263`) **550 s** →
  v15 with the offspring search forced on **178 s** (3.1x) → v15 default **31 s** (17.7x). All three charged
  12 LORD++ tests. v15-on reproduces v14's ledger (11 pre-registrations scored and lost, 20 offspring
  logged), and the default run records the same 11 pre-registration rows — verdict, rejection class and
  charge (pinned by `test_prereg_ledger_rows_do_not_depend_on_the_offspring_search`).

## Operator decisions (not changed — each is a gate or a live-store edit)

* **POWER-LORD-01 is not wired.** Wiring it is stricter and, on the guard's own grid, flips us_equity to
  REFUSE (the 2026-08-09 memo). Decide together with a finer calibration grid.
* **LORD++ spending sequence.** γ_k ∝ k^-1.6 puts 44% of all wealth on the first test (after 8 tests the
  level is 6.5e-4, z≈3.2); the tick also tests every spec at the batch's TIGHTEST level while charging each
  its own. Both are conservative-valid; a flatter sequence and per-spec levels would each add power.
* **Lockbox criterion** (63 bars, forward SR >= 0.30) barely separates edge from noise: P(clear) 0.44 at
  IR 0 vs 0.64 at IR 1.0. Needs a power-derived horizon/threshold or a sequential test.
* **Cohort `enforce_funnel_feasibility: true`** mirrors a turnover cull the funnel no longer applies to
  pre-registrations (it culled 96 of 140 us_equity members).
* **Historical stores.** `scripts/research/crucible_v15_reopen.py --store <dir>` (dry-run) lists the
  definitely-untested pre-registrations and the LORD++ phantom charges; `--apply-reopen` /
  `--apply-fdr-refund` repair them (backups written). On `results/crucible_orchestrator/real`: 40 rows,
  us_equity 98 phantoms. The declared-(-1) LLM specs need no repair — their folded formulas hash anew.
* **Shared ledgers.** The ledger's primary key is the formula, so a formula tested on one substrate cannot
  be tested on another in the same store; run one `--out` per substrate.
* **Execution convention.** Every book enters at the close the signal reads; a one-bar lag halves the
  5-day reversal IC on Taiwan small-cap. Add a lag check before any capital decision.
* **F3 capturability** uses one rebalance phase (holder_conc -0.045 at offset 0; positive on 16 of 21
  offsets). Staggered books would remove the arbitrary phase.

## Not fixed (low / latent)

T86 cache key omits the ticker set; turnover annualization hardcoded at 252 (intraday only);
taiwan_smallcap base book charged the stock-tax cost; `coverage.max_nan_frac` / `universe.min_adv_usd`
still unwired; WQ101 #29's `min(x, 5)` evaluated as a clamp rather than `ts_min`; GP crossover can splice
constants into coefficient slots (zero genomes); `balance_util` / `asof_grid` build cost (taiwan_smallcap
~217 s, 192 s of it one function); bridge fetches every series twice. The GP's train fitness
(`combination_fitness`) and the cohort path (`cohort.py`, `cohort_eval.py`, `cohort_mc.py`) still build the book
with the raw combiner, so the EW fallback remains in train fitness values and cohort books; they were not
re-audited under the v15 `augmented_book` semantics (under the default `prereg_only` policy train fitness decides no
verdict and no routing; degenerate genomes are no longer bred).

## CRU-1 / CRU-2

No frozen gates file changed (signal_eval `519158fa1450`, taiwan_signal_eval `22a18172be1a`,
taiwan_smallcap_altdata `0ccf6dd584f0`); the only gates edit is a new key in the unfrozen
`crucible_altdata.gates.yaml`. The agent view gains one column, `stat_hash` — a hash of the formula text and
its declared type, no score. MAJOR bump: v15.0 changes which hypotheses are tested, the gate's degenerate
handling, two data inputs and the LORD++ accounting.

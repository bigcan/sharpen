# TAILWIND forward runner (Tier-2 roadmap X2) — 2026-09-29

**What X2 asked for.** The batch executor books each month-end rebalance at the month-end close under `execution.decision_lead_bars: 1`, but nothing could run that book forward. Four gaps:
- Month-end flags came from the data window, whose last row is never a confirmed month-end.
- The documented live target, `linear_core_weights(window)[-1]`, filled one or two closes late (T2-01, T1-04).
- The "forward recompute" read precomputed rows, so it was tautological (T2-02).
- There was no runner, order path, log or persisted book (T6-04).

X2's acceptance test: a growing-window replay in which the live fill date equals the batch fill date at 100% of the 2006-2026 month-ends, forward-vs-batch L1 stays under 0.05 at every step, and N5(b) turns green.

**Verdict: all three pass.** Every forward read reproduces the batch exactly: L1 = 0 at all 5,174 steps, against a 0.05 gate.

## Design

| Piece | Where | What it does |
|---|---|---|
| Ex-ante NYSE calendar | `sharpen/data/trading_calendar.py` | Rule-based full closures (Good Friday; New Year's Saturday rule; Juneteenth from 2022; observed rules; dated special closures from 9/11 to Carter 2025). No network. It reproduces all three cached panels' session calendars exactly (5,176 / 5,149 / 5,133 sessions). |
| Forward read | `sharpen/paper/forward_runner.py` `compute_forward_decision` | At close *s*, see the read rule below. |
| Signal stage on given prices | `cross_asset_loader.prepare_two_sleeve_payload` | Split out of `load_two_sleeve_data`, which now only fetches then calls it. Every array is byte-identical on all three linear-core configs. |
| Read-only cache access | `cross_asset_loader.read_cached_ohlcv` | Never fetches or writes. Replays and tests read certifying caches through it. |
| Daily cycle | `ForwardRunner.run_session` | Decide; settle every session since the last mark (normally one); apply the kills; append to the log; persist the book. |
| Target log | `targets.jsonl` | Append-only and hash-chained (tamper → FAIL). Re-logging a session is idempotent. A kill's flatten supersedes that session's target. |
| Book | `paper_state.parquet` + `runner_state.json` | `PaperState` filled through `SimFillEngine`, including financing carry and borrow. Entry prices are re-based by `new_mark / old_mark` on every refetch, so a vendor dividend re-adjustment is never booked as P&L (T1-10). |
| Incremental parity | `ForwardRunner.evaluate` | Every run recomputes the whole history on today's data and compares the book with it session by session, under `paper_soak.parity`. Each logged target was computed from prices truncated at its own as-of, so a signal that read the future would show up as drift (T2-02). |
| Kills (terminal: flatten at the next session, stop) | `ForwardRunner._kill_reason` | A kill file (`safety.kill_file` or `<state>/KILL`); drawdown ≥ `max_drawdown_kill_pct`; a session loss ≥ `daily_loss_halt_pct`; target gross > `max_gross_exposure`; per-sleeve action-drift CRIT when `gates.safe_mode.crit_triggers_flatten`. |
| Drift | `ForwardRunner.drift_reports` | One `ActionDriftTracker` per sleeve, fed the held conviction of every logged target, against `drift.baseline_path`. A missing baseline is REVIEW, never silent. |
| Exit codes (N3 map) | `exit_code` | PASS 0, REVIEW 3, FAIL 1. A soak whose only gap is its horizon exits 0. An unevaluable hard group, a kill, a broken log or stale data exits 1. |
| CLI | `scripts/run_forward_paper.py` | One-shot (cron after the close) or `--loop` (daemon; serves `PaperMetrics`). Live mode fetches into `<state>/data`, never a certifying cache. `--offline` reads the config's cache read-only. |

**The read rule** (`compute_forward_decision`), at close *s*:
1. Truncate the prices at *s*. A later bar is in progress and is dropped. The data must reach *s*, and its sessions must equal the calendar.
2. Append the next two calendar sessions as empty rows.
3. Recompute every signal from those prices.
4. Drive the unchanged executor and read row −2, the step that fills at *s*+1.

Two sessions, because the drive treats a window's last row as unconfirmed: the lead's final-row hold and the alpha's monthly flag both key on it. With *s*+1 and *s*+2 present, the flags the read step uses are the calendar's, for conviction and alpha alike. Row *s*+1's conviction and vol need data ≤ *s* only (cutoff lag 1).

**Failure handling in the CLI:**
- A stale data source means no target: the next session holds and the job retries (exit 1).
- An engine error during decide, settle or log triggers the declared `safety.emergency_flatten_on_error`.
- A gates file the evaluator cannot read is a reported FAIL, with parity and drift still scored, and no flatten.

## Acceptance (`scripts/research/tailwind_forward_replay.py`, `configs/tailwind_v1.yaml`, 13 workers, 19 min)

Artifact: `results/tailwind_v1/forward_replay_tailwind_v1.json`.

| X2 criterion | Result |
|---|---|
| Forward vs batch weights, every step | 5,174 decisions (as-of 2006-01-04 … 2026-07-30); **max L1 = 0.0** at every step and at all 247 month-end fills (gate 0.05) |
| Live month-end fill date = batch | **368 / 368** sleeve conviction switches fill on the same session as the batch, and all 368 fill on the month-end session itself |
| N5(b) | `test_forward_prefix_consistency_all_configs` passes for `cross_asset_momentum` (lead 0), `tailwind_v1` and `tailwind_v1_challenge` (lead 1), through the forward read |

**The same decisions booked through the persisted runner, 5,173 sessions:**
- Incremental parity is exact: L1 6e-17, return tracking error 0 bps, 0 missed month-ends, cost ratio 1.00. Parity PASS.
- Risk PASS: no kill fired; max drawdown 21.9% against the 25% kill; max gross 2.80.
- Drift PASS and horizon PASS.
- **Performance FAIL** (a review group), so overall **REVIEW, exit 3**: the trailing 12-month Sharpe is below its 0.30 floor, consistent with the negative excess returns since 2023.
- Total return +153% (financed, share-exact accounting after T4-10 `c6f61bb3`).

This is a historical replay. It proves the forward path, not the edge.

## Tests

- `tests/data/test_trading_calendar.py` (35): every rule and closure on hand-checked dates, Easter, and a Good Friday month-end (2024-03-28, which a weekday calendar gets wrong). Also as-of resolution across time zones, fail-closed inputs, and exact reproduction of the 2006-2026 panel.
- `tests/paper/test_forward_runner.py` (20). Covers:
  - forward reads equal the batch at every month-end eve and mid-month cutoff, for lead 0 and lead 1;
  - the switch fills on the month-end session under the lead, and one session later without it;
  - the old `w[-1]` reader misses month-end fills (the teeth);
  - N5(b);
  - the hash-chained log (idempotent re-log, conflict, order, tamper, flatten supersede);
  - the 45-session daily cycle, with zero parity and exit 0;
  - idempotent re-runs, stale data, in-progress bars and calendar mismatches;
  - kill file → flatten → stop;
  - a vendor re-adjustment is re-based, not booked;
  - a missed run holds and is recorded;
  - the exit-code map and the drift config.
- `tests/scripts/test_run_forward_paper.py` (2, real data, offline). Two consecutive sessions for the own-capital config: exit 0, parity PASS, and the certifying caches are byte-identical afterwards. The challenge config fails closed on its gates without flattening.
- Mutations: 14 of 14 meaningful mutants are killed. Among them:
  - one appended session;
  - an off-by-one conviction row;
  - the in-progress bar kept;
  - no calendar check;
  - no re-base;
  - the kill file ignored;
  - a missed session booked;
  - no hash check;
  - a literal-UNKNOWN exit map;
  - no flatten supersede;
  - four calendar rules.

  A 15th, as first written, was a no-op, because holidays are bucketed by year. Rewritten as a real mutation, it is killed by 3 tests.
- Suites: 853 affected tests pass; the audit floor and tripwires pass.

## Still open

- **The challenge gates cannot be scored** (T6-05). `configs/tailwind_v1_challenge_v2.gates.yaml` lacks `paper_soak.performance.rolling_sharpe_floor` (and siblings), so the challenge book's verdict is FAIL (exit 1). **[Operator: a gates-file change.]**
- **The kill semantics are the declared ones: terminal flatten.** Whether a challenge phase should kill at all is still R3. **[Operator.]**
- **Paper-sim only.** There is no broker order path (`IBFillEngine`, rung-2), and no docker service or scheduler entry yet. The CLI is ready for cron or `--loop`.
- **Drift feed.** The trackers are fed daily held convictions against a baseline sampled monthly on the research basis (230 rebalances). The bases match (the same mean-of-signs and BAB conviction), but vol-regime bucketing is not wired (no cutpoints are passed).
- **Unscheduled closures.** Add them to `SPECIAL_CLOSURES` when announced. Until then the freshness gate fails the run closed.
- **Cost.** Each run recomputes the full history (about 4 s): fine daily, heavy for replays (parallelized there).
- **N3 is not done.** A `promotion_check` across all enforcing artifacts still does not exist. This runner does exit non-zero on FAIL.

## Incident: the certifying cache was overwritten, then restored

While checking the loader split, I pointed a 19-asset config at `results/tailwind_v1` to stay offline. `fetch_and_clean` treated the asset-superset miss as a stale cache and refetched in place at 17:12, overwriting the 2026-07-31-vintage certifying OHLCV cache (T1-03 / T6-08, the trap the audit named). No backup existed.

Recovery:
- The fresh pull is kept in `results/tailwind_v1/_refetch_2026-09-29T1712/`.
- The certifying window (≤ 2026-07-31, 18 assets) was rebuilt from it, with a `reconstructed` block in the manifest.
- On the code as committed at the time, the recorded executor numbers reproduce to 4 d.p.: SR 0.6351 / 0.4983, P(pass) 14/21 / 14/23.
- The values are a newer vendor vintage: about 2% of daily returns differ by 1e-6 to 1e-4.

`read_cached_ohlcv` now exists so that a check that only reads can never write.

# Leaks found

Three defects that made this project's own results look better, or more reliable, than they
were. Each one passed code review, produced plausible numbers, and survived for weeks to months.
None raised an error.

They are published because these are the mistakes most backtesting projects make and never
detect. For each: what the bug was, how it was found, what it invalidated, and the test that now
fails if it comes back.

| ID | Class | Lived for | Effect |
|---|---|---|---|
| [X2](#1-x2--the-coarse-bar-look-ahead) | Temporal look-ahead in multi-timeframe features | Months, from the file's creation | **Was the entire edge** of one strategy: PF 1.69 → 1.02 |
| [GATE-CAUSAL-01](#2-gate-causal-01--the-gate-that-read-the-bar-it-traded) | Same-bar look-ahead in a signal gate | From the file's creation | Optimistic bias across 54 configs |
| [SEED-01](#3-seed-01--the-seed-that-never-reached-the-environments) | Reproducibility | Until 2026-08-18 | Every earlier multiseed result measured noise |

---

## 1. X2 — the coarse-bar look-ahead

**The bug.** The multi-timeframe feature handler builds 15-, 60- and 240-minute bars from a
fine base series, and for each base bar looks up "the current coarse bar" with
`np.searchsorted(coarse_ts, base_ts, ...)`. Coarse bars are stamped with their **start** time,
so a base bar at 10:07 was mapped to the 10:00–10:15 bar: the bar that **closes eight minutes
in the future**. Every base bar inside a coarse interval saw that interval's final high, low and
close, up to `scale − 1` minutes ahead.

The live engine could not do this. Live only ever sees the in-progress coarse bar. So the
simulator and live trading were running different strategies, and nobody had checked.

**Why it survived.** The buggy line was written when the file was created and never appeared in
another diff. Routine audits read the diff, so they were structurally blind to it across hundreds
of green runs. The only leakage invariant at the time covered normalization statistics, so
"leakage check: PASS" gave false coverage of a temporal leak. The existing sim↔live parity test
compared observations only at **end of data**, the one point where the in-progress coarse bar and
the completed one coincide.

**How it was found.** A gap between simulated and live P&L triggered a full lifecycle audit of
the strategy (independent finder and skeptic per pillar). Finding P2-01 in
[`research/sg1_btc_strategy_audit_2026-05-29.md`](research/sg1_btc_strategy_audit_2026-05-29.md)
reproduced it numerically.

**What it invalidated.** An apples-to-apples re-run, identical except for the fix (same costs,
seeds, evaluation and training budget):

| Code | Best-solo median walk-forward PF | Folds clearing PF ≥ 1.10 |
|---|---|---|
| Leaky | **1.689** | 8 / 8 |
| Fixed | **1.016** | 0 / 8 |

The look-ahead was worth about 0.67 of profit factor. It **was** the edge. The strategy was
shelved. A second strategy on the same handler collapsed to PF 0.79–0.85; a third kept real but
regime-dependent signal that no longer cleared its deployment gate.

**The fix.** Map each base bar to the last **closed** coarse bar:
`np.searchsorted(coarse_ts, base_ts - scale_ns, side='right') - 1`, applied identically in the
simulator (`sharpen/data/multiscale_handler.py`) and the live observation builder, in one commit.

**Tripwires.** `tests/data/test_multiscale_causality.py` asserts that the coarse bar assigned to
a base bar at time `t` closed at or before `t`. It was committed first as a strict expected failure
reproducing the leak, and flipped to a plain test when the fix landed. `tests/crypto/test_live_obs_parity.py`
asserts that simulator and live coarse-scale features agree; with only the simulator fixed it
failed at a max difference of 1.21.

**Lesson.** A parity test that checks only where the two paths trivially agree proves nothing.
State the causal invariant directly (every input to bar `t` was knowable at `t`) and test that.

---

## 2. GATE-CAUSAL-01 — the gate that read the bar it traded

**The bug.** A signal-gated wrapper only let the agent trade when volatility features crossed a
threshold. It evaluated the gate at `handler._ptr`, the bar **not yet consumed**, and when the
gate opened, the agent's next action traded that same bar. The features involved (normalized ATR,
Parkinson volatility, volume z-score) are computed from bar `i`'s own high, low, close and volume
with no shift. The z-score's `shift(1)` made its *running statistics* causal, never the value
being scored.

So the gate decided whether to trade a bar using that bar's own range and volume. It was also a
sim↔live mismatch: live reads the last **closed** bar, so live gated on bar `k` while the
simulator gated on `k + 1`. A code comment claimed the live path "mirrors" the simulator. It did
not.

**How it was found.** Auditing the fix for a *different* gate bug. The escalation rule
(a correctness-critical module gets a whole-module re-review when touched, not a diff review)
reached the line. Proven empirically: inject an ATR spike at bar 60; the gate inspected bar 60
while the last consumed bar was 59, and the next step traded bar 60.

**What it invalidated.** 54 configurations, every variant of that strategy family across four
instruments. The bias is **optimistic**, so the family's existing NO-GO verdict only hardened.
But every pre-fix number from it is void as a performance floor.

**The fix.** Gate on `handler._ptr - 1`, the last closed bar
(`sharpen/envs/signal_gated_wrapper.py`). That is also the parity fix, because it is the only bar
live can see.

**Tripwires.** `TestGateCausal01` in `tests/test_signal_gated_wrapper.py`: three cases, each
verified to fail when the look-ahead is reintroduced. Before this there was no causality test for
the gate at all.

**Lesson.** Leakage hides in *filters*, not just features. Anything that decides *whether* to act
needs the same causality test as the thing that decides *how*. And a comment that asserts parity
is a claim to test, not a fact.

---

## 3. SEED-01 — the seed that never reached the environments

**The bug.** `run_full_pipeline.py --seed N` seeded torch, NumPy and `random` in the parent
process. It did not make a run reproducible, because four independent gaps each let entropy in:

- The vector-env factory had **no seed parameter**.
- The trainer called `env.reset()` with no seed.
- `reset(seed=None)` left Gymnasium's generator unset, so it lazily **self-seeded from OS
  entropy** and drew the random episode start from that stream.
- Environments ran under `AsyncVectorEnv(context="spawn")`: fresh interpreters that inherit
  nothing from the parent's global seed, whatever it is.

The same audit found one legacy environment and one wrapper drawing from the global `np.random`
or an unseeded generator, which no seed passed to an env can ever reach.

**How it was found.** Re-running a recorded result. The identical config at the identical seed
returned **−7.70%** where the earlier run had recorded **+3.40%**, an 11.1-point divergence at
nominally the same seed.

**What it invalidated.** Every "multiseed" result before the fix. They measured run-to-run
variability, not a seed effect, and no individual run can be reproduced. A reported "18.7-point
seed spread" was never a seed effect. Any fold or model ranking decided by less than ~10 points
was inside the noise floor. Separately, hyperparameter trials had each been scored on a
**different** random 500-bar validation window, so trial ranking was partly a window lottery.

**The fix.** Seed environments at **construction**, not only at reset. A factory-seeded env stays
reproducible for every caller that resets without a seed, which fixed the evaluation, backtest and
trainer paths without editing them. Per-worker fan-out uses `seed + i`, matching Gymnasium's own
derivation.

**Tripwires.** `tests/test_env_seed_reproducibility.py`: seven cases, including a static check
that names, by file and line, any env code calling the global RNG.

**Lesson.** A seed flag is a behavior claim. Verify it by running twice and diffing the outputs,
not by reading where the flag is passed.

---

## What the three have in common

- **No error, plausible output.** Each produced numbers in a believable range. Two biased results
  upward; the third made noise look like signal.
- **Diff review could not see them.** The bugs sat in stable code. Finding them took whole-module
  audits triggered by stakes (a sim↔live gap, a fix in a neighboring line, a re-run), and each is
  now guarded by a test that **executes** the property rather than inspecting the code.
- **Every guard is a negative test, checked in both directions.** It must pass on the fixed code
  and fail when the bug is reintroduced. A test that has never been seen to fail has not been
  shown to test anything.

The invariants these became are `LEAK-1` (normalization isolated per split) and `LEAK-2`
(temporal causality, including sim↔live parity) in the project's working rules.

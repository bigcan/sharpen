# Methodology

How this project decided that a result was real, and why the negative results in
[NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md) can be trusted as far as they go.

The short version: **state the question and the kill criterion before looking, measure whether the
test can see a realistic effect at all, deflate for everything you tried, prove the data could not
see the future, and have someone other than the author try to break the result.** Each section
below is one of those, with the code that enforces it.

Numeric thresholds are deliberately not repeated here. They live in `configs/*.gates.yaml` and are
read by code, never hardcoded; quoting them in prose is how prose and code drift apart.

---

## 1. Principles

1. **Falsify before optimizing.** The first test of an idea is the cheapest one that could kill it,
   usually a CPU probe, run before any hyperparameter search or GPU time. Most rows in the negative
   ledger died there.
2. **One question, one verdict.** A stage answers one pre-specified question and writes one decision
   artifact. Fused pipelines that train, tune and evaluate in one job are rejected by
   `scripts/validate_config.py`.
3. **Ask whether the gate is reachable before running.** If the test cannot detect a plausible
   effect, or the action space cannot physically clear the bar, the run is a budget spend against
   a foregone conclusion. Measure power and ceilings first.
4. **A NO-GO is a result.** It is recorded with its evidence, and closed families are not reopened
   without new evidence (a different venue, better data, a structurally new signal).
5. **Fail closed.** An unmeasured gate leg, a missing power curve or an undefined statistic counts as
   a failure, never as a pass.

---

## 2. Pre-registration

A hypothesis is written down, with its gates and kill criterion, **and committed before any result
exists.**

- **Research probes:** `docs/research/<topic>_preregistration_<date>.md`, committed before the run.
  Git history then shows whether the criterion moved after the result.
- **Signals:** a `SignalSpec` (`sharpen/signals/spec.py`) is content-hashed before evaluation and the
  hash is recorded in the scorecard. A post-hoc edit changes the hash.
- **Crucible:** the proposing agent writes the spec and the hash locks it before it sees
  out-of-sample data (principle CR-2 in the
  [discovery spec](research/crucible_agentic_discovery_spec.md)).
- **Gates:** the gates files are themselves hashed. Crucible's funnel `gates_hash` is frozen, and a
  new capability may not change any past verdict (invariant `CRU-1`).

Pre-registration is what separates a test from a search. Without it, every threshold is a free
parameter.

---

## 3. Leakage invariants

Two invariants cover the defects that manufactured this project's worst false results
([LEAKS_FOUND.md](LEAKS_FOUND.md)).

**`LEAK-1`: normalization statistics are fit per split.** Running z-scores, EMA statistics and
scalers reset at every train / validation / test boundary. Nothing learned on test data reaches
training.

**`LEAK-2`: temporal causality.** No input at bar *t* may carry data stamped after *t*. In practice
that means:

- multi-timeframe features map to the last **closed** coarse bar, never the one in progress;
- resample `label` / `closed` conventions, `searchsorted` alignment, rolling windows and warmup carry
  are all causal;
- anything that decides *whether* to act, such as a signal gate or a filter, reads only closed bars;
- external data joins on **release time**, not the period it describes (CFTC COT describes Tuesday
  and publishes Friday);
- the simulator sees exactly what live trading could see.

Every `LEAK-2` guard has a **negative test**: a test that fails if the look-ahead is reintroduced.
The signal funnel's Tier 0 runs a truncation tripwire on every candidate, and Crucible enforces
causality on generated formulas rather than asserting it in a docstring.

Free equity data adds a third trap: **survivorship bias is a time gradient**, so training and holdout
windows contain different universes. Results on such data are labelled upper bounds; see
[DATA.md](DATA.md).

---

## 4. Statistics

### Multiplicity

A best-of-N result is inflated by N. The project corrects for it at every level.

| Control | What it answers | Where |
|---|---|---|
| **Deflated Sharpe** (Bailey & López de Prado 2014) | Probability the true Sharpe is > 0, after deflating the benchmark to the expected maximum of the N trials and adjusting for skew and kurtosis | `deflated_sharpe_ratio` in `sharpen/crypto/eval/statistics.py` |
| **Trial count** | N is `max(batch pool, declared hypotheses)`, so submitting in small batches cannot shrink it | `configs/crucible_multiplicity.gates.yaml` |
| **Probability of backtest overfitting** (CSCV) | How often the in-sample winner lands below the median out of sample | `probability_of_backtest_overfitting`, same module |
| **Benjamini-Hochberg / BHY FDR** | False-discovery rate across a batch | `bh_fdr`, `bhy_fdr` in `sharpen/signals/_ic.py` |
| **Harvey-Liu-Zhu hurdle** | A t-statistic bar that already prices in the literature's multiple testing | `configs/signal_eval.gates.yaml` |
| **Online FDR (LORD++)** | Error budget across an unbounded stream of Crucible runs, without an ever-rising global N | Crucible spec §6.1 |

A single global trial count that grows forever is **forbidden**: the deflation benchmark then rises
without bound while sample length does not, so any fixed edge eventually becomes undetectable by
construction.

### Out-of-sample structure

- **Combinatorial purged path resampling** with an embargo after each train→test boundary: a
  distribution of Sharpes across paths rather than one path. For the signal funnel nothing is refit
  (signals are fixed rules), so this measures stability across paths, not out-of-sample fit; trained
  policies get real out-of-sample tests from walk-forward below.
- **Embargoed holdout** for final validation, touched once.
- **Walk-forward** across calendar windows for trained policies, reporting the median across seeds.
- **Recent out-of-sample** on the most recent window before anything is deployed.
- **Forward incubation**: a Crucible survivor is judged on bars timestamped after the hypothesis was
  written (the lockbox, `configs/crucible_lockbox.gates.yaml`).

### Uncertainty

Confidence intervals use a **circular block bootstrap** (`block_bootstrap_sharpe_ci`), because
overlapping and autocorrelated returns inflate a naive t-statistic. Read the interval, not the t.
Since `crucible-v14.0` the funnel's IC t-statistic and deflated-Sharpe observation count also use a
Newey-West effective count when forward-return labels overlap (see
[LEAKS_FOUND](LEAKS_FOUND.md#found-in-the-pre-release-audit-crucible-v140)).

### Power, before anything else

Before trusting a silence, compute the test's **minimum detectable effect** at 80% power and compare
it with a realistic edge. The project adopted this after its first campaigns, and two findings from it
shape how every negative result should be read:

- the unit is ΔSharpe per 252-**bar** year, so switching to intraday bars buys no power;
- the deployed multi-leg contract costs roughly 2.5× the power of an idealised single test.

`scripts/research/planted_sweep.py --bar-only` and `scripts/research/forward_power.py` reproduce
both without data. Crucible's power guard refuses to mine a substrate that cannot resolve a plausible
edge (`configs/crucible_power.gates.yaml`).

---

## 5. Controls that decide what a result means

Deflation says whether a number is distinguishable from luck. These controls say whether it is the
thing it claims to be.

| Control | Question | Example that it decided |
|---|---|---|
| **Negative control** | Does a zero-information position, on the same cost basis, come out negative? A control that cannot go negative controls nothing | Validated on three substrates in the free-data probe campaign |
| **Circular-shift timing null** | Shift each asset's positions in time, keeping exposure, trade count and holding periods. Does the rule know *when*? | HMA crossover: re-timed copies earned more (p = 0.582) |
| **Edge-sign flip** | Re-score with the signal inverted. Does an overlay's benefit survive? | Risk overlays: every helpful feature reversed; 0/54 arms reach PF ≥ 1 |
| **Matched exposure** | Compare against the same gross exposure, not against zero | DD throttles and gross caps turned out to be de-levering |
| **Different venue or series** | Does the effect survive a change in how it is measured? | Commodity session premium reversed in spot gold |
| **Per-subperiod reporting** | Is the gain spread across time or one episode? | Vol-managed overlay: the gain was one crisis |
| **Frictionless vs net** | Is there gross structure at all, and does it survive cost? | Crucible gates on frictionless Sharpe; net is reported against a stated cost model |
| **Exposure fraction** | Does the strategy actually trade? A book that stays flat looks low-risk | `exposure_frac` in the allocator metrics |

---

## 6. The signal validation funnel

`sharpen/signals/eval_harness.py` runs every candidate through tiers that each target a different
way a backtest lies: causality and hygiene, predictive power, capturability after cost, stability
and recent out-of-sample, purged cross-validation, deflation, and orthogonality to a factor book.
Thresholds live in `configs/signal_eval.gates.yaml`. Design:
[signal_eval_system_design.md](research/signal_eval_system_design.md).

A candidate that clears everything is `PROMISING`, not discovered. Almost everything lands on
`LOGGED`: fully measured, below the bar.

---

## 7. Automated search (Crucible)

Automating hypothesis generation multiplies every risk above. The design rules that contain it:

- **The agent proposes, statistics dispose.** The language model writes hypotheses; it never scores,
  ranks or gates.
- **The agent is blind to outcomes.** It reads only a deduplication view and the list of killed
  families, never verdicts, Sharpes or holdout results (invariant `CRU-2`). Otherwise it becomes an
  oracle that hill-climbs the holdout.
- **Pre-registered specs are tested, not screened.** A cheap upstream filter once culled every
  pre-registration, so the binding test never ran while the run still reported "0 promising". The
  number actually tested is now reported alongside the number that passed.
- **Rejections are classified.** A rejection is `DECISIVE` only if the test could have resolved the
  smallest acceptable edge; otherwise it is `UNDERPOWERED` and parked, not killed.
- **Time is the final arbiter.** Survivors wait in the lockbox for data that did not exist when they
  were proposed.

The decision contract, in execution order, is §12 of the
[discovery spec](research/crucible_agentic_discovery_spec.md).

---

## 8. Training protocol for RL

[`docs/protocol_v2.md`](protocol_v2.md) is the full contract. The parts that matter for credibility:

- **Staged runs**: `data-prep` → `hpo` → `l1-multiseed` → `ensemble-confirm` → `wf` → `oos` →
  `paper-deploy`. Each stage is its own tracked run that writes a manifest, and a downstream stage names
  the upstream manifest it depends on, which must have passed.
- **Multi-seed median, not max.** Seed choice is not a free parameter. (Until the seed was seeded at
  environment construction, this was not true; see SEED-01 in
  [LEAKS_FOUND.md](LEAKS_FOUND.md).)
- **Budget rule.** Training steps are sized against the length of the training window, because
  replaying a short window too many times overfits.
- **HPO objective is fixed** and reward parameters are locked during search.
- **No hindsight in backtests.** Any reward term that reads future prices is zero outside training.
- **Beat the linear baseline.** RL is only admissible as an overlay that beats a validated linear
  core out of sample, on identical costs and conventions.

---

## 9. Verification

### Tripwire tests, checked in both directions

A test that has never been seen to fail has not been shown to test anything. For every negative test:

1. break the code deliberately and confirm the test fails;
2. restore it and confirm the test passes;
3. confirm the edit actually applied (a surviving mutant is usually a no-op edit);
4. confirm the test can exhibit the failure at all.

Formula-level tripwires pin numerical results; `tests/test_dsr_formula_tripwire.py` fails if the
deflated Sharpe implementation drifts. See [guides/testing.md](guides/testing.md).

### Two-tier audit

- **Tier 1, diff review**, runs on every change. It cannot find a bug in code that has not changed,
  and the worst leak in this project sat in an unchanged line through hundreds of green reviews.
- **Tier 2, the deep lifecycle audit**, re-derives a strategy's correctness from data to execution,
  independent of any diff. It is triggered by **stakes**: before any promotion to capital, before
  reading a deploy-gating verdict, or when a result looks too good. Eleven pillars, each with a
  *finder* that lists findings with file-and-line evidence and an independent *skeptic* that
  confirms or refutes each one. Template:
  [deep_lifecycle_audit_template.md](audit/deep_lifecycle_audit_template.md).

A Tier-2 audit found the coarse-bar look-ahead that was one strategy's entire edge, and Tier-2 audits
of the momentum book found its "declared but not wired" gates. The reports are published under
`docs/research/*_deep_lifecycle_audit_*.md`.

---

## 10. Limits of this methodology

- **It is only as strong as its power.** On free daily data many realistic edges sit below the
  detectable threshold. A NO-GO in that regime is a statement about the instrument.
- **Declared is not wired.** Several audits found gates that were documented, configured and even
  unit-tested but read by nothing on the live path. Every safeguard above is claimed only where a
  consumer was verified; treat any new one as unwired until you see the code that reads it.
- **Free data is survivorship-biased.** Equity results on it are upper bounds.
- **Earlier records predate some fixes.** Results produced before a defect was fixed are void as
  evidence and were re-run or marked, not silently inherited. The version ledger in the Crucible spec
  records which changes preserved past verdicts and which required re-scoring.
- **Automation raises the stakes.** An agent that proposes thousands of hypotheses makes every
  multiplicity and leakage control above load-bearing, not optional.

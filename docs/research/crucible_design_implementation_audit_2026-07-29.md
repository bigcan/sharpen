# Crucible — deep design + implementation audit (2026-07-29)

**Scope:** the whole Crucible discovery machine on `July2026` @ `af32b797`, system version `crucible-v5.0`
(`finrl_pro_ds/crucible/**`, `finrl_pro_ds/signals/**`, `scripts/research/crucible_*.py`,
`configs/crucible_*.gates.yaml`), plus the lifetime run record in
`results/crucible_orchestrator/**`.

**Question:** why has Crucible never produced a useful alpha, and what should change?

**Verdict:** the record is an instrument artifact, and the instrument's defect is now measurable rather
than inferred. Across the entire lifetime scored record (n=170), **the economic leg passed 170/170 and
both significance legs passed 0/170.** The composite gate has no leg operating in a discriminating
regime, so `P(PROMISING) = 0` independently of what the market contains. Separately, since 2026-07-12
the substrate-power guard refuses **every** substrate, so the loop currently cannot mine at all.

The single highest-value change requires no new research: a correctly specified replacement decision
layer already exists in-tree, is E1-calibrated and E2-powered (**power 0.81 at ΔSR 0.5 vs the shipped
funnel's 0.00**), and is wired to nothing.

---

## 1. Evidence — the lifetime record

Read directly from `results/crucible_orchestrator/**/{trial_ledger,orchestrator,lockbox,governance}.db`.

| Quantity | Value |
|---|---|
| Orchestrator instances | 4 (`real`, `taiwan_manual`, `taiwan_v2`, `taiwan_llm_diversity_probe`) |
| Ticks recorded | 23 |
| Trial-ledger rows (lifetime) | **403** |
| Verdicts | `LOGGED` 170 · `SCORED_NOT_SELECTED` 69 · unscored/`NULL` 164 · **`PROMISING` 0** |
| Candidate types | `overlay` 282 (70%) · `cross_sectional` 121 |
| Lockbox enrollments (all DBs) | **0** |
| Governance handoffs | **0** |

### 1.1 Per-leg pass rates on the 170 scored candidates

`combination_fitness(...).passes_gate` is a 6-way AND
(`finrl_pro_ds/signals/generation/fitness.py:396`).

| Leg | Threshold | Observed range | **Pass** |
|---|---|---|---|
| `delta_mean` (uplift) | ≥ 0.10 | min 0.368 · med 0.540 · max 0.986 | **170 / 170 (100%)** |
| `dsr_aug` | ≥ 0.90 | min 0.000 · med **0.000** · max 0.674 | **0 / 170** |
| `marginal_t` (HLZ) | ≥ 3.0 | min −1.399 · med 1.278 · max **2.116** | **0 / 170** |

One leg is inert, two are absolute. This is the whole result. Everything below explains *why* each of
the three is in the regime it is in, and what to do about it.

---

## 2. Root causes

### RC-1 (CRITICAL) — `marginal_t` measures the wrong quantity

`fitness.py:360` builds the "marginal contribution" stream as `marg = b_aug − b_base` and tests
`Sharpe(marg)·√N_eff ≥ 3`.

The combiner (`envs/allocator_factory.dynamic_sleeve_alphas`, with the shipped
`tilt_strength = 0.0`, `redundancy_strength = 0.0`) is a **convex, sum-to-1 inverse-vol** weighting:
`α_s ∝ 1/σ_s`, normalized. Adding a candidate `c` therefore scales every incumbent weight by the same
factor `(1 − w_c)`, giving the exact identity

```
b_aug  = (1 − w_c)·b_base + w_c·r_c
marg   = b_aug − b_base   =  w_c·(r_c − b_base)
```

(Derived independently here; the project's own F3 artifact
`results/crucible_marginal_seal/marginal_seal.json` confirms it numerically —
`max_abs_dev = 3.5e-18`, `identity_holds: true`.)

So `marginal_t ≥ 3` is a **paired mean-dominance test**: *does the candidate's own mean return exceed
the whole base book's mean return by three standard errors?* That is not the marginal-Sharpe question
the funnel is built to ask, and for the candidate class the machine is designed to find it is
anti-correlated with the objective:

- a variance-reducing **diversifier** normally has `μ_c < μ_base`, so `E[marg] < 0` and
  `marginal_t → −∞` as N grows — *more data makes a good candidate fail harder*;
- a dollar-neutral **cross-sectional L/S spread** is structurally lower-mean than a directional
  TSMOM+carry book — near-guaranteed fail;
- an **overlay** has `E[m] ≈ 0` by construction (causal z-score → `tanh`), so its mean return is a
  covariance term — also below `μ_base`.

It is also not scale-invariant. From the same artifact, one fixed signal re-levered:

| leverage | 0.25× | 0.5× | 1× | 2× | 4× | 8× |
|---|---|---|---|---|---|---|
| `marginal_t` | −2.304 | −2.099 | −1.680 | −0.862 | +0.471 | +1.860 |
| `delta_sr_oos` | 0.469 | 0.570 | 0.703 | 0.825 | 0.900 | 0.929 |

A candidate with annualized own-Sharpe 1.91 and ΔSR +0.70 scores `t = −1.68`. **The verdict is a
function of how the candidate is scaled**, and across the whole sweep it never passes.

Note a nuance the earlier (cont-131) narrative overstated: on the *full* 170-row record `marginal_t`
is mostly **positive** (160/170) but ceilinged at 2.116. So the empirically binding failure is a
magnitude ceiling, with sign-inversion demonstrated as a live hazard rather than the universal
pattern. Both point at the same fix.

This defect is documented in-code as a deliberate "Tier-C known seal" (`fitness.py:352-359`) and
pinned by `tests/signals/test_tier_c_seals_registered.py`.

### RC-2 (CRITICAL) — `dsr_aug` is a book-level bar, not a candidate test

`fitness.py:322-342` deflates the **absolute** augmented-book Sharpe. The candidate barely controls
this quantity:

- on a base book with ~0 Sharpe over the train era (Taiwan: base train SR ≈ −0.007), no candidate can
  lift the *whole book* to a deflated-Sharpe posterior of 0.90 — hence median `dsr = 0.000` across the
  record, max 0.674;
- on a *good* base book the failure inverts: the leg passes on the base's own merit and provides no
  protection at all. The calibration YAML says this in as many words
  (`configs/crucible_calibration.gates.yaml:18-23`).

Either way it does not measure the candidate. A leg that is unpassable on one substrate and
non-binding on another is not a gate.

### RC-3 (CRITICAL) — the loop is hard-stopped by its own power guard

`configs/crucible_power.gates.yaml`: `plausible_delta_sr_max: 0.50`, `action: refuse` (flipped from
`warn` on 2026-07-12). The orchestrator refuses when `implied_mde_delta_sr > ceiling`
(`orchestrator.py:215`). Against the measured sweep
(`results/crucible_calibration/calibration_mde_sweep.json`, grid top = holdout 1011 bars):

| substrate depth | MDE | vs ceiling 0.50 |
|---|---|---|
| below grid (`extrapolated_low`) | ≥ 3.63 | refuse |
| **on grid** (holdout 1011 — where every daily substrate sits) | **1.4025** | refuse |
| above grid (`unmeasured_high`, v5.0 fail-closed) | `+inf` | refuse |

**Every substrate is refused.** Nothing can be mined today without `--force-underpowered`. The config
comment states this is the intended damning signal, not a bug — correct, but it means any "improve
Crucible" plan has to clear this gate explicitly or it changes nothing.

### RC-4 (HIGH) — alt-data can only ever be a market-timing overlay

Three coupled facts:

1. `crucible/data/altdata_bridge.py` emits `{terminal: (T,) array}` — **broadcast series only**, never
   per-name.
2. `evolve.py:284` — `inputs = available_terminals(panel) if is_overlay else INPUTS`. The
   cross-sectional path draws `INPUTS` = OHLCV only (`grammar.py:65`). Feature slots are reachable
   **only** by the overlay path.
3. `_overlay_returns` (`evolve.py:139`) collapses the cross-section with
   `g[t] = nanmean_n(scores[t])` → one scalar per day → `tanh` tilt on the existing book.

So every non-price dataset the project has connected (TWSE T86 institutional flow, TAIFEX
large-trader OI, CFTC COT, FRED, GDELT, EDGAR) can only ever answer *"should the book be bigger or
smaller today?"* — `T` observations instead of `T×N`, the lowest-information-density and
lowest-capacity use of that data. **70% of all lifetime trials (282/403) ran through this channel.**

The v2.9 C2-06 change that produced this over-corrected: the real problem was that *broadcast* slots
rank() to dead constants in a cross-sectional genome; the fix excluded **all** slots rather than
filtering by shape. Corroborating evidence that the cross-sectional channel is the productive one: the
only PROMISING the project has ever recorded (`tw_smallcap_mom_rev`, month-revenue drift) came through
the per-name cross-sectional scorecard path, **not** through the miner.

### RC-5 (HIGH) — the search has no negative feedback

`ledger.KILLED_VERDICTS = {"NO_GO", "NO_ADD", "GATE_FAIL"}` (`ledger.py:29`), but the funnel's only
failure verdict is `LOGGED`. The lifetime record contains **zero** rows in `KILLED_VERDICTS`, so
`killed_families()` returns `[]` by construction and the LLM proposer's anti-rediscovery line renders
`(none)` on every tick (`llm_proposer.py:144-150`, already flagged in-code). 403 trials generated no
learning signal.

Compounding it: dedup is by **exact formula hash**, so a semantically identical re-derivation is a
fresh charge, while a hypothesis burned once at MDE ≈ 3.6 is never re-admitted as the substrate
deepens.

### RC-6 (MEDIUM) — the binding holdout gate has never executed

`evolve.py:366` pre-filters on `c.result.passes_gate` computed on the **train** split, using the
*identical* 6-way gate as the holdout re-score. Since the two sealed legs fail on train, no candidate
ever reaches the holdout stage. The expensive, correctly designed stage (full file-drawer `N`,
cross-search dispersion pool) sits behind the broken one and is dead code in production. The docstring
calls the train step a "cheap pre-filter" — it is not; the train split is *longer* than the holdout, so
for `marginal_t` it is strictly the harsher test.

### RC-7 (MEDIUM) — multiplicity accounting is batch-shaped, not hypothesis-shaped

`tier4_deflation` (`eval_harness.py:315-319`) sets `n_trials = |batch with finite IC-IR|`. The same
100 alphas deflate against `N=100` submitted together and against `N=10` submitted in ten batches —
identical evidence, different verdict, chosen by submission shape. Cross-tick multiplicity is meant to
be LORD++, which `fdr.py:26-36` openly documents as **"ACCOUNTED, not CONTROLLED"** (nothing consumes
`next_level()`; `alpha_floor` defaults to 0.0).

### RC-8 (MEDIUM) — Tier-0 causality is not enforced on generated genomes

`assert_causal` / `tier0_hygiene` (the truncation-equivalence backstop, the project's best anti-leak
device) are called only from `scorecard.evaluate_signal` and from tests. `evolve.score()` →
`eval_on_panel` never truncation-tests a genome; causality is asserted by construction in a docstring
and spot-checked on one signal in `tests/signals/test_generation_grammar.py:170`. This fails in the
*false-positive* direction, so it is not a cause of 0-PROMISING — but it is the hole that would turn a
future discovery into a mirage, and it is cheap to close.

### RC-9 (MEDIUM) — the uplift threshold was never calibrated against its own null

Every one of 170 GP genomes produced `ΔSR ≥ 0.368` (median +0.540). Random genomes should not add half
a Sharpe point to a book. That the floor is 0.10 says it was set without reference to the null
distribution it has to separate from — the combiner-based ΔSR estimator carries a large positive bias
on this machinery (partly addressed by the F14-4 degenerate-vol cull, evidently not removed). A leg
that passes 100% of the time is not protecting anything, and it is the *only* leg standing between the
corrected contract and a false discovery once the seals are dropped.

---

## 3. What is already right (do not regress these)

- **Tier-0 truncation-equivalence causality** (`eval_harness.assert_causal`) is a stronger anti-leak
  device than most research stacks have. Its problem is coverage, not design.
- **CPCV purge/embargo** is implemented correctly — left-seam embargo plus right-seam purge with
  `t + horizon < b` so a label window never crosses a train gap (`fitness.py:175-201`,
  `eval_harness.py:612-619`).
- **The overlay z-score is genuinely causal** — expanding-window `cumsum` mean/var, `_OVERLAY_MIN_OBS`
  warmup, then an explicit one-bar lag (`evolve.py:146-168`). It passes truncation-equivalence by
  construction.
- **`_ar1_effective_n` clamps negative ρ₁ to 0** — it can only deflate, never inflate, a t-stat.
- **Version + provenance discipline** (`crucible/version.py`, raw-byte `gates_hash`, the frozen
  `519158fa1450` / `22a18172be1a` / `0ccf6dd584f0` moats) is exemplary and is what made this audit
  cheap to run.
- **The v5.0 fail-closed repair** of `interp_mde` was the right call, correctly reasoned, and correctly
  refused the tempting re-fit to the measured `N^-0.21`.
- **The self-documentation is unusually honest** — `fdr.py` says its FDR is accounted not controlled;
  the power YAML calls its own signal damning; `fitness.py` registers its Tier-C seal in a test. None
  of the critical findings above required catching a claim that the code denied.

---

## 4. The fix that already exists and is wired to nothing

`finrl_pro_ds/crucible/corrected_contract.py` implements exactly the right replacement:

- **Significance:** the **Jobson–Korkie–Memmel** Sharpe-difference z on the full panel
  (`_sharpe_diff_z`, Memmel 2003 paired variance). This is the correct statistic — it tests
  `H0: SR(b_aug) = SR(b_base)` directly, it is *paired* (so ρ ≈ 0.99 between the two books shrinks the
  SE rather than swamping it), it is **scale-invariant**, and it deflates by AR(1) `n_eff`.
- **Multiplicity:** a **binding** LORD++ p-threshold (`p_value ≤ OnlineFDR.next_level()`) — the F13 fix,
  realized by *consuming* the audited recurrence rather than editing it.
- **Retains** the three cheap guards (uplift / fragility / collinearity); **drops** both seals.
- Thresholds live in `configs/crucible_corrected_contract.gates.yaml`, so no frozen funnel byte moves.

Measured (`results/crucible_corrected_contract/corrected_calibration_both.json`):

| | shipped funnel | corrected contract |
|---|---|---|
| E1 null FPR (T=2048, 500 candidates) | — | **0.000** (Clopper-Pearson upper-95% 0.0060 ≤ 0.01 ceiling) |
| Power @ realized ΔSR 0.5 (T=4044) | **0.000** | **0.813** |
| Power @ realized ΔSR 0.8 | 0.825 | 0.991 |
| Power @ ΔSR 0.21 (β=0.002) | 0.000 | 0.200 |

It clears both halves of the pre-registered joint criterion in
`configs/crucible_calibration.gates.yaml:130` (E1 ≤ `null_fpr_max`, power ≥ 0.80 at ΔSR ≤
`mde_delta_sr_max` = 1.00).

**And `grep` finds it imported by exactly two things: its own research script and its tests.**
`evolve.py`, `fitness.py`, and `orchestrator.py` never call it. It was built as a parallel artifact
for the ICAIF paper and deliberately kept out of production to preserve CRU-1.

The highest-value change available to this project is therefore an **integration plus an operator
decision**, not new research.

---

## 5. Ranked roadmap

### U1 — Promote the corrected contract to the production decision layer *(do first)*

Wire `corrected_contract_fitness` into `evolve()` as the binding holdout gate, behind a
`contract: shipped | corrected` switch on `Substrate`. Make the train step an actually-cheap
pre-filter (uplift + fragility + collinearity, **no** significance leg) so the holdout stage becomes
reachable — closing RC-6 in the same change.

This is a **MAJOR** bump (`crucible-v6.0`): it changes the verdict function. It moves no frozen gate
byte (thresholds already live in the parallel gates file), but unlike v3.0/v4.0/v5.0 it is **not**
monotone-stricter, so CRU-1's "preserves every recorded verdict" must be re-argued explicitly rather
than asserted: the honest framing is that the recorded 0-PROMISING was produced by a contract now shown
to have zero power, and the record should be re-scored under the new contract and re-published, not
silently inherited.

*Acceptance test:* re-run E1/E2 **through `evolve()`** (not the standalone scorer) and require
E1 CP-upper ≤ 0.01 **and** power ≥ 0.80 at ΔSR ≤ 0.50. Re-score the 170-row lifetime record under the
corrected contract and report the leg-pass table from §1.1 again — that table is the acceptance
artifact.

*Cost:* small. Scorer, gates file, unit tests and calibration harness all exist.

### U2 — Re-calibrate the power guard against the corrected contract *(must ship with U1)*

`calibration_mde_sweep.json` measured the **shipped** funnel's MDE. Keeping it after switching
contracts leaves the machine refusing every substrate for a reason that no longer applies — U1 without
U2 yields zero discoveries just as surely as today.

- Re-run `mde_sweep` under the corrected contract.
- Extend `t_grid` above 4044 so deep substrates **interpolate between measured anchors** instead of
  hitting `unmeasured_high` — this is exactly the "extend the sweep" unblock v5.0's docstring names as
  the intended forcing function.
- Only then is `action: refuse` at `plausible_delta_sr_max: 0.50` a meaningful gate.

*Acceptance:* the cross-asset substrate stamps `implied_mde_delta_sr ≤ 0.50` in `interp_mode: grid`
or `interpolated`, or the operator records an explicit logged override.

### U3 — Open the cross-sectional channel to per-name alt-data *(IMPLEMENTED — `crucible-v7.1`)*

Two coupled changes:

1. `crucible/data/panel_bridge.py` / `altdata_bridge.py` gain the ability to emit `(T, N)` slots, not
   only `(T,)`.
2. `evolve.py:284` selects terminals **by slot shape** rather than by `candidate_type`: a `(T,N)` slot
   joins the cross-sectional leaf set; only `(T,)` slots stay overlay-only.

This restores what the v2.9 C2-06 fix over-corrected away, and points the miner at the one channel
that has ever produced a PROMISING in this project.

**Implemented 2026-07-29 as `crucible-v7.1` (MINOR).** `grammar.cross_sectional_terminals(panel)`
returns `INPUTS` + the `(T,N)` slots only; `(T,)` slots stay excluded from cross-sectional genomes, so
the whole of C2-06's protection against dead broadcast terminals is retained. `available_terminals`
(the overlay registry) is unchanged. On a panel with no `(T,N)` slot the draw is *exactly* `INPUTS`, so
no existing search trajectory moves — the change is reachable only by a panel that carries per-name
data.

The evaluation path needed nothing: `eval_on_panel` already broadcasts `(T,)` and passes `(T,N)`
through, and `Panel._sliced_slots` already slices both on axis 0, so the Tier-0 truncation tripwire
holds for a per-name slot (asserted). Verified end-to-end rather than by construction: a
`rank(twse:inst_net)` genome is scored by the real cross-sectional search — not culled — returning a
measured marginal ΔSR over 15 CPCV paths.

`panel_bridge.build_panel_feature_slot` lands alongside it: per-ticker series assembled into one
`(T,N)` slot, column order following `Panel.tickers`, an absent ticker becoming an **all-NaN column
rather than a 0.0** (a zero is a tradeable value), and the `assert_asof_join_causal` PIT gate running
**per column** — a per-name panel is exactly where one late-reporting name could smuggle look-ahead
into an otherwise clean matrix. Wiring a concrete connector to it (TWSE T86 institutional net flow is
the obvious first) is data work that remains.

*Correction to this section's original framing:* it claimed the move buys `T×N` observations and that
this is "the only lever that materially changes MDE". The V3 measurement says otherwise — MDE in ΔSR
units is flat in breadth. What breadth buys is a larger ΔSR for the same per-name signal, which clears
a fixed ceiling rather than lowering it. The change is still right; the mechanism is not the one
originally stated.

### U4 — Give the search a memory

- Write a terminal verdict (`NO_GO`) when a candidate is decisively rejected — or redefine
  `killed_families()` over `LOGGED` plus a decisiveness criterion — so the proposer prompt stops
  rendering `(none)`.
- Replace exact-formula-hash dedup with semantic dedup (canonicalized AST, or IC-correlation to an
  already-scored genome).
- Add **power-aware re-admission**: a hypothesis burned at implied MDE 3.6 should become re-testable
  once the substrate is deep enough to test it.

### U5 — Fix the shape of multiplicity accounting

Replace batch-size `n_trials` with a per-substrate cumulative hypothesis count, and let LORD++ actually
bind in the funnel (the corrected contract already does this via `lord_pass`). This closes the
"submit in small batches" loophole in RC-7.

### U6 — Enforce Tier-0 on genomes

Call `assert_causal` (or a 2–3-row truncation probe) inside `evolve.score()` before fitness — the cost
is negligible next to the CPCV fitness evaluation. Add a mutation tripwire: a deliberately leaky DSL
operator must make the test fail.

### U7 — Re-calibrate the uplift floor against its own null *(gate on this before U1 goes live)*

Once the seals drop, `uplift_min` is the *only* remaining economic leg, and it currently passes 100%
of candidates. Measure the ΔSR null distribution on the real base books (the machinery already exists:
`crucible_matched_null.py`) and set `uplift_min` at a real quantile of it. Shipping U1 with a 100%-pass
uplift leg would swap a zero-power machine for a poorly-controlled one.

### U8 — The honest ceiling

Even corrected, the contract needs ΔSR ≈ 0.3–0.5 at T ≈ 4000 to reach 0.8 power. The lever that moves
MDE is **breadth (T×N) and forward accumulation**, not more ticks on the same panel. Crucible's future
is as a cross-sectional per-name miner (U3) on deeper substrates, not as a timing-overlay miner. This
does not disturb the standing direction (TSMOM → paper); it re-points the discovery machine at the
substrate class where its statistics can actually work.

---

## 6. Findings table

| ID | Sev | File | Issue |
|---|---|---|---|
| RC-1 | CRITICAL | `signals/generation/fitness.py:360` | `marginal_t` is a substitution-residual mean-dominance test (identity `marg = w_c(r_c − b_base)`), sign-inverted for diversifiers, not scale-invariant. 0/170 lifetime pass. |
| RC-2 | CRITICAL | `signals/generation/fitness.py:322` | `dsr_aug` deflates the *absolute* augmented-book Sharpe — unpassable on a ~0-Sharpe base era, non-binding on a good one. 0/170 lifetime pass, median 0.000. |
| RC-3 | CRITICAL | `configs/crucible_power.gates.yaml:30,41` | `refuse` @ ceiling 0.50 vs measured MDE ≥ 1.4025 everywhere ⇒ every substrate refused; the loop cannot mine. |
| RC-4 | HIGH | `crucible/data/altdata_bridge.py:7`, `signals/generation/evolve.py:284` | Alt-data is `(T,)`-only and reachable only by the overlay path ⇒ restricted to market timing; 70% of trials spent there. |
| RC-5 | HIGH | `crucible/ledger.py:29,186` | Funnel emits `LOGGED`, never a `KILLED_VERDICTS` value ⇒ `killed_families()` structurally empty ⇒ no negative feedback to the proposer. |
| RC-6 | MEDIUM | `signals/generation/evolve.py:366` | Train pre-filter uses the identical sealed gate ⇒ the binding holdout gate has never executed in production. |
| RC-7 | MEDIUM | `signals/eval_harness.py:315`, `crucible/orchestrator/fdr.py:26` | `n_trials` = batch size (verdict depends on submission shape); LORD++ accounts but does not control. |
| RC-8 | MEDIUM | `signals/generation/evolve.py:320` | `assert_causal` never runs on a generated genome; causality asserted by docstring. |
| RC-9 | MEDIUM | `configs/crucible_calibration.gates.yaml:14` | `uplift_min = 0.10` passes 170/170 — never calibrated against its own null; becomes the sole economic leg after U1. |
| RC-10 | HIGH | `tests/crucible/test_reproduce_p5.py:90`, `test_reproduce_cohort.py:57` | Both end-to-end reproducibility tests are RED: the power guard refuses the synthetic substrate, so no manifest is ever mined. `--force` forces the *dirty* gate, not the *power* gate. |

---

## 7. Test baseline (2026-07-29, this worktree)

`pytest tests/crucible tests/signals tests/research` — **3 failures**, none of which is a
crucible *logic* defect, but one of which is RC-3 breaking CI:

| Test | Cause | Class |
|---|---|---|
| `tests/crucible/test_reproduce_p5.py::test_synthetic_orchestrator_run_reproduces_via_cli` | power guard refuses the mine (below) | **RC-10 — real** |
| `tests/crucible/test_reproduce_cohort.py::test_cohort_run_reproduces_and_cohort_drift_is_detected` | same | **RC-10 — real** |
| `tests/research/test_vwap_falsification.py::test_swing_verdict_no_go_and_not_suppression_bug` | `FileNotFoundError: results/xsec_momentum/ohlcv_daily.parquet` — `results/` is gitignored and absent from this worktree | environment, not code |

### RC-10 — the reproducibility contract cannot be exercised

Running the exact command the test issues:

```
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 1 --t 320 --n 6 \
       --max-proposals 8 --start-ts 2020-01-04T00:00:00 --out <tmp> --force
```

```
substrate synthetic UNDERPOWERED: implied MDE 5.58 ΔSR > ceiling 0.50 (T=320, holdout=80, extrapolated_low)
night 1/1 ... dirty=False mined=False status=OK promising=0 ... (UNDERPOWERED — skipped)
done: 1 nights, mined 0 (only when dirty), 0 PROMISING total
```

`--t 320` ⇒ holdout 80 bars, below the sweep's low anchor (189), so
`interp_mde` returns `3.63·√(189/80) = 5.58` ≫ ceiling 0.50 ⇒ refuse. The test's
`assert proc.returncode == 0` passes because **a refused tick is a successful tick**; the failure only
surfaces at `len(manifests) == 1`.

So RC-3 is not merely a statement about production substrates — it has already broken the v2.6
"reproduce payoff", the one mechanism that verifies a past run re-derives its verdicts.

**Two corrections to the record this forces:**

1. `--force` forces the `substrate_dirty` eligibility gate only. The power gate has its own flag,
   `--force-underpowered`, which neither reproduce test passes.
2. `project_crucible_tier_b_c_remediation_s553` records this loose end as "RESOLVED (2026-07-16) —
   stated power-guard cause was wrong; was CRLF `gates_hash`, fixed by cont-134." The CRLF fix **is**
   holding — verified in this worktree, `core.autocrlf=true` yet both gates files are LF and hash to
   the frozen moats `519158fa1450` / `22a18172be1a` — and the tests are red anyway, for the
   power-guard reason originally suspected. That memory needs correcting.

*Fix (trivial, ship with U2):* pass `--force-underpowered` in both reproduce tests — the synthetic
substrate is a **mechanics fixture**, not a discovery run, so its statistical power is irrelevant to
whether a manifest round-trips — or make `--force` imply it under `--mode synthetic`.

---

## 8. U1 + U2 — implemented 2026-07-29 (`crucible-v6.0`)

### U1 — the corrected contract is wired

`evolve(contract="shipped"|"corrected")`, threaded through `run_hypothesis_loop` → `Substrate.contract`
→ `crucible_orchestrator.py --contract corrected`. Under `corrected`:

- the **train step becomes a cheap pre-filter** (uplift / fragility / collinearity / non-degenerate
  only — `_passes_cheap_prefilter`), so the certified holdout stage is reachable for the first time
  (RC-6 closed);
- the **holdout decision** is `corrected_contract_fitness(...).passes_corrected` — one JKM
  Sharpe-difference z ≥ `t_min` **and** a binding LORD++ p-gate — with `marginal_t` and `dsr_aug`
  dropped (RC-1, RC-2 closed);
- the orchestrator supplies the **tightest** LORD++ level of the tick (`_tick_lord_level`: simulate the
  tick on a copy of the account assuming no discovery, take the min), so nothing is promoted at a
  level looser than one it could actually be charged;
- the run manifest now pins `contract` + `corrected_gates_hash`, so no verdict can be read without
  knowing which gate produced it.

`contract` defaults to `"shipped"`, so every existing call path is byte-identical to v5.0. Nine
tripwires in `tests/crucible/test_contract_corrected_evolve.py` (all green), including the headline:
on a planted substrate the shipped contract emits **0** PROMISING and the corrected contract emits
**> 0**, while both emit **0** on a pure null.

**CRU-1 is explicitly not claimed.** This bump is not monotone-stricter — that is the point. The
0-PROMISING record must be re-scored under the corrected contract, never pooled with it; see the
`version.py` v6.0 note.

### U2 — the power sweep is re-measured against the corrected contract

`crucible_calibration.py --exp mde_sweep --contract corrected` writes
`results/crucible_calibration/calibration_mde_sweep_corrected.json`;
`crucible_power.gates.yaml` gained `calibration_sweep_path_corrected` and the loader selects the curve
matching the substrate's contract (a missing curve disables the stamp loudly rather than substituting
the other contract's). `t_grid` extended `[756…4044]` → `[…, 32256]`, pushing measured coverage from
holdout **1011 → 8064** so deeper substrates **interpolate between measured anchors** instead of
returning `unmeasured_high`/+inf — the structural unblock v5.0's own docstring asked for.

Two caveats that must travel with that claim, both now recorded in the gates file:

- **Coverage is still finite, by design.** Past the top anchor `interp_mde` still returns +inf and the
  guard still refuses. "Deep substrates interpolate" holds only up to the deepest *measured* holdout.
- **The axis is bars, not time.** The generator is a daily-scale DGP with `periods_per_year=252`, so a
  row means "MDE at N bars of *this* per-bar signal-to-noise". Reading a high-N row as an intraday
  substrate's MDE additionally assumes intraday per-bar SNR resembles daily — a modelling assumption
  this sweep does not measure.

**One deliberate semantic change, and it matters.** The shipped curve scores on the FULL panel while
recording `holdout_bars` — so it understates MDE, i.e. claims more power than the gate has. The
corrected curve scores on the **binding holdout window**, the gate's real N. The shipped curve is left
byte-stable (it is the provenance v5.0 reads).

Measured (30 seeds, power target 0.80):

| T | holdout | shipped MDE *(full-panel)* | corrected MDE *(full-panel)* | **corrected MDE *(deployed, holdout)*** |
|---|---|---|---|---|
| 756 | 189 | 3.63 | — | 1.79 |
| 1512 | 378 | — | — | 1.73 |
| 2782 | 696 | 1.94 | 0.797 | 1.23 |
| 4044 | 1011 | 1.4025 | **0.507** | **1.28** |
| 6048 | 1512 | *unmeasured_high* | — | 0.79 |
| 8064 | 2016 | *unmeasured_high* | — | 0.86 |
| 16128 | 4032 | *unmeasured_high* | — | 0.55 |
| 32256 | 8064 | *unmeasured_high* | — | **0.39 ✓** |

**A near-miss worth recording: the first pass of this table showed 0.55 at BOTH deep anchors and I
almost wrote it up as the curve flattening into a floor.** It was the beta grid running out. MDE is
read as "the first grid point reaching power 0.80", and at holdout 4032/8064 the contract scored power
0.20/0.43 at `beta=0.002` and 0.97/1.00 at `beta=0.003` with nothing in between — so both depths
reported `beta=0.003`'s realized ΔSR (≈0.55). The tell that it was not a floor: **power at the fixed
`beta=0.002` point kept climbing with depth (0.033 → 0.20 → 0.43)**, which a data-independent floor
forbids. Densifying the low end (`beta_grid` + 0.0005…0.0025) resolved holdout 8064 to **0.393** at
`beta=0.0025` (power 0.90).

Note the error direction: a grid-pinned MDE **over-states** the detection limit, so the guard refuses
substrates that are in fact adequately powered — fail-safe, but wrong in exactly the direction that
keeps the loop switched off.

**The floor cont-131 pinned to `marginal_t` is gone with the leg.** That audit identified a
data-independent MDE floor near ΔSR 0.4 arising from the `marginal_t` specification, which made the
0.50 ceiling unreachable at any N. No such floor is visible under the corrected contract in the
measured range. Consequence, verified against the live guard:

```
T= 4044 (cross_asset daily)  holdout=1011  MDE=1.281  mode=grid          -> REFUSE
T=24000 (between anchors)    holdout=6000  MDE=0.455  mode=interpolated  -> ALLOW
T=32256 (deep)               holdout=8064  MDE=0.393  mode=grid          -> ALLOW
```

**This is the first substrate configuration in Crucible's history that clears its own power guard.**
The crossover sits at holdout ≈ 5,000–6,000 bars, and depths between measured anchors now
`interpolate` rather than returning +inf — the structural unblock, demonstrated end-to-end.

> ⚠️ **Qualified by V4 below.** That ALLOW is measured on the **overlay** curve. A substrate that mines
> *both* candidate types is judged by the worse of the two curves, and the cross-sectional surface only
> reaches holdout 2016 — so a mixed-type substrate is still REFUSED (`unmeasured_high`) at those
> depths. The ALLOW stands for an overlay-only substrate; it does not yet stand for a general one.

### A limitation of the power stamp — and a correction to my own first reading of it

**What is real:** the curve is measured on the **overlay path only**. `_e2_power_curve` plants
`macro:plant` and scores it through `_overlay_returns`. There is no cross-sectional power curve, yet
the stamp is applied to every substrate regardless of `candidate_type`. That is an unmeasured claim of
exactly the family v5.0 was written to close, and it is why V2/V3 below build one.

**What I got wrong, and am correcting here.** My first reading of this was that the guard's `holdout_bars`
axis is "blind to cross-sectional breadth", so a per-name `(T,N)` panel carrying ~N× the effective
observations would be refused despite being adequately powered — making U3's payoff invisible to the
gate. I built the planted cross-sectional fixture to measure that, and **the measurement does not
support the claim.** Probe at 10 seeds, corrected contract, holdout window:

| | N=12 | N=100 |
|---|---|---|
| holdout 378 — realized ΔSR at the detection transition | ~1.3–2.4 | ~0.8–2.6 |
| holdout 1011 — same | ~0.7–1.8 | ~0.9–2.7 |
| realized ΔSR at a FIXED `beta=0.001`, holdout 1011 | +1.80 | **+5.88** |

**MDE expressed in ΔSR units does not fall with breadth — and it should not.** ΔSR is already
risk-adjusted, and the standard error of a Sharpe *difference* is set by the number of **time**
observations, not by the cross-section. So keying the guard on bars is *correct* for the quantity it
measures.

What breadth actually buys is the bottom row: **the same underlying per-name signal produces a much
larger ΔSR on a wide panel** (+5.88 vs +1.80 at identical `beta` and identical bars). Breadth moves a
real alpha *above* a fixed ceiling rather than moving the ceiling down. That is a better argument for
U3 than the one I made, and it arrives through the `plausible_delta_sr_max` comparison rather than
through the MDE curve.

The residual, genuine issue is narrower than I first stated: the overlay and cross-sectional curves are
numerically different even though both key on bars, so a cross-sectional substrate should be judged
against a cross-sectional curve.

### V3/V4 — the cross-sectional surface, and what it changes

Measured (24 seeds, 10-point beta grid, corrected contract, binding holdout window):

| holdout | N=12 | N=25 | N=50 | N=100 | **overlay curve** |
|---|---|---|---|---|---|
| 378 | 2.15 | 2.66 | 2.43 | 1.93 | 1.73 |
| 696 | 1.85 | 1.79 | 1.37 | 1.33 | 1.23 |
| 1011 | 1.83 | 1.18 | 1.35 | 1.40 | 1.28 |
| 2016 | 0.83 | 1.01 | 0.91 | 1.19 | 0.86 |

Two readings:

1. **MDE is flat in breadth**, within noise — confirming the correction above. It is *not* flat in
   depth. So pooling over `n` (taking the worst value at each depth) is the honest consumer: indexing
   by `n` would claim a resolution the measurement does not support.
2. **The cross-sectional path is EQUAL-OR-WORSE than the overlay path at matched depth.** Since every
   curve before this was measured on the overlay path and applied to both, the guard has been
   **under-stating MDE for cross-sectional mines by ~35–40%** — claiming more power than those
   substrates have. That is the fail-OPEN direction v5.0 exists to close, and it was latent for the
   same reason v5.0's bug was: nothing had yet been mined where it mattered.

V4 repairs it. `stamp_substrate_power` now takes the **worst MDE across the candidate types the
substrate will actually mine**, and a type with no measured curve returns
`(+inf, 'unmeasured_candidate_type')` — refuse — rather than borrowing another type's curve. Verified
against the live curves:

| T | holdout | overlay-only | xsec-only | **both (worst)** |
|---|---|---|---|---|
| 4044 | 1011 | 1.281 REFUSE | 1.833 REFUSE | 1.833 REFUSE |
| 8064 | 2016 | 0.862 REFUSE | 1.185 REFUSE | 1.185 REFUSE |
| 24000 | 6000 | **0.455 ALLOW** | `unmeasured_high` | **REFUSE** |
| 32256 | 8064 | **0.393 ALLOW** | `unmeasured_high` | **REFUSE** |

So V4 **tightens** the guard, and specifically withdraws the unqualified "first ALLOW" claim: the ALLOW
holds for an overlay-only substrate, not for one mining both types.

### The extended surface — and the deep ALLOW does not survive

The cross-sectional surface was re-measured out to holdout 8064 to close that gap. It does not close
the way I expected:

| holdout | N=12 | N=25 | N=50 | N=100 | **pooled (worst)** | overlay | **worst-across-types** | |
|---|---|---|---|---|---|---|---|---|
| 1011 | 1.83 | 1.18 | 1.35 | 1.40 | 1.83 | 1.28 | **1.83** | REFUSE |
| 2016 | 0.83 | 1.01 | 0.91 | 1.19 | 1.19 | 0.86 | **1.19** | REFUSE |
| 4032 | 0.66 | 0.60 | 0.83 | 0.73 | 0.83 | 0.55 | **0.84** | REFUSE |
| 8064 | 0.42 | 0.64 | 0.54 | 0.64 | 0.64 | 0.39 | **0.64** | REFUSE |

**A substrate mining both candidate types is refused at every measured depth.** The cross-sectional
path still needs ΔSR ≈ 0.64 at holdout 8064 against a 0.50 ceiling, so the overlay curve's 0.393 ALLOW
is overridden. Extending the surface therefore did not unlock the deep mine — it converted an
*unmeasured* refusal (`unmeasured_high`) into a *measured* one. That is a strictly better state, and a
less exciting one than the interim result suggested.

These deep rows are **not** grid-limited — checked explicitly, having made that mistake once already.
Each has several sub-0.80 power points beneath the reported beta (N=50 at holdout 8064: power 0.083 →
0.750 → 1.000 across beta 0.0001 → 0.00015 → 0.0002), so the transition is properly bracketed.

One honest cost to note: pooling by the worst N spends real headroom. At holdout 8064 the four
breadths give 0.42 / 0.64 / 0.54 / 0.64 — a spread that is MC noise at 24 seeds if MDE is truly
N-independent, and pooling reports 0.64 against a mean of ~0.56. The right fix is more seeds, not a
looser pooling rule; and it would not change this verdict either way, since even the mean exceeds 0.50.

**Two honest readings, and the second corrects a headline in §4.**

1. **Like-for-like, the contract is ~2.8× more sensitive**: at T=4044 on the same full-panel
   convention, MDE 1.4025 → **0.507**. (This reproduces `corrected_calibration_both.json`'s
   "power 0.813 @ ΔSR 0.5" exactly — that figure is a full-panel measurement.)
2. **Deployed, most of that is given back by the holdout window.** The gate decides on 1011 bars, not
   4044, so the corrected contract's *deployed* MDE at daily depth is **1.28**, not 0.507. It still
   beats the shipped 1.4025 — and does so measured on 4× fewer bars, a conservative comparison — but
   §4's "power 0.00 → 0.81 at ΔSR 0.5" is a **full-panel** claim and must not be read as a deployed one.

**Consequence: the guard still refuses the daily substrates, and the ceiling was NOT moved.** At
holdout 1011 the corrected MDE 1.28 still exceeds `plausible_delta_sr_max: 0.50`. Editing that
threshold to make the mine pass after seeing the result is precisely the goal-post move this project's
own gate-repair-before-freeze rule forbids, so `crucible_power.gates.yaml`'s thresholds are untouched.

What changed is that the refusal is now (a) measured against the right gate on the right window, and
(b) **no longer structural**. The corrected curve reaches ≈0.79 by holdout 1512 and ≤0.55 by holdout
4032, with no evidence of the data-independent floor that made the shipped contract's ceiling
unreachable at any N. So clearing 0.50 is now a question of substrate **depth and breadth** rather than
an asymptote — U3 (per-name `(T,N)` alt-data, which multiplies effective N by the cross-section) and U8
territory, and the first concrete reason to expect the guard could ever say yes.

### U1e — E1 through the search: a new finding, and it is RED

U1's own acceptance criterion said "re-run E1/E2 **through `evolve()`**, not the standalone scorer".
That distinction turned out to matter, for a reason worth stating plainly:

**Dropping `dsr_aug` also dropped the funnel's only file-drawer deflation.** The shipped contract paid
for search multiplicity by deflating the augmented-book Sharpe against `gen_n_eff` — *every genome ever
scored*. The corrected contract has no such term; its multiplicity control is the LORD++ p-gate, which
charges **one test per pre-registered spec** while a GP search evaluates hundreds of evolved offspring
that charge nothing. So the corrected contract's published E1 (FPR 0.000 over 500 candidates) is a
**per-candidate, one-hypothesis-at-a-time** measurement and does **not** bound the FPR of the same
contract *inside a search*. That is precisely the audit §5 caveat — the corrected contract was designed
for "pre-registered low-N single-hypothesis probes", not for mass mining.

Measured (`crucible_calibration.py --exp e1 --contract corrected`, 60 null panels driven through the
real `evolve`, unchanged `e1_null` ceilings):

| bound | value | ceiling | |
|---|---|---|---|
| per-candidate FPR | 0.0004 (CP-upper95 **0.0020**) | 0.01 | **PASS** |
| per-tick FPR | 0.017 — 1 of 60 panels (CP-upper95 **0.077**) | 0.05 | **FAIL** |

At n=60 the per-tick Clopper-Pearson bound is **0.0487 even at k=0**, so a single false positive is
arithmetically sufficient to fail a 0.05 ceiling — the reading was bound-slack-limited, not
necessarily rate-limited. `n_panels` is experimental design, not a threshold, and raising it tightens
the bound in *both* directions (it can only make a genuinely over-ceiling FPR **easier** to detect), so
re-measuring at higher n is the sharper reading, not the softer one. `--e1-panels` was added for it.

**Re-measured at n=240 (7 PROMISING over 9,236 genomes / 1,750 holdout evaluations, 6 panels affected):**

| bound | n=60 | **n=240** | ceiling | |
|---|---|---|---|---|
| per-candidate FPR | 0.0004 (CP≤0.0020) | 0.00076 (CP≤**0.0014**) | 0.01 | **PASS** |
| per-tick FPR | 0.017 (CP≤0.077) | 0.025 (CP≤**0.0487**) | 0.05 | **PASS** |

**Verdict: GREEN — but read the margin, not the label.** The point estimate went *up* (0.017 → 0.025)
while the bound came *down* (0.077 → 0.0487): the extra panels resolved the rate better and it landed
just under the ceiling, with **~2% headroom**. One more false positive in those 240 panels (7/240)
would put CP-upper at ≈0.055 and flip it back to RED. The point estimate sitting at half the ceiling
says the true rate is probably fine; the bound says we do not yet know that sharply.

Two things follow, and neither is "ship it":

- **Operationally, 2.5% per-tick is not nothing.** On a nightly cadence a *pure-null* substrate would
  emit a false PROMISING roughly nine times a year. That clears the pre-registered ceiling and is still
  a real review burden — the shipped contract's answer to this was the `gen_n_eff` deflation that v6.0
  removes.
- **The exposure is specifically search multiplicity**, so the clean fix is structural rather than a
  tighter threshold. Neither is implemented here:
  1. **`prereg_only`** — only pre-registered seeds (never evolved offspring) are eligible for the
     corrected holdout gate. This exactly aligns the decision with the LORD++ charging unit and is the
     audit's own §5 design ("pre-registered low-N single-hypothesis probes"). It removes the exposure
     rather than bounding it.
  2. **Split the tick's α** — keep offspring eligible but threshold at `lord_level / n_holdout_tests`,
     so the search pays for its own multiplicity.

### V1 — `prereg_only` implemented, and it closes the exposure outright

Fix (1) is now the shipped default: `eligibility.offspring_policy: prereg_only` in
`crucible_corrected_contract.gates.yaml`. Under it an evolved offspring is still generated, scored and
ledgered — it belongs in the file drawer — but it can never be **promoted**, because it charges no
LORD++ wealth. That makes the decision unit identical to the charging unit: one pre-registered
hypothesis, one test, one charged level. `seed_formulas` *is* the pre-registration (the orchestrator
passes exactly the tick's fresh specs), so the restriction needs no new plumbing.

Re-measured, same 240 null panels through the real `evolve`:

| bound | `all` (as first wired) | **`prereg_only` (default)** | ceiling |
|---|---|---|---|
| per-candidate FPR | 0.00076 (CP≤0.0014) | **0.0000 (CP≤0.0003)** | 0.01 |
| per-tick FPR | 0.025 (CP≤0.0487) | **0.0000 (CP≤0.0120)** | 0.05 |

**Zero false PROMISING in 240 null panels**, and the per-tick bound goes from ~2% headroom to roughly
4× under the ceiling. This is a direct confirmation of the diagnosis: the residual FPR *was* offspring
being promoted on a hypothesis the FDR account never paid for, and removing that removes it.

`offspring_policy: all` remains available for a substrate where the search's discovery upside is worth
the unpaid multiplicity — but it must be re-measured at that substrate's own search budget first, and
the gates file says so.

**Recommendation:** `contract: corrected` with the default `prereg_only` is now calibrated for
production use on a pre-registered hypothesis stream.

### Turned ON — `crucible-v8.0`

`--contract` now defaults to **`corrected`**. `shipped` remains selectable for reproducing a pre-v6.0
run; its verdicts are not comparable to corrected ones and the two records must never be pooled.

Flipping the default exposed a precondition that had to be fixed first, and it is worth stating plainly
because it would have been the worst possible combination. `_load_power_guard` returned
`(None, None, "")` when a calibration curve was missing, which made `_process_substrate` skip the guard
**entirely** — the substrate mined unguarded. `results/` is gitignored, so the curves are absent on
**every fresh clone**. "Contract on + curves missing" is therefore the newly-powered contract mining
with no power gate at all. Verified directly before changing anything:

```
xsec curve MISSING  : guard= False sweeps= None
  => power_gate is None -> _process_substrate skips the guard entirely -> MINES UNGUARDED
```

Now the loader returns the guard with an empty sweep map, and `stamp_substrate_power` reads an empty
candidate-type set as unmeasured (`+inf`) ⇒ refuse:

```
curve MISSING -> guard present: True | sweeps: {}
  stamp: mde=inf mode='unmeasured_empty' -> REFUSE
```

The same fix closes a latent fail-open in the v7.0 fold itself: it initialised the worst-across-types
maximum at `-inf`, so an empty type set would have passed **any** ceiling. `--no-power-guard` and
`--force-underpowered` remain the only ways through, and both are explicit.

**What turning it on does today: nothing mines.** With both curves measured, every real substrate is
refused — worst-across-types MDE 1.833 at holdout 1011 against a 0.50 ceiling, and still 0.64 at
holdout 8064. The contract being live changes the verdict function that *would* apply; the power guard
independently decides no current substrate is worth mining. Both statements hold at once, and the
ceiling was not moved to manufacture an ALLOW.

A reporting defect found while doing this and fixed: under `--contract corrected` the
`per_leg_null_pass_rate` diagnostic still tallies the **shipped** legs off the train `FitnessResult`
(the only per-genome record `evolve` surfaces), so its `all_pass=0.000` is *not* the corrected FPR. The
output now carries `per_leg_is_shipped_train_legs: true` so it cannot be misread.

### Also fixed

RC-10: `test_reproduce_p5` and `test_reproduce_cohort` now pass `--force-underpowered` (the synthetic
substrate is a mechanics fixture; `--force` only overrides `generation.enabled`).

---

## 9. Relationship to prior audits

This audit confirms the direction of the 2026-07-07 design audit and the 2026-07-14 independent audit
("0 PROMISING is a machine property"), and adds three things they did not have:

1. **A leg-by-leg pass table on the full 403-row lifetime record** — the prior audits argued the gate
   was unpassable; §1.1 measures it (uplift 170/170, both seals 0/170).
2. **A correction to the cont-131 mechanism claim.** Across the full record `marginal_t` is
   *predominantly positive* (160/170) and ceilinged at 2.116, rather than universally sign-inverted;
   sign inversion is a demonstrated hazard (F3 scale sweep), not the record's dominant pattern.
3. **RC-3 and RC-4 as first-class blockers.** The power guard's flip to `refuse` (2026-07-12) and the
   `(T,)`-only alt-data channel post-date or sit outside the earlier audits' framing, and both must be
   addressed for any gate fix to change the outcome.

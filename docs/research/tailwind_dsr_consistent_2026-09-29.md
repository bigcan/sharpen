# TAILWIND deflated Sharpe, recomputed consistently — 2026-09-29

> **Superseded for deploy purposes — read this first.** The same day's Tier-2 audit (`docs/research/tailwind-v1_deep_lifecycle_audit_2026-09-29.md`, verdict **BLOCK**) reproduced every number below exactly. It found that the executor's 0.967 / 0.667 describe an idealized batch book: unfinanced total returns, a month-end fill no live code can schedule, fixed-notional labels, and firm limits with the declared kills off.
>
> Corrected executor figures, each below its gate:
> - **Recorded (E0) method:** DSR 0.936.
> - **Excess of T-bill on net exposure:** DSR 0.907–0.934, P(pass) 0.632. The financing leg was implemented the same day (§8). T-bill only gives DSR 0.929 and P(pass) 0.632 (12/19). The declared model (T-bill plus 25 bp short borrow) gives **DSR 0.913 and P(pass) 0.609**, both FAIL.
> - **Realizable timing** (documented `w[-1]` target, fill d+2): DSR 0.923, P(pass) 0.615; delayed fills d+1..d+4 give 0.920–0.940.
> - **Holdings-based orders:** DSR 0.909, P(pass) 0.606.
> - **Declared kills on:** P(pass) 0.524.
> - **Sampling:** P(pass) 14/21 has Wilson 95% interval [0.454, 0.828].
>
> The research basis still clears when financed: DSR 0.972 with T-bill only, 0.961 with the declared 25 bp. At 50 bp borrow it is 0.945, which fails, so the borrow assumption now carries that claim. The binding record stays `BLOCK_multiplicity` 0.896 until an operator memo (roadmap N1).

**Question:** does TAILWIND (momentum TSMOM + BAB) clear the own-capital gate `min_dsr 0.95` (`configs/tailwind_v1.gates.yaml`) when the deflated Sharpe is computed the way Bailey & López de Prado define it? The two recorded figures, 0.896 and 0.871, both block it.
**Script:** `scripts/research/tailwind_dsr_consistent.py` (local caches only, ~40 s). **Artifact:** `results/tailwind_v1/dsr_consistent_2026-09-29.json`. **Tests:** `tests/research/test_tailwind_dsr_consistent.py`. Math-skill verified (§5).

## 1. Answer

| Book | Recorded DSR | Consistent DSR (N = 24) | Across N = 18–31 | Gate 0.95 |
|---|---|---|---|---|
| **Research basis** (the book as validated, Sharpe 0.732) | 0.896 | **0.988** | 0.987–0.990 | **PASS** |
| **Executor path, pre-fix** (the book as wired, Sharpe 0.554) | 0.871 | **0.930** | 0.922–0.938 | **FAIL** |
| **Executor path, lag fixed** (same day, §7; Sharpe 0.635) | — | **0.967** | 0.963–0.972 | **PASS** |

- **The research-basis block does not hold.** The recorded 0.896 charges selection twice.
- **The pre-fix executor fails, and a code defect was most of the gap.** Its Sharpe was 24% below the research book (0.554 vs 0.732). The main cause was an extra bar of execution lag (09-23 audit R-3). Fixing it the same day (§7) lifts the executor to Sharpe 0.635 and DSR 0.967, clear of 0.95 across the whole N bracket. The rest of the gap includes the binding gross cap (S564) and daily re-vol-scaling.
- **Out-of-universe evidence supports the momentum sleeve.** The frozen selected signal, run on the 14 instruments of the wide panel it never saw, earns Sharpe **0.441** with **PSR 0.974**. This is the reproducible, like-for-like replacement for the external 0.389.
  - It is out-of-universe but not out-of-time: the same 2006–2026 period.
- **Not a capital decision.** Clearing the DSR does not authorise capital. The fixed executor's challenge P(pass) reads 0.667 (vs the 0.65 floor) on only 21 disjoint windows, and the needless-termination gate still fails (§7). A Tier-2 audit of the changed executor is required before any promotion.

## 2. What was inconsistent

**Research basis 0.896** (`audit_tailwind_book.py:174-192`)
- **Selection is charged twice.** The script cuts the momentum sleeve's in-sample Sharpe (0.601) to 0.389, an out-of-universe figure from a different, uncached 32-ETF universe with no FX leg. It recombines that with BAB, then deflates the result against E[max of N=24] as if it were the in-sample best of 24 trials. An out-of-sample number has already paid for selection; it takes PSR, not DSR.
- **The estimand is mismatched.** The trial pool is momentum-only books (cross-sectional Sharpe sd 0.152), but the observed Sharpe is momentum+BAB. The same 18 books combined with BAB, as the deployed book is, have sd 0.116.

**Executor path 0.871** (`tailwind_executor_recompute.py:101-157`)
- The same momentum-only pool is used against the executor's momentum+BAB Sharpe.
- This computation has no haircut, so there is no double charge.

## 3. The consistent computation

| Rule | Applied as |
|---|---|
| R1 | Deflate the **in-sample** Sharpe of the selected book (0.732 research, 0.554 executor) |
| R2 | Trial pool = each of the 18 graded momentum books combined with BAB via `portfolio_frontier.risk_parity`; the deployed book ranks 1st of 18 |
| R3 | Trial count = the declared N = 24, with no correlation discount (§4) |
| R4 | Out-of-universe evidence scored by PSR, never deflated again |

- Skew, kurtosis and T come from the observed book's daily returns.
- PSR is called with `periods_per_year=1`. The default feeds an annualized Sharpe into the per-period formula (09-23 audit S-10).

**Reproduction.** Research honest reproduces as **0.8960** and curated as **0.9744**, both exact.

The executor reproduces as **0.8756** against the recorded 0.871. Four facts bound that gap:
- The recorded artifact no longer exists.
- The executor is deterministic: two runs are bit-identical.
- Realized vol is 9.83% vs a recorded 9.82%.
- The residual (Sharpe 0.554 vs 0.549) moves no conclusion.

**Single-fix variants of 0.896**, for traceability to the 09-23 audit:

| Variant | DSR / PSR |
|---|---|
| Like-for-like haircut 0.563 instead of 0.389, N = 24 | 0.966 |
| 0.389 haircut kept, PSR with no deflation | 0.996 |
| 0.389 haircut kept, trial count discounted to n_eff | 0.952 (rejected, fail-open, §4) |

## 4. Why the trial count is not discounted — measured

FORMULAS.md RS01, the repo's `use_effective_n` path (`eval_harness.py:343-352`) and the 09-23 audit (S-7, and its "0.964 with N_eff 4.9") all discount N by the participation-ratio n_eff when trials are correlated. `deflated_sharpe_ratio`, however, builds SR* from the **empirical** cross-sectional variance of the trial Sharpes, and that variance already shrinks with correlation. Discounting N as well counts the correlation twice.

**Test.** Under H0, the trial Sharpe estimates are z/√T with z ~ MVN(0, R). I simulated that null on this pool's own correlation matrix (200k draws) and compared the estimator σ_cs·e(N) against the true E[max]:

| Pool | n_eff | N at which the estimator is exact | Estimator / true max at N = 18 · 24 · n_eff · repo-scaled n_eff |
|---|---|---|---|
| Momentum-only | 4.89 | 14.7 | 1.05 · 1.12 · **0.67** · **0.79** |
| Combined with BAB | 1.98 | 14.9 | 1.05 · 1.12 · **0.29** · **0.48** |

- **The raw and declared counts overstate the null maximum,** i.e. they fail closed (conservative).
- **Every n_eff discount understates it, by 21–71%,** i.e. it fails open.

The tests pin both directions on synthetic structures:
- **iid:** the raw count is exact.
- **Common factor:** the raw count is exact, and n_eff fails open.
- **Pure clusters:** the raw count is conservative.

The primary therefore uses the declared 24, and n_eff results are reported only under `REJECTED_fail_open`.

**The same fault is dormant elsewhere.** `use_effective_n` is off by default, but the 09-23 audit recommends turning it on. Doing that without also moving the variance to the cluster level would loosen Crucible's DSR in the fail-open direction.

## 4a. Executor basis — why the pool is used unscaled

There is no executor-basis trial pool: the executor cannot run the XSMOM or per-class books. The research combined pool stands in, **unscaled**:
- Under H0, a Sharpe estimate's noise is set by sample length and trial correlation, not by execution mechanics.
- The pool's observed dispersion (0.152 momentum, 0.116 combined) sits at its simulated pure-noise level (0.178, 0.121).

Shrinking the pool by the executor's Sharpe ratio (0.757) would give 0.958. That understates the null, so it is rejected.

## 5. Math-skill verification (RS01, S08, RS06, XMATH-20, Phase E)

| ID | Result |
|---|---|
| RS01 `deflated_sharpe_ratio` | PASS: canonical form, per-period inputs, ddof=1, fails closed (None) when undefined, clamps |
| S08 PSR | PASS for this caller (`periods_per_year=1`) |
| RS06 `effective_n_trials` | Implementation PASS; **consumption as RS01's N with empirical σ is fail-open** (§4) |
| XMATH-20 | PASS: per-period Sharpe throughout |
| Phase E | E1 trial count measured on this pool; E2 degenerate inputs crash, not pass; E3 executor pool is a flagged stand-in |
| Tests | `tests/crypto/test_deflated_sharpe.py`, `test_psr_mintrl.py`, `tests/signals/test_deflation_wiring.py`, `test_evaluator_holes_c2.py`, `test_tier_b_anticonservative_f14.py`: 41 passed. New `tests/research/test_tailwind_dsr_consistent.py`: 4 passed |

Findings:
- **M-1 (CRITICAL):** 0.896 is a wrong verdict at a capital gate.
- **M-2 (HIGH):** the RS01/S-7 n_eff guidance is fail-open.
- **M-3 (MEDIUM):** the executor pool mismatch does not flip the verdict.
- **M-4 (LOW):** the executor reproduction residual.

## 6. What this does not change, and what needs a decision

- **Unchanged.** The other open Tier-2 items are untouched by the recomputation: borrow and financing (P3-03), the end-to-end executor test (P3-02), and the drift/kill wiring (P10-03/04). §7 has P(pass) after the lag fix.
- **No gate edited.** `tailwind_v1.gates.yaml` still records `honest_dsr_n24: 0.896` / `decision: BLOCK_multiplicity`, and `audit_tailwind_book.py` still computes the hybrid. Changing the enforcing computation is a gate change. **[Operator decision.]**

## 7. Executor lag fixed — same day

**The defect (R-3).** The linear-core arrays are already causal at `t−1`: row `t` of `conviction_ary` / `vol_ary` reads data `<= t−1`. The env then fills a step-`k` decision at close `k+1`, so feeding row `k` at step `k` put two bars between the data cutoff and the fill. The month-end rebalance therefore filled at the first close of the next month, one day staler than the research convention (month-end `d` weights from data `<= d−1`, filled at close `d`).

**The fix.** New config key `execution.decision_lead_bars: 1`, now set in both TAILWIND configs.
- The linear-core drive reads row `k+1` at step `k` (`allocator_factory._apply_decision_lead`), applied once inside `_linear_core_drive` so the sim oracle, the forward-recompute path and every caller share it.
- The data cutoff is close `k` and the fill is close `k+1`: one bar, under the env's own latency, with no same-close idealization.
- **Environment, `PaperState` and replay accounting are untouched.**

The fix fails closed in four places:
- **Causality must be proven.** Each array builder declares `conviction_cutoff_lag` / `vol_cutoff_lag` (momentum and BAB: 1, each proven by a current-bar crash tripwire at load). Rates-carry declares 0, because its tripwire never crashes the current bar, so the lead refuses it.
- **The final bar is never acted on.** The window's last bar is always flagged a "month-end" (P2-01), so the last decision holds instead.
- **α is aligned.** The two-sleeve α monthly hold keys on fill-bar stamps under the lead, so month-ends trade once.
- **The RL gate baseline refuses the lead.** `evaluate_linear_core*` rejects it: a lead baseline would read one bar more than the RL policy it is scored against.

**Result** (`results/tailwind_v1/dsr_consistent_2026-09-29.json`, executor block):

| Executor | Sharpe | vs research 0.732 | Consistent DSR (N 18–31) | P(pass), disjoint windows | Needless-termination share |
|---|---|---|---|---|---|
| Pre-fix (`decision_lead_bars: 0`) | 0.554 | 76% | 0.930 (0.922–0.938) FAIL | 0.615 = 16/26 (reproduces the record) | 0.125 |
| **Lag fixed** (`decision_lead_bars: 1`) | **0.635** | **87%** | **0.967 (0.963–0.972) PASS** | **0.667 = 14/21 (≥ 0.65)** | 0.214 (still fails its cap) |

**Why the gain is the timing effect, not an artifact.**
- **The research book pays the same cost for one extra day.** Delaying it one day drops its Sharpe from 0.732 to 0.660 (−0.072); the executor gains +0.081 from removing that day.
- **Costs did not fall.** Turnover is unchanged (11.33 vs 11.34 per year) and gross exposure is identical. Fees and slippage are *higher* ($14.8k vs $13.2k), because the book is larger.
- **Nothing in the changed code can see the future.** A two-bar-lead mutant (genuine look-ahead) is caught by the causality tripwire on both sleeves.

**Tests.** `tests/paper/test_decision_lead.py` (13 tests) covers:
- the timing at a month-end;
- a default that is byte-identical to the legacy drive;
- fail-closed refusals;
- the final-bar hold;
- real-builder LEAK-2 tripwires under a future bump and a fill-bar crash;
- a same-bar "teeth" test;
- α alignment;
- the gate-baseline refusal;
- the volatility-row pin.

Batch vs forward-recompute parity stays at exactly 0 under the lead. The full affected suites pass (695), as does the audit-addendum floor (635). Audit verdict: **PASS WITH NOTES**.

**What still blocks capital.**
1. **P(pass) is thin.** 0.667 rests on 21 disjoint challenges; its Wilson 95% interval is roughly [0.45, 0.83], so it is a point estimate, not a pass.
2. **The needless-termination gate fails**, before and after. The internal risk kills would stop 3 of the 14 paths that pass the firm's rules (0.214), against 2 of 16 pre-fix (0.125); the cap is 0.05. The change is one path, so it is noise, but both sit far above the cap: the kills are too tight for this book's normal drawdowns.
3. **A Tier-2 deep lifecycle audit is required.** The executor that produces these deploy-gating numbers changed. The fixed `deep_strategy_audit` workflow is present; it needs non-RL pillars for `tailwind-v1`.
4. **The gate record is unchanged.** It still says `BLOCK_multiplicity` / 0.896. **[Operator decision.]**

**Knock-on.**
- Other configs keep the legacy drive: the default is 0 and only the TAILWIND configs opt in. That covers the RL allocator and the cross-asset rung-1 paper (whose linear core the 09-23 audit measured at 0.403 → 0.528 for the same fix).
- Re-running the executor-path scripts now reflects the fixed timing:
  - `tailwind_executor_recompute.py` and `tailwind_dsr_consistent.py`;
  - the paper-validation runners;
  - `run_combiner_selection.py` and `execution_overlay_runner.py`.

  Each artifact stamps its `decision_lead_bars` and financing model (`allocator_factory.execution_stamp`, Tier-2 N4). Set `decision_lead_bars: 0` to reproduce a pre-fix artifact. The forward-path render, sizing lab, drift baseline and Stage-4 scripts work on the research basis; they never run the drive, so the lead does not change them. *(Corrected 2026-09-29: an earlier version of this line listed them as executor scripts.)*
- A pre-existing one-bar offset in the paper executor's per-sleeve P&L attribution (`two_sleeve._sleeve_attribution`) was found during the audit and queued separately. It is reporting-only, not a gate.

## 8. Financing leg added — same day (Tier-2 roadmap N2)

**The defect (Tier-2 S1).** Every certifying Sharpe, DSR, P(pass) and PSR used total returns, with cash at 0 and leverage financed free, on a book that is net long throughout.

**The fix.** A new opt-in config block, `financing:`, handled by `sharpen/data/financing.py`. Both TAILWIND configs declare `model: tbill`, tenor `3m`, act/360, and `short_borrow_bps: 25`.
- **Carry.** `carry_ary = −rf` on every position. The env and `PaperState` already accrued carry (a long pays it, a short earns it), which charges the book `−net·rf`.
- **Borrow.** A new `borrow_ary` fee on short notional, wired through the env, `PaperState`, the replay and the execution-overlay path.
- **Rate.** The interval (t−1, t] accrues at the 3m yield known at close t−1 (read as-of from the DATA-CLEAN'd research curve), times calendar days / 360.
- **Notional basis.** Carry is charged on the notional carried *into* the bar, not the end-of-bar notional. Carry was zero everywhere before, so no recorded number moves: every unfinanced block of the JSON is identical to the pre-change run.
- **Fails closed** on an unknown key or model, on a missing prior yield, and in the single-sleeve loader, which refuses a financing block instead of dropping it.
- **Research basis.** Every graded momentum book and the BAB sleeve are restated on excess returns from their own held weights before the risk-parity combine. The trial pool is financed the same way.

**Result** (`results/tailwind_v1/dsr_consistent_2026-09-29.json`, `*_financed` blocks; N = 24, bracket N 18–31):

| Book | Short borrow | Sharpe | DSR, matched excess pool | DSR, unfinanced pool | P(pass), disjoint |
|---|---|---|---|---|---|
| Research basis | 0 bp | 0.620 | 0.972 (0.969–0.976) PASS | 0.961 | — |
| Research basis | **25 bp (declared)** | 0.587 | **0.961 (0.956–0.965) PASS** | 0.947 | — |
| Research basis | 50 bp | 0.555 | 0.945 (0.939–0.951) FAIL | 0.929 | — |
| Executor, lead 1 | 0 bp | 0.519 | 0.929 (0.922–0.936) FAIL | 0.906 | 0.632 = 12/19 |
| **Executor, lead 1** | **25 bp (declared)** | **0.498** | **0.913 (0.905–0.922) FAIL** | 0.888 | **0.609 = 14/23 FAIL** |
| Executor, lead 0 | 25 bp | 0.418 | 0.840 FAIL | 0.803 | 0.583 = 14/24 |

- **It reproduces the Tier-2's financed row.** P(pass) 12/19 and needless share 0.25 match exactly. DSR is 0.929 against the audit's 0.934 (matched pool) and 0.906 against 0.912 (doc pool). The research basis is 0.972 / 0.961 against 0.973 / 0.962. The residuals come from the day-count convention.
- **Needless-termination still fails** at the declared model: 0.071 against the 0.05 cap.
- **Since 2023 the executor earned less than T-bills:** excess Sharpe −0.37 at the declared model.
- **Out-of-universe momentum** on the 14 unseen ETFs: Sharpe 0.441 → 0.291 and PSR 0.974 → 0.902 on excess returns.
- **The drag checks out in situ.** Replaying the same weights with and without the leg gives 1.12%/yr. The book's actual notional net exposure averages 0.545 (its weight label is 0.488); at a mean rf of 1.70% that is 0.93%/yr. The book is also more net long when rates are high, which adds 0.23%/yr.

**Verdict.**
- The executor fails both gates on the certifying estimand: DSR 0.913 < 0.95 and P(pass) 0.609 < 0.65.
- The research basis clears at ≤ 25 bp borrow and fails at 50 bp, so the borrow level is now load-bearing for the research-basis claim.
- For a prop account these figures are a floor on the financing drag: the balance earns no interest, and CFD swap markups are roadmap X4.

**Tests.** `tests/paper/test_financing.py` has 27 tests:
- The N2 acceptance test: a net-long book with a short leg, whose assets earn exactly the cash rate, has zero excess return bar by bar. It runs through the env, and through the real union builder and the replay.
- Borrow is charged on shorts only, and the replay charges it exactly as the env does.
- Env↔`PaperState` lockstep parity holds with non-zero, asset-varying carry and borrow. This was never exercised before, because every carry array was zero.
- The rate construction, including a LEAK-2 negative test: a yield printed at bar t cannot move that bar's accrual.
- The fail-closed spec, and a pin on both TAILWIND configs' declared model.
- Loader wiring, with fetch and curve stubbed and no network.
- The research-side `held_weights` / `excess_returns` helpers.

All 10 mutations were killed:
- re-zeroed builder carry;
- end-of-bar notional, in the env and in `PaperState`;
- same-close yield;
- borrow on every position;
- a flipped short-carry sign;
- borrow dropped in the replay, the factory or `PaperState`;
- an unfinanced union.

The affected suites pass (769), as do the audit floor (635) and the causality tripwires (11). Tier-1 audit: **PASS WITH NOTES**.

**Still open.**
- **[Operator]** Declare the financing model in `tailwind_v1.gates.yaml` (the N2 operator item).
- **[Operator]** Confirm the borrow level. 25 bp is an assumption; general-collateral ETF borrow plus the short-credit spread at a retail broker is closer to 50 bp, which fails the research basis.
- The research curve cache ends 2026-06-12, so the last 49 days carry the last yield forward. This is surfaced as a curve-manifest WARN, which the paper-validation integrity block now shows for TAILWIND.
- Return-stream sleeves (VRP) are not financed by this leg. TAILWIND has none.

## 9. Lead and financing pinned (Tier-2 roadmap N4) — same day

- **Config pins.** Tests assert that both TAILWIND configs declare `decision_lead_bars: 1` and the financing model above. Deleting either fails a test.
- **Validation.** `validate_config` now requires a linear-core allocator config (`execution.shadow: linear_core_*`) to declare the lead as the integer 0 or 1.
  - A lead over a sleeve with a 0 conviction cutoff lag (`rates_carry`) FAILs.
  - The two dormant rung-1 paper configs now declare 0 explicitly.
  - A malformed financing block FAILs, as does one on a path that does not carry it; return-stream sleeves WARN as unfinanced.
  - The lags come from one table, `cross_asset_loader.CONVICTION_CUTOFF_LAG`, which both the builders and the validator read.
- **Parsing.** The run-time parser rejects non-integers (`True`, `1.0`, `"1"`) instead of coercing them into a lead.
- **Stamps.** Every executor-path artifact records its lead and financing model (`execution_stamp`).
- **Not done:** N4's other half, stamping the enforcing research-basis scripts, is moot. They never run the drive.

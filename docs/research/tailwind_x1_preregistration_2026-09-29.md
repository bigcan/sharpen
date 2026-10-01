# TAILWIND-v1 X1 re-certification — pre-registration (2026-09-29)

**Status: REGISTERED 2026-09-30.** The operator delegated the decisions ("make the best decisions for me"). All recommended defaults in §4 are adopted, and M1/M2 are handled as in the Sign-off section. The certifying pass runs after this commit.

**What X1 is.** Tier-2 roadmap row X1 (`docs/research/tailwind-v1_deep_lifecycle_audit_2026-09-29.md`) calls for one pre-registered pass on the realizable, financed executor. The audit expects DSR ≈ 0.91–0.93 and a Wilson-lower P(pass) well under 0.65, so BLOCK is expected to stand. The pass reports whatever it finds and is never tuned toward a pass.

**Files.**
- Method, ledger and pins: `configs/tailwind_v1_x1.prereg.yaml`. The script reads every setting from this file.
- Thresholds: read from the gates files the yaml names. None is authored here.
- Script: `scripts/research/tailwind_x1_recertification.py` (`--mode dry-run | certify`).
- Tests: `tests/research/test_tailwind_x1_recertification.py`.
- Artifact: `results/tailwind_v1/x1_recertification_<date>.json`.

## 1. The frozen method

| Item | Registered |
|---|---|
| Book | The real `TwoSleeveExecutor` on `configs/tailwind_v1_challenge.yaml`, lead 1. X2 showed this batch series equals the forward runner at all 5,174 steps. |
| Estimand | Excess of the 3m T-bill (act/360, prior-close yield) minus borrow on short notional. 25 bp is primary; 0 and 50 bp are reported. |
| Window | 2007-04-30 .. 2026-05-29: the research book's first non-zero day through the last research-panel bar. Verified in-run; a mismatch aborts the pass. |
| Data | Four files pinned by full sha256, plus the executor manifest's `content_sha256_16` (`cbdca6fc55773b87`). Snapshot at `results/_snapshots/tailwind_x1_2026-09-29T2246/`. A mismatch aborts. Every network fetch path is stubbed to raise, and the pins are re-hashed after the pass. |
| DSR | BLdP, deflated once. Observed = the executor's in-sample Sharpe over the window. The pool is the 18 graded momentum books, each risk-parity-combined with BAB, every leg financed on its own held weights at the same borrow, with Sharpes measured over the window. |
| N | From the trial ledger (§2); never discounted by n_eff. The bracket is **{64, 77, 96}** and every value must clear. N = 24 and the n* from `null_max_calibration` are diagnostics. |
| Momentum engine | The same executor with the BAB sleeve removed, deflated against the 18 financed momentum-only books. Reported beside the combined DSR; not gating. |
| P(pass) | See §3. |
| Render | Executor basis, over the window. Reports path stats, the needless-termination share, the daily-breach rate and gross exposure (max vs the 4.7 render ceiling and the env's 3.0). The vol frontier is not run: re-sizing on the result would be tuning. Not gating. |
| Stage-4 | Executor basis. OOS runs from the 2026-06-01 cutoff to the cache end (2026-07-31); the baseline is the pre-cutoff subperiod median PF. Buckets come from `tailwind_v1_stage4.gates.yaml`. Two rebalances, so `power_sufficient` is expected to be false. `deploy_gating: false`. |
| Verdict | **PASS** needs DSR ≥ `min_dsr` (0.95) at every bracket N **and** Wilson-lower P(pass) ≥ `min_p_pass_step1` (0.65) at every MAE. Anything else is **BLOCK**. A PASS carrying an integrity warning is **REVIEW**: the curve manifest is WARN, or executor-path code changed between the pre-registration commit and the pass. A pin, lineage or guard failure is **FAIL**. Exit codes follow N3: 0 / 3 / 1 / 1. |
| Lineage | The artifact records the file sha256s, the manifest hash, config and gates shas, the script sha, the git HEAD, the dirty flag and paths, the dependency paths changed since the pre-registration commit, the window, lead, financing, Python and library versions. |
| One pass | `certify` refuses unless the yaml is `registered`, signed off, committed and clean. It also refuses if any `x1_recertification_*.json` already exists. A refused attempt is logged as `x1_attempt_FAIL_*.json`. |

## 2. Trial ledger (N12 draft)

Every count is a floor. Sources and dates are in the yaml.

| ID | Family | Count | Dates |
|---|---|---|---|
| F1 | 18-ETF momentum grid | 18 | 06-03 |
| F2 | Strengthen pass: 3 signal variants, the combined core, and a 9-cell lookback × vol-window grid | 13 | 06-03 |
| F3 | Wide 32-ETF universe, graded books | 20 | 07-01 |
| F4 | Sleeve candidates: rates carry, value, XLG, BAB, commodity carry, and commodity, country and crypto TSMOM | 8 | 06-12 .. 07-31 |
| F5 | Sleeve combiners: static RP, sleeve momentum, drawdown control, dynamic tilt | 4 | 06-18 .. 06-26 |
| F6 | Multi-speed TSMOM blend | 1 | 08-01 |
| F7 | Execution and sizing forks: lead 0/1, labels vs holdings, vol targeting, gross cap | 4 | 08-28 .. 09-29 |
| F8 | DSR-method variants | 9 | 07-01 .. 09-29 |
| | **Strategy-only (F1–F6)** | **64** | |
| | **Full (F1–F8)** | **77** | |

- **Not counted:** the closed-form sleeve extensions in `frontier.json`, and sensitivity reads that selected nothing (delayed fills, MAE, borrow).
- **Survivorship:** the 18- and 32-ETF universes are hand-picked survivors (T1-01), and this N does not deflate that.
- **The σ_cs gap:** σ_cs comes only from F1 × BAB, the one family that can be rebuilt on the same financed estimand and window. The other families enter through N only.

## 3. P(pass) estimator (N10)

Disjoint windows from the window start:
- A window runs `challenge_simulator.simulate_path` for at most `render_horizon_days` (756) sessions.
- The next window starts the session after the current one resolves. The walk advances on the run being scored.
- TIMEOUTs are right-censored: a horizon-cap TIMEOUT jumps the horizon, a data-end TIMEOUT stops the walk. Both leave the denominator, and both counts are reported.
- The gate run uses the declared terminal kills (8% static drawdown, 4% daily; `challenge_v2` `paper_soak.risk`), the 10% target and a 5-day minimum.
- The same walk gives the daily-breach rate.
- Diagnostics:
  - firm-only (10% / 5%) on its own walk;
  - the needless share, from the firm walk with the kills paired on each slice.
- It runs at `intraday_mae_mult` 1.0 and 1.4.
- The interval is two-sided 95% Wilson, and the lower bound binds.

## 4. Operator decisions (recommended defaults)

- **(a) N1, the DSR method.** Adopt the consistent-DSR rules: in-sample deflation once; a matched financed pool on the same window; the declared or ledger N, never an n_eff discount. The n_eff discount was measured fail-open by 21–71% (`tailwind_dsr_consistent` §4). **Recommend: adopt.**
- **(b) N rule.** The ledger's full count (77) is primary. The bracket {64, 77, 96} must clear at every N; 96 is a 25% allowance for unenumerated search (T3-05: "treat N as a floor"). N = 24 and n* are diagnostics. **Recommend: adopt.** Alternative: primary 64, which leaves out the method and execution forks.
- **(c) Borrow.** 25 bp as declared; 0 and 50 bp as sensitivity. **Recommend: adopt.** Note that 50 bp already fails the research basis (0.945).
- **(d) Kill policy (R3).** The declared terminal kills are the gate; firm-only is a diagnostic. **Recommend: adopt.**
- **(e) Estimand and MAE.** Disjoint walk, TIMEOUTs censored, Wilson lower bound at MAE 1.0 **and** 1.4, both of which must clear. **Recommend: adopt.** Two notes from the Math skill, both fail-open toward P(pass), so neither can manufacture a BLOCK:
  - **M1.** The MAE inflates only the daily-loss check (simulator semantics). The static drawdown floor stays close-based.
  - **M2.** Censoring horizon-cap TIMEOUTs follows N10 but raises P(pass) if they are common. The fail-closed alternative is to count them as non-pass.

  If you want either M1 or M2 closed, say so before sign-off and it goes into the registration.

## 5. Disclosed before registration

These were measured during orientation on the current tree. None is the registered estimand: N, pool window and estimator all differ.

- **Declared financing, full span, N = 24:** SR 0.506, DSR 0.919 (0.911–0.927), P(pass) 0.600 = 15/25 on the old firm-only estimator.
- **Trimmed to the window:** SR 0.520, DSR 0.920.
- **T-bill only:** DSR 0.933, P(pass) 0.619.
- **Momentum-only executor on the window:** SR 0.502, no DSR computed.
- **A blinded end-to-end `run_pass`:** output discarded unread; it confirmed only that the pass completes.

## 6. Machinery validation (no certifying number)

- **Tests:** 23 X1 tests pass, all data-free. They cover:
  - Wilson against the audit's 14/21 interval, and against statsmodels on 5 cases;
  - disjointness, censoring of both kinds, the MAE passthrough, the paired needless share;
  - the fail-closed verdict and the N3 exit map;
  - pin refusal and the offline guard;
  - ledger arithmetic, and every threshold source resolving;
  - certify refusals: draft, uncommitted, second pass.
- **Mutations:** 6 of 6 killed.
- **Math skill:** PASS WITH NOTES (M1–M3). **Audit:** PASS WITH NOTES.

## 7. After the pass

The pass writes the artifact and `docs/research/tailwind_x1_recertification_<date>.md`, with every registered number and any deviations. A PASS promotes nothing. A Tier-2 audit on the final tree must come first, and the change to the gate record (`configs/tailwind_v1.gates.yaml`, still `BLOCK_multiplicity`) is the operator's decision.

## Sign-off

Registered 2026-09-30. The operator delegated the calls in chat, and the yaml records them under `operator_signoff`:
- **(a)–(e):** adopted as recommended.
- **M1:** kept as a declared limitation, because X1 specifies the challenge simulator's MAE semantics.
- **M2:** the gate is unchanged (N10 censoring). Each walk also reports a P(pass) and Wilson lower bound with horizon-cap TIMEOUTs counted as fails. That diagnostic can only be lower than the gate read, never higher.

The pass cites the commit that registered the yaml.

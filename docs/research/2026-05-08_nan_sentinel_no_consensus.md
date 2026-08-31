# Fix 2: NaN sentinel for "no consensus" in ensemble aggregators

> **Created:** 2026-05-08 | **Session:** 538-cont
> **Question:** Replace `np.zeros_like` "no-majority" return in `ens_agreement` with a NaN sentinel propagated through aggregator → env → live engine → drift/agreement-decay metrics. Ship or stay on Fix 1 indefinitely?
> **Verdict:** **CONDITIONAL GO** — ship the inference-side path (aggregator + live engine + monitors + offline-eval) plus a re-evaluation pass; defer env-side change + checkpoint retrain unless re-evaluation reveals material PF regression on fold_07.

## Executive summary

The `np.zeros_like` fallback in `ens_agreement` conflates "no consensus, hold" with "deliberate flat = liquidate". The bug exists at four sites (one live aggregator, two offline aggregators, one env consumer). It already hit sg1-btc (S538-cont — 6.62% drawdown in 26h). It is **already present** on sg1-xauusd: the recent recal (S535-R1) baked a 0.7966 deadband_frac into the baseline, the same +0.27 jump signature that fired CRIT on sg1-btc; the strategy survived only because XAU's higher solo deadband-frac masked the jump under the CRIT threshold and trades less often.

A NaN sentinel propagates "no signal" cleanly through the live engine if every consumer is taught to filter NaN. The inference-side path is small (~6 files, no retrain). The training-side path is bigger (env semantic change → re-evaluation of FU-3 checkpoints; potential retrain). Recommendation: ship the inference-side fix and re-evaluate fold_07 first. Decide on retrain after seeing the re-eval delta.

## Prior art (internal)

| When | Workstream | Event | File ref |
|---|---|---|---|
| 2026-04-23 | sg1-xauusd ensemble paper-deploy | `ens_agreement` chosen, deadband 0.35, 3 seeds — never investigated dispersion-collapse | memory `project_sg1_xauusd_ensemble_paper_deploy_s489.md` |
| 2026-04-23 | GMGP1-BTC L1 S495 | Val-argmax-PF showed `ens_mean` (+8.13%) > `ens_agreement` (+4.10%) on trending BTC — first crack in canonical-rule premise | memory `decision_ensemble_val_selection_s495.md` |
| 2026-05-06 | sg1-xauusd | drift CRIT, R1 recal baked live deadband_frac 0.7966 (Feb baseline q1-q4 0.51-0.57). Recal treated cause as baseline staleness | memory `project_sg1_xauusd_drift_recal_s535.md` |
| 2026-05-07 | sg1-btc | FU-3 8-fold WF PROMOTE under `ens_agreement`; ensemble_v2 baked + live-swapped | memory `project_sg1_btc_fu3_wf_promote.md` |
| 2026-05-08 | sg1-btc | drift_crit at warmup boundary; root-caused to dispersion-collapse; Fix 1 (config override → ens_mean) shipped without rebake | memory `project_sg1_btc_drift_crit_rule_revert_s538.md` |

## 1 — Codebase grounding (PROVEN — code read 2026-05-08)

### The bug — 4 sites

| Component | File | Line | Returns on no-majority |
|---|---|---|---|
| Live aggregator | `sharpen/agents/sac/ensemble_agent.py` | 51 | `np.zeros_like(actions[0])` |
| Offline aggregator (canonical, multi-workstream) | `scripts/sg1_xauusd_ensemble_eval.py` | 157 | `np.zeros_like(stacked[0])` |
| Offline aggregator (BTC fork) | `scripts/sg1_btc_velotrade_ensemble_eval.py` | 148 | `np.zeros_like(stacked[0])` |
| Env step semantic interpreter | `sharpen/envs/continuous_swing_env.py` | 243-313 | `delta = target_position - current_position; if |delta| > deadband: trade` |

### Live engine consumption path (`sharpen/crypto/live/live_engine.py`)

This file serves both sg1-btc (Bybit broker) and sg1-xauusd (cTrader broker). NaN behavior of each consumer if the aggregator emitted `np.nan` today:

| Line | Step | Behavior on `target_position = NaN` |
|---|---|---|
| 942 | `_predict(obs)` | passes through (no validation) |
| 946 | `policy_target_position = float(target_position)` | `float(np.nan)` = `nan`, OK |
| 952-957 | `ActionDriftTracker.observe(target_position, …)` | NaN appended to deque; histogram drops it; `(abs(live) < deadband).mean()` underreports because NaN ≮ deadband |
| 981-985 | `AgreementDecayTracker.observe(target_position)` | `abs(NaN) < deadband` is False — NaN counted as **not flat**, the opposite of what we want |
| 1010 | `np.clip(target_position * multiplier, -1.0, 1.0)` | NaN propagates through multiply and clip → still NaN |
| 1017-1025 | Deadband filter `abs(delta) < deadband_threshold` | False (NaN comparison) → does NOT skip — falls through to risk manager |
| 1027-1039 | `risk_manager.check(action=np.array([NaN]), …)` | unspecified behavior; some checks use `>`/`<` which return False on NaN, some use `np.clip` which preserves NaN |
| 1080-1101 | `broker.execute_position_change(target_position=NaN)` | broker would receive NaN; behavior depends on broker (Bybit/cTrader) input validation — almost certainly an order-format crash or zero-quantity rejection |

**Conclusion:** changing only the aggregator to emit NaN without teaching every consumer about NaN is a **regression** — engine would propagate NaN into a broker call. The minimum atomic unit is aggregator + drift trackers + deadband filter (treat NaN-delta as "skip bar entirely"). This is six file edits.

### Other assets

- `sharpen/reporting/eval_distribution.py:75` — `(abs_a < deadband).mean()` — same NaN-undercount as live ActionDriftTracker. Used by Stage 2/2.5 baseline writers, so eval baselines change if NaN-bars become a thing.
- `sharpen/monitoring/agreement_decay.py:155-159` — docstring explicitly says "For ens_agreement the aggregator returns 0 on consensus failure, so |action| < deadband captures both naturally-flat and consensus-failed bars consistently with the eval baseline." This contract is the bug pinned in code comments — Fix 2 must rewrite this contract.

## 2 — NaN sentinel semantic contract (rule-agnostic)

| Producer | When NaN is emitted |
|---|---|
| `ens_agreement` / `ens_majority` | no 2/3 directional majority |
| `ens_mean` / `ens_median` | only if **all** seed inputs are NaN (won't happen in practice — solo predict never produces NaN) |
| `ens_pf_weighted` | same — only on all-NaN inputs |
| Solo agent | never |

For non-consensus rules NaN is effectively unreachable, which keeps the contract uniform without behavior change for `ens_mean`/`ens_pf_weighted` users.

| Consumer | Rule on NaN |
|---|---|
| `EnsembleAgent.predict` | emit NaN scalar (already would; no per-batch averaging conflict because action_dim=1 today) |
| `ContinuousSwingEnv.step` | `if np.isnan(target).any(): hold current_position; no fee; do NOT increment trade_count; reward = DSR.compute(0)` (zero step return, no PnL impact) |
| `live_engine._trading_step_inner` | new branch right after line 946: `if np.isnan(target_position): log_step(skip_reason="no_consensus", policy_target_position=NaN); return`. Trackers still observe NaN (so they can count it). |
| `ActionDriftTracker.observe` | accept NaN; exclude from `deadband_frac` numerator AND denominator; surface `no_consensus_frac` as a new field on `DriftReport` |
| `AgreementDecayTracker.observe` | accept NaN; new metric `no_consensus_frac` replaces flat-bar conflation; baseline schema gains `no_consensus_frac` field, `flat_bar_frac` becomes "naturally-flat-only" |
| `eval_distribution.summarize_scalar_actions` | filter NaN before `np.histogram` and `(abs_a < deadband).mean()`; emit `no_consensus_frac` field |
| Bundle baker (`bake_*_ensemble_v*.py`) | bundle `ensemble_eval_distribution` schema gains `no_consensus_frac`; protocol version bumps to `v2.2_…__nan_sentinel` |

The rule-agnostic phrasing: **"NaN means no actionable signal — every consumer holds the prior decision and excludes the bar from distribution metrics."** This intentionally matches `np.nan`'s semantics in pandas/numpy aggregations (drop-by-default).

## 3 — Cross-strategy impact: sg1-xauusd

**sg1-xauusd is hit by the same bug today.** Evidence:

- Feb baseline `ensemble_eval_distribution.deadband_frac` per quartile q1-q4 = 0.5696 / 0.5079 / 0.5184 / 0.5643 (memory `project_sg1_xauusd_drift_recal_s535.md`). Solo SAC on a 0.35-deadband XAU regime would not naturally sit at 50%+ deadband; that is the `ens_agreement` consensus filter dropping bars.
- Live deadband_frac = 0.7966 across 13 days, std 0.0112 (steady state, not drift in progress) → +0.27 over Feb baseline.
- sg1-btc had +0.29 over its (much lower) Feb baseline. **Same magnitude, same mechanism.** sg1-xauusd survived because (a) starting point was already high, so live didn't cross the +0.30 CRIT threshold, and (b) XAU bar cadence is 3-min on a market open ~12h/day — fewer flatten-cycles per calendar day than BTC's 24h.
- The R1 recal on 2026-05-06 baked 0.7966 into the baseline as "the new normal" — the dispersion collapse is now invisible to the drift detector.

**Measurement plan to confirm (≤2 GPU-hours):**
1. Pull the 1043 logged `policy_target_position` values from the pre-recal sg1-xauusd run via WandB API (run id `live-sg1-xauusd-ctrader-paper-20260422`, _step 2100-4506).
2. Replay the 3 seeds offline on the same XAU bars, classify each bar as "long-majority / short-majority / no-consensus" using deadband 0.35.
3. Compute `no_consensus_frac` directly. Hypothesis: ≥50% of "deadband" bars are actually no-consensus.

If confirmed, sg1-xauusd's 43 trades over 1043 bars likely include the same build-and-flatten waste sg1-btc was paying. The `+$330 PV` is the surviving signal **after** that drag.

## 4 — Training implications

The env consumes `target_position=0` *today*, both during HPO and in the WF eval scripts that ran the FU-3 8-fold PROMOTE numerics. So the FU-3 fold_07 PROMOTE was conditional on the disagreement-flatten cost. Two questions:

- **Inference-only or also training?** Inference-only is sufficient if we change only the *live* and *offline-eval* code paths and leave the env alone. The agents will still produce continuous actions (they don't see "consensus" — that's an aggregation artifact). The aggregator is downstream of the policy.
- **Re-evaluation needed?** Yes. We must re-run the fold_07 evaluation under the new semantics (NaN-bar = hold, no fee, no trade) using the existing seeds {456, 3141, 123} and report the new test PF. If the new PF degrades materially vs the original 2.46 ens-PF, the picture changes — that would mean the FU-3 PROMOTE was load-bearing on the disagreement-flatten cost. Either way it's a 30-minute backtest, not a retrain.
- **Retrain?** Probably not. SAC seeds learned to size positions assuming the env's deadband + fee logic — they don't know about the aggregator. A retrain is only justified if (a) re-eval shows large degradation AND (b) we want to optimize for the new env semantics. Defer that decision until re-eval lands.

## 5 — Migration plan + GPU cost

| Stage | Scope | Rebake? | GPU hours |
|---|---|---|---|
| 5.1 | Aggregator NaN emit (3 files: `ensemble_agent.py`, both `*_ensemble_eval.py`) | no | 0 |
| 5.2 | Live engine NaN handler (1 file: `crypto/live/live_engine.py` — new `if isnan` branch + drift/agreement-decay observe filters) | no | 0 |
| 5.3 | Drift + agreement-decay tracker filters + `no_consensus_frac` field | no | 0 |
| 5.4 | `eval_distribution.summarize_scalar_actions` NaN filter + new field | no | 0 |
| 5.5 | Re-evaluate fold_07 (3 seeds × 1 window) on existing checkpoints with new semantics | no rebake; reports only | ~0.5 GPU-h |
| 5.6 | Optionally re-bake bundle if `no_consensus_frac` is wired into manifest schema | yes (sg1-btc + sg1-xauusd) | ~0.1 GPU-h each |
| 5.7 | Live swap on sg1-btc back from ens_mean to ens_agreement (NaN-aware) | optional | 0 |
| 5.8 | Live swap on sg1-xauusd to NaN-aware ens_agreement | yes (image rebuild) | 0 |
| 5.9 | (Optional) env-side NaN-hold + Stage 2.5 re-bake on fold_07 if 5.5 shows regression | yes (retrain optional) | up to ~24 GPU-h if retrain |

Total committed cost: **~1 GPU-hour** for stages 5.1-5.8. Stage 5.9 is conditional and gated by 5.5 outcome.

## 6 — Test plan

Unit tests (new):
- `tests/agents/test_ensemble_agent.py::test_agreement_no_majority_emits_nan` — replaces `test_agreement_split_stays_flat`; expect `np.isnan(out).all()`.
- `tests/agents/test_ensemble_agent.py::test_agreement_all_flat_emits_nan` — replaces `test_agreement_all_flat`.
- `tests/envs/test_continuous_swing_env.py::test_step_nan_action_holds_position` — current_position remains, trade_count unchanged, reward equals DSR(0).
- `tests/monitoring/test_action_drift.py::test_observe_nan_excluded_from_deadband_frac` — feed N=600 mixed bars with 200 NaN; expect `deadband_frac` computed on n=400 and `no_consensus_frac=0.333`.
- `tests/monitoring/test_agreement_decay.py::test_observe_nan_no_consensus_metric` — same, with new `no_consensus_frac` field.

Replay fixture (S538-cont):
- 504 logged `policy_target_position` values from WandB run `live-sg1-btc-bybit-demo-paper-ensemble-v2`.
- Pull per-seed actions (logged `action_456`, `action_3141`, `action_123` during S535-cont-2 FU-3 WF eval).
- Re-run aggregator under new semantics. Verify: of 230 originally-zero bars (45.6% × 504), ≥220 should now emit NaN. Engine replay verifies they would all have been `skip_reason="no_consensus"`.

## 7 — Risks + go/no-go

**Risks of NaN-sentinel:**

| Risk | Mitigation | Residual |
|---|---|---|
| NaN reaches broker (broker crash or zero-fill) | Stage 5.2 mandatory in same PR as 5.1 — never ship aggregator change without engine guard | Low |
| NaN persists indefinitely → policy stuck if seeds chronically disagree | `AgreementDecayTracker` already exists for this; fires CRIT on sustained "no consensus". Schema change makes the metric explicit | Low |
| Type errors elsewhere in numpy ops on NaN-stamped arrays | Aggregator output is a 1-element `np.float32` tensor; downstream code doesn't propagate it as a vector. Audit the 6 listed files only | Low |
| Existing baselines (`ensemble_report.json`) don't have `no_consensus_frac` field — backwards-compat | Default to `0.0` when missing; live tracker logs WARN once if absent | Low |
| Re-eval (5.5) shows fold_07 PROMOTE no longer holds | Surfaced as decision artifact; fall back to ens_mean (Fix 1 already deployed) | Acceptable |
| Bias: changing eval semantics retroactively favors agreement-rule decisions | The new semantic is *more permissive* (hold-on-disagree) than the old (flatten-on-disagree). Other rules (`ens_mean`) are unchanged. Net: at most +X% PF for agreement-rule, no degradation elsewhere | Acceptable |

**Go/no-go.** Ship Fix 2 conditionally. The inference-side stages 5.1-5.5 are <1 GPU-hour, fix a real cross-strategy bug, and surface a new `no_consensus_frac` metric that closes the agreement-decay observability gap. The env-side change (5.9) is the bigger commitment and should be gated by the 5.5 re-eval result. Fix 1 (config override to `ens_mean`) remains the live-trading state for sg1-btc until 5.5 lands. sg1-xauusd should also be migrated to ens_mean as an interim if 5.5 is delayed beyond ~7 days, since it is currently running with a baked-in dispersion-collapse baseline that masks the same bug.

## References

1. Memory: `project_sg1_btc_drift_crit_rule_revert_s538.md` (incident)
2. Memory: `project_sg1_btc_fu3_wf_promote.md` (bundle that got swapped)
3. Memory: `project_sg1_xauusd_ensemble_paper_deploy_s489.md` (cross-strategy)
4. Memory: `project_sg1_xauusd_drift_recal_s535.md` (recal that masked the same bug)
5. Memory: `project_drift_baseline_audit_snapshot_20260506.md` (fleet snapshot)
6. Memory: `decision_ensemble_val_selection_s495.md` (Stage 2.5 rule selection)
7. Code: `sharpen/agents/sac/ensemble_agent.py:51`
8. Code: `scripts/sg1_xauusd_ensemble_eval.py:157`
9. Code: `scripts/sg1_btc_velotrade_ensemble_eval.py:148`
10. Code: `sharpen/envs/continuous_swing_env.py:243-313`
11. Code: `sharpen/crypto/live/live_engine.py:942-1101`
12. Code: `sharpen/monitoring/{action_drift,agreement_decay}.py`
13. Code: `sharpen/reporting/eval_distribution.py:75`
14. Code: `sharpen/live/agent_loader.py:245` (config-overrides-bundle precedence)

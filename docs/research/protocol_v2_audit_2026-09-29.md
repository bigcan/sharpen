# Protocol v2 deep audit — 2026-09-29

**Question asked:** audit `docs/protocol_v2.md`, suggest improvements, and report whether the protocol is so strict that strategies can never make it through.
**Rename note:** later the same day (commit `919ed393`), Protocol v2 was renamed **SharpOps** and the document moved to `docs/sharpops.md`. Read every `docs/protocol_v2.md` reference below as `docs/sharpops.md`; the findings are unchanged.
**Scope:** the protocol (v2.7, 865 lines, including the uncommitted §3.1 provenance block) and everything that implements it: `scripts/validate_config.py`, the stage launchers, the ensemble / walk-forward eval scripts, the gate YAMLs, and the live safety code. Branch `agentic-trading-lab` @ `7a2c2e75`.
**Method.** Three independent strands:
1. **Operating characteristic.** A seeded Monte Carlo of the gates, as documented and as implemented, giving P(pass) for a strategy of known true Sharpe. Script: `scripts/research/protocol_v2_gate_power.py` (~10 s).
2. **Outcome ledger.** Every candidate that entered the protocol since 2026-04-18: where it stopped, and whether the result held up later.
3. **Conformance.** Each normative rule traced to code and classed ENFORCED / WARN / DORMANT / DOC-ONLY / CONTRADICTED. Every headline claim was re-checked at the cited line.

This builds on `full_codebase_audit_2026-09-23.md` (R-1 bar-level PF gates, S-2 conjunctive stacks). It confirms those findings and extends them to the protocol as a whole, without repeating them.

---

## 1. Answer

**Yes, the protocol is too strict. As written and as run, it cannot pass a realistic edge.** But that is not why the strategies on record failed, and loosening the thresholds would not by itself produce a profitable strategy. The protocol is miscalibrated in both directions: it blocks realistic edges and passed every artifact put in front of it.

1. **No realistic edge can pass.** Every RL gate grades profit factor *per bar*. The floors in use (1.10–1.5) correspond to a true annualized Sharpe of **3–67**, depending on bar size and time in market. Realistic intraday edges for a small operator are 0.5–2.
   - At the gate-YAML floors, the documented chain passes with probability below 0.1% for any true Sharpe ≤ 5.
   - The walk-forward gate that actually runs (G1) passes a Sharpe-1 strategy **0.1%** of the time and a Sharpe-3 strategy **4%** (§2).
2. **Strictness did not kill the record.** Every RL candidate either failed on direct evidence of no edge, or first "passed" on an artifact.
   - The clean GMGP1-BTC canary's pooled out-of-sample (OOS) Sharpe was −4.8 to −6.0.
   - Separately, the shipped SAC stack cannot learn a planted IC-0.05 signal.
   - The single-asset RL NO-GOs stand.
3. **It was too lax where it mattered.** Every PASS on record was measured on leaked or artifact-bearing data. No protocol gate caught any of them (§3).
   - Stage 4 graded a **+14,432%** 60-day return as HOLD.
   - The ratio gates go inert at realistic profit factors. In the model, the 60-day gate returns HOLD in 100% of paths at every true Sharpe, zero included.
4. **Most of it is not enforced.** `validate_config.py` checks config shape before a run. It enforces none of the stage-outcome gates (§4):
   - No stage manifest is ever written, and Stage 4 does not exist for RL.
   - `hpo_pf_floor`, `wf_pf_floor` and `recent_oos_days` are read by no code.
   - The one gate that ever killed a candidate (walk-forward G1) lives in per-workstream eval scripts. There, a failure was overridden, a threshold was lowered after the result was seen, and a walk-forward failure deploys the best solo seed instead of stopping.
5. **It governs a closed program.** It was written for single-asset intraday RL on prop-firm accounts; that program is falsified and its fleet has been offline since 2026-07-03.
   - The active direction (linear core first, RL only as an overlay behind a beat-linear-OOS gate) has no protocol, and the one gate that direction depends on is not in it.
   - Each linear workstream built its own AND-stack instead. That is where strictness has plausibly blocked a real edge. Recomputed consistently (`tailwind_dsr_consistent_2026-09-29.md`), TAILWIND's research-basis DSR is 0.988, not the recorded 0.896, so that block does not hold. The executor-path book (the one that would trade) failed at 0.930 against the recorded 0.871, because of an extra bar of execution lag (R-3). With that fixed the same day, it reaches 0.967, but only on unfinanced total returns. The same day's Tier-2 audit measured 0.907–0.934 on excess-of-T-bill returns, still BLOCK.
6. **The structural reason nothing makes it: paper is gated as if it were capital.**
   - Confirming a Sharpe-1 edge takes about four years of OOS data, whatever the protocol says (§2.4).
   - For an intraday strategy, fresh OOS data comes only from running forward, and running forward on paper requires passing the whole stack first.
   - The dormant Phase-β gates wait to be calibrated on a passing candidate, which cannot appear.
   - Both loops are closed.

---

## 2. Operating characteristic: what the gates can and cannot pass

**Model.** A window of T years yields a realized annualized Sharpe `s + training dispersion + noise/√T`.
- Seeds on a shared window share the market term (correlation ρ).
- Bar-level PF follows from the Sharpe via `PF = (1+kx)/(1−kx)`, where x is the Sharpe per in-market bar and k = σ/E|r|.
- The analytic map matches direct per-bar simulation to about 0.004 PF, for both normal and t(4) tails.
- It also agrees with the 09-23 audit's empirical map from 849 saved trajectories: bar-PF [1.01, 1.02) ↔ Sharpe ≈ 1, ≥ 1.10 ↔ ≈ 8.
- Legs not modelled (fixed-lot DD stress, MDD breach, compliance, obs-noise, sensitivity) can only lower the pass rates, so **every chain number below is an upper bound**.

### 2.1 A bar-level PF floor is a Sharpe floor far above any real edge

True annualized Sharpe at which the PF point estimate equals the floor (normal tails; t(4) tails ≈ 11% lower):

| Bars | Time in market | PF 1.10 | PF 1.20 | PF 1.50 |
|---|---|---|---|---|
| 15 m crypto 24/7 | 30% | 3.9 | 7.4 | 16.4 |
| 15 m crypto 24/7 | 100% | 7.1 | 13.6 | 29.9 |
| 15 m gold | 30% | 3.2 | 6.1 | 13.5 |
| 15 m gold | 100% | 5.9 | 11.2 | 24.7 |
| 3 m crypto 24/7 | 30% | 8.7 | 16.6 | 36.6 |
| 3 m crypto 24/7 | 100% | 15.9 | 30.4 | 66.8 |

The floor is not even a fixed Sharpe:
- It falls with lower exposure and fatter tails, so it rewards trading less, not trading better.
- It rises with finer bars, so the 3-minute SG-1 floors were stricter than GMGP1's at the same number.
- The doc defaults (HPO and L1 at 1.5) are stricter still than the YAML values most configs use (1.2).

### 2.2 The documented chain (15 m crypto, 60% in market, 4 × 1-month WF, 10 seeds with ρ = 0.5)

| True SR | Doc floors 1.5 / 1.5 | YAML floors 1.2 / 1.2 / 1.10 | Floors removed | L1: every seed PF > 1 | WF: all 4 windows > 0 | Stage 4 HOLD |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.5% | 8.5% | 6.1% | 100% |
| 0.5 | 0 | 0 | 1.2% | 13% | 9.4% | 100% |
| 1 | 0 | 0 | 2.7% | 20% | 13.9% | 100% |
| 2 | 0 | <0.1% | 9.4% | 35% | 26.6% | 100% |
| 3 | 0 | <0.1% | 22.5% | 54% | 41% | 100% |
| 5 | 0 | <0.1% | 63% | 86% | 73% | 100% |
| 8 | 0 | 7.6% | 95% | 99.5% | 96% | 100% |
| 15 | <0.1% | 99% | — | — | — | — |

- **With floors:** nothing below Sharpe ~8 passes. That band is where every artifact in the record sat (Sharpe 20–86).
- **Without floors:** the "every unit must win" rules (every seed profitable, every window profitable) still pass a zero-edge strategy 0.5% of the time and a Sharpe-1 strategy 2.7%. That is a discrimination ratio of about 5. With near-identical seeds (ρ 0.9, dispersion 0.1) the figures are 2.0% vs 6.6%. With 3-minute bars and 8 folds, Sharpe 3 passes 9.9%.
- **The HPO floor is noise, not a test.**
  - At 1.2 it passes 78% of the time at Sharpe 0 (best of 50 trials, each scored on 500-bar slices), and 99% on 3-minute bars.
  - At 1.5 it is unreachable.
  - No code applies it anyway (§4).
- **Stage 4 is inert.** At realistic PF (≈ 1.01–1.05), "PF_oos ≥ 0.8 × WF PF" is satisfied by any PF above ~0.82, which includes a losing strategy. It binds only at artifact scale (PF ~2), where it demands PF_oos ≥ 1.6. Principle 5 ("most-recent OOS is non-negotiable") therefore protects against nothing at realistic scale.

### 2.3 The walk-forward gate as actually run (G1)

`sg1_btc_velotrade_ensemble_eval.py:526-674`: PASS iff at least 2 seeds clear `pf_floor` in at least N folds. There is no Sharpe criterion, and one losing fold out of four still passes.

| Rule | SR 0 | 0.5 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|---|
| Canary: 2 of 5 seeds, ≥ 3/4 folds, PF ≥ 1.10 | 0 | 0 | 0.1% | 0.9% | 4.1% | 32% | 93% |
| Same rule, PF ≥ 1.05 | 2.4% | 5.3% | 9.6% | 26% | 50% | 90% | ~100% |
| Same rule, PF ≥ 1.00 | **45%** | 58% | 69% | 87% | 96% | ~100% | 100% |
| SG-1-BTC: 2 of 3 seeds, ≥ 7/8 folds, PF ≥ 1.10 | 0 | 0 | 0 | 0 | 0 | 0.3% | 38% |
| SG-1-BTC rule, PF ≥ 1.00 | 1.1% | 3.0% | 7.0% | 24% | 52% | 93% | 100% |

The structure is lax (it takes the second-best of five seeds) and the floor is strict. The floor alone sets the outcome, and no floor value gives a usable test on four one-month folds.

### 2.4 The limit no protocol can change

Power of **one** pre-registered one-sided test on pooled OOS Sharpe, `Φ(SR·√T − z_α)`:

| α (false pass at SR 0) | Pooled OOS | SR 0.5 | SR 1 | SR 2 | SR 3 | Sharpe detectable at 80% power |
|---|---|---|---|---|---|---|
| 0.20 | 4 months | 0.29 | 0.40 | 0.62 | 0.81 | 2.92 |
| 0.20 | 1 year | 0.37 | 0.56 | 0.88 | 0.98 | 1.68 |
| 0.10 | 1 year | 0.22 | 0.39 | 0.76 | 0.96 | 2.12 |
| 0.10 | 2 years | 0.28 | 0.55 | 0.94 | 1.00 | 1.50 |
| 0.10 | 4 years | 0.39 | 0.76 | 1.00 | 1.00 | 1.06 |
| 0.05 | 4 years | 0.26 | 0.64 | 0.99 | 1.00 | 1.24 |
| 0.05 | 8 years | 0.41 | 0.88 | 1.00 | 1.00 | 0.88 |

- The standard error of an annualized Sharpe is 2.47 on 60 days and ≈ 1.7 on the 4 × 1-month walk-forward.
- An RL overlay that must beat the linear core by ΔSR 0.3 on the same path needs **10 years** of OOS at ρ = 0.9 (5 years at ρ = 0.95) for 80% power at α 0.10.
- **Consequence:** a gate on a few months of data can separate artifacts from everything else, and nothing finer. The protocol has to be built around where confirming data comes from: long history for daily books, forward running for intraday ones. §6 does that.

---

## 3. Outcome ledger

All PFs are bar-level. "Leak-era" means measured before the X2 fix `6c027e19` (2026-05-30). X2 entered in `7b1d9907` (S121) and was present for every V7 multiscale run after the protocol's ratification on 2026-04-18. GATE-CAUSAL-01 was fixed 08-26.

| Workstream | Furthest point | What decided it | Later |
|---|---|---|---|
| GMGP1-BTC | Paper (live checkpoint was a pre-v2 L1 seed) | HPO 2.26, L1 1.95, WF 4/4 median 2.03, Stage 4 PASS at +5,438% | X2: fold-0 PF 0.79–0.85 de-leaked; Stage 4 re-graded with slippage: PF 0.545 |
| GMGP1-BTC clean canary | Died at WF (06-06) | G1 0/4 at PF ≥ 1.1; best-solo median 0.977; stress DD −33% | Direct evidence of no edge: pooled OOS Sharpe −4.8 to −6.0, IC negative in 20/20 |
| SG-1-BTC | Paper → shelved 06-01 | Leak-era WF G1 8/8; de-leaked X1: 3/10 L1 seeds PF < 1, WF G1 0/8, best solo 1.016 | X2 (1.689 → 1.016) plus GATE-CAUSAL-01; pooled −0.1 to −2.1 (SE ≈ 1.5) |
| GMGP1-Gold | Paper | L1 2.58, 2.5-R PROMOTE P = 1.0, WF 4/4 median 2.99 | X2 (de-leaked G1 2/4); stale prints ≈ 21% of the best fold; bid-ask bounce 70–86% of gross |
| GMGP1-XAUUSD, SG-1-XAUUSD, SG-1-EURUSD | Paper, halted | Leak-era PASS chains; Q1 OOS HOLD at PF 3.54 / +240% | VOID (X2; SG-1 also GATE-CAUSAL-01); never re-tested clean |
| GMGP1-EURUSD | Stage 2.5-R PROMOTE | Fleet halted before WF | Leak-era |
| Funding-Arb DSAC | Died at WF (04-29) | 4/8 windows profitable, 8/8 required | No leak; later direct regime-decay evidence (−0.59% 2025, −3.15% 2026) |
| GMGP1-SPX500 (post-X2) | Died at L1 / WF | L1 0.97 / 0.96 vs 1.5 | Long-only OOS Sharpe +1.21 failed PF 1.10, but 0/5 seeds beat buy-and-hold: beta, rightly killed |
| Execution-overlay SAC | Died | +2.0 bps uplift gate | Gate above the action space's +1.70 bps ceiling: unreachable by construction |
| CMGP1, AlphaSeek, RL allocator | Never ran the chain | Killed outside the protocol | Direct evidence of no edge |

**Every PASS was an artifact, and no gate caught one.**
- Passing runs had annualized Sharpe 20–86, or +6 to +9 on gold's stale prints.
- The Q1 recent-OOS sweep (`results/q1_2026_oos_blindspot/*/verdict.json`) graded these as HOLD:
  - GMGP1-BTC: +14,432%.
  - Gold: +570%.
  - SG-1-XAUUSD: +240%.
  - SG-1-BTC: +7,774% at daily Sharpe 78, flipped to RETRAIN only by a hand-written "implausible return" override (the plausibility ceiling §6 proposes, applied manually once).
- The artifacts were found outside the protocol:
  - X2 by a sim-vs-live P&L gap.
  - Stale prints by the Fable forensics.
  - Bid-ask bounce by the 09-23 audit.

**Gates that decided without evidence either way:**
- **SG-1-BTC X1:** pooled −0.1 to −2.1 with SE ≈ 1.5 cannot exclude Sharpe ≈ 1.
- **Funding-Arb:** required 8/8 profitable windows. P(8/8) ≈ 5% at Sharpe 1 even with quarter-long windows.
- **SPX500 long-only:** its PF-floor failure happened to be right for a different reason (it was beta).

**The protocol was not followed as written:**
- **GMGP1-BTC (S528):** the WF uplift gate failed at 1.078 < 1.10. An operator override then lowered it to 1.05, on a gates file whose header said not to edit after first launch. This was documented in `project_gmgp1_btc_along_wf_uplift_threshold_revision.md`, and is now obsolete as leak-era.
- **Funding-Arb:** L1 had CV 0.383 (above the 0.38 band ceiling) and a seed at PF 0.97, an immediate FAIL under §4. It proceeded with two seeds relabelled as outliers.
- **SG-1-BTC X1:** 3/10 seeds had PF < 1 (0.67–0.69), another immediate FAIL, and CV 0.248 sat in the pre-committed escalation band. `seed_report.json` instead reports `stats_healthy_seeds` on the 7 winners (CV 0.015). That is the "median, not max" principle inverted.
- **Any walk-forward failure:** it returns `reject_ensemble_use_best_solo`, which deploys a solo seed. Only G1 ever stopped a candidate.

**Every calibration anchor in the protocol is leak-era.**
- **§3.5.3 "validated cells"** (PF 2.44 / 1.95 / 2.76 / 2.00, S500–S542): codified 05-21, nine days before the X2 fix, never re-derived. De-leaked, the same cells read 0.79–0.85 and 1.016.
- **§4's ensemble evidence (+17.6%, +17.15%):** leak-era. The +17.6% is mislabelled as L1 (it is SG-1-XAUUSD's walk-forward). `ens_agreement`'s apparent consensus edge came from a flatten-on-disagreement bug: fixing it cut that rule's PF by 18.8%, while `ens_mean` matched the original.
- **`hpo_pf_floor: 1.2`:** set on leak-inflated PFs (`project_hpo_leak_contaminated_at_selection_s553.md`).
- **The CV rationale ("PF = 2.0 ± 1.0"):** assumes the leak-era PF level.
- **§3.5's multiplicity REJECT (> 50×):** a hard block resting on this void evidence.

**The non-RL strategies never used the protocol.** Each built its own stack:

| Strategy | Stack and verdict |
|---|---|
| TSMOM | `cross_asset_momentum.gates.yaml`: DSR 0.918 < 0.95; paper rung REVIEW (PSR 0.84 < 0.95, when the verdict's own minimum track record for 0.95 is 32 months; corr to SPY 0.53 > 0.40) |
| TAILWIND | `tailwind_v1*.gates.yaml`: DSR 0.896 / 0.871 < 0.95, P(pass) 0.615 < 0.65. Recomputed consistently 2026-09-29: research-basis DSR 0.988 (PASS); executor-path DSR 0.930 (FAIL), then 0.967 once the R-3 lag was fixed. That 0.967 is unfinanced: on excess-of-T-bill returns it is 0.907–0.934 (FAIL; Tier-2 2026-09-29). P(pass) 0.667 on 21 windows (CI about [0.45, 0.83]) is 0.524 with the declared kills; the needless-termination gate fails |
| keel-v1 | `finrlx_strategy.gates.yaml`: alpha t 1.996 < 2.0 (pre-registered; now in lockbox) |
| ATL × Jev | Crucible funnel + `atl_jev.gates.yaml`: P4 not run at ~20–25% power |
| BALLAST | `ballast_v1.gates.yaml`: survivorship NO-GO |

---

## 4. Conformance: the document vs the code

| Rule | Status | Evidence |
|---|---|---|
| Line 7: "the stage contract is what `validate_config.py` enforces" | **False** | The validator checks config shape; stage-outcome gates live in eval scripts that write `verdict.json` and exit 0 (`sg1_xauusd_ensemble_eval.py:1768-1771`, `sg1_btc_velotrade_ensemble_eval.py:870-877`) |
| P1: one stage = one run; fused pipelines banned | **Contradicted** | `run_full_pipeline.py:997-1063` runs HPO (on by default) → train → val/test backtest in one run; `--stage` only triggers validation (`:881-899`). The GMGP1-Gold A2 chain ran this way (operator-aware, under §5's backward-compat default) |
| P2/P3, §2: manifests, upstream PASS, resume | **Doc-only** | Nothing writes `<run_id>.manifest.json`, `env_code_sha`, `training_health` or `replay_buffer`; no `--upstream-run`/`--resume`. State passes through hand-edited YAML (`ensemble.seeds`, `chosen_rule`) |
| P5: `test_end ≥ today − 60d` | Doc-only | No check |
| P6: median, not max | **Contradicted** | G1 passes on the 2nd-best of 5 seeds; the fallback deploys the test-argmax seed (`sg1_xauusd_ensemble_eval.py:1075-1077`); "healthy seeds" stats drop losers |
| P7: ≥ 10 stochastic eval episodes | Doc-only | Every eval is one deterministic rollout; `eval_episodes` is in no config |
| §3 recency | **Contradicted** | HPO data may be 180 days old (doc: 7), WF 90 (`validate_config.py:616-636`) |
| §3.5 multiplicity; 22-month window | Partial / doc-only | Multiplicity is computed only for crypto and the allocator, WARN below 15 (doc: 10); SG-1-BTC WF trained on 6- and 18-month windows |
| S1 `hpo_pf_floor`; training-health hard-fail | **Doc-only** | No code reads the floor. Only NaN is caught; the entropy / Q-div / saturation keys are unread and disagree with the doc (0.01 vs −3.0, 100 vs 10, 0.9 fraction vs 95.0 percent) |
| S1 HPO budget | Contradicted | Validator checks `hpo.trials`; launchers read `n_trials` |
| S2 L1 gate, N = 20 escalation | One-off | Only `auto_queue_wf_after_l1.py` (SG-1-XAUUSD, hardcoded seeds) applies it; no auto-extend; CV in (0.30, 0.38] is FAIL there, not AMBIGUOUS |
| S2.5 Phase 0 diversity selection | **Contradicted** | Audit-only, ranked on **test** PF (`:1139`); `ens_pf_weighted` weights are L1 test PFs, used even in the val bake-off (`:1017`); drift baselines come from the same window |
| S3 "all windows profitable, median Sharpe > 0, no MDD breach" | **Not implemented** | G1–G5 instead; `wf_pf_floor` (in 69 config files) is read by no code |
| `run_walk_forward.py` | **Broken** | Reads `sharpe_ratio` / `max_drawdown`, but `run_backtest` returns `wf_fold_XX/*` keys, so every fold metric is None. It also seeds only the parent process: SEED-01 still lives here |
| S3 PF-XCHECK per window | Doc-only | Exists only in the sensitivity audit, report-only, on (H+L)/2 (which cannot see bid-ask bounce: 09-23 M-1/M-2) |
| S4 recent-OOS HOLD/WATCH/RETRAIN; compliance | **Not implemented for RL** | `recent_oos_days` is read by no code; only one-off scripts (the Q1 sweep, the TAILWIND research script) bucket; `ftmo_compliance_report.py` exists but nothing calls it |
| §5 CLI, "validate before every run" | Doc-only | Remote launches never pass `--stage`; distributed HPO, `launch_l1_multiseed.py` and `run_walk_forward.py` never call the validator |
| §7 source parity; §8.1 feature / Mahalanobis drift | Doc-only | No implementation |
| §8.2 action / agreement drift; §8.3 kill file | Enforced at runtime | Lockout thresholds hardcoded in `kill_file.py:38-39`, `gates.safe_mode.*` unread |
| "No hardcoded gate thresholds" | **Violated** | At least 12 sites, e.g. `sg1_xauusd_ensemble_eval.py:590-592,1111`, `sensitivity_audit.py:55-66`, `fixed_lot_stress.py:169-170`, `ftmo_compliance_report.py:55-56`, `validate_config.py:616-624,803-812` |
| Beat-linear-OOS gate (project direction) | **Absent** | Opt-in `hpo_require_beat_buy_hold` falls back with a WARN and is skipped on the distributed path |
| Seed-divergence tripwire | Unit test only | `tests/test_env_seed_reproducibility.py`; nothing checks at run time that seeds differ |
| Document structure | Defect | The `### Stage 3` heading is missing, so the Stage 3 gates sit under Stage 3.5 |

**The in-flight change.** The uncommitted provenance block (§3.1, `check_data_provenance`, the `build_data_manifest.py` flags, and `tests/test_data_manifest_provenance.py`) is sound.
- It ships dormant: WARN unless a workstream sets `gates.provenance_required`.
- It resolves gates through the same overlay as its sibling checks.
- The 219 validator and provenance tests pass.
- It does not affect strictness.

---

## 5. What to keep

- **Staged runs with one decision per stage.** The principle is right, even though the default launcher breaks it.
- **The causality invariants.** LEAK-1/LEAK-2 with their negative tripwire tests, and the bundle SHA-256 contract with its all-or-nothing live swap (enforced in `sharpen/live/ensemble_bundle.py`).
- **The provenance block** (in flight).
- **Pre-commitment.** Declaring the ambiguous band before launch is the right discipline; it was attached to the wrong statistic.
- **Runtime safety.** Live action-drift and kill-file flatten.
- **A Tier-2 audit before real capital.**

---

## 6. Recommendations

Ordered by leverage. Items marked **[operator]** change a standing rule and need a decision.

1. **A promotion ladder keyed to the cost of a false positive [operator].** This is the change that most directly answers "nothing ever makes it".
   - Rungs: research → paper incubation (no capital) → prop-firm challenge (fee at risk) → live capital.
   - **Each rung has one pre-registered primary test**, with its α and its power at a stated target Sharpe published before the run.
   - **Paper:** α ≈ 0.20 on pooled OOS, plus the tripwires in item 2 and an integrity-scoped audit, not the full Tier-2.
   - **Prop challenge:** the forward paper record must pass a sequential, always-valid test on daily net P&L, plus a simulated challenge P(pass) with a confidence interval.
   - **Live:** α ≤ 0.05 on backtest plus forward evidence, deflated against the trial ledger, plus the full Tier-2.
   - Paper becomes the instrument that generates the confirming evidence the backtest cannot. This amends CLAUDE.md's "capital (live/paper)" rule.
2. **Artifact tripwires as hard gates at every rung.** These are the checks with a track record. Each artifact on record would have tripped at least the ceiling.
   - A plausibility ceiling: single-asset intraday net Sharpe above ~3, or a window return beyond a stated bound, is presumed an artifact until audited.
   - A frictionless-vs-net split.
   - Mark-to-mid, and PF-XCHECK on the true LOB mid.
   - A runtime check that seeds diverge and that the same seed reproduces.
   - A placebo run on shuffled returns and a circular-shift timing null.
   - Planted-signal recovery for any training stack before it is used for discovery.
3. **Retire bar-level PF gates [operator].**
   - Grade daily-aggregated net Sharpe on OOS pooled across all walk-forward folds, with its CI. Keep bar-PF as a diagnostic.
   - Extend the walk-forward over the full available history instead of the last four months.
   - Re-express the ratio gates (CV, OOS hold, sensitivity, obs-noise) on excess return or Sharpe, not PF, so they stop going inert at realistic scale.
4. **Replace AND-of-units rules with the pooled statistic.** This covers every-seed, every-window, and 2-of-5 seeds in 3/4 folds; seed and fold dispersion become diagnostics.
5. **Calibrate every gate on planted signals of known Sharpe, never on candidates.** Publish its false-pass rate at Sharpe 0 and its power at the target. This also unblocks the dormant Phase-β gates (sensitivity, obs-noise, provenance) without waiting for a passing candidate.
6. **Selection hygiene.**
   - Rank seeds and set ensemble weights on val, never test.
   - Read the test window once.
   - Embargo HPO windows before the first walk-forward test fold. The SG-1-BTC HPO val window may overlap WF folds 0–1; unconfirmed.
   - No post-hoc exclusion of losing seeds.
7. **A trial ledger and frozen gates.**
   - Count every HPO trial, retrain and variant.
   - Hash gate files, as CRU-1 does for Crucible, so a post-hoc edit is visible.
   - A waiver is a written decision that cites the ledger.
   - Loosening without this step would turn the current false negatives into false positives.
8. **The linear path and a feasible overlay gate.**
   - Linear books use the same ladder.
   - The RL overlay gate becomes non-inferiority on net Sharpe (margin δ) plus superiority on a precisely measurable secondary: cost, turnover or drawdown.
   - Superiority on Sharpe itself needs ~10 years of OOS (§2.4).
9. **Replace the 60-day recent-OOS rule** with rung 3's forward sequential test. Keep model age as an ops check, not as evidence.
10. **Make the document true.**
    - Split it into a short promotion standard, the RL training contract, and live ops.
    - Move the leak-era anchors to a marked-void appendix. Demote §3.5 to unvalidated guidance and drop its REJECT.
    - Fix line 7, the missing Stage 3 heading, the recency mismatch and the training-health keys.
    - Implement or delete each DOC-ONLY section, and move the hardcoded thresholds into YAML.
    - Fix `run_walk_forward.py`: the metric keys, and seeding the envs.
11. **A cheap immediate re-read.**
    - Done 2026-09-29 (`tailwind_dsr_consistent_2026-09-29.md`). The research basis clears (0.988). The executor path failed (0.930) until its extra bar of lag (R-3) was fixed the same day; it now clears (0.967).
    - That made TAILWIND look like the natural first paper-rung candidate. The same day's Tier-2 audit (`tailwind-v1_deep_lifecycle_audit_2026-09-29.md`) returned **BLOCK** on every rung, including forward paper-sim incubation, because no forward runner exists. The executor's 0.967 is unfinanced and batch-timed. Its corrected figures all fail: the recorded (E0) method gives 0.936, excess of T-bill 0.907–0.934, and realizable timing 0.920–0.940. P(pass) 14/21 has Wilson interval [0.454, 0.828], and is 0.524 with the declared kills. The research basis still clears when financed (0.962–0.973).
    - SG-1-BTC X1 and Funding-Arb do not merit re-runs: nothing in their record points to an edge.

---

## 7. Reproduce

```bash
python scripts/research/protocol_v2_gate_power.py --section all   # every table in §2 (seeded, ~10 s)
python scripts/research/protocol_v2_gate_power.py --section map   # PF <-> Sharpe map vs direct per-bar simulation
```

**Model limits.**
- Window-level normal noise, independent across windows.
- Training dispersion 0.5 Sharpe, with sensitivity shown at 0.1.
- HPO scored on 3 × 500-bar slices, which the canary config produces via `episode_length: 500`.
- Omitted legs make every chain figure an upper bound.
- The conclusions do not depend on these choices. Floors in use need Sharpe ≥ 3 under every setting tried, and short windows cannot confirm Sharpe ≤ 2 under any threshold.

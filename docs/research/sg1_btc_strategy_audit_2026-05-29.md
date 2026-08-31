# sg1-btc RL Strategy — Deep Lifecycle Audit

**Date:** 2026-05-29 · **Strategy:** sg1-btc (3-min BTC SAC, Velotrade prop-firm path) · **Protocol:** v2.5.1 · **Scope:** 11 pillars (data → live) + external SOTA benchmark · **Method:** finder + skeptic per pillar; severity = skeptic-adjusted.

---

## Executive Summary

The sg1-btc pipeline is **engineered well below the waterline and weak above it**: the low-level correctness machinery (LEAK-1 normalization, DSR formula, equity/short accounting, bundle SHA256 chain, paired block-bootstrap) is genuinely strong, but the *decision and validation layer* that is supposed to certify robustness is riddled with documented-but-unimplemented gates and single-point calibrations. Live capital is protected from blow-up (the in-engine drift WARN/CRIT path and bundle integrity are solid), but the strategy is being run on a known-decayed, cost-unaware policy whose promotion evidence does not transfer.

**The dominant cross-cutting theme — the single-cost-regime fault.** The entire HPO → reward → seed-selection → ensemble-rule chain was optimized and selected under a near-cost-free, turnover-blind regime (taker 5.0 bps, slippage **0**). The S552 RCA later proved the honest cost is ~10.5 bps round-leg and reconciled the live loss to the penny — but the correction was applied *downstream* (to the WF re-run config only, still uncommitted) **without re-deriving the selection**. So the deployed seed-456 policy was never chosen net-of-cost; the cost-corrected WF inherits the cost-free top-3 verbatim; and the OOS gate still bakes the optimistic costs. This is the AlphaSeek frictionless-artifact trap, and it propagates through every stage of the lifecycle.

**Four supporting themes:**
1. **Statistical robustness claims are over-stated by construction.** Overlapping WF folds (94% train overlap → ~8 correlated trials, not 8 independent ones), thin 15-day L1 / 1-month WF test windows, 3× val reuse with no multiplicity correction, and a degenerate seed pool (test-PF CV 0.8%, action corr 0.84-0.86) mean the CV/8-of-8 gates measure *seed-convergence and adjacency*, not regime robustness. The ensemble adds ~0 diversification (correctly fell to SOLO_BEST_FALLBACK).
2. **The reward does not train the policy to economize cost.** Cost is subtracted inside R_t, but DSR is scale-invariant so the churn disincentive is structurally muted, and there is **no turnover penalty term** anywhere — trade frequency is governed by one static deadband.
3. **Documented safeguards do not exist in code.** The training-health kill gates, the [15,40] multiplicity check, PF-XCHECK for the V7 path, the fixed-lot stress sub-report, and the §4.5 retrain triggers (cost-drift, 28d staleness) are all declared in the protocol/configs but have **no Python consumer** — false assurance at five separate stages.
4. **Missing standard overfitting controls.** No deflated-Sharpe / PBO / purged-embargoed CV anywhere; nothing quantifies that PF=2.47 survives the 50-trial × top-3-of-10 × best-of-7-rule × 8-fold search.

**Credit where due:** LEAK-1 causal EMA-Z with warmup carry-forward (train==live bit-for-bit), the manifest/bundle SHA contract, block-bootstrap-primary verdicts, atomic swap bundles with operator handshake, the NaN no-consensus sentinel, the two-mode drift handling, the penny-accurate sim-live RCA, and pre-committed do-not-edit gate YAMLs are all real strengths and should be preserved.

**Verdict:** The pipeline will protect capital but will **not reliably build a robust, consistent strategy** until selection is re-derived under honest cost and the documented gates are actually wired. No bad ensemble is live (deploy is solo seed-456), so the fixes are about the *next* promotion, not an emergency.

---

## Per-Pillar Findings

Severity scale: S1 = will cause live loss / leakage · S2 = materially weakens robustness or rigor · S3 = rigor/hygiene gap · S4 = minor / positive. REFUTED findings dropped (count noted per pillar). IDs as emitted by the pillar runs (note: several pillars internally reused the `P4-xx` prefix; preserved verbatim for traceability).

### P1 — Data preparation & integrity
*Functional and LEAK-1-safe, but integrity gates are weaker than the protocol claims; provenance (sha256) and regime-coverage gates are no-ops.* (11 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P1-01 | S3 | PF-XCHECK (mid vs close, >30% halt) declared invariant but not implemented for V7/SG-1; the same-named run_full_pipeline check is bar-PF vs trade-PF on one close stream | protocol_v2.md:437,707; run_full_pipeline.py:484,642,681-682 | Add (high+low)/2 pseudo-mid cross-check or formally scope PF-XCHECK out for OHLCV pipelines | Next |
| P1-02 | S2 | DATA-CLEAN not enforced: build_data_manifest self-certifies clean_ohlcv_passed via a high<low/close≤0 check; bybit file not in clean_ohlcv.FILES | build_data_manifest.py:66-72; clean_ohlcv.py:311-322 | Call clean_ohlcv.detect_outliers; record threshold + n_repaired + cleaned_by | Now |
| P1-03 | S2 | regime_quartiles degenerate (self-quantile → always ~0.25); the >0.10 gate can never fail | build_data_manifest.py:55-64; all manifests:11-14 = 0.25 | Bin against fixed/absolute vol bands; emit regime_cutpoints | Now |
| P1-04 | S2 | Manifest omits sha256/source/gap_count/feature_distribution/built_at; no JSON schema for data manifest | protocol_v2.md:138-160 vs emitted fields; schemas/manifest.schema.json:4 | Emit sha256 + full §3 fields; add schema; reject on hash mismatch | Next |
| P1-05 | S3 | gap_detection:false + unconditional df.ffill() would silently bridge real outages (bitfinex sibling has 8.3h gaps) | yaml:127; multiscale_handler.py:278 | Gate ffill on flag + max-gap; emit gap_count; mask synthetic bars | Next |
| P1-06 | S4 | buffer_days:0 + inclusive filters → 1-bar train/val overlap at every fold boundary (16 bars / 8 folds) | splitter.py:56-62; multiscale_handler.py:268,304-305 | buffer_days≥1 or half-open filters | Now |
| P1-07 | S3 | clean_ohlcv has dead no-op lines (135/158) and a 5% clamp that erases legit BTC liquidation wicks | clean_ohlcv.py:135,158,48,454,147-149 | Remove dead lines; percentile/MAD outlier rule; neighbor-median repair; record n_clamped | Now |
| P1-08 | S4 | Freshness gate trusts manifest.last_ts without verifying against parquet (no hash binding) | validate_config.py:337-357 | Read parquet max-ts or enforce P1-04 sha256 | Now |
| P1-09 | S4 | **POSITIVE:** LEAK-1 EMA-Z isolation correct end-to-end, train==live | run_full_pipeline.py:905,918; multiscale_handler.py:159(WARMUP=200),161-171 | Add perturb-train regression test | Now |
| P1-10 | S4 | **POSITIVE:** Calendar-anchored splits + causal resampling correct | splitter.py:56-62; multiscale_handler.py:53-67 | Pass explicit label='left',closed='left' | Now |
| P1-11 | S4 | No drop_duplicates / monotonic assert in manifest or handler (append occurred: 35,983 bars) | build_data_manifest.py:43; multiscale_handler.py:270 | drop_duplicates(keep='last') + assert; emit duplicate_ts_count | Now |

### P2 — Feature engineering & normalization / leakage
*EMA-Z machinery is excellent, but a true coarse-bar forward look-ahead inflates every metric and creates a train↔live divergence.* (9 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P2-01 | **S1** | Coarse-bar alignment leaks future: every base bar in a coarse interval is fed the FULLY-COMPLETED coarse bar (close up to 14/59 min ahead). Live sees the partial in-progress bar → train/live divergence. Numerically reproduced | multiscale_handler.py:59,356,394-401; live_obs_builder.py:444,744; live cadence cfg:131 | Make coarse alignment causal (map to last COMPLETED coarse bar); apply identically in handler + live builder → re-HPO/L1/WF | Next |
| P2-02 | S2 | Features 0/1/2 + base ATR computed full-series; atr_norm rolling(14)/ema(50) carry pre-cutoff state across split (causal warm-start leak; drives the signal gate) | multiscale_handler.py:131-154,335-344; signal_gated_wrapper.py:183 | Route ATR through the warmup-buffer carry, or rename LEAK-1 to 'norm-stat isolation' | Now |
| P2-03 | S3 | MultiScaleHandler never drops the ~200 unstable EMA-warmup rows (ParquetHandler does) | parquet_handler.py:125-128 vs multiscale_handler.py:304-311 | Drop WARMUP_BUFFER rows or set start_date earlier | Now |
| P2-04 | S3 | No sg1-btc feature ablation; only FQ-1 artifact is gmgp1-v5 (flags volume_z 44% LOO); 8th feature kept for TC alignment only | results/fq1_gmgp1-v5-*.json; multiscale_handler.py:178-181 | Run FQ-1 on sg1-btc; A/B replacement 8th feature | Research |
| P2-05 | S2 | No V7 no-look-ahead / mid-coarse-bar parity test; existing tests compare only at end-of-data, masking P2-01 | tests/data/test_compute_features_with_warmup.py; tests/crypto/test_live_obs_parity.py:95-143 | Add truncation + mid-interval bootstrap parity test; wire to Audit chain | Now |
| P2-06 | S4 | WARMUP_BUFFER comment wrong: '~1.7 half-lives' is actually ~4.8 (buffer is generously adequate) | multiscale_handler.py:159; parquet_handler.py:124 | Fix comment to ~4.8 | Now |
| P2-07 | S4 | Stale docstrings say '7 features'; env defaults features_per_scale=7 (config rescues with 8) — latent footgun | multiscale_handler.py:5,377-379; continuous_swing_env.py:47,482,506 | Update docstrings; default 8; assert input_size==features_per_scale | Now |
| P2-08 | S4 | **POSITIVE:** EMA-Z normalization-stat isolation correct + train/live bit-equal | multiscale_handler.py:161-171; tests pass rtol 1e-6 | Extend carry to ATR (P2-02) | Now |
| P2-09 | S4 | **POSITIVE:** Bounded SymLog→EMA-Z→tanh transforms; log_return clip never binds (0/408479 bars >10%) | multiscale_handler.py:25-27,50,137,149,154 | Keep | — |

### P3 — Environment mechanics & execution realism
*Step ordering and short accounting are correct; the fill model and the total absence of a real-env unit test are the liabilities.* (13 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P3-01 | **S1** | Flat-bps slippage defaults 0 and was unset in the L1 config that selected deployed seed-456; ensemble chosen under zero-slippage. RCA: ~4.58 bps/leg unmodeled (Gap A) | continuous_swing_env.py:58; l1_decay01.yaml; sim_live_gap.md:43-48 | Re-run L1+WF at taker 0.00055 + slippage 5.0; re-select seeds | Next |
| P3-02 | S2 | No test instantiates the real ContinuousSwingEnv; equity/PnL/deadband/ATR-cap/DD paths uncovered despite 4 load-bearing audit-fix comments | tests/* use mocks; env audit fixes :292,347,383,393 | Add golden-trajectory test_continuous_swing_env.py | Now |
| P3-03 | S2 | Inner env runs a tick-by-tick running-peak 10% DD termination live never runs (live = static-peak) — dual DD authority in training | continuous_swing_env.py:390,395; crypto_risk_manager.py:192-198 | Add dd_termination_mode: static/running/off; mirror live | Next |
| P3-04 | S3 | Conditional ATR-cap (clip ±0.5 when atr>p90) is sim-only; no live counterpart → sim/live position divergence in high-vol bars (PRISM 'divergence' is inert — disabled) | continuous_swing_env.py:293-305; live_engine.py has no ATR-cap | Port ATR-cap to live OR disable in deploy-matched training | Next |
| P3-05 | S2 | Signal-gate thresholds (atr 0.15 / park 0.0009 / vol 0.60) held from S479; never OOS-revalidated for gate-open-rate drift; gate is the turnover lever (cost multiplier) | yaml:61-65,105 | Add per-fold gate-open-rate diagnostic + target band to gates YAML | Next |
| P3-06 | S2 | validate_config enforces taker_fee presence but ZERO slippage checks — the exact S552 failure passes clean | validate_config.py:120-163 (ran: PASS exit 0) | Require slippage_base_bps>0 for prop-firm live-broker configs | Now |
| P3-07 | S3 | Sim instant DD/daily-loss trip vs live 3-pt-median+2-bar smoothing; empirical bound never computed for decay01 run | risk_shaping_wrapper.py:220-245; audit_daily_loss_gate_alignment.py:38-42 | Re-run alignment audit with decay01 run added | Now |
| P3-08 | S4 | **POSITIVE:** BUG-04 structurally inapplicable to V7 (mark-to-market reward, no switch term) | continuous_swing_env.py:347,357,385 | Document V7 invariant | Now |
| P3-09 | S4 | **POSITIVE:** Step ordering causally sound (decide t, earn t→t+1); RCA-confirmed | continuous_swing_env.py:250,351; sim_live_gap.md:97 | Add ordering comment + golden timing test | Now |
| P3-10 | S2 | Fill model is taker-at-close + single flat-bps; no LOB/queue/partial-fill, no vol/size-dependence; live can miss fills | continuous_swing_env.py:55-58,364-366; bybit_perp_broker.py:337-339 | Vol/size-scaled slippage + unfilled-order prob (keep flat as floor) | Research |
| P3-11 | S4 | **POSITIVE:** SHORT-ACCT satisfied — symmetric, debt-free, compounds on equity; scale-invariant | continuous_swing_env.py:385,388 | Add SHORT-ACCT golden case | Now |
| P3-12 | S3 | DSR + normalize_accumulated_reward: sums per-bar DSR then ÷(1+skipped); not DSR of aggregated return; reward magnitude correlates with unobservable skip count | signal_gated_wrapper.py:143-144; dsr.py:89-91 | Ablate true/false; document intended credit semantics | Research |
| P3-13 | S3 | PF-XCHECK inert for V7 (close-only pricing, no mid path) | continuous_swing_env.py:351; multiscale_handler.py:419 | Scope out in CLAUDE.md or add (high+low)/2 cross-check | Now |

### P4 — Reward design
*DSR formula is textbook-correct; the gaps are no turnover penalty, cost-model staleness, and reward↔PF misalignment.* (11 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P4-01 | S2 | No explicit turnover penalty; DSR scale-invariance mutes the in-reward cost disincentive; only brake is static deadband | continuous_swing_env.py:359-370; dsr.py:83-96; randd_archive/2026-03.md:86 | Add reward.turnover_penalty (off by default), tune as HPO axis | Research |
| P4-02 | S2 | Deployed policy trained on cheap cost (0.0005, slip 0); S552 correction uncommitted + un-retrained (~10.5 bps gap) | git HEAD config; live_sg1_btc_bybit.yaml:139,150,116 | Commit cost block, propagate to live, re-run HPO→L1→WF | Next |
| P4-03 | S2 | HPO objective=PF but reward=DSR; documented rho≈0.08 weak alignment | run_full_pipeline.py:189; protocol_v2.md:272; randd_archive:86 | Reward-ablation; demote DSR if rho stays <0.3 | Research |
| P4-04 | S3 | dsr_eta half-life ~825 bars = 27% of 3000-bar episode → unconverged DSR for a large fraction of each random-start episode | objective.py:143; yaml:130 (eta 0.000843) | Prime A/B at reset or constrain eta lower bound | Next |
| P4-05 | S3 | normalize_accumulated_reward averages summed DSR over hold (= P3-12) | signal_gated_wrapper.py:116,143-144 | Accumulate raw R_t then compute DSR once, or drop the division | Next |
| P4-06 | S3 | DD penalty (equity-fraction quadratic, scale 5.0) summed onto dimensionless DSR; scale never HPO-tuned | risk_shaping_wrapper.py:252-256; objective.py search space | Add dd_penalty_scale to HPO or normalize terms | Next |
| P4-07 | S3 | No numeric DSR formula regression test (only plumbing tests) | tests/test_prism_rcrp_dsr.py | Add hand-computed DSR assertion + warmup/var-guard cases | Now |
| P4-08 | S4 | DSR clip [-10,10] and raw-PnL clip [-50,50] hardcoded; dsr_scale not searched | dsr.py:96; continuous_swing_env.py:380 | Expose as config; log clipped-fraction | Now |
| P4-09 | S4 | **POSITIVE:** DSR faithful to Moody & Saffell (pre-update A/B, 3/2 exp, 0.5 factor, var clamp, guard) | dsr.py:83-96 = FORMULAS.md MATH-R01 | Lock with P4-07 test | Now |
| P4-10 | S4 | **POSITIVE:** Reward cost & equity use one price_return/total_delta; DSR resets cleanly per episode | continuous_swing_env.py:351,385-388; dsr.py:49-54 | Preserve single-source pattern | Now |
| P4-11 | S4 | **POSITIVE:** Leverage-invariant deadband prevents fee-drag PF artifact | continuous_swing_env.py:281-289 | Scale any future turnover penalty by max_leverage too | Now |

### P5 — Training & HPO
*BUG-01 + trade-level PF + min-trades are good; three documented gates have no code consumer.* (10 confirmed, 1 NEEDS-DATA, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P5-01 | S2 | Training-health kill gates (entropy_floor/q_div_max/action_sat_max) declared everywhere but NEVER read; validator only WARNs if missing; config values ≠ protocol defaults | yaml:116-118; protocol_v2.md:278-280; no Python consumer | Implement kill in objective.py mirroring min_trades; reconcile defaults | Next |
| P5-02 | S2 | HPO trains ~1.9× multiplicity (500K/262K) — far below productive [15,40] and even <10× WARN; HP ranked on under-trained noise (claim that #44 is luck = NEEDS-DATA) | yaml:111; protocol_v2.md:182 | Raise steps_per_trial or document override + measure HPO↔L1 rank rho | Research |
| P5-03 | S2 | The §3.5.5 multiplicity check 'enforced by validate_config' does not exist in the validator (verified by 2 runs) | protocol_v2.md:222,226; validate_config.py 379-405,555-577 | Add check_multiplicity to STAGE_CHECKS[hpo, l1-multiseed] | Now |
| P5-04 | S3 | reward↔PF Spearman diagnostic runs only in serial pipeline; sg1-btc ran distributed → never computed | evaluate.py:334; distributed_hpo_worker.py:339-342 | Move into shared finalization; persist rho to best_params | Now |
| P5-05 | S3 | Trial training unseeded (no torch/np/env seed) → HPO non-reproducible, ranks on one noisy realization | objective.py:430,443; sac_trainer.py (no seeding) | Seed per-trial or train K=2-3 seeds/trial | Next |
| P5-06 | S3 | HPO selects on a single fixed 3-month val window; val/test divergence explicitly unmeasured | objective.py:395-400; rehpo report:83-84 | Score on rolling sub-windows; assert val/test gap | Research |
| P5-07 | S4 | No hpo.search_space declared → falls back to hardcoded ranges; lr_alpha pinned at 2e-6 floor (optimum at edge) | yaml (no search_space); objective.py:126-144; best_params:7 | Declare search_space; widen lr_alpha floor | Now |
| P5-08 | S4 | **POSITIVE:** NopPruner (no early-kill) correct for SAC convergence; consistent across paths | coordinator:1030; run_full_pipeline.py:103 | Keep until P5-05 lands | — |
| P5-09 | S4 | **POSITIVE:** BUG-01 (objective=PF, trade-level, reward locked) enforced across config/validator/both paths | yaml:109; validate_config.py:386-390; evaluate.py:253-260 | Keep | Now |
| P5-10 | S4 | min_trades gate value (30) contradicts docstring (100); low floor for a 3-min strategy | objective.py:459; evaluate.py:19 | Make gates.hpo_min_trades; fix docstring | Now |
| P5-11 | S4 | **POSITIVE:** Signal gate applied to train AND eval via shared make_env; thresholds identical across stages | env_factory.py:137-140; objective.py:363,395 | Centralize gate block to prevent drift | Now |

### P6 — Validation L1 multiseed (Stage 2)
*Well-specified protocol, but the gate code is XAUUSD-hardcoded and the decay01 verdict came from a manual recovery; seeds converged → CV measures degeneracy.* (9 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P6-01 | S2 | L1 gate computed in committed code ONLY for XAUUSD (hardcoded constants, top-3 by TEST PF ≠ configs' val_argmax); sg1-btc verdict = manual 'S542 recovery' | auto_queue_wf_after_l1.py:44-64,158-191; seed_report.json:7 | Extract workstream-agnostic eval_l1_gate.py reading gates from config | Next |
| P6-02 | S2 | N=10 seeds converged (test-PF CV 0.82%, action corr >0.83) → CV gate measures seed-convergence, ensemble adds ~0 diversification (SOLO_BEST_FALLBACK) | seed_report.json:24-30; verdict.json:36-47,142-158 | Add inter-seed action-corr floor to Stage-2 gate; surface full NxN | Research |
| P6-03 | S3 | val_argmax top-3 ([456,42,2025]) nearly disjoint from test top-3 ([3141,456,1337]); seed 42 is worst test seed; rank rho≈-0.29 | seed_report.json:10-31 | Use diversity selector on full N=10; document low-CV caveat | Next |
| P6-04 | S3 | eval_distribution baseline covers only 3 selected seeds (protocol: all N); challenge_target_hit_rate_by_seed absent | ensemble seed_report.json:5-9; protocol_v2.md:290-291 | Regenerate full-N distribution + hit-rate; promote WARN→FAIL | Now |
| P6-05 | S3 | ens_pf_weighted uses TEST PFs as weights (latent test→deploy leak) | yaml:177; ensemble_eval.py:155-167 | Use VAL PFs or exclude rule from val bake-off | Now |
| P6-06 | S4 | N=10 ran in 3 batches (5+4+1) over 3 days; static parquet so no data drift; provenance bookkeeping only | seed_report.json created_at | Record per-batch parent-run id | Now |
| P6-07 | S3 | Stage-2.5 bootstrap on thin ~2926-bar (2wk) single-regime window; P(PF)=0.65 low-power; no window-widening rule | verdict.json:50,39 | Lengthen window or document WF bootstrap as binding; add min-bars warning | Next |
| P6-08 | S3 | validate_config --stage l1-multiseed checks only key PRESENCE, never CV/all-profitable computation | validate_config.py:555-577,1228 | Add --seed-report arg that recomputes the gate | Next |
| P6-09 | S4 | **POSITIVE:** Pre-committed escalation band, val-split rule selection, block-bootstrap primary, overlay regression test (4×4 decision-matrix test already exists) | yaml:131; ensemble_eval phase-2; tests/test_stage_2_5_resolver.py:67-96 | Preserve | — |

### P7 — Walk-forward + stress (Stage 3)
*Structurally reasonable but inherits cost-free seeds, has no fixed-lot stress, truncates DD at 8%, and grades the wrong aggregation rule.* (13 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P7-01 | S2 | Cost-free top-3 [456,42,2025] inherited verbatim into cost-corrected WF (config header admits §4.5 Stage 2.5-R trigger) | l1_decay01.yaml:75; wf yaml:15-18,190 | Re-run L1 under corrected cost, re-derive seeds before reading WF | Next |
| P7-02 | S2 | Seed selection statistically null (test-PF CV 0.8%); best-test seed 3141 not picked | seed_report.json:24-30 | Diversity/DD-tail tiebreak when CV<floor | Research |
| P7-03 | S2 | Fixed-lot stress sub-report (mandated post-S466) ABSENT; eval terminates at 8% trailing DD → per-fold DD and G4 buffer structurally capped | protocol_v2.md:438-441; risk_shaping_wrapper.py:220-227; config_utils.py:126 | Add no-early-term stress pass + stress_dd/leverage gates | Next |
| P7-04 | S3 | 94% train overlap (18mo train, 1mo step) → 8 correlated trials; G5 CV optimistic | wf yaml:90-93; splitter.py:56-77 | Report effective-fold-adjusted CV or pooled-OOS bootstrap | Research |
| P7-05 | S3 | WF gates evaluate ens_agreement (gates YAML) but Stage 2.5 chose ens_mean/SOLO — grading the wrong object (proven: WF G3 median = ens_agreement PFs) | ensemble_eval.py:449; gates.yaml:19; verdict.json:26; sg1_btc_ensemble_wf/all_folds_long.csv | Read chosen_rule/solo from verdict.json | Now |
| P7-06 | S3 | Block-bootstrap only at Stage 2.5 single window; WF (8×14,880 bars) judged by point-estimate G1-G5 only (matches protocol, but richer evidence unused) | verdict.json:36-47; ensemble_eval.py:297-404 | Add pooled-OOS block-bootstrap to WF verdict | Next |
| P7-07 | S3 | G4 Velotrade buffer computed from the 8%-truncated DD → can never read <~2pp; near non-binding | ensemble_eval.py:271,277-282; gates.yaml:96 | Tie G4 to P7-03 stress trajectory | Next |
| P7-08 | S3 | No 'every fold profitable' / per-fold return>0 gate; no PF-XCHECK; validator only checks wf_windows≥4 | protocol_v2.md:436-437; ensemble_eval.py:439-570; validate_config.py:748-751 | Add total_return>0 (≥7/8) + close-vs-mid PF gate | Now |
| P7-09 | S4 | No per-fold trade-count / active-days floor (min_trades_window:20 precedent exists) | sg1_arm_gate_backtest.py:226; gates.yaml:163 | Add min_trades_per_fold | Now |
| P7-10 | S2 | Cost-corrected WF mid-training; no ensemble_wf verdict exists; only completed verdict is cost-FREE Stage 2.5-R | results/sg1_btc_velotrade_decay01_ensemble_wf absent; partial manifests 20260529_* | Block promotion read until WF verdict exists; stamp cost model into verdict.json | Now |
| P7-11 | S4 | **POSITIVE:** LEAK-1 compliant norm_cutoff=val_end == test_start; warmup carry | ensemble_eval.py:195,293; runtime fold_0:9-10 | Assert norm_cutoff≤test_start | Now |
| P7-12 | S4 | **POSITIVE:** Gates fully externalized; _require_gate hard-fails on missing field | gates.yaml:1-14; ensemble_eval.py:409-425 | Stamp gates git SHA into verdict | Now |
| P7-13 | S4 | **POSITIVE:** Sequential fold dispatcher with fresh-subprocess WandB polling + abort-on-fail | launch script:259-313,249-250 | Persist resolved checkpoint paths into manifest | Now |

### P8 — Recent-OOS & compliance (Stage 4)
*The weakest, stalest link: OOS verdict tests a RETIRED seed under falsified cheap cost, the CLI stage is unimplemented, and absurd metrics pass.* (12 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P8-01 | **S1** | On-file OOS verdict tests RETIRED seed-42 SOLO, not deployed DECAY-01 seed-456; live policy has NEVER passed a Stage-4 gate | oos_backtest.yaml:2; verdict.json:4; aggregate script:70; live config:72 | Re-point OOS config + aggregator at seed-456 bundle; re-run | Next |
| P8-02 | **S1** | OOS runs falsified cheap cost (0.0005, slip 0); on-file return 1.35M% is a frictionless artifact | oos_backtest.yaml:41; continuous_swing_env.py:365-366,387-388 | Set taker 0.00055 + slippage 5.0; inherit cost from bundle manifest | Now |
| P8-03 | S3 | Bucket thresholds hardcoded (0.20/0.40) in aggregator; no oos_* keys in gates YAML (anti-pattern) | aggregate script:126-130; gates.yaml | Add oos block; read from YAML | Now |
| P8-04 | S2 | Bucketizer ratios OOS PF vs L1 VAL PF (or hand-typed baseline), not WF-median PF as protocol mandates | aggregate script:187,114; verdict.json:16; protocol_v2.md:447 | Use WF-median PF baseline | Now |
| P8-05 | S2 | Stage 4 not wired: run_full_pipeline has no --stage; STAGE_CHECKS['oos']=[]; documented CLI non-existent | run_full_pipeline.py:757-771; validate_config.py:1235 | Implement oos stage check (ckpt/cost/report match) or fix doc | Next |
| P8-06 | S3 | OOS window ~96d fixed historical block, not rolling 60d; misses recent (May-2026) regime | oos_backtest.yaml:17-18; protocol_v2.md:444 | Parameterize window from manifest last_ts | Now |
| P8-07 | S2 | Bucket accepts 1.35M% return as HOLD; no sanity bounds; PF-XCHECK skipped | verdict.json:12,19; summary.md:25 | Add plausibility bounds + mid-vs-close PF | Now |
| P8-08 | S3 | Velotrade OOS compliance skipped (compliance_must_pass:false); min-active-days possibly dropped (NEEDS-DATA: rulebook) | gates.yaml:96-99; ensemble_eval.py:528-541 | Verify Velotrade 2-Step rulebook; set active_days_min if applies | Research |
| P8-09 | S3 | FTMO consistency denominator uses positives-only (generous); matters for FTMO siblings | ftmo_compliance_report.py:227,238-243 | Make denominator selectable; default stricter net-total | Research |
| P8-10 | S3 | Aggregator picks glob-latest checkpoint + latest shared dir (mtime heuristic) — fragile/non-deterministic | ensemble_eval.py:65-76; aggregate script:255,261-263 | Resolve by bundle SHA; workstream-scoped dir | Next |
| P8-11 | S4 | **POSITIVE:** BUG-03 (hindsight=0) + full-window determinism + LEAK-1 enforced via shared prep | config_utils.py:143-147; summary.md:25 | Add post-prep assertion | Now |
| P8-12 | S4 | **POSITIVE:** Deployed=solo and OOS=single-checkpoint → architecture-matched (no solo/ensemble mismatch) | live config:62-63,73 | Maintain alignment | Now |

### P9 — Ensemble formation (Stage 2.5)
*Mechanics excellent; diversity selector is dead code, seed_pfs are metric-inconsistent, and the WF rule mismatch is proven against artifacts.* (13 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P9-01 | S2 | Diversity-aware selector never does selection — always called on pool==K==3 (note_pool_too_small) | sg1_xauusd_ensemble_eval.py:1123-1129; ensemble_eval.py:701-704; verdict.json:164-166 | Move diversity selection upstream to full N=10 pool | Next |
| P9-02 | S2 | Top-3 by val_argmax on 15-day window where all seeds within noise; #2 val pick is worst test seed | seed_report.json:10-31 | Lengthen windows or diversity-fallback when CV<floor | Next |
| P9-03 | S3 | ens_pf_weighted weighted by stale L1 trade-level PFs (~3.17) that disagree ~29% with the script's bar-PFs (~2.46) | yaml:177-180; solo_456_metrics.json; sg1_arm_gate_backtest.py:164-171 | Weight from recomputed bar-PF; assert config match | Now |
| P9-04 | S3 | val reused 3×; chosen_rule = argmax over 4 noisy PFs (top-2 within 0.16%) with no multiplicity correction | sg1_xauusd_ensemble_eval.py:1033-1034; verdict.json:21-24 | Require argmax margin; default to ens_median on tie | Now |
| P9-05 | S2 | decay01 top-3 selected under wrong cost model, inherited into cost-corrected WF (confirmed firing on inherited seeds) | wf yaml:3-18; manifest 20260529_191828 | Re-run L1 at corrected cost; re-pick | Next |
| P9-06 | S2 | Canonical gates aggregation_rule (ens_agreement) ≠ deployed rule (ens_mean) — WF grades ens_agreement (proven: G3 median = ens_agreement PFs); sibling dir named *_stale_ens_agreement* confirms recurring | gates.yaml:19; verdict.json:26; sg1_btc_ensemble_wf/* | WF reads chosen_rule from verdict; single source of truth | Now |
| P9-07 | S3 | Diversity proxy = action-signal correlation, not return correlation; lambda mixes incomparable units | sg1_xauusd_ensemble_eval.py:647-689,753 | Option for return-corr; normalize PF before lambda | Research |
| P9-08 | S3 | Agreement-decay monitor no-op for ens_mean (only consensus rules); current rule choice determines monitoring posture | agreement_decay.py:43; live_engine.py:179-181 | Document mean-family relies on ActionDriftTracker; add rule-agnostic utilization floor | Now |
| P9-09 | S3 | Block-bootstrap (primary promote gate) on single 2926-bar window (~54 blocks) → within-regime noise only | verdict.json:48-53; sg1_xauusd_ensemble_eval.py:488-489 | WF cross-fold bootstrap binding; n-effective-blocks floor | Now |
| P9-10 | S4 | **POSITIVE:** Paired stationary block-bootstrap, MDD-less-negative, NO_DATA guard, deterministic seed | sg1_xauusd_ensemble_eval.py:497-508,478-486 | Emit n-effective-blocks | Now |
| P9-11 | S4 | **POSITIVE:** Gates hard-fail-on-missing; bundle per-file SHA256 + MISSING sentinel | ensemble_eval.py:409-425; write_ensemble_swap_bundle:844-872 | Keep | Now |
| P9-12 | S4 | **POSITIVE:** NaN no-consensus sentinel (Fix-2) distinguishes hold from silent fail; dual-signal monitor | ensemble_eval.py:152,218-222; agreement_decay.py:202-355 | Bake all baselines post-Fix-2 | Now |
| P9-13 | S4 | Cadence-arithmetic inconsistency: gates YAML says both '3-min' and '5-min'; retrain windows (2000 bars '~7d at 5-min') = 4.2d at true 3-min | gates.yaml:118,134,155,166 | Fix comments to 3-min; resize windows if 7d intended | Now |

### P10 — Live / drift / sim-to-live consistency
*Capital protected, but the deployed policy is cost-unaware and decayed, and the retrain-cadence machinery is documentation.* (12 confirmed, 0 refuted)

| ID | Sev | Finding | Evidence | Improvement | Effort |
|----|----|---------|----------|-------------|--------|
| P10-01 | **S1** | Live cost model ≠ corrected training cost; policy seed-selected cost-free; S552 correction uncommitted+un-run; broker charges 0.00055 + 5bps cross | live config:139,150; l1_decay01.yaml:75; sim_live_gap.md:125; git diff (19 unstaged) | Commit cost; bump live taker→0.00055; re-run L1+2.5+WF to re-select | Next |
| P10-02 | S2 | 28d sub-5m staleness cadence absent from gates YAML (carries 90d); deployed fold_07 train_end 2026-02-01 is ~118d stale | gates.yaml:159-171; protocol_v2.md:620-624; live config:107-113 | Add max_age_days:28 + train_end field | Now |
| P10-03 | S3 | §4.5 retrain triggers schema-disjoint: check_retrain_triggers reads retrain_policy:, gates YAML uses gates.retrain.* — sg1-btc live config lacks retrain_policy block; cost_drift unimplemented | check_retrain_triggers.py:255-257; live config (no block); gmgp1 configs have it | Add retrain_policy to live config; reconcile schemas | Next |
| P10-04 | S3 | cost_drift trigger (1.20×) would have caught the ~2.1× gap but is not computed (order_fee logged, no rolling ratio) | live_engine.py:2829; gates.yaml:170; no consumer | Add rolling realized-cost-vs-config tracker → drift/cost_ratio | Next |
| P10-05 | S3 | Mode-A vs Mode-B drift diagnostic (trailing-30d std of live_frac) not automated; mode discrimination is manual | no impl (grep); action_drift.py:391-404 recommends only; recal:238 'next deliverable' | Implement std-of-live_frac classifier in DriftReport | Next |
| P10-06 | S2 | norm_warmup pkl outside bundle SHA chain AND resolve_norm_warmup_path fail-OPEN → silent degrade to legacy EMA-Z (S509 skew) | live config:74; ensemble_bundle.py:236-244; live_obs_builder.py:78-83 | Fail-closed for prop-firm; or hash pkl in bundle | Now |
| P10-07 | S3 | Drift-threshold precedence reads top-level drift: before gates.drift:; works only by omission; adding a key there silently shadows gates (S551-cont-9 class) | live_engine.py:124-128; live config:207-245 | Single source: read only gates.drift OR fail on dual-declare | Now |
| P10-08 | S4 | Solo deploy carries ensemble aggregation_rule + agreement-decay machinery (no-op, correct by design) | live config:73; agent_loader.py:251-257 | Re-bake baseline on any future consensus swap | Now |
| P10-09 | S4 | **POSITIVE:** Bundle SHA256 chain all-or-nothing (per-ckpt + config), operator swap handshake for prop-firm | ensemble_bundle.py:152-234; swap_handshake.py:159-201 | Extend chain to norm pkl (P10-06) | Now |
| P10-10 | S4 | **POSITIVE:** Sim-live RCA reconciles Gap A to the penny + de-dup ledger; Gap B isolated as cadence problem | sim_live_gap.md:64-73,208-216,107-115 | Encode 'soak blocked on cost-corrected WF PASS' as a gate | Next |
| P10-11 | S4 | **POSITIVE:** Action-drift baseline regime-conditioned with Fix-2 no_consensus_frac marker | baseline ensemble_report.json; live_engine.py:204-216 | Preserve fields on recal | Now |
| P10-12 | S3 | Live taker_fee 0.0005 under-models broker 0.00055 → engine fee/PV estimate optimistic by 0.5 bps/leg | live config:139,150; bybit_perp_broker.py:516 | Set 0.00055 or read broker's exposed rate | Now |

### P11 — External SOTA benchmark
*Above median academic practice on costs/normalization/bootstrap; the two material un-mitigated gaps are overfitting statistics and purged/embargoed CV.* (10 confirmed, 0 refuted) — detailed in the Gap Analysis table below.

---

## Prioritized Roadmap

Ordered within bucket by (robustness × consistency impact) / effort.

### NOW — config / test / doc, no retrain

| # | Change (file · key/function) | Why | Confirming metric |
|---|------------------------------|-----|-------------------|
| N1 | `validate_config.py` → add `check_multiplicity` to STAGE_CHECKS[hpo, l1-multiseed]; add slippage>0 check to cost-model section | The two protocol guardrails that would have caught both the under-training (1.9×) and the zero-slippage selection are inert | Running validator WARNs on 1.9× and exits non-zero on missing slippage |
| N2 | `configs/sg1_btc_velotrade_ensemble.gates.yaml` → add `retrain.max_age_days: 28` + `oos:` block (hold/watch ratios); fix 5-min cadence comments to 3-min | 3-min strategy run on a quarterly freshness assumption; deployed ckpt 118d stale; OOS thresholds hardcoded | `grep max_age_days` → 28; staleness check flags current deploy |
| N3 | `sg1_btc_velotrade_ensemble_eval.py` (WF) → read `chosen_rule`/SOLO from Stage 2.5 `verdict.json` instead of `gates_cfg['aggregation_rule']` | Proven: WF grades ens_agreement while 2.5 chose ens_mean — Stage 3 validates the wrong object | WF verdict's evaluated rule == 2.5 chosen_rule |
| N4 | `tests/` → add `test_continuous_swing_env.py` (golden trajectory) + `test_multiscale_no_lookahead.py` (truncation + mid-bar parity) | Most accounting-critical code is mock-tested only; no test catches the P2-01 coarse leak | Reverting an audit fix / introducing a leak fails the test |
| N5 | `aggregate_q1_2026_oos_blindspot.py` + `oos_backtest.yaml` → cost 0.00055+5bps; baseline = WF-median PF; add sanity bound on total_return; PF cross-check | OOS green-lights a 1.35M% frictionless artifact against a noisy single-window baseline | OOS return collapses to realistic; baseline = WF median; absurd return → WARN |
| N6 | `live_engine.py` `_pick` → read only `gates.drift`; `live config` → taker 0.00055 | Dual-source threshold precedence is the S551-cont-9 / 2026-05-28 drift_crit class; engine fee optimistic | validate_config fails on dual-declare; engine fee == broker fee |
| N7 | `tests/test_prism_rcrp_dsr.py` → numeric DSR formula assertion | Insidious bug class — a wrong exponent/factor silently corrupts the training signal with no failing test | Flipping the 3/2 exponent fails the new test |

### NEXT — requires a stage re-run

| # | Change | Why | Confirming metric |
|---|--------|-----|-------------------|
| X1 | **Re-run L1 N=10 → Stage 2.5 → WF under corrected cost** (taker 0.00055 + slippage 5.0 in `l1_decay01.yaml` AND committed `wf` config); re-select top-3/solo | The headline fault — deployed policy never selected net-of-cost; cost-corrected WF currently inherits cost-free seeds. Gate for everything else | Cost-corrected verdict PF (expect <2.53); chosen seed re-derived from cost-aware cohort; per-leg cost reconciles to live |
| X2 | Make coarse-bar alignment causal (`multiscale_handler._resample_ohlcv` + `_scale_index_map` AND `live_obs_builder`); fold into X1 re-run | S1 forward look-ahead inflates all sim metrics and diverges train↔live; fixing it in the cost re-run avoids a second retrain | New no-lookahead test passes; sim PF drops on coarse-heavy folds and tracks live more closely |
| X3 | Add fixed-lot stress pass + `stress_dd`/`stress_leverage` gates to WF eval; tie G4 to stress trajectory | DD truncation at 8% makes per-fold DD and the only Velotrade gate near non-binding (S466 mode) | Un-truncated worst DD exceeds reported trailing DD; G4 recomputed from honest DD |
| X4 | Move diversity selection upstream to full N=10 pool (L1 reconcile) + add inter-seed action-corr floor | Diversity selector is dead code on pool==K; ensemble adds ~0 diversification | diversity_audit.pool_size==10; selected-seed corr <0.7; bootstrap P(PF) rises |
| X5 | Implement training-health kill gates in `objective.py` (entropy/q-div/action-sat) | Degenerate collapsed trials can win HPO; the protocol's only pathology filter doesn't run | A known-collapsed checkpoint is killed (status=killed_health) |
| X6 | Wire retrain automation for sg1-btc (add `retrain_policy` block; reconcile with gates.retrain; add cost_drift tracker → WandB `drift/cost_ratio`) | No automated degradation detector for the strategy whose dominant risk is alpha decay | check_retrain_triggers returns a real decision; injected 5.5bps fills fire cost_drift |

### RESEARCH — open questions

| # | Question | Why it matters |
|---|----------|----------------|
| R1 | Does PF=2.47 survive a Deflated-Sharpe / PBO (CSCV) computation given 50×top-3-of-10×best-of-7×8-fold multiplicity? | The single biggest missing overfitting control; quantifies selection-bias risk before live capital |
| R2 | Does a CVaR / Calmar reward branch cut worst-fold DD without regressing median PF vs DSR? | Aligns the objective with the binding Velotrade trailing-DD cap; DSR is variance-symmetric |
| R3 | Does a turnover-penalty term (or DSAC / regime-adaptive DSR) reduce cost-regime fragility and seed correlation? | Reward currently does not train the policy to economize cost; seeds correlated 0.85 |
| R4 | Does purged/embargoed CPCV yield lower PBO than the overlapping WF for sg1-btc? | SOTA financial CV; current buffer_days:0 has no embargo at the feature lookback boundary |
| R5 | Is the val/test seed-rank decoupling fundamental at 3-min, or an artifact of 15-day windows? | Determines whether seed identity is ever load-bearing or windows must lengthen |

---

## External Benchmark Gap Analysis (P11)

| Best practice (source) | Do we do it? | Gap | Recommendation |
|------------------------|--------------|-----|----------------|
| Cost embedded in per-bar reward (cost-aware DRL) | **Yes** | None — honest all-taker model reconciled to broker | Preserve; verify cost_drift trigger is actually consumed (it is not — wire it) |
| LEAK-1 causal EMA-Z normalization, per-split cutoff | **Yes** | Feature-side only; selection-side has no embargo | Pair with purged/embargoed CV (R4) |
| Stationary block bootstrap for MDD/PF CIs (Politis-Romano) | **Yes** | Only paired ens-vs-solo, single window | Add single-sample P(PF>1) + pooled-OOS WF bootstrap |
| Multi-seed selection + training-budget multiplicity rule | **Yes** | Budget multiplicity ≠ statistical-selection multiplicity; seeds correlated 0.85 | Add seeds_per_trial≥2; enlarge/diversify pool |
| Deflated Sharpe / PBO / CSCV (Bailey & López de Prado) | **No** | Rejected as *replacement*, never added as *complement* | Add audit-only `overfitting_audit` block; gate PBO<0.5, DSR p<0.05 (R1) |
| Purged k-fold / CPCV with embargo (López de Prado) | **No** (overlapping WF, buffer_days:0) | Boundary leakage + 94% train overlap | Embargo ≥ feature lookback (~2-3d); add CombinatorialPurgedSplitter (R4) |
| Risk-sensitive reward (CVaR / Calmar / distributional RL) | **No** (DSR only; DSAC unused here) | Sharpe-family tolerates rare large DD that breaches the prop cap | Trial CVaR/Calmar reward branch + DSAC arm (R2/R3) |
| Domain randomization / obs-noise / exec-failure stress (Pinto, Tobin, SA-MDP) | **No** (diagnosed, not built) | Path-fragility invisible; v2.7 plan exists (S553) | Implement v2.7-A sensitivity sweep then Stage 3.5 obs-noise (R3) |
| Regime-conditioned policy | **No** (regime_adaptive_dsr OFF) | Single policy must generalize across regimes | Enable regime-adaptive DSR + re-HPO eta |
| Pre-committed, versioned, no-post-hoc gate thresholds | **Yes** | None | Add DSR/PBO thresholds to the same pre-committed YAML |

---

## Appendix — Verification Log

**Per-pillar verdict counts** (skeptic-adjudicated):

| Pillar | Confirmed | NEEDS-DATA | Refuted |
|--------|-----------|-----------|---------|
| P1 Data | 11 | 0 | 0 |
| P2 Features | 9 | 0 | 0 |
| P3 Env | 13 | 0 | 0 |
| P4 Reward | 11 | 0 | 0 |
| P5 HPO | 10 | 1 (P5-02) | 0 |
| P6 L1 | 9 | 0 | 0 |
| P7 WF | 13 | 0 | 0 |
| P8 OOS | 12 | 0 (P8-08/09 code-confirmed; external-rulebook claims flagged NEEDS-DATA inline) | 0 |
| P9 Ensemble | 13 | 0 | 0 |
| P10 Live | 12 | 0 | 0 |
| P11 Benchmark | 10 | 0 | 0 |
| **Total** | **123** | **1** | **0** |

**REFUTED findings:** None. No finding was dismissed outright. The skeptic instead **downgraded severity** on several and **corrected evidence** on many; the notable adjudications a reader should see:

- **P3-01 / P4-02 / P7-01 / P9-05 / P10-01 (the cost fault)** — all CONFIRMED. The deployed object is **solo seed-456**, not the 3-seed ensemble some finders implied; the live venue is **Bybit demo/paper** and promotion is gated ("soak NOT promotable", S552). Several finders rated S1; skeptic split: P3-01/P8-01/P8-02/P10-01 held at **S1** (will-cause / has-caused live-style loss + frictionless artifact), while P4-02/P7-01/P9-05 lowered to **S2** (demo, not real capital; documented known issue).
- **P2-09 (log_return clip caveat)** — the flash-crash saturation worry was **data-refuted**: 0 / 408,479 bars exceed |log_ret|>0.10 (max 0.0481); positive finding stands, caveat materiality nil.
- **P5-02 (HPO under-training → trial #44 is luck)** — arithmetic CONFIRMED (1.9×) but the causal ranking claim is **NEEDS-DATA** (requires the proposed HPO↔L1 Spearman study); held S2 for the unverified-ranking risk.
- **P8-03 (hardcoded OOS thresholds)** — downgraded **S2→S3**: functionally equivalent to the protocol 0.8/0.6 ratios, a declarativeness gap not a robustness break.
- **P3-04 (PRISM 'divergence')** — partially corrected: PRISM is **disabled** (inert), so it is not an active live divergence; the genuine gap is the sim-only ATR-cap.
- **P10-03 (no automated degradation detector)** — the blanket claim was **overstated**: `check_retrain_triggers.py` exists and is wired into sibling configs; downgraded S2→S3 (schema-reconciliation + per-config-wiring gap, not absence of all automation).
- **Citation corrections carried forward:** WARMUP_BUFFER value is **200** (finder mis-stated 159 = line number); splitter path is `sharpen/data/splitter.py` (not `training/`); FORMULAS.md is user-scope (`~/.claude/skills/math/`); mid_price exists in 20 sharpen files (not AlphaSeek-only); WF baseline doc lives in live config header, not a non-existent `_ensemble_wf` verdict; P9 4×4 decision-matrix regression test **already exists** (`tests/test_stage_2_5_resolver.py`).

**Severity distribution (confirmed):** S1 ×4 (P2-01, P3-01, P8-01, P8-02, P10-01 — note P3-01/P10-01 are the same cost fault surfaced in two pillars) · S2 ×30 · S3 ×41 · S4 ×48 (incl. ~20 explicit positives credited).

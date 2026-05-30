# Training → Live Protocol v2.6

> **Status:** Active. Standardizes the training-to-live workflow across all FinRL-Pro_DS workstreams (GMGP1, SG-1, CMGP1, AlphaSeek, Funding-Arb).
> **Reference run:** GMGP1 staged approach. **Anti-pattern:** AlphaSeek `k28l6ef8` monolithic 5.7-day run.
> **Owner:** R&D. **Last updated:** 2026-05-30 Session 553 (Stage 2.5-R Sensitivity Audit — Protocol v2.7-A C1-C5 shipped).
>
> **Version history:**
> - **v2.6** (2026-05-30, S553) — Protocol v2.7-A Stage 2.5-R **Sensitivity Audit** sub-step added (new §4.5.1). 3×3 (`deadband_threshold` × `max_leverage`) eval-time config-sensitivity sweep on PROMOTE candidates; edge-stability gate `pf_inner_min / pf_center ≥ floor` (default **0.70**) across 8 deployable, non-halted NEIGHBOR cells. Verdict `schema_version` bumped to `"2.6"` with additive `sensitivity_audit` block + new top-level `deployed_config` field. Phase α (S553+) — keys present in all 9 ensemble gates yamls, `sensitivity_audit_required: false`, validator dormant for legacy `protocol_version != "2.6"` configs. Phase β (post-backfill calibration) — operator bumps 5 active live workstreams to `protocol_version: "2.6"` + `sensitivity_audit_required: true`. New validator rule `check_sensitivity_audit` enforces SENS-1 (grid center matches deployed config). Standalone CLI `scripts/stage_2_5_r_sensitivity_audit.py` is the canonical entrypoint (mirrors `recompute_stage_2_5_verdicts.py` precedent — `run_full_pipeline.py --stage sensitivity-audit` not added; monolithic launcher does not support `--stage` dispatch). Backfill sidecar `scripts/backfill_deployed_config_field.py` populates the new field on legacy v2.5 verdicts. **Researcher artifact** `.agent/artifacts/mc_robustness_methods_research.md` ranked Method #5 CONDITIONAL-GO. **Architecture artifact** `.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md` (8 ADRs, 3 SENS-* invariants, 5-commit breakdown). **Forward-declared:** v2.7-B (Stage 3.5 obs-noise robustness, Method #4 GO) and v2.7-C (Stage 3 exec-failure stress sub-block, Method #3 CG re-framed) are deferred follow-ups — slots reserved in §4. **NO live-strategy impact during Phase α** — gates schema is additive, validator is dormant. See `decision_protocol_v25_bootstrap_primary.md` for the v2.5 bootstrap precedent that v2.6 extends.
> - **v2.5.1** (2026-05-21, S545) — Training-budget & data-window rule (new §3.5) locking in findings from six studies: GMGP1 Volume Study v1 (Gold), Data-Window Study (BTC Axis A), Volume Study v2 (BTC, 2M ceiling), Gold Steady-State, SG-1-XAUUSD Volume Study v2 Phase 1 + Phase 2, SG-1-BTC A1 extended audit. Empirically-derived **multiplicity rule** (`total_timesteps / bars_in_train_window ∈ [15, 40]`) and **calendar-anchored data window** (NOT bar-anchored — preserves regime coverage across timeframes) replace per-asset budget guessing. Validated cells for GMGP1-BTC (22mo × 2M, 32×) and SG-1-XAUUSD (24mo × 4M, 17×) become protocol defaults. §9 retrain cadence extended with **timeframe-dependent intervals** (sub-5m: 14–28d; 5–15m: 30–45d; ≥15m: 60–90d) to handle microstructure decay without shrinking the training window. Non-breaking: existing configs with valid multiplicity remain compliant; `validate_config.py` adds a multiplicity preflight (WARN below 10×, REJECT above 50× — requires fresh study). See `decision_training_budget_multiplicity_rule.md` and `decision_calendar_anchored_training_window.md`.
> - **v2.5** (2026-05-05, S526) — Stage 2.5 ensemble-confirm gate refinement: block-bootstrap probability (`P(ens_PF > solo_PF)`, `P(ens_MDD better)`) is now the PRIMARY decision criterion for prop-firm / live-capital workstreams; the legacy point-estimate `ensemble_uplift_min` is demoted to audit-only metadata. New `gates.ensemble_bootstrap_p_pf_ambiguous` (default **0.75**) introduces an `AMBIGUOUS_BOOT` band (mirrors S495 `AMBIGUOUS_RERUN` for the noise-aware criterion). v2.5 matrix promotes on PF dominance (`P_PF ≥ promote`) regardless of `P_MDD`, fixing a v2.3 misclassification (Funding-Arb DSAC `P_PF=1.0, P_MDD=0.36` was previously SOLO_BEST_FALLBACK; v2.5 makes it PROMOTE). Validator promoted: bootstrap keys are FAIL-on-missing for prop-firm; uplift demoted to WARN-on-missing. Verdict JSON bumped to `schema_version: "2.5"` with new `bootstrap` / `legacy_uplift` blocks (additive — v2.1/v2.3 readers parse cleanly). See `decision_protocol_v25_bootstrap_primary.md` and the architecture artifact `.agent/artifacts/stage_2_5_bootstrap_primary_architecture.md`.
> - **v2.4.1** (2026-05-01, S512) — Operational-hygiene patch from external simplification review (`.agent/artifacts/protocol_v2_simplification.md`). Two non-breaking refinements adopted; three rejected. **Adopted:** (a) §2 + §4 Stage 2.5 specify **post-PROMOTE replay-buffer purge** for non-top-K seeds (purged seeds retain SHA256 in manifest for audit); collapses Stage-2 buffer retention from `N=10` to `K=3` for typical prop-firm runs (~70% disk reduction without affecting Stage-3 warm-resume). (b) §8.1 joint-feature drift Mahalanobis pinned to **Ledoit-Wolf shrinkage covariance** (`sklearn.covariance.LedoitWolf`) before any `live_obs_builder.py` implementation lands; eliminates the ill-conditioned-Σ false-positive risk on collinear LOB/MA features without changing χ² thresholds or replacing Mahalanobis with autoencoder/PCA alternatives. New gate key `gates.drift.mahalanobis_top_k` (default **8**). **Rejected (with rationale):** MedianPruner on Q-divergence/TD-error (contradicts BUG-01 + NopPruner gotcha — RL learning curves are non-monotonic, kills late-bloomers); CRIT-flatten replacement with TWAP/limit-chase (inverts cost asymmetry for FTMO/Velotrade — DD-breach termination dwarfs slippage); CSCV/Deflated-Sharpe replacing block bootstrap in Stage 2.5 (answers a different statistical question — single-strategy overfitting vs paired ensemble-vs-solo dominance — and the v2.3 bootstrap costs ~30s, not "massive"). See `decision_protocol_v241_simplification_review.md` for full review and counter-evidence.
> - **v2.4** (2026-04-24, S495-cont) — Prop-firm challenge decoupling shipped. Schema change: training configs use `env.risk:` (phase-invariant DD + daily-loss shaping — no `profit_target_pct`, no `success_bonus`, no terminate-on-profit); deploy-stage configs use `configs/deploy/<firm>/<phase>.yaml` overlays layered onto the base training YAML via `deep_merge(allowlist=)` in `finrl_pro_ds/config_utils.py`; `challenge:` block gates `ChallengeStateMachine` (live only, `challenge.enabled=true`). Training matrix collapses from **N strategies × M firms × 3 phases** to **N checkpoints + 3M overlays**. Legacy `env.prop_firm:` deprecated (validator WARN elsewhere, FAIL at paper-deploy). See `decision_prop_firm_decoupling_s495.md` + `.agent/artifacts/prop_firm_decoupling_architecture.md` (rev 2 LIVE). Four new validator checks: `check_legacy_prop_firm_block`, `check_challenge_block`, `check_static_peak_consistency`, `check_overlay_allowlist`. `REASON_PHASE_COMPLETE` kill-file reason distinct from `REASON_DRIFT_CRIT` (preserves §8.3 repeat-CRIT lockout math). First customer: GMGP1-XAUUSD FTMO seed 42, solo A/B PASS 7/7 gates on Q1 2026 OANDA OOS.
>   - **2026-04-24 amendment (Open Questions Q1–Q5 resolved):** Q1 ensemble-checkpoint timing → val-argmax-PF from WF-OANDA (S495 rule). Q2 adapter deprecation → gate-on-migration-completion (calendar-agnostic). Q3 The5ers/HyroTrader/FundingPips overlays → defer until those workstreams activate. Q4 `ChallengeStateMachine` re-homed `finrl_pro_ds/crypto/live/` → `finrl_pro_ds/live/` (asset-agnostic). Q5 manifest extension shipped: `challenge_target_hit_rate_per_window` per seed (in `seed_report.json` and `ensemble_report.json`) + `ensemble_challenge_target_hit_rate` for the chosen rule (in `ensemble_report.json`), at both L1 and WF stages. New `finrl_pro_ds/reporting/challenge_target.py`; backfill via `scripts/backfill_eval_distribution_v22.py --phase-spec ...`; new validator `check_drift_baseline_manifest_schema` (best-effort WARN at paper-deploy).
> - **v2.3** (2026-04-23, S495-cont) — Ensemble methodology amendment from external expert review (see `decision_ensemble_bootstrap_diversity_s495.md`). Five changes: (a) §4 Stage 2.5 uplift gate replaced with **block-bootstrap on per-bar PnL** — `P(ens_PF > solo_PF) > 0.90` AND `P(ens_MDD < solo_MDD) > 0.90`. Point-estimate `ensemble_uplift_min` retained as a sanity-check secondary, not the primary decision rule. (b) §4 top-3 seed selection extended from "median test PF" to **diversity-aware**: top-1 by PF, then #2/#3 maximize `PF − λ·max_action_corr` against already-selected. Reduces same-local-minimum collapse. (c) New §4.5 **Stage 2.5-R (ensemble re-eval after retrain)** with explicit triggers (HP change, scheduled cadence, live PF degradation, **agreement-decay** capital-starvation, action KL drift, cost drift) and a 6-step protocol. (d) §8 adds **agreement-decay live monitor** for ensemble-deployed strategies (`ens_agreement` flat-rate vs baseline) — silent capital-starvation failure mode. (e) §5 / new §11.x specify the **atomic ensemble swap artifact** (`ensemble_v{N}.tar.gz`): 3 ckpts + per-seed normalizer states + resolved config + chosen rule + SHA256 manifest, container reads/swaps all-or-nothing.
> - **v2.2** (2026-04-23, S495-cont) — RLOps amendment from external expert review: §8 extended with live action-distribution drift (regime-conditioned KL / deadband-delta), §8.3 adds tiered WARN/CRIT safe-mode with graceful flatten (replaces prior "auto-halt" which was ambiguous and left open positions at risk), §2 manifest schema adds `eval_distribution` block required in stage-2 seed reports and stage-2.5 ensemble reports, §11 moves drift check from deferred to blocking for prop-firm / live-capital workstreams. See `decision_protocol_v22_rlops_drift_safemode.md`.
> - **v2.1** (2026-04-23, S495) — Stage 2.5 ensemble rule selection moved from hardcoded `ens_agreement` (canonical per S493) to val-argmax-PF per-workstream selection. Motivated by GMGP1-BTC L1 falsification (S495 — ens_agreement was the *worst* ensemble on trending BTC, +4.10% vs ens_mean +8.13%). First customer: **GMGP1-BTC** (retro-applied). See `decision_ensemble_val_selection_s495.md`.
> - **v2.0** (2026-04-18, S475) — initial post-audit revision: 6 core stages + Stage 2.5 ensemble-confirm (with hardcoded `ens_agreement` canonical rule per S493). Superseded the DRAFT.
> - **v2-DRAFT** (2026-04-17, S474) — first codified protocol after AlphaSeek monolithic-HPO incident.

---

## 0. Principles

1. **One stage = one WandB run = one decision artifact.** Never fuse HPO, walk-forward, and ensemble eval into a single multi-day job.
2. **Every stage is independently runnable, queueable, and resumable.** Failure in stage N does not invalidate stages 0..N-1. For off-policy RL (SAC, IQN), resumability requires the replay buffer artifact — see §5.
3. **Inter-stage contract is a manifest file**, not in-memory state. All inputs and outputs are explicit. Manifest schema is enforced (see §11 must-haves).
4. **Mandatory gates exit non-zero on failure.** No "advisory" checks for production-bound work.
5. **Most-recent OOS is non-negotiable.** Any checkpoint deployed to paper must have `test_end ≥ today − 60d`.
6. **Multi-seed median, not max.** Single-seed cherry-picking is forbidden for graduation candidates.
7. **Stochastic policy = stochastic eval.** Stages 2–5 must report mean ± std over ≥10 eval episodes when the policy is stochastic. Single-rollout PF on a sampling policy is meaningless.

### Anti-pattern (what v2 forbids)

AlphaSeek `k28l6ef8` (Phase 6 HPO, 5.7 days, terminated S474):
- HPO + walk-forward training + ensemble eval fused into one WandB run
- No inter-stage checkpointing — failure at hour 130 would have lost everything
- No per-window decision artifact — could not separately diagnose which window/seed/HP set failed
- No mid-run halt criterion when intermediate windows underperformed

v2 splits this into discrete stages with PASS/FAIL gates between each.

---

## 1. Stage Taxonomy

Each stage launches as its own WandB run with naming `<workstream>-stage{N}-<purpose>_<ISO-ts>` and tags `stage:{N}`, `workstream:{ws}`, `depends_on:<upstream_run_id>`.

| # | Stage | Purpose | Typical runtime | Output artifact |
|---|---|---|---|---|
| 0 | data-prep | Build cleaned parquet + manifest | minutes | `<dataset>.manifest.json` |
| 1 | hpo | Hyperparameter search (Optuna) | hours–days | `best_hp.json` + `study.db` |
| 2 | l1-multiseed | Validate best HPs across N≥3 seeds | hours | per-seed checkpoints + `seed_report.json` |
| 2.5 | ensemble-confirm (prop-firm / live-capital; else advisory) | Diversity-aware top-3 selection + bootstrap-gated multi-seed aggregation vs best-solo on L1 val→test windows | minutes | `ensemble_report.json` (solo + 4 ensemble rules + bootstrap verdict + correlation matrix) + `ensemble_v{N}.tar.gz` swap bundle |
| 2.5-R | ensemble re-eval after retrain (any workstream that previously ran Stage 2.5) | Re-pick top-3 + re-run bootstrap gate on freshly trained seeds; produce next-gen swap bundle | minutes (after L1 retrain) | new `ensemble_report.json` + `ensemble_v{N+1}.tar.gz` |
| 3 | walk-forward (+ fixed-lot stress sub-report) | Temporal robustness across K windows; full-window fixed-lot replay attached as sub-artifact | hours | per-window checkpoints + `wf_report.json` (includes `stress` block) |
| 4 | recent-oos (+ compliance filter sub-report) | OOS test on `today−60d → today−1d`; FTMO/Velotrade compliance filter applied to candidate set | minutes | `oos_report.json` (includes `compliance` block + final selection) |
| 5 | paper-deploy | Live container on `finrl-desktop` (reads `ensemble_v{N}.tar.gz` if ensemble was promoted; prop-firm / live-capital applies `configs/deploy/<firm>/<phase>.yaml` overlay per v2.4) | continuous | live engine emits its own runs |

**Stage count:** 8 (stages 0, 1, 2, 2.5, 2.5-R, 3, 4, 5). Stages 2.5 and 2.5-R are eval-only (no training, no new WandB run by default — they write sub-artifacts under the stage-2 / stage-2-R parent). 2.5-R is *not* a fresh run-from-zero re-derivation; it's the re-evaluation that follows any L1 retrain whose downstream is a live-deployed ensemble. Fixed-lot stress and compliance filter are *replay/selection* operations on prior outputs, not new compute stages — kept as sub-reports inside stages 3 and 4 to reduce CLI/manifest plumbing without losing rigor.

---

## 2. Manifest Schema

Every stage writes `<run_id>.manifest.json` to `results/<run_id>/`:

```json
{
  "stage": 1,
  "workstream": "gmgp1-xauusd",
  "run_id": "abc123de",
  "run_name": "gmgp1-xauusd-stage1-hpo_20260418_140000",
  "depends_on": null,
  "git_sha": "11c5d38c",
  "config_path": "configs/gmgp1_xauusd_ftmo_hpo.yaml",
  "config_sha": "<sha256 of resolved config>",
  "env_code_sha": "<sha256 of envs/<env_file>.py>",
  "random_seeds": [42, 1337, 2024],
  "inputs": {
    "data": {
      "path": "data/xauusd_ctrader_15min.parquet",
      "manifest": "data/xauusd_ctrader_15min.manifest.json",
      "sha256": "..."
    }
  },
  "outputs": {
    "best_hp": "results/<run_id>/best_hp.json",
    "study_db": "results/<run_id>/study.db",
    "checkpoint": "results/<run_id>/checkpoint_final.pth",
    "replay_buffer": "results/<run_id>/replay.pkl"
  },
  "training_health": {
    "final_actor_entropy": -1.23,
    "q_target_max_div_ratio": 2.4,
    "action_saturation_pct": 18.7,
    "max_grad_norm": 4.2,
    "diverged": false
  },
  "eval_distribution": {
    "histogram_bins": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
    "counts": [212, 180, 310, 1420, 1510, 340, 205, 198],
    "mean": 0.02, "std": 0.41, "entropy": 1.12,
    "deadband_frac": 0.38, "saturation_frac": 0.04,
    "by_vol_quartile": {
      "q1": {"mean": 0.01, "std": 0.28, "deadband_frac": 0.52, "saturation_frac": 0.01},
      "q2": {"mean": 0.02, "std": 0.37, "deadband_frac": 0.41, "saturation_frac": 0.03},
      "q3": {"mean": 0.03, "std": 0.44, "deadband_frac": 0.32, "saturation_frac": 0.05},
      "q4": {"mean": 0.02, "std": 0.52, "deadband_frac": 0.24, "saturation_frac": 0.08}
    }
  },
  "challenge_target_hit_rate_per_window": {
    "step1": {"target_pct": 0.10, "window_bars": null, "n_windows": 1, "n_hits": 1, "hit_rate": 1.0, "bars_to_target_median": 1245.0, "max_cum_return": 0.124},
    "step2": {"target_pct": 0.05, "window_bars": null, "n_windows": 1, "n_hits": 1, "hit_rate": 1.0, "bars_to_target_median": 412.0, "max_cum_return": 0.124}
  },
  "started_at": "2026-04-18T14:00:00Z",
  "ended_at": "2026-04-19T02:14:00Z",
  "status": "PASS",
  "gates": {
    "median_pf": {"value": 2.31, "threshold": 1.5, "pass": true}
  }
}
```

**Required fields by stage:**
- `env_code_sha` — every stage. Silent edits to env files invalidate prior manifests; downstream rejects on mismatch unless `--allow-env-drift` set.
- `random_seeds` — every training stage. Eval reproducibility.
- `outputs.replay_buffer` — stages 1, 2, 3 for off-policy algos (SAC, IQN). Required for `--resume` to work correctly; cold-buffer resume changes effective sample distribution. **Stage 2 retention policy (v2.4.1):** all N seeds write their buffer at training time; on Stage 2.5 verdict (PROMOTE / DD-only PROMOTE / SOLO_BEST_FALLBACK), buffers for **non-selected seeds** are purged from disk (the diversity-aware top-K survives, plus the diversity-Phase-0 top-1 in the SOLO branch). Purged seeds keep their entry in the manifest with `replay_buffer.purged: true` and the original `sha256` retained for audit. Stage-3 warm-resume only ever reads buffers for seeds that survived Stage 2.5, so the purge is lossless for the documented `--resume` path. `--allow-cold-replay` is the documented escape hatch if a purged seed is later needed.
- `training_health` — every training stage. Used by stage gates (see §4 stage 1).
- `eval_distribution` — **stage 2 per-seed** (required for prop-firm / live-capital; advisory elsewhere) and **stage 2.5 ensemble** (required when Stage 2.5 is mandatory). Consumed by §8.2 live-action-drift check; the bucketed `by_vol_quartile` form is the reference for regime-conditional baselining. Aggregated ensemble distribution must be logged under `ensemble_report.json → ensemble_eval_distribution` with an additional `composition_rule` field (the S495 chosen aggregation rule). For multi-dim action spaces (CryptoPerp, Funding-Arb), log per-asset marginal histograms under `by_asset.<asset_key>` instead of a scalar histogram.

Status enum: `PASS`, `FAIL`, `WARN`, `RUNNING`. Downstream stages refuse to launch if upstream `status != "PASS"`.

---

## 3. Data Manifest (stage 0)

Written by `scripts/build_data_manifest.py`. Co-located with the parquet file as `<dataset>.manifest.json`.

```json
{
  "dataset": "xauusd_ctrader_15min",
  "path": "data/xauusd_ctrader_15min.parquet",
  "sha256": "...",
  "source": "ctrader",
  "broker_account": "demo",
  "rows": 184320,
  "first_ts": "2022-01-03T00:00:00Z",
  "last_ts": "2026-04-17T22:45:00Z",
  "freq_seconds": 900,
  "nan_count": 0,
  "gap_count": 12,
  "max_gap_bars": 96,
  "regime_quartiles": {
    "vol_q1": 0.27, "vol_q2": 0.26, "vol_q3": 0.24, "vol_q4": 0.23
  },
  "feature_distribution": {
    "<feature_name>": {"mean": 0.0, "std": 1.0, "p01": -2.3, "p99": 2.4}
  },
  "clean_ohlcv_passed": true,
  "built_at": "2026-04-18T13:55:00Z"
}
```

`feature_distribution` is consumed by §8 live-feed drift check; precompute it once per dataset.

**Rejection rules** (enforced by `validate_config.py`):
- `clean_ohlcv_passed != true` → reject
- `last_ts < today − 7d` for HPO; `< today − 1d` for backtest → reject
- Any `regime_quartile < 0.10` → reject (insufficient regime coverage)
- `nan_count > 0` → reject
- `max_gap_bars > 1 day` without explicit `gap_detection: true` in config → reject

---

## 3.5 Training Budget & Data Window Rule (v2.5.1 NEW)

Locks in the empirical findings from six volume/budget studies (S500–S543; see study memories listed at end of section). Replaces per-asset budget guessing — future asset workstreams pick `total_timesteps` and `train_window` by applying the rules below, not by running fresh studies.

### 3.5.1 The Multiplicity Rule (primary)

**Definition.** Replay-buffer multiplicity = `total_timesteps / bars_in_train_window`. Empirically governs the overfitting/generalization trade-off across all studied asset classes and timeframes.

**Productive zone:** multiplicity ∈ **[15, 40]**. Pick `total_timesteps` so the ratio lands here. Buffer size stays at 500K (held constant across all studies); only `total_timesteps` varies.

| Zone | Multiplicity | Behavior | Action |
|------|--------------|----------|--------|
| Under-trained | < 10× | Policy hasn't learned robust features | `validate_config.py` WARN; OK for smoke runs |
| Borderline | 10–15× | Marginal; works on some assets (XAU 2M @ 24mo = 8.6×) | Allowed; flag for N=10 validation |
| **Productive** | **15–40×** | Buffer revisits rescue overfitting; CV tightens; val-test gap < 1.5× | **TARGET** |
| Cliff | > 50× | All seeds collapse; CV widens; val-test gap ≥ 1.5× | `validate_config.py` REJECT — requires fresh study |

**Why this works:** Longer history → lower multiplicity for the same budget → the buffer-revisiting effect that exploits regime variety, before noise/overfitting dominates. Mechanism transferred cleanly across **BTC@15m (Bybit)** and **XAU@3m (OANDA)** — portable.

### 3.5.2 Calendar-Anchored Data Window (NOT bar-anchored)

**Rule:** `train_window` is set in **calendar months**, NOT in bars. Minimum **22 months**, preferred **24 months**, regardless of timeframe.

**Why calendar, not bars:** Macro regime cycles (bull/chop/bear), Fed cycles, halvings, ETF flows, seasonality — all calendar-anchored phenomena. A bar-anchored window (e.g., "always 60K bars") on a 3m strategy gives only ~4 calendar months, missing regime variety the agent must generalize to. The multiplicity rule absorbs the bar-density difference between timeframes via `total_timesteps`, not via window length.

**Empirical anchor:** SG-1-XAUUSD (3m, 24mo, ~233K bars, 4M steps, 17× mult) passed Phase 2 N=10 with PF 1.95, CV 6.55%. GMGP1-BTC (15m, 22mo, ~63K bars, 2M steps, 32× mult) peaked at PF 2.44, CV 2.17%. Both calendar-equivalent; both in productive zone via their own bar-density-appropriate budgets.

### 3.5.3 Validated Cells (protocol defaults)

| Workstream | Timeframe | Train window | Bars (approx) | `total_timesteps` | Mult | Test PF | Study |
|---|---|---|---|---|---|---|---|
| GMGP1-Gold | 15m | 10mo | ~21K | 500K–1M | 24–48× | 2.00 (cliff above 1M) | Volume Study v1 (s500) |
| **GMGP1-BTC** | **15m** | **22mo** | **~63K** | **2M** | **32×** | **2.44** | Data-Window + Volume v2 (s522) |
| **SG-1-XAUUSD** | **3m** | **24mo** | **~233K** | **4M** | **17×** | **1.95 (N=10)** | VS-v2 Phase 1 + Phase 2 (s540) |
| SG-1-BTC | 3m | 22–24mo | ~330K | 2M + decay | ~6× (extended) | 2.76 (regime-fragile) | A1 extended + DECAY-01 (s533, s542) |

**Bold rows are the canonical protocol defaults** for prop-firm new-asset launches in the same class. Gold cell at 10mo is intentionally short-window per the steady-state-fee study; treat as IB-specific exception.

### 3.5.4 Asset-Class Fee Models (workstream-locked)

| Asset class | Fee model | Source |
|---|---|---|
| Bybit BTC perp | `steady_state_5bps_from_step_0` | Bybit taker tier-0 |
| OANDA XAU spot | `steady_state_2.35bps_oanda_from_step_0` | OANDA retail XAU bid-ask mid |
| IB Gold (GC/MGC) | `steady_state_6.8bps_ib_mgc_from_step_0` | S536/S538-cont-2 — see `configs/gmgp1_sac_gc_15min_steadystate.yaml` + `project_gmgp1_gc_steady_state_fee_audit.md`. Legacy `configs/gmgp1_sac_gc_15min.yaml` is the v5 paper baseline only (curriculum 0→2bp preserved); do NOT use for new HPO. |

Curriculum fee schedules (0 → 2bp → 5bp ramp) are forbidden outside HPO sensitivity-analysis runs (BUG-01 lock-down; `decision_steady_state_fees_pattern.md`).

### 3.5.5 New-Asset Launch Preflight (enforced by `validate_config.py --stage l1-multiseed`)

```
□ bars_in_train_window computed and printed at preflight
□ multiplicity = total_timesteps / bars_in_train_window ∈ [15, 40]   → WARN below 10, REJECT above 50
□ train_window ≥ 22 calendar months                                   → REJECT if shorter (no exceptions outside GMGP1-Gold IB legacy)
□ l1_seeds = 10 (prop-firm / live-capital) or 5 (exploratory)         → see §4 Stage 2
□ l1_pf_cv_ambiguous = [0.22, 0.38] pre-committed                     → see §4 Stage 2
□ fee_model matches venue from §3.5.4                                 → REJECT on curriculum schedule (BUG-01)
□ ensemble_uplift_min matches baseline tier (1.10 low, 1.05 high)     → see §4 Stage 2.5
□ retrain cadence configured per §9 timeframe tier
```

**Axes NOT studied (do not lock in via this protocol):** `batch_size` (512 fixed by HPO inheritance), `update_interval` (4 fixed; 1:4 env:policy ratio standard SAC), `num_envs` (20 fixed; cheap 3-cell sweep `{5, 20, 50}` queued but not run), `buffer_size` (500K — DO NOT CHANGE; multiplicity is computed against this constant). Entropy α / actor LR / critic LR / grad clip remain HPO-tuned per asset.

### 3.5.6 When to run a fresh study (the protocol does NOT cover this)

Required if any of the following:
- Timeframe outside the studied set (e.g., sub-1m LOB, 1H+ swing). Mechanism is expected to transfer but multiplicity productive-zone bounds are not yet validated outside 3m–15m.
- New asset class with materially different bar-density profile (e.g., illiquid CFD with frequent gaps, or futures roll bars). Multiplicity formula assumes contiguous bars.
- Multiplicity outside [15, 40] is *operationally desired* (e.g., wall-clock constraint forces under-trained run). Requires explicit OVERRIDE block in gate YAML + study justification.
- `buffer_size ≠ 500K`. Multiplicity rule is defined against 500K; changing this invalidates the productive-zone bounds.

### 3.5.7 Study Memory References

| Slug | Topic |
|---|---|
| `decision_training_budget_multiplicity_rule.md` | The 15–40× rule + this section |
| `decision_calendar_anchored_training_window.md` | Calendar-vs-bar anchoring rationale |
| `decision_volume_axis_v2_2m_ceiling.md` | GMGP1-BTC 2M ceiling at A_long |
| `decision_volume_axis_v2_sg1_xauusd.md` | SG-1-XAUUSD 4M productive zone |
| `project_volume_study_5budget_verdict_s509.md` | GMGP1-Gold v1 (5-budget verdict) |
| `project_sg1_xauusd_volume_study_v2_phase1_verdict.md` | XAU Phase 1 CLIFF triple-confirm |
| `project_sg1_xauusd_volume_study_v2_phase2_n10_verdict.md` | XAU Phase 2 N=10 PASS |
| `project_a1_extended_audit_findings_s533.md` | SG-1-BTC extended-window regime audit |
| `decision_l1_multiseed_n_seeds_s488.md` | N=10 + ambiguous-band rule |
| `project_gmgp1_btc_along_wf_uplift_threshold_revision.md` | Tier-dependent ensemble uplift threshold |

---

## 4. Per-Stage Gates

All numeric gate thresholds live in `configs/<workstream>.gates.yaml` and are referenced by name below. Defaults shown in parentheses are the project-wide fallback when a workstream omits the key.

### Stage 0 — data-prep
- Manifest written
- All rejection rules above pass

### Stage 1 — hpo
- Steady-state `taker_fee` from step 0 (no `fee_schedule` curriculum) — see `decision_steady_state_fees_pattern.md`
- HPO objective = `profit_factor` (BUG-01 invariant)
- `hindsight_weight = 0.0` outside HPO (BUG-03)
- `study.db` retained for downstream analysis
- **Per-trial seeding:** trials run with **single seed** by default (noise-fit risk acknowledged; L1 multiseed in stage 2 is the validation filter). Workstreams may override `hpo.seeds_per_trial: ≥2` in gate config to average across seeds at the cost of longer HPO. This is an explicit trade-off, not an oversight.
- **HPO budget** declared per workstream in gate config: `hpo.trials`, `hpo.steps_per_trial`, `hpo.wall_clock_hours_estimate`. `validate_config.py` rejects unbounded HPO. Pick `hpo.steps_per_trial` per §3.5.1 multiplicity rule (target 15–40×); 500K is the project-wide default for HPO trials and yields ~24× on a 21K-bar window, ~8× on a 63K-bar window (the latter under-trained for L1 but acceptable for HPO since L1 multiseed is the real validation filter).
- **Training-health hard-fail** (any trial that hits these is auto-pruned and excluded from best-trial selection):
  - actor entropy collapses (final entropy < `gates.entropy_floor`, default `-3.0` for SAC)
  - Q-target divergence (`q_target_max_div_ratio > gates.q_div_max`, default `10.0`)
  - action saturation (`action_saturation_pct > gates.action_sat_max`, default `95.0`)
  - any NaN/Inf in losses
- Gate: Optuna best trial PF ≥ `gates.hpo_pf_floor` (default 1.5)

### Stage 2 — l1-multiseed
- **`total_timesteps` and `train_window` must satisfy §3.5 multiplicity rule** (15–40× target; REJECT above 50×, WARN below 10×). For prop-firm / live-capital workstreams without a dedicated budget study, use the validated cells in §3.5.3 directly.
- N ≥ `gates.l1_seeds` (default 5; **prop-firm / live-capital workstreams must set N ≥ 10** — see S488 rationale below)
- Each seed evaluated over `gates.eval_episodes` (default 10) — for stochastic policies (SAC), report **mean and std across episodes** per seed; deterministic eval (greedy action) is logged additionally for diagnostic purposes
- Report **median** PF, Sharpe, MDD across seeds (not max)
- Gate: median PF ≥ `gates.l1_pf_floor` (default 1.5), all seeds profitable, **CV (std/mean) of PF ≤ `gates.l1_pf_cv_max` (default 0.30)** — tightened from prior 0.5 because CV=0.5 admits PF=2.0 ± 1.0 which is operationally unstable
- **`eval_distribution` per seed (v2.2 RLOps requirement)** — emitted into `seed_report.json` for every seed. Mandatory for prop-firm / live-capital; advisory otherwise. Bucketing by vol quartile uses the `regime_quartiles` from the data manifest (§3). This is the live-monitoring baseline referenced in §8.2; missing `eval_distribution` means §8.2 action-drift is log-only for that workstream.
- **`challenge_target_hit_rate_per_window` per seed (v2.4 prop-firm amendment, S495 Open Question #5 resolution 2026-04-24)** — emitted into both `seed_report.json` (key `challenge_target_hit_rate_by_seed`) and at WF stage. Computed by `finrl_pro_ds/reporting/challenge_target.py` from the per-bar `portfolio_value` series. Default phase specs: step1 (target=0.10) + step2 (target=0.05); funded skipped (target=∞). Used as the S495 val-selection tiebreaker when ensemble-rule PFs are within Δ < 0.1%, and as a deploy-readiness signal (workstreams with `hit_rate < 0.5` on step1 should not paper-deploy). Mandatory for prop-firm / live-capital; advisory otherwise.

- **Pre-committed escalation rule (mandatory for prop-firm / live-capital, N ≥ 10):**
  - Declare `gates.l1_pf_cv_ambiguous: [low, high]` (default `[0.22, 0.38]`) in the workstream gate YAML **before launch**, not after seeing results
  - If measured CV ∈ [low, high] the verdict is **AMBIGUOUS** — auto-extend with a second batch of seeds to reach N=20, then re-evaluate the gate. Point estimate outside this range → clear PASS/FAIL at N=10, stop
  - Any seed with PF < 1.0 → immediate FAIL (short-circuit, no need to finish remaining seeds)
  - Rationale: the CV estimator has 95% CI ≈ [0.15, 0.45] at N=10 (McKay/Vangel, assumed approximate normality of PF across seeds). With a gate at 0.30, a measured CV of 0.30 is consistent with true CV anywhere in that interval; N=20 tightens to [0.20, 0.40]. Without the escalation rule the Type I/II error rates of the CV gate are poorly controlled. Literature anchors: Henderson et al. 2018 (*Deep RL That Matters*); Agarwal et al. 2021 (*Statistical Precipice*) both recommend N ≥ 10 for variance-based claims
  - Escalation batches use **different seeds** from the first batch (no overlap) so that CV estimate pools independent samples

### Stage 2.5 — ensemble-confirm (mandatory for prop-firm / live-capital; advisory elsewhere)
**Amended v2.3 — bootstrap-gate + diversity-aware seed selection.** Supersedes the S495 point-estimate `ensemble_uplift_min`. Motivation: with N=10 seeds and a single test window, a +5–10% PF uplift is statistically indistinguishable from noise, while diversity-blind top-3-by-PF risks selecting three seeds that converged on the same local minimum (giving 0% diversification benefit at 3× live-inference cost). The two prior amendments (S493 canonical rule, S495 val-argmax) addressed *which rule wins*; v2.3 addresses *whether the win is real* and *whether the seeds being combined actually disagree*. See `decision_ensemble_bootstrap_diversity_s495.md`.

#### Inputs
- All N L1 seeds and per-seed checkpoints from stage 2 (`seed_report.json`).
- L1 **val** window and L1 **test** window declared in `data:` block (`val_start_date`, `val_end_date`, `test_start_date`, `test_end_date`).
- Per-seed test trajectories (action_<seed> column) used for action-correlation analysis.

#### Method (5 phases, all driven by `scripts/*_ensemble_eval.py` → `sg1.run_stage_2_5_val_selection()`)

**Phase 0 — diversity-aware top-K selection (v2.3 NEW).** Replaces "top-3 by median test PF":
  ```
  candidates = sort N seeds desc by test_PF
  selected = [candidates[0]]                          # always keep #1 by PF
  while len(selected) < K:                            # K = gates.ensemble_top_k (default 3)
      best, best_score = None, -inf
      for c in candidates not in selected:
          max_corr = max(action_corr(c, s) for s in selected)
          score = c.PF - λ * max_corr                 # λ = gates.ensemble_diversity_lambda (default 1.0)
          if score > best_score:
              best, best_score = c, score
      selected.append(best)
  ```
  - `action_corr` = Pearson correlation on the bar-aligned test-window action time series. For multi-dim action spaces, take the mean across asset dims.
  - The full N×N correlation matrix and per-step diversity scores are written to `ensemble_report.json → diversity_audit` for review.
  - **Failsafe:** if the diversity-aware selection differs from naive top-K-by-PF, both candidate sets are evaluated through Phases 1–4 and the verdict logs both. The diversity-aware set wins iff its bootstrap P(PF) and P(MDD) gates both PASS.

**Phase 1 — val bake-off:** run all 4 ensemble rules + K solos on the **val window** (norm cutoff = `train_end_date`). Reports bar-level PF for `ens_mean`, `ens_median`, `ens_agreement`, `ens_pf_weighted`, plus solos.

**Phase 2 — rule selection (S495):** `chosen_rule = argmax_{r ∈ ensembles}(val_PF[r])`. Test split is NOT consulted for this choice. Solo PFs on val are logged for audit but don't enter the selection.

**Phase 3 — test eval:** run only `chosen_rule` + K solos on the **test window** (norm cutoff = `val_end_date`). Save full per-bar trajectories (`portfolio_value`, `action_agg`, `action_<seed>`, timestamps) for Phase 4.

**Phase 4 — bootstrap gate (v2.3 introduced, v2.5 PRIMARY):**
- For each rule (chosen ensemble, each solo) compute the per-bar return series `r_t = (pv_t − pv_{t-1}) / pv_{t-1}`.
- Run **stationary block bootstrap** (Politis-Romano 1994) on the per-bar returns. `B = gates.ensemble_bootstrap_resamples` (default **10000**) resamples; mean block length `gates.ensemble_bootstrap_block_len` (default `sqrt(n_bars)`, capped at 0.1 × n_bars). Block bootstrap is required because per-bar returns are autocorrelated.
- For each resample, compute (PF, trailing-MDD) under that resample. This produces empirical distributions over PF and MDD for every rule.
- Compute paired statistics: `P_PF = P(PF_ens > PF_best_solo)` and `P_MDD = P(MDD_ens > MDD_best_solo)` (less negative = better; for MDD "better" means smaller magnitude). Pairing is by resample index (same block draw used for both rules).
- **Verdict logic (v2.5 matrix — PRIMARY for prop-firm):**
  - `P_PF ≥ gates.ensemble_bootstrap_p_pf_promote` (default **0.90**) → **PROMOTE ensemble** for paper-deploy regardless of `P_MDD` (PF dominance; MDD wash). Stage 3 WF runs all K seeds with `chosen_rule`; stage 5 reads `ensemble_v{N}.tar.gz`.
  - `P_PF ∈ [gates.ensemble_bootstrap_p_pf_ambiguous, p_pf_promote)` (default `[0.75, 0.90)`) AND `P_MDD ≥ p_mdd_promote` → **PROMOTE_DD_ONLY** (prop-firm allowed; non-prop-firm advisory). PF gain is in the noise floor but DD tightening is statistically real and matters for prop-firm DD-buffer compliance.
  - `P_PF ∈ [p_pf_ambiguous, p_pf_promote)` AND `P_MDD < p_mdd_promote` → **AMBIGUOUS_BOOT** (v2.5 NEW; mirrors S495 `AMBIGUOUS_RERUN` — second data point required before promote/reject).
  - `P_PF < p_pf_ambiguous` (any `P_MDD`) → **SOLO_BEST_FALLBACK**. Stage 3/5 use the diversity-Phase-0 top-1 seed.
  - **Audit-only secondary (v2.5 demoted):** the legacy `uplift = chosen_test_PF / best_solo_test_PF` is still computed and logged in the verdict's `legacy_uplift` block, but does NOT enter the decision when bootstrap gates are present. Large bootstrap-vs-uplift discrepancies (`legacy_uplift ≥ 1.20` but bootstrap SOLO, or vice versa) flag for manual review — usually indicate a test window dominated by a few outlier bars, which is itself a finding.

#### Gate thresholds (from `gates:` block, no code defaults; missing keys raise per CLAUDE.md anti-pattern)
- **Diversity selector:** `gates.ensemble_top_k` (default **3**), `gates.ensemble_diversity_lambda` (default **1.0**).
- **Bootstrap (v2.5 PRIMARY):** `gates.ensemble_bootstrap_resamples` (default **10000**), `gates.ensemble_bootstrap_block_len` (default `null` → auto = `sqrt(n_bars)` capped at 10% of bars), `gates.ensemble_bootstrap_p_pf_promote` (default **0.90**), `gates.ensemble_bootstrap_p_mdd_promote` (default **0.90**), `gates.ensemble_bootstrap_p_pf_ambiguous` (v2.5 NEW; default **0.75**) — defines the AMBIGUOUS_BOOT band floor.
- **Audit-only secondary (v2.5 demoted):** `gates.ensemble_uplift_min` retained (default **1.10**) — logged in `verdict.legacy_uplift.uplift_ratio` and NOT consulted when bootstrap gates are present. Falls back to primary decision only for pre-v2.3 configs missing the bootstrap keys (`LEGACY_GATE_DEFER` path).
- **Selection rule:** `gates.ensemble_rule_selection: val_argmax_pf` (S495 default). Other methods (`static:<rule>`, `val_argmax_sharpe`) reserved for future amendments.

#### Atomic swap artifact (v2.3 NEW)
On PROMOTE, `run_stage_2_5_val_selection()` writes `results/<run_id>/ensemble_v{N}.tar.gz` containing:
- `checkpoints/seed_<id>/checkpoint_final.pth` × K
- `normalizers/seed_<id>/ema_state.pkl` × K (per-seed EMA-Z normalizer state — LEAK-1 invariant applied to deploy: forgetting the normalizer = silent live hallucination)
- `config.resolved.yaml` (env config, fee schedule, deadband, action mapping; the exact config used in Phase 3)
- `ensemble_manifest.json`:
  ```json
  {
    "schema_version": "v2.3",
    "workstream": "<ws>",
    "version": "v{N}",
    "chosen_rule": "ens_mean",
    "seeds": [42, 789, 456],
    "selected_via": "diversity_aware",
    "diversity_audit": {"action_corr_matrix": [...], "selection_score": [...]},
    "checkpoint_sha256": {"42": "...", "789": "...", "456": "..."},
    "normalizer_sha256": {"42": "...", "789": "...", "456": "..."},
    "config_sha256": "...",
    "bootstrap_verdict": {"p_pf": 0.94, "p_mdd": 0.92, "decision": "PROMOTE"},
    "produced_at": "2026-04-23T..."
  }
  ```
- Bundle SHA256 written alongside the tar.gz as `.sha256`.
- **Live container contract:** the engine reads `ensemble_manifest.json` first, validates each `*_sha256` matches the unpacked file, and refuses to start if any file is missing or has a mismatched hash. Container swap is **all-or-nothing**: never load 2 new ckpts + 1 old, never load ckpts without their matching normalizer state.
- Bundle path declared in live config: `agent.ensemble.bundle_path: results/<run_id>/ensemble_v{N}.tar.gz`. The legacy `agent.ensemble: {rule, seeds}` form is rejected for new deploys.

#### `ensemble_eval_distribution` (v2.2 carry-over)
`ensemble_report.json` records the aggregated action distribution produced by `chosen_rule` on the **test window**, using the same schema as stage-2 `eval_distribution` plus a `composition_rule` field. This is what live monitoring compares against post-deploy; per-seed distributions are **not** valid baselines for an ensemble-deployed strategy because the aggregated distribution is a convex combination of per-seed distributions under the chosen rule.

#### `ensemble_challenge_target_hit_rate` (v2.4 prop-firm amendment)
`ensemble_report.json` also records per-phase `challenge_target_hit_rate_per_window` for the chosen ensemble rule's aggregated PV trajectory, plus the per-seed block (`challenge_target_hit_rate_by_seed`) carried over from stage 2. The ensemble-level hit rate is the operationally meaningful number when the strategy is ensemble-deployed; per-seed hit rates inform tiebreaking when val-rule PFs are nearly identical. Backfill for paper-deployed checkpoints whose reports predate this amendment via `scripts/backfill_eval_distribution_v22.py --phase-spec ...`.

#### Operational notes
- **Checkpoint collision guard:** if multiple seeds share a checkpoint dir (e.g. concurrent deploys hit `deploy_bare_metal.py` timestamp-collision), substitute the next-best distinct seed. Ensemble eval **requires** per-seed distinct weight provenance; manifest records checkpoint SHA256s.
- **Replay-buffer purge (v2.4.1):** on PROMOTE / DD-only PROMOTE, `run_stage_2_5_val_selection()` retains the K (default 3) replay buffers for the diversity-aware selected seeds and **deletes** the remaining `N − K` buffers from `results/<stage2_run_id>/seed_<id>/replay.pkl`. On SOLO_BEST_FALLBACK, only the diversity-Phase-0 top-1 buffer is retained. The Stage-2 manifest is rewritten in-place with `outputs.replay_buffer.<seed>.purged = true` for the deleted ones (sha256 preserved for audit). Rationale: with N=10 seeds at ~400MB–1GB per buffer, this collapses Stage-2 buffer disk from ~10GB to ~3GB per workstream without affecting any documented `--resume` path. Implementation lands in the same script as `run_stage_2_5_val_selection`; `--no-purge` is reserved for ad-hoc reruns. Originated from external simplification review S512 (see v2.4.1 entry above).
- **Bias control:** val is now used three times (HP selection, seed selection, rule selection). Diversity-aware Phase 0 adds *no* val use (correlation is computed on test trajectories — bias-budget consumed in Phase 4 bootstrap). Verdict logs all 4 val PFs so audits can detect near-tie rule choices that may be overfit.
- **Cost:** ~10 min wall-time on CPU for 2mo val + 2mo test windows (Phase 0 N×N correlation = seconds, Phase 4 bootstrap with B=10K = ~30s for typical 30K-bar test). Up from ~8 min in v2.2.
- **When not to run:** research-tier workstreams (e.g. CMGP1 crypto) may skip Stage 2.5 — ensemble remains a per-workstream option, not a protocol requirement. `validate_config.py --stage ensemble-confirm` only hard-fails on prop-firm / live-capital tags (`prop-firm`, `FTMO`, `Velotrade`).
- **Empirical evidence anchoring v2.3 (3 workstreams + 1 venue split):**
  - SG-1 XAUUSD L1 OANDA (S490): `ens_agreement` +17.6% — strong PROMOTE under both legacy uplift and v2.3 bootstrap (rerun pending).
  - GMGP1 XAUUSD CME Path A (S493): `ens_agreement` +17.15% — same.
  - GMGP1 XAUUSD OANDA Path B (S494): `ens_agreement` +8.47% — AMBIGUOUS under legacy gate; v2.3 expectation is that bootstrap P_PF will land near 0.7–0.8 (below 0.90 promote) but P_MDD likely passes (intraday DD tighter), triggering the **DD-only PROMOTE** branch.
  - GMGP1 BTC Velotrade L1 (S495, val-rule winner = `ens_mean` +8.13%): falls in same AMBIGUOUS band; bootstrap will arbitrate.
- **First customers for v2.3:** SG-1 BTC L1 (launched S494-cont, parent `ighx368o` — first to use bootstrap gate from day one) + retro-application to GMGP1-XAUUSD-OANDA-Path-B (currently AMBIGUOUS under v2.2).
- **Retro-apply policy:** Workstreams already paper-deployed under v2.1/v2.2 verdicts keep their `agent.ensemble: {rule, seeds}` form until next L1 retrain. Stage 2.5-R (§4.5) is the migration path — first retrain produces a v2.3 `ensemble_v{N}.tar.gz` bundle and switches the live container contract.

### Stage 2.5-R — ensemble re-eval after retrain (v2.3 NEW)
**Mandatory for any prop-firm / live-capital workstream that previously promoted an ensemble (Stage 2.5 verdict = PROMOTE) AND is undergoing any L1 retrain.** Discrete protocol stage, not an ad-hoc rerun.

#### Triggers (any one fires Stage 2.5-R; declared in `gates.retrain.*`)
1. **L1 retrain** — Any change in HPs, training data, env code, or feature set that produces a new `seed_report.json`. The new seeds are almost certainly a different set from the old top-K, so the ensemble must be re-derived. **No exceptions:** even a "minor" HP nudge invalidates the prior diversity-correlation and bootstrap analysis.
2. **Scheduled cadence** — Every `gates.retrain.ensemble_recheck_days` (default **90**) on every live ensemble. Drift insurance against silent regime evolution.
3. **Live PF degradation** — Live PF < `gates.retrain.live_pf_ratio` × paper PF (default **0.8**) for `gates.retrain.live_pf_window_days` (default **5**) consecutive trading days, with ≥ `gates.retrain.min_trades_window` (default **20**) trades. Routes through §8.3 CRIT flatten before retrain.
4. **Agreement-decay (capital starvation, v2.3 NEW)** — For `ens_agreement` and similar consensus-filter rules: live `flat_bar_frac` (no-trade fraction) drift > `gates.retrain.agreement_decay_delta` (default **+0.20**) above baseline `flat_bar_frac` from `ensemble_eval_distribution`, sustained for ≥ `gates.retrain.agreement_decay_window_bars` (default **2000** live bars). **Critical failure mode:** when seeds diverge under regime shift, `ens_agreement` stays flat → no losses (PF degradation alarm doesn't fire) but capital utilization → 0. Silent death. Trigger is mandatory for `ens_agreement`/`ens_majority`-deployed ensembles; no-op for `ens_mean`/`ens_median`/`ens_pf_weighted`.
5. **Action KL drift** — §8.2 live action drift WARN sustained for ≥ `gates.retrain.action_kl_warn_window_bars` (default **5000**); CRIT goes through §8.3 immediately.
6. **Cost drift** — Realized slippage + taker fees > `gates.retrain.cost_drift_ratio` × backtest assumption (default **1.20**) over rolling `gates.retrain.cost_drift_window_trades` (default **100**). HPs were tuned for a specific fee level; if execution cost moves materially, the diversity/bootstrap analysis was done under wrong friction.

#### Steps (each = one decision artifact; sequence matches Stage 1→2.5 but inputs are post-retrain)
1. **L1 multiseed N≥10** with new HPs/data (Stage 2 of v2 protocol). New `seed_report.json`.
2. **Phase 0 diversity-aware top-K from scratch.** Do NOT carry over old top-K seeds — they were optimal for the prior loss landscape. The new seed pool is a fresh population. Continuity, if any, falls out naturally when the new diversity-aware selector happens to pick the same seed IDs.
3. **Phases 1–4 from Stage 2.5** (val bake-off → rule selection → test eval → bootstrap gate). May produce a different `chosen_rule` than the prior ensemble — this is expected when regime has shifted (e.g. choppy→trending switches `ens_agreement` → `ens_mean`).
4. **Decision matrix** (same as Stage 2.5 §4 verdict logic):
   - PROMOTE → produce `ensemble_v{N+1}.tar.gz`, proceed to Stage 3 WF re-confirm.
   - DD-only PROMOTE → produce `ensemble_v{N+1}.tar.gz` flagged `purpose: dd_buffer_only`, proceed to Stage 3.
   - SOLO_BEST_FALLBACK → demote ensemble for this workstream, produce `solo_v{N+1}.tar.gz` (single-ckpt bundle, same schema), proceed to Stage 3 with the diversity-Phase-0 top-1 seed.
5. **Stage 3 walk-forward re-confirmation** on the winning rule (or solo). Required even if Stage 2.5-R verdict matches the prior verdict — temporal robustness on new data is not transitive.
6. **Atomic live swap** — the live container hot-swaps from `ensemble_v{N}.tar.gz` to `ensemble_v{N+1}.tar.gz` via the engine's bundle-load path. Swap protocol:
   - Engine reads `ensemble_v{N+1}.tar.gz`, validates SHA256 manifest.
   - **Drain phase:** existing positions held; no new entries for `agent.swap.drain_bars` (default **20** live bars, ~1 hour at 3-min) so any in-flight signal from old ensemble settles. Watchdog suppresses no-trade alerts during drain.
   - **Cutover:** new ensemble takes over at the next bar boundary; old ensemble discarded. **Never run two ensembles in parallel** — coordinated wrong-way trades from a transitional state are worse than either alone.
   - **Rollback path:** if the new bundle fails SHA256 validation OR the engine fails to load any of the K ckpts/normalizers, retain old ensemble and post CRIT to Telegram. No partial swap.
   - Operator must explicitly approve cutover via `kill_file.swap_approved` handshake for prop-firm workstreams (live-capital risk gate).

#### Manifest schema additions
- `ensemble_manifest.json → predecessor_version`: the bundle being replaced (e.g. `"v3"` → swapping to `"v4"`). Audit chain.
- `ensemble_manifest.json → trigger`: which §4.5 trigger fired (`l1_retrain`, `scheduled_cadence`, `live_pf_degradation`, `agreement_decay`, `action_kl_drift`, `cost_drift`). Tracked for retrospective analysis of trigger predictive value.

#### When NOT to fire Stage 2.5-R
- A live container restart (no model change) does NOT fire 2.5-R. Same bundle reloaded.
- A solo-deployed strategy (Stage 2.5 verdict was SOLO_BEST_FALLBACK or never ran ensemble) does NOT fire 2.5-R on retrain — it fires regular Stage 2.5 if the workstream is prop-firm tagged. The "-R" suffix specifically denotes a re-eval of an existing ensemble's continued validity.

### Stage 2.5-R Sensitivity Audit (v2.6 NEW)

**Mandatory for any prop-firm / live-capital workstream at `protocol_version: "2.6"` with `gates.sensitivity_audit_required: true`.** Eval-time 3×3 config-perturbation sweep on the PROMOTE'd ensemble/solo. Gates Stage 3 paper-deploy candidacy on edge-stability of the trained policy in nearby YAML space.

Upstream research: `.agent/artifacts/mc_robustness_methods_research.md` (Method #5 CONDITIONAL-GO, S553).
Architecture: `.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md` (8 ADRs).
Canonical entrypoint: `scripts/stage_2_5_r_sensitivity_audit.py`.

#### Grid
- 3×3 over `(deadband_threshold, max_leverage)` YAML knobs (the two config knobs that change the env's action-processing layer at [`continuous_swing_env.py:245,286,298-303`](../finrl_pro_ds/envs/continuous_swing_env.py#L243-L313) without retraining).
- Center cell = deployed config (SENS-1 invariant; validator FAIL on mismatch). Defaults: deadband `[0.20, 0.25, 0.30]`, max_leverage mults `[0.5, 1.0, 1.5]` (mults relative to deployed `env.max_leverage`).
- **Axes are NOT orthogonal in env effect** (B5 fix: `effective_deadband = deadband × max_leverage`). The 3×3 characterizes config sensitivity *as deployed*, not in orthogonal-axis space. Documented per ADR-6.
- Cells with `max_leverage > gates.sensitivity_deployable_max_leverage_cap` flagged `deployable: false`; reported but excluded from the edge-stability gate.

#### Per-cell rollout
- Each cell re-runs the trained policy through the test split with cell-specific env knobs. Frozen policy weights (no retraining). Test split = same window as Stage 2.5 chose.
- Aggregation rule = the PROMOTE'd rule from verdict.json (`ens_pf_weighted` / `ens_agreement` / `ens_mean` / `ens_median` / `solo_<seed>`).
- Per-cell trajectory parquet at `results/<ws>_ensemble/sensitivity_audit/<cell_label>/<rule>_trajectory.parquet`.
- **PF-XCHECK per-cell** (CLAUDE.md invariant): mid_price vs close-marked PF; >30% divergence flags `halt: true`. Note: env-side dual-equity-curve recording is deferred — until that lands, PF-XCHECK status is `SKIPPED` (divergence=0, pass=true) and the halt path is exercised only by synthetic test fixtures.

#### Gate (ADR-4 + ADR-5)
- **Eligibility filter:** `deployable == true AND halt == false` (across the 8 NON-CENTER neighbors).
- **Minimum:** `n_deployable_neighbors ≥ gates.sensitivity_audit_required_min_deployable_neighbors` (default **4**); fewer → `UNKNOWN_INSUFFICIENT_NEIGHBORS`.
- **Center halt** → `UNKNOWN_INSUFFICIENT_NEIGHBORS`.
- **Metric:** `pf_ratio = pf_inner_min / pf_center` where `pf_inner_min = min(PF over eligible neighbors)`.
- **Decision:** `pf_ratio ≥ gates.edge_stability_pf_ratio_floor` (default **0.70**) → `PASS`; else `FAIL`.

#### Phase α / Phase β rollout (S553 operator decision)
| Phase | Configs | Validator behavior | When to advance |
|---|---|---|---|
| **α — calibration** | `protocol_version` defaults to `"2.5"`; gates yamls declare keys with `sensitivity_audit_required: false`. | Dormant (back-compat exempt). | After all 5 deployed strategies are backfilled and the operator reviews empirical `pf_ratio` distribution to calibrate `edge_stability_pf_ratio_floor`. |
| **β — enforcement** | Operator bumps L1 multiseed `protocol_version: "2.6"` AND flips `gates.sensitivity_audit_required: true`. | Validator FAILs prop-firm configs missing the gates keys; Stage 3 launcher refuses without `sensitivity_audit` block in upstream manifest. | Permanent. |

#### Operator backfill flow (Phase α calibration)
```
for ws in gmgp1-gold gmgp1-btc gmgp1-xauusd sg1-btc sg1-xauusd; do
  python scripts/backfill_deployed_config_field.py \
      --workstream $ws --config configs/<ws>_l1_multiseed.yaml
  python scripts/stage_2_5_r_sensitivity_audit.py \
      --workstream $ws --config configs/<ws>_l1_multiseed.yaml
done
```
Backfill FAIL policy (operator decision Q3): **flag-and-document only**. Live container keeps running; randd_log entry + memory FLAG; optionally queue retrain with sensitivity-aware HPO. NO auto-unship.

#### Verdict schema 2.6
Additive `sensitivity_audit` block alongside the v2.5 `bootstrap` / `legacy_uplift` blocks; new top-level `deployed_config` field (REQUIRED for v2.6). v2.5 readers parse v2.6 verdicts. v2.6 reader accepts `schema_version ∈ {"2.1", "2.3", "2.5", "2.6"}`.

#### When NOT to fire
- Workstreams with `protocol_version: "2.5"` (or unset) are exempt — back-compat preserved.
- Non-prop-firm workstreams: advisory only (validator WARN-on-missing instead of FAIL).
- SOLO_BEST_FALLBACK verdicts: sensitivity audit runs informational-only against the solo policy; gate not blocking.

### Stage 3.5 — observation-noise robustness (v2.7-B forward-declared)
**Reserved slot.** Researcher Method #4 GO: OHLC multiplicative log-noise → re-run policy → distribution of PF/MDD across noise seeds × WF windows. Direct prop-firm tail-DD signal; structural answer to the sim-to-live class of incidents ([project_sim_to_live_gap_audit_s470], [project_live_obs_normalization_mismatch_s509]). Engineering: ~250 LOC core + new env wrapper. Not yet shipped — pending operator approval after v2.7-A backfill calibration completes. See `.agent/artifacts/mc_robustness_methods_research.md` §Method #4.


- K ≥ `gates.wf_windows` (default 4) rolling windows
- Window split: `train: 12mo, val: 1mo, test: 1mo`, slide by 1mo (workstream may override under `wf.split` in gate config)
- Stochastic eval rules from stage 2 apply per window
- Gate: all windows profitable, median Sharpe > 0, no window MDD breach
- PF cross-check via `mid_price` AND `close` per window; >30% divergence → halt (PF-XCHECK invariant)
- **Fixed-lot stress sub-report** (attached to `wf_report.json` as `stress` block):
  - Replay full WF window with **fixed lot size** (no PropFirm early-term truncation)
  - Required after S466 SG-1 XAUUSD incident (apparent 1.09% DD was 3-sim-day artifact; fixed-lot revealed 4.59%)
  - Stress gate: worst intraday DD ≥ `gates.stress_dd_buffer_pp` (default 0.5pp) from FTMO/Velotrade cap, peak leverage ≤ `gates.stress_leverage_max` (default 1.0×)
- **Execution-failure stress sub-block** (v2.7-C forward-declared, NOT YET SHIPPED): Bernoulli(p) miss-fill DR sweep `p ∈ {0.05, 0.10, 0.20}` × N_seeds=5 per WF window. Researcher Method #3 (re-framed) CONDITIONAL-GO. Slot reserved as `wf_report.json → stress.exec_failure`; gate `gates.stress_exec_failure_pf_floor` (default **0.85**). Pending operator approval after v2.7-A backfill calibration. See `.agent/artifacts/mc_robustness_methods_research.md` §Method #3.

### Stage 4 — recent-oos (+ compliance filter sub-report)
- Test window: `today − gates.recent_oos_days` (default 60) `→ today − 1d`
- Hard rule for paper-deploy candidates (closes Q1 2026 blind spot)
- Gate verdict buckets (per `project_paper_checkpoints_q1_2026_blindspot.md`):
  - PF ≥ `gates.oos_hold_ratio` × WF median PF (default 0.8) → HOLD
  - PF ≥ `gates.oos_watch_ratio` × WF median PF (default 0.6) → WATCH
  - else → RETRAIN-required
- **Compliance filter sub-report** (attached as `compliance` block; skipped for workstreams without prop-firm caps such as Velotrade):
  - Wraps `scripts/ftmo_compliance_report.py`
  - Filters on: min `gates.ftmo_min_active_days` (default 4) active trading days, no single day > `gates.ftmo_max_day_share` (default 0.50) of profit
  - Final selection from stage 2/3 candidates
  - Gate: ≥ 1 candidate passes filter

### Stage 5 — paper-deploy
- Pre-flight: `validate_config.py` re-run on live YAML (checks include v2.4 `check_legacy_prop_firm_block` / `check_challenge_block` / `check_static_peak_consistency` / `check_overlay_allowlist` for prop-firm / live-capital tags)
- `monitor_fleet.py` exits non-zero on resource contention
- Live config has `risk.static_peak: true`, `kill_file` path set, `taker_fee` matches training manifest, `gap_detection: true` if data manifest required it
- **Prop-firm / live-capital (v2.4):** deploy YAML = base training config + `configs/deploy/<firm>/<phase>.yaml` overlay layered via `finrl_pro_ds/config_utils.deep_merge(allowlist=...)`. Base must use `env.risk:` (not deprecated `env.prop_firm:`). Overlay supplies the `challenge:` block (`challenge.enabled=true`, `challenge.phase ∈ {step1, step2, funded, custom}`, `challenge.advance_rule: "manual_ack"`). `challenge.advance_rule: "auto"` FAILs validator. Overlay keys restricted by `configs/deploy/ALLOWLIST.yaml` — touching `env.reward.*` or any HPO-sensitive key FAILs before merge.
- Container starts via `./scripts/manage_strategies.sh up <target>`

---

## 5. CLI Surface (`run_full_pipeline.py`)

Refactor from current monolithic flow to staged DAG launcher:

```bash
# Stage 0 — data prep (idempotent)
python scripts/build_data_manifest.py --data data/xauusd_ctrader_15min.parquet

# Stage 1 — HPO (own WandB run)
python scripts/run_full_pipeline.py --config configs/gmgp1_xauusd_ftmo_hpo.yaml \
  --stage hpo --trials 50

# Stage 2 — multiseed L1 (own WandB runs, fan out)
python scripts/run_full_pipeline.py --config configs/gmgp1_xauusd_ftmo_hpo.yaml \
  --stage l1-multiseed --hp-run <stage1_run_id> --seeds 5

# Stage 2.5 — ensemble confirmation (eval-only, ~5 min CPU)
# Reuses the stage-2 parent run — does not spawn a new WandB run by default.
python scripts/gmgp1_xauusd_ensemble_eval.py \
  --config configs/gmgp1_xauusd_ftmo_rehpo_l1_multiseed.yaml
# (Or the SG-1 equivalent: scripts/sg1_xauusd_ensemble_eval.py)

# Stage 2.5-R Sensitivity Audit (v2.6 NEW; mandatory under Phase β protocol_version: "2.6")
# Standalone CLI mirrors recompute_stage_2_5_verdicts.py precedent — run_full_pipeline.py
# does NOT support --stage sensitivity-audit (monolithic launcher; v2.7-A scope deviation).
python scripts/stage_2_5_r_sensitivity_audit.py \
  --workstream <ws> --config configs/<ws>_l1_multiseed.yaml
# Backfill sidecar (one-shot per workstream, populates v2.6 deployed_config field):
python scripts/backfill_deployed_config_field.py \
  --workstream <ws> --config configs/<ws>_l1_multiseed.yaml

# Stage 3 — walk-forward + fixed-lot stress (own WandB runs per window)
python scripts/run_full_pipeline.py --config configs/gmgp1_xauusd_ftmo_hpo.yaml \
  --stage wf --hp-run <stage1_run_id> --seed <best_seed> --windows 4

# Stage 4 — recent OOS + FTMO compliance filter
python scripts/run_full_pipeline.py --config configs/gmgp1_xauusd_ftmo_hpo.yaml \
  --stage oos --wf-run <stage3_run_id>

# Convenience: all stages sequential (each = own WandB run)
python scripts/run_full_pipeline.py --config configs/gmgp1_xauusd_ftmo_hpo.yaml \
  --stage all
```

**CLI rules:**
- Each stage accepts `--upstream-run <id>` flags pointing at depended-on WandB runs
- Stage refuses to launch if upstream manifest `status != "PASS"`
- `--stage all` runs sequential, halting on first non-PASS
- `--resume <run_id>` re-enters a `RUNNING` or `FAIL` stage from checkpoint. **Off-policy algorithms (SAC, IQN) require the upstream `outputs.replay_buffer` to be present** — resume from cold buffer changes effective sample distribution and is rejected unless `--allow-cold-replay` is set with explicit acknowledgment.
- `--allow-env-drift` required if `env_code_sha` of current env file differs from upstream manifest. Default behavior is reject; this flag forces explicit operator acknowledgment that env semantics may have changed.
- Backward-compat: today's bare `python scripts/run_full_pipeline.py --config X` defaults to `--stage all`

---

## 6. Skill-Chain Integration

Each stage transition fires a skill hook automatically:

```
stage 0 → Audit          (data manifest sanity, regime coverage)
stage 1 → WandB          (HPO study summary), Math (if reward changed)
stage 2 → multiseed report (training-health summary across seeds)
stage 3 → Audit          (WF results + stress sub-report), Math (PF cross-check)
stage 4 → Audit          (regime drift vs training manifest, compliance filter)
stage 5 → Live-Trading + Docker pre-flight + Live-Monitor
```

Implementation: each stage's exit hook writes `results/<run_id>/skill_chain.txt` listing skills to dispatch. Auto-dispatch wiring is opt-in until tooling matures (see §11).

---

## 7. Cross-Source Parity (when training source ≠ live broker source)

Run `scripts/source_parity.py` once per (training_source, live_source) pair:

- Sample 1-month overlap between training parquet and live broker historical
- Compute close-to-close correlation and KS-test on spread distribution
- Verify bar-timestamp convention (close-stamped vs open-stamped)

**Asset-class-calibrated thresholds** (configurable per pair under `parity.<pair>` in workstream gate config):

| Asset class | Default min correlation | Default max KS p-value |
|---|---|---|
| Futures (GC ↔ MGC, IB ↔ Databento) | 0.995 | 0.01 |
| FX / metals spot (cTrader ↔ training) | 0.990 | 0.01 |
| Crypto cross-venue (Bitfinex ↔ Binance, etc.) | 0.970 | 0.05 |

Required pairs to validate:
- Databento GC ↔ IB GC (futures)
- Bitfinex BTC ↔ Binance BTC (crypto)
- IC Markets cTrader XAUUSD ↔ training source (FX/spot — provenance gap per S470 #7)

Output: `data/parity/<training_source>_<live_source>.json`. Referenced by `validate_config.py`.

---

## 8. Live-Feed + Live-Action Drift (post-deploy)

Extension to `live_obs_builder.py`. Three independent drift signals (feature, joint-feature, action) feed a single tiered safe-mode state machine (§8.3). **Blocking for prop-firm / live-capital workstreams; advisory for research-tier.** Grace period: drift checks suppressed until `gates.drift.min_bars_before_check` (default 500) live bars have accumulated.

### 8.1 Feature drift (environment / input side)

**Per-feature drift** (catches obvious schema/scaling breaks):
- Rolling 1000 live bars compute z-score per feature against training-distribution mean/std (from `feature_distribution` in data manifest)
- WARN if > 3 features drift > 3σ
- CRIT if > 3 features drift > 5σ

**Joint-feature drift** (catches correlated drift that per-feature misses):
- Compute Mahalanobis distance of the rolling 1000-bar feature mean against the training-distribution multivariate mean. **Covariance estimator (v2.4.1):** `sklearn.covariance.LedoitWolf` shrinkage on the training-distribution sample covariance — guarantees positive-definite Σ even on the highly-collinear feature families we use (LOB ladder, MA family, ATR/realized-vol cluster) and eliminates the false-positive risk that motivated proposed autoencoder/PCA replacements (rejected — see v2.4.1 review). LW is asymptotically consistent with the sample covariance under correctly-specified models, so the χ² thresholds below are unchanged.
- Top-K feature selection: `gates.drift.mahalanobis_top_k` (default **8**); features picked by descending training-distribution variance after EMA-Z normalization, with explicit deny-list in `gates.drift.mahalanobis_exclude_features` for redundant pairs (e.g. include `realized_vol_30` only if not also including `realized_vol_60`). The chosen feature set is frozen in the data manifest (§3) so live and training share the same K-set; mismatch → reject at Stage 5 pre-flight.
- WARN if Mahalanobis > `gates.drift.mahalanobis_warn` (default χ²(K, 0.99))
- CRIT if > `gates.drift.mahalanobis_crit` (default χ²(K, 0.999))

This catches silent feed schema changes, broker feed degradation, and training-to-live distribution shift. Per-feature checks miss correlated drift (all volatility features shifting together within 3σ individually but jointly anomalous).

### 8.2 Action drift (policy / output side) — NEW in v2.2

Feature drift is necessary but not sufficient. An RL agent can receive in-distribution observations yet respond to them in a way that has shifted — most commonly **collapse to a single safe action** (V7-style deadband-flat at >99% of bars). This is the dominant RL-specific failure mode and is invisible to §8.1.

- **Baseline source:** `eval_distribution` from §2 — per-seed from `seed_report.json` (solo-deployed) or `ensemble_eval_distribution` from `ensemble_report.json` (ensemble-deployed, post-S495). Missing baseline → action drift runs in log-only mode, no WARN/CRIT.
- **Regime conditioning:** the `by_vol_quartile` sub-distributions are the reference. Live rolling-window realized vol maps to a quartile using `data_manifest.regime_quartiles` cutpoints; the comparison uses the matching-quartile baseline, not a single global distribution. Required to avoid false positives on legitimate regime flips (trending vs choppy).
- **Window:** rolling `gates.drift.window_bars` (default 1000) live actions.

**Scalar action space (V7, `Box(-1,1,(1,))`):**
- `deadband_frac_delta = |live_frac(|a|<0.25) − baseline_frac(|a|<0.25)|`
- `saturation_frac_delta = |live_frac(|a|>0.95) − baseline_frac(|a|>0.95)|`
- WARN if either delta > `gates.drift.deadband_frac_warn` / `gates.drift.saturation_frac_warn` (default 0.15)
- CRIT if either delta > `gates.drift.deadband_frac_crit` / `gates.drift.saturation_frac_crit` (default 0.30)

**Multi-dim action space (CryptoPerp, Funding-Arb, `Box(-1,1,(n_assets,))`):**
- Per-asset marginal KL-divergence between live action histogram and baseline histogram (same bins as manifest)
- `max_asset_kl = max_{asset} KL(live_asset || baseline_asset)`
- WARN if `max_asset_kl > gates.drift.action_kl_warn` (default 0.5)
- CRIT if `max_asset_kl > gates.drift.action_kl_crit` (default 1.0)
- Joint (full-action-vector) KL is intractable at 10–20 dimensions and is not computed; the marginal-max is the informative view.

**Cheap free signal (not a gate, log only):** workstreams with an oracle/signal gate (SG-1 `oracle_signal_gate`) emit live gate-firing rate alongside eval baseline rate. A delta > 0.3 against eval is a strong precursor even if §8.2 gates haven't fired.

**Agreement-decay (v2.3 NEW, ensemble-deployed only):** for `ens_agreement` / `ens_majority` rules, track live `flat_bar_frac` (fraction of bars where the rule returned zero action). Baseline = `ensemble_eval_distribution.deadband_frac` from `ensemble_report.json` (the post-aggregation flat fraction on the test window). This is a **silent-death** detector — when seeds diverge under regime shift, the consensus filter stays flat → no losses (PF gates don't fire) but capital utilization → 0. Drift > `gates.drift.agreement_flat_delta_warn` (default **0.20**) sustained for `gates.drift.agreement_flat_window_bars` (default **2000**) → WARN; > `agreement_flat_delta_crit` (default **0.40**) → CRIT routed through §8.3 flatten. Also fires the §4.5 Stage 2.5-R retrain trigger #4. No-op for non-consensus rules (`ens_mean`, `ens_median`, `ens_pf_weighted`).

### 8.3 Tiered safe-mode — NEW in v2.2

Replaces prior "auto-halt" language, which was ambiguous (halt ≠ flatten in trading) and left open positions at risk for the human-response window.

| State | Trigger (any of) | Action |
|---|---|---|
| **WARN** | feature 3σ / Mahalanobis 99% / action-KL > `action_kl_warn` / deadband-delta > `deadband_frac_warn` / saturation-delta > `saturation_frac_warn` | Engine flag `no_new_entries: true` (existing positions held). Telegram WARN. Continue monitoring. |
| **CRIT** | feature 5σ / Mahalanobis 99.9% / action-KL > `action_kl_crit` / deadband-delta > `deadband_frac_crit` / saturation-delta > `saturation_frac_crit` | Engine writes `kill_file` with body `{state: CRIT, cause, ts, signal_values}`. Existing FTMO-style force-close path (S422) flattens all open positions. Container exits non-zero. **Watchdog does NOT auto-restart.** Telegram CRIT. |
| **REPEAT-CRIT LOCKOUT** | CRIT fires ≥ `gates.safe_mode.crit_repeat_count_before_lockout` (default 2) within `crit_repeat_window_hours` (default 24) | Watchdog refuses manual restart until `kill_file.override` written by human. Prevents restart thrash on genuine regime breaks. |

**Implementation notes:**
- Existing FTMO flatten code (S422, `PropFirmWrapperV7.force_close`) is reused — no new flatten primitive. The hook is: drift callback → `kill_file` writer → engine's existing kill-file observer → force-close → exit.
- Watchdog reads `kill_file` metadata (timestamp + count within 24h) to implement lockout. Kill-file format extended with a JSON body (currently a bare sentinel file).
- Stage 5 pre-flight (`validate_config.py`) rejects any live YAML missing `risk.flatten_on_kill_file: true` or `kill_file` path for prop-firm / live-capital configs.
- Cross-reference from §9 retrain cadence: CRIT → halt-paper-for-retrain path always goes through this §8.3 flatten, never through "stop new entries only."

---

## 9. Retrain Cadence

Automatic triggers (watchdog cron). Thresholds are per-workstream in `configs/<workstream>.gates.yaml` under `retrain.*`.

**Timeframe-dependent `retrain.max_age_days` default (v2.5.1):** Microstructure decays faster than price-action regimes, so faster-timeframe strategies require shorter scheduled retrain intervals — but the *training window* stays calendar-anchored at 22–24mo regardless (§3.5.2). The fix for microstructure decay is cadence, not window shrinking.

| Timeframe tier | `retrain.max_age_days` default | Rationale |
|---|---|---|
| ≥ 15m | **90** | Price-action regimes are stable over months; project-wide default |
| 5m–15m | **45** | Mixed dependence; halve the cadence |
| < 5m | **28** | Microstructure regimes (spread/depth/MM rotation) shift week-to-week; SG-1-BTC alpha-decay 2026-05-09 is the cautionary anchor (`project_sg1_btc_alpha_decay_audit_20260509.md`) |

| Trigger | Action |
|---|---|
| Live PF < `retrain.live_pf_ratio` × paper PF (default 0.8) for `retrain.live_pf_window_days` (default 5) consecutive trading days, AND ≥ `retrain.min_trades_window` (default 20) trades in window | Open re-HPO ticket |
| `today − checkpoint.train_end > retrain.max_age_days` (default per timeframe tier above) | Open re-HPO ticket |
| Stage 4 recent-OOS verdict = WATCH at next quarterly review | Open re-HPO ticket |
| Stage 4 recent-OOS verdict = RETRAIN-required | Halt paper via §8.3 CRIT path (graceful flatten, not halt-only), mandatory re-HPO before redeploy |
| §8.1/8.2 CRIT drift alarm | §8.3 CRIT: flatten + kill_file + no auto-restart. Retrain ticket auto-opened. |

The `min_trades_window` floor prevents low-frequency strategies (e.g., funding-arb at hourly cadence) from triggering on 5 trades.
All "halt paper" actions route through §8.3 — no retrain trigger may bypass the flatten step when open positions exist.

Quarterly review: every 90 days, all paper strategies re-run stage 4 (recent-OOS) with current data.

---

## 10. Migration Plan

### Existing workstreams

| Workstream | Current state | Migration action |
|---|---|---|
| GMGP1-XAUUSD | Already staged (HPO → L1 → paper). Static_peak fixed S468. **Stage 2.5 PASS S493** (ens_agreement +17.15% uplift, study `gmgp1_xauusd_ftmo_rehpo_20260421`). | Run stage 3 WF with top-3 seeds using `ens_agreement` aggregation, then stage 4 + compliance before paper swap. |
| GMGP1-BTC | Same. Velotrade no daily cap. | Same; stage 4 compliance sub-report skipped (Velotrade has no daily-loss compliance). |
| GMGP1-GC | Paper running on MGC. | Stage 4 quarterly review at 90-day mark. |
| SG-1 XAUUSD | Constrained-RL Arm B running (`01imgvm7`). | Once ablation gate passes, re-HPO under v2 with stages 0–4. |
| CMGP1 | v2 config drafted (S474). Universe verify pending. | Full v2 from stage 0 (data manifest first). |
| AlphaSeek | TERMINATED pending replan. | If resurrected: refactor `phase6-hpo-4w` mega-loop to stages 1+3, ensemble eval becomes a stage 2/3 sub-report. |
| Funding-Arb | v3 DSAC implemented. HPO pending. | Full v2 from stage 1. |

### New workstreams

Default to v2 from inception. No fused-pipeline pattern allowed.

---

## 11. Open Items

### Must-have for v2 ratification (blocking)

These pieces are required for v2 to be operationally enforceable rather than aspirational:

- `validate_config.py` implementation (encodes all stage 0/5 rules + manifest schema validation, including v2.2 drift/safe_mode gate enforcement for prop-firm / live-capital tags)
- `build_data_manifest.py` implementation (stage 0 producer)
- Manifest JSON-schema file at `docs/schemas/manifest.schema.json` (schema enforcement for §2, including `eval_distribution` block)
- Per-workstream `gates.yaml` files (so §4 thresholds are config-driven, not hardcoded)
- Stage refactor of `run_full_pipeline.py` (currently monolithic at line 745) — minimum viable: `--stage {hpo,l1-multiseed,wf,oos,all}` dispatch with manifest read/write
- **(v2.2, blocking for prop-firm / live-capital only)** Live-feed + live-action drift in `live_obs_builder.py` (§8.1 + §8.2) wired to §8.3 tiered safe-mode
- **(v2.2, blocking for prop-firm / live-capital only)** `eval_distribution` writer in `seed_report.json` (Stage 2) and `ensemble_report.json` (Stage 2.5); backfill for SG-1 XAUUSD + GMGP1 XAUUSD CME paper-deployed checkpoints
- **(v2.2, blocking for prop-firm / live-capital only)** Kill-file JSON-body extension + watchdog repeat-CRIT lockout (`kill_file.override` handshake)
- **(v2.3, blocking for prop-firm / live-capital only)** Block-bootstrap + diversity-aware top-K extension to `scripts/sg1_xauusd_ensemble_eval.py` (writes `ensemble_report.json → bootstrap_verdict` and `diversity_audit`). First customer = SG-1 BTC L1 currently in flight (parent `ighx368o`).
- **(v2.3, blocking for prop-firm / live-capital only)** Atomic ensemble swap bundle (`ensemble_v{N}.tar.gz` writer in `run_stage_2_5_val_selection`; live-engine bundle reader with SHA256 validation + drain-and-cutover swap protocol; `kill_file.swap_approved` handshake).
- **(v2.3, blocking for any ensemble-deployed prop-firm strategy)** Live agreement-decay tracker in `live_obs_builder.py` / `live_action_drift.py` (§8.2 extension); fires WARN/CRIT and §4.5 Stage 2.5-R trigger #4.
- **(v2.4, SHIPPED S495-cont, blocking for prop-firm / live-capital only)** `RiskShapingWrapper` (training) + `ChallengeStateMachine` (live) split replacing `PropFirmWrapperV7`. `finrl_pro_ds/config_utils.py` with `deep_merge(allowlist=)` + `_prep_backtest_config`. Deploy-overlay family `configs/deploy/{ftmo,velotrade}/{step1,step2,funded}.yaml` + `configs/deploy/ALLOWLIST.yaml`. Four new `validate_config.py` checks: `check_legacy_prop_firm_block` (WARN / paper-deploy FAIL), `check_challenge_block`, `check_static_peak_consistency` (S422 train/live divergence guard), `check_overlay_allowlist`. `REASON_PHASE_COMPLETE` kill-file reason distinct from `REASON_DRIFT_CRIT`. `scripts/prop_firm_ab_compare.py` (solo / ensemble / train_parity arms, 7 Q1 metrics) for first-customer A/B. See `decision_prop_firm_decoupling_s495.md`.
- **(v2.4, REMAINING, blocking for prop-firm / live-capital only)** Per-workstream Step-5 rollout (SG-1-XAUUSD FTMO, GMGP1-BTC Velotrade, SG-1-BTC Velotrade, GMGP1-MGC, SG-1-EURUSD — each = rename `env.prop_firm:`→`env.risk:`, drop `success_bonus`/`profit_target_pct`, attach overlay, 48h paper smoke). Ensemble A/B (A2/B2) for GMGP1-XAUUSD FTMO gated on WF-OANDA completion. Step 6 retirement of `PropFirmWrapperV7` adapter gated on ≥30 live-trading days clean across all migrated workstreams.
- **(v2.4.1, blocking for off-policy workstreams that run Stage 2.5)** Replay-buffer purge step in `run_stage_2_5_val_selection()` — rewrites Stage-2 manifest with `replay_buffer.<seed>.purged: true` and unlinks the file for non-selected seeds. Touches all 6 ensemble_eval scripts (gmgp1_btc, gmgp1_xauusd, gmgp1_xauusd_oanda, sg1_xauusd, sg1_btc_velotrade, funding_arb_dsac). Smoke test: rerun on a completed Stage-2 study with `--dry-run` first; verify only top-K + diversity-top-1 buffers remain.
- **(v2.4.1, blocking for prop-firm / live-capital only — supersedes the v2.2 §8.1 stub)** Joint-feature drift implementation in `live_obs_builder.py` uses `sklearn.covariance.LedoitWolf`. `build_data_manifest.py` (§3) extended to write the LW shrinkage parameter and the top-K feature whitelist into the manifest; live path loads them and refuses to start on hash mismatch. Gate-key migration: `gates.drift_mahalanobis_warn`/`_crit` (legacy underscored) → `gates.drift.mahalanobis_warn`/`_crit` (dotted, matches the rest of `gates.drift.*`). `validate_config.py` accepts both for one minor version, then hard-fails legacy form.

### Deferred to v2.5 (nice-to-have)

- `source_parity.py` implementation
- Fixed-lot stress as a callable sub-report (currently inline in stage 3 spec)
- Watchdog cron for retrain triggers
- Skill-chain auto-dispatch wiring (file-based hand-off works in the meantime)
- Re-calibration of action-drift KL defaults (0.5 / 1.0) per asset class after 2–3 workstreams publish `eval_distribution` baselines
- Conditional baselines extended beyond vol quartile (e.g. vol × spread, vol × session) — pending false-positive rate data from vol-only conditioning

---

## 12. Invariants Reaffirmed

The following CLAUDE.md invariants remain authoritative across all stages:

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params during HPO. |
| BUG-03 | `hindsight_weight = 0.0` during backtesting |
| BUG-04 | Dense reward on switch bars uses direction BEFORE switch |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt` |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x) |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`; >30% divergence = halt |

---

## 13. References

- `decision_steady_state_fees_pattern.md` (S464)
- `project_ftmo_risk_manager_fix.md` (S422)
- `project_xauusd_rehpo_prelaunch_checklist.md` (S468)
- `project_paper_checkpoints_q1_2026_blindspot.md` (S470)
- `project_sim_to_live_gap_audit_s470.md` (S470)
- `project_constrained_rl_research.md` (S472)
- `project_sg1_xauusd_q1_ftmo_fixed_lot.md` (S466)
- `.agent/artifacts/sim_to_live_gap_audit_s470.md`
- `decision_alphaseek_terminated_pending_replan.md` (anti-pattern reference)

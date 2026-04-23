# Training → Live Protocol v2.1

> **Status:** Active. Standardizes the training-to-live workflow across all FinRL-Pro_DS workstreams (GMGP1, SG-1, CMGP1, AlphaSeek, Funding-Arb).
> **Reference run:** GMGP1 staged approach. **Anti-pattern:** AlphaSeek `k28l6ef8` monolithic 5.7-day run.
> **Owner:** R&D. **Last updated:** 2026-04-23 Session 495 (Stage 2.5 val-split amendment).
>
> **Version history:**
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
| 2.5 | ensemble-confirm (prop-firm / live-capital; else advisory) | Verify multi-seed `ens_agreement` aggregation beats best-solo on L1 test window | minutes | `ensemble_report.json` (solo + 4 ensemble rules + uplift verdict) |
| 3 | walk-forward (+ fixed-lot stress sub-report) | Temporal robustness across K windows; full-window fixed-lot replay attached as sub-artifact | hours | per-window checkpoints + `wf_report.json` (includes `stress` block) |
| 4 | recent-oos (+ compliance filter sub-report) | OOS test on `today−60d → today−1d`; FTMO/Velotrade compliance filter applied to candidate set | minutes | `oos_report.json` (includes `compliance` block + final selection) |
| 5 | paper-deploy | Live container on `finrl-desktop` | continuous | live engine emits its own runs |

**Stage count:** 7 (stages 0, 1, 2, 2.5, 3, 4, 5). Stage 2.5 is eval-only (no training, no new WandB run by default — it writes a sub-artifact under the stage-2 parent). Fixed-lot stress and compliance filter are *replay/selection* operations on prior outputs, not new compute stages — kept as sub-reports inside stages 3 and 4 to reduce CLI/manifest plumbing without losing rigor.

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
- `outputs.replay_buffer` — stages 1, 2, 3 for off-policy algos (SAC, IQN). Required for `--resume` to work correctly; cold-buffer resume changes effective sample distribution.
- `training_health` — every training stage. Used by stage gates (see §4 stage 1).

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
- **HPO budget** declared per workstream in gate config: `hpo.trials`, `hpo.steps_per_trial`, `hpo.wall_clock_hours_estimate`. `validate_config.py` rejects unbounded HPO.
- **Training-health hard-fail** (any trial that hits these is auto-pruned and excluded from best-trial selection):
  - actor entropy collapses (final entropy < `gates.entropy_floor`, default `-3.0` for SAC)
  - Q-target divergence (`q_target_max_div_ratio > gates.q_div_max`, default `10.0`)
  - action saturation (`action_saturation_pct > gates.action_sat_max`, default `95.0`)
  - any NaN/Inf in losses
- Gate: Optuna best trial PF ≥ `gates.hpo_pf_floor` (default 1.5)

### Stage 2 — l1-multiseed
- N ≥ `gates.l1_seeds` (default 5; **prop-firm / live-capital workstreams must set N ≥ 10** — see S488 rationale below)
- Each seed evaluated over `gates.eval_episodes` (default 10) — for stochastic policies (SAC), report **mean and std across episodes** per seed; deterministic eval (greedy action) is logged additionally for diagnostic purposes
- Report **median** PF, Sharpe, MDD across seeds (not max)
- Gate: median PF ≥ `gates.l1_pf_floor` (default 1.5), all seeds profitable, **CV (std/mean) of PF ≤ `gates.l1_pf_cv_max` (default 0.30)** — tightened from prior 0.5 because CV=0.5 admits PF=2.0 ± 1.0 which is operationally unstable

- **Pre-committed escalation rule (mandatory for prop-firm / live-capital, N ≥ 10):**
  - Declare `gates.l1_pf_cv_ambiguous: [low, high]` (default `[0.22, 0.38]`) in the workstream gate YAML **before launch**, not after seeing results
  - If measured CV ∈ [low, high] the verdict is **AMBIGUOUS** — auto-extend with a second batch of seeds to reach N=20, then re-evaluate the gate. Point estimate outside this range → clear PASS/FAIL at N=10, stop
  - Any seed with PF < 1.0 → immediate FAIL (short-circuit, no need to finish remaining seeds)
  - Rationale: the CV estimator has 95% CI ≈ [0.15, 0.45] at N=10 (McKay/Vangel, assumed approximate normality of PF across seeds). With a gate at 0.30, a measured CV of 0.30 is consistent with true CV anywhere in that interval; N=20 tightens to [0.20, 0.40]. Without the escalation rule the Type I/II error rates of the CV gate are poorly controlled. Literature anchors: Henderson et al. 2018 (*Deep RL That Matters*); Agarwal et al. 2021 (*Statistical Precipice*) both recommend N ≥ 10 for variance-based claims
  - Escalation batches use **different seeds** from the first batch (no overlap) so that CV estimate pools independent samples

### Stage 2.5 — ensemble-confirm (mandatory for prop-firm / live-capital; advisory elsewhere)
**Amended S495 — val-split rule selection supersedes S493 hardcoded canonical rule.** See `decision_ensemble_val_selection_s495.md`. Motivation: GMGP1-BTC L1 Stage 2.5 (S495) showed `ens_agreement` — the S493 canonical rule based on XAUUSD evidence — is the *worst* ensemble on trending BTC (+4.10% vs `ens_mean`'s +8.13%). A hardcoded canonical rule encodes an asset-microstructure bet; val-split selection lets each workstream discover the regime-appropriate rule without leaking test-split information.

- **Inputs:** top-3 seeds by L1 test PF from stage 2 (`seed_report.json`), their checkpoints, L1 **val** window, and L1 **test** window. Both windows must be declared in the config's `data:` block (`val_start_date`, `val_end_date`, `test_start_date`, `test_end_date`).
- **Method:** run `scripts/*_ensemble_eval.py` (all delegate to `sg1.run_stage_2_5_val_selection()`) with `profit_target_pct` disabled and `episode_length=0` (full-window eval):
  - **Phase 1 — val bake-off:** run all 4 ensemble rules + 3 solos on the **val window** (norm cutoff = `train_end_date`). Reports bar-level PF for `ens_mean`, `ens_median`, `ens_agreement`, `ens_pf_weighted`, plus solos.
  - **Phase 2 — rule selection:** `chosen_rule = argmax_{r ∈ ensembles}(val_PF[r])`. Test split is NOT consulted for this choice. Solo PFs on val are logged for audit but don't enter the selection.
  - **Phase 3 — test eval:** run only `chosen_rule` + 3 solos on the **test window** (norm cutoff = `val_end_date`).
  - **Phase 4 — uplift gate:** `uplift = chosen_rule_test_PF / best_solo_test_PF`
- **Gate thresholds (from `gates:` block, no code defaults):**
  - `gates.ensemble_uplift_min` (default **1.10**) → PROMOTE ensemble; stage 3 WF runs all three seeds with `chosen_rule` aggregation, stage 5 live-config declares `agent.ensemble: {rule: <chosen>, seeds: [...]}`
  - `gates.ensemble_ambiguous_min` (default **1.05**) → AMBIGUOUS_RERUN; keep best-solo in the interim, confirm on a second workstream before codifying for that workstream
  - `uplift < ambiguous_min` → SOLO_BEST_FALLBACK; stage 3/5 use best-solo seed (advisory log only for non-prop-firm; hard fallback for prop-firm)
- **Selection rule `gates.ensemble_rule_selection: val_argmax_pf`** must be declared in the config. Other methods (`static:<rule>` for reproducing S493 behavior, `val_argmax_sharpe` etc.) may be added in future amendments.
- **Checkpoint collision guard:** if multiple seeds share a checkpoint dir (e.g. concurrent deploys hit `deploy_bare_metal.py` timestamp-collision), substitute the next-best distinct seed. Ensemble eval **requires** per-seed distinct weight provenance; manifest must record checkpoint SHA256s.
- **Empirical evidence for the amendment (S495, N=3 workstreams):**
  - SG-1 XAUUSD L1 OANDA (S490): `ens_agreement` +17.6% — regime-flippy gold, agreement filter dominates
  - GMGP1 XAUUSD L1 CME (S493): `ens_agreement` +17.15% — same asset class, same rule wins
  - GMGP1 BTC Velotrade L1 (S495): `ens_agreement` +4.10% (below ambiguous floor); `ens_mean` +8.13% — trending BTC, noise-averaging beats consensus-filter. **This is the motivating case.**
  - With val-split selection, BTC would have chosen `ens_mean` at the val stage (hypothesis; falsifiable on next re-run), avoiding the protocol-hardcoded wrong answer.
- **Bias control:** val is already used twice (HP selection, seed selection). Rule selection adds a third, low-entropy use (log₂(4)=2 bits). Verdict artifact logs all 4 val PFs so audits can detect near-tie rule choices that may be overfit.
- **Retro-apply to BTC, leave paper-deployed XAUUSD alone:** Workstreams already paper-deployed on `ens_agreement` keep their S493 verdict (no container churn): SG-1 XAUUSD (paper live per S489), GMGP1 XAUUSD CME (Stage 3 WF already launched on `ens_agreement`). **First customer under S495 = GMGP1-BTC Velotrade L1** (retro-applied — val was never consulted in the S493 run, so running S495 val-selection now is a legitimate use of val, not post-hoc bias). Second customer = SG-1 BTC L1 (launched 2026-04-23 S494-cont, parent `ighx368o`).
- **Cost:** ~8 min wall-time on CPU for 2mo val + 2mo test windows (7 rules on val + 4 rules on test). Previous single-window eval was ~5 min.
- **When not to run:** research-tier workstreams (e.g. CMGP1 crypto) may skip Stage 2.5 — ensemble remains a per-workstream option, not a protocol requirement. `validate_config.py --stage ensemble-confirm` only hard-fails on prop-firm / live-capital tags (`prop-firm`, `FTMO`, `Velotrade`).

### Stage 3 — walk-forward (+ fixed-lot stress sub-report)
- K ≥ `gates.wf_windows` (default 4) rolling windows
- Window split: `train: 12mo, val: 1mo, test: 1mo`, slide by 1mo (workstream may override under `wf.split` in gate config)
- Stochastic eval rules from stage 2 apply per window
- Gate: all windows profitable, median Sharpe > 0, no window MDD breach
- PF cross-check via `mid_price` AND `close` per window; >30% divergence → halt (PF-XCHECK invariant)
- **Fixed-lot stress sub-report** (attached to `wf_report.json` as `stress` block):
  - Replay full WF window with **fixed lot size** (no PropFirm early-term truncation)
  - Required after S466 SG-1 XAUUSD incident (apparent 1.09% DD was 3-sim-day artifact; fixed-lot revealed 4.59%)
  - Stress gate: worst intraday DD ≥ `gates.stress_dd_buffer_pp` (default 0.5pp) from FTMO/Velotrade cap, peak leverage ≤ `gates.stress_leverage_max` (default 1.0×)

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
- Pre-flight: `validate_config.py` re-run on live YAML
- `monitor_fleet.py` exits non-zero on resource contention
- Live config has `risk.static_peak: true`, `kill_file` path set, `taker_fee` matches training manifest, `gap_detection: true` if data manifest required it
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

## 8. Live-Feed Drift (post-deploy)

Extension to `live_obs_builder.py`:

**Per-feature drift** (catches obvious schema/scaling breaks):
- First 1000 live bars compute z-score per feature against training-distribution mean/std (from `feature_distribution` in data manifest)
- WARN if > 3 features drift > 3σ
- CRIT (auto-halt) if > 3 features drift > 5σ

**Joint drift** (catches correlated drift that per-feature misses):
- Compute Mahalanobis distance of the rolling 1000-bar feature mean against the training-distribution multivariate mean (covariance from manifest top-K features)
- WARN if Mahalanobis > `gates.drift_mahalanobis_warn` (default χ²(K, 0.99))
- CRIT (auto-halt) if > `gates.drift_mahalanobis_crit` (default χ²(K, 0.999))

This catches silent feed schema changes, broker feed degradation, and training-to-live distribution shift before the strategy bleeds capital. Per-feature checks miss correlated drift (e.g., all volatility features shifting together within 3σ individually but jointly anomalous), which is the more dangerous failure mode.

---

## 9. Retrain Cadence

Automatic triggers (watchdog cron). Thresholds are per-workstream in `configs/<workstream>.gates.yaml` under `retrain.*`:

| Trigger | Action |
|---|---|
| Live PF < `retrain.live_pf_ratio` × paper PF (default 0.8) for `retrain.live_pf_window_days` (default 5) consecutive trading days, AND ≥ `retrain.min_trades_window` (default 20) trades in window | Open re-HPO ticket |
| `today − checkpoint.train_end > retrain.max_age_days` (default 90) | Open re-HPO ticket |
| Stage 4 recent-OOS verdict = WATCH at next quarterly review | Open re-HPO ticket |
| Stage 4 recent-OOS verdict = RETRAIN-required | Halt paper, mandatory re-HPO before redeploy |

The `min_trades_window` floor prevents low-frequency strategies (e.g., funding-arb at hourly cadence) from triggering on 5 trades.

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

- `validate_config.py` implementation (encodes all stage 0/5 rules + manifest schema validation)
- `build_data_manifest.py` implementation (stage 0 producer)
- Manifest JSON-schema file at `docs/schemas/manifest.schema.json` (schema enforcement for §2)
- Per-workstream `gates.yaml` files (so §4 thresholds are config-driven, not hardcoded)
- Stage refactor of `run_full_pipeline.py` (currently monolithic at line 745) — minimum viable: `--stage {hpo,l1-multiseed,wf,oos,all}` dispatch with manifest read/write

### Deferred to v2.1 (nice-to-have)

- `source_parity.py` implementation
- Fixed-lot stress as a callable sub-report (currently inline in stage 3 spec)
- Live-feed drift z-score + Mahalanobis check in `live_obs_builder.py`
- Watchdog cron for retrain triggers
- Skill-chain auto-dispatch wiring (file-based hand-off works in the meantime)

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

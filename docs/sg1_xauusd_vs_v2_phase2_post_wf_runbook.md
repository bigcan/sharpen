# SG-1-XAUUSD Volume Study v2 Phase 2 — post-WF runbook

Operator playbook for steps 2–5 of the S540-cont chain. Fires *after* the
Stage 3 WF dispatcher (`launch_sg1_xauusd_vs_v2_phase2_wf_multiseed.py`)
returns clean — i.e., all 8 fold cells `wandb_finished_ok=3`.

Pre-staged here so the swap is one-shot ops rather than half a session of
hand-edits. Each block is a checkpoint; halt and surface to operator if any
fails.

---

## 0. Pre-flight

```bash
# Confirm dispatcher exited clean and manifest agrees.
DISP_DIR=$(ls -dt results/sg1_xauusd_vs_v2_phase2_wf_* | head -1)
jq '{n_folds, folds: [.folds[] | {fold_idx, exit_code, wandb_finished_ok}]}' \
   "$DISP_DIR/manifest.json"
# All folds must show exit_code=0 and wandb_finished_ok=3.
```

```bash
# Confirm all 24 ckpts present locally (auto-collected by launch_l1_multiseed).
find checkpoints -path "*sg1-xauusd-vs-v2-phase2-wf-fold*-seed*/checkpoint_final.pth" | wc -l
# Expect: 24
```

If either fails: re-pull via `python scripts/auto_collect_checkpoints.py
--all_instances` and verify all 24 before continuing.

---

## 1. Stage 3 WF gate evaluation (G1–G5)

```bash
python scripts/sg1_xauusd_ensemble_eval.py \
    --wf_config configs/sg1_xauusd_volume_study_v2_phase2_wf_multiseed.yaml \
    --gates_file configs/sg1_xauusd_volume_study_v2_phase2_ensemble.gates.yaml \
    --output_dir results/sg1_xauusd_vs_v2_phase2_wf_ensemble \
    --device cpu
```

Output: `results/sg1_xauusd_vs_v2_phase2_wf_ensemble/`
- `fold_{00..07}/{solo_*,ens_*}_trajectory.parquet` + `*_metrics.json`
- `verdict.json` — top-level `overall: PASS|FAIL`, `decision`, per-gate
  details for G1–G5

Decision matrix (per `configs/sg1_xauusd_volume_study_v2_phase2_ensemble.gates.yaml`):
- **ALL_PASS** → proceed to step 2.
- **ANY_FAIL** → swap aborted; fall back to solo seed with highest
  `wf_median_pf` (`decision.any_fail_action = reject_ensemble_use_best_solo`).
  Surface to operator with the failed gate's details.

Sanity: `chosen_rule` in verdict should match the Stage 2.5-R chosen rule
(`ens_mean` for Phase 2). If WF flips the choice (val_argmax_pf differs at
WF granularity), update `--rule` in step 3 to match.

---

## 2. Bake drift baseline from WF fold-07 ens_mean trajectory

```bash
python scripts/bake_sg1_xauusd_vs_v2_phase2_drift_baseline.py
```

(Defaults: rule=ens_mean, input-root=`results/sg1_xauusd_vs_v2_phase2_wf_ensemble`,
bundle=`results/sg1_xauusd_volume_study_v2_phase2_stage25r/ensemble_v2.tar.gz`.)

Output: `baselines/sg1_xauusd_vs_v2_phase2_fold_07/ensemble_report.json`
- Schema matches `baselines/sg1_xauusd_wf_fold_07_ens_mean/` (top-level
  `ensemble_eval_distribution` with `deadband_frac`, `saturation_frac`,
  `by_vol_quartile`, `regime_cutpoints`).
- Aggregates 8 WF folds (~9k bars) for breadth — robust to single-fold
  regime quirks.
- Bundle SHA recorded in `bundle_metadata` for swap-handshake parity.

Sanity check: print summary's `deadband_frac` and confirm it lands in
[0.30, 0.55] (current live baseline 0.7966 post-S535-R1 was driven by
deadband-threshold-magnification — Phase 2 should land closer to the
ensemble-report-natural ~0.39).

---

## 3. Live config diff (`configs/live_sg1_xauusd_ctrader.yaml`)

Apply this diff (the only file changed in the live config repo for the swap):

```yaml
# --- agent.ensemble: seeds + ckpt pattern + seed_pfs + rule ---
# Predecessor (v1, S488 OANDA WF, 2026-04-22 → swap):
#   seeds: [42, 2025, 3141]
#   checkpoint_pattern: "checkpoints/WF_seed{seed}_fold_{fold:02d}_*/checkpoint_final.pth"
#   seed_pfs: {42: 1.566, 2025: 1.584, 3141: 1.619}
agent:
  ensemble:
    seeds: [2025, 42, 9999]                              # Phase 2 top-3 val_argmax_pf
    checkpoint_pattern: "checkpoints/sg1-xauusd-vs-v2-phase2-wf-fold{fold:02d}-seed{seed}_*/checkpoint_final.pth"
    fold: 7                                              # WF fold-07 trained through Feb 2026
    aggregation_rule: "ens_mean"                         # ← confirm matches Step 1 verdict.chosen_rule
    seed_pfs:                                            # ← from verdict.json wf_median_pf per seed
      2025: <fill from verdict>
      42:   <fill from verdict>
      9999: <fill from verdict>

# --- drift.baseline_path ---
drift:
  baseline_path: "/app/baselines/sg1_xauusd_vs_v2_phase2_fold_07/ensemble_report.json"

# --- wandb.run_id (distinct from predecessor for auditability) ---
wandb:
  run_id: "live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520"
  tags: [..., "vs-v2-phase2", "ensemble-v2"]              # drop "rule-revert", add Phase 2 tags
```

Validate:

```bash
python scripts/validate_config.py \
    --config configs/live_sg1_xauusd_ctrader.yaml \
    --stage paper-deploy
# Expect: status=PASS
```

(Live config is BASE only; full validation pairs it with `configs/deploy/ftmo/<phase>.yaml`
overlay, but the base alone should pass paper-deploy stage gates.)

---

## 4. Build + deploy (atomic-swap with sentinel)

```bash
# 4a. Sentinel — only needed if prev_path is missing on the volume.
# Check first:
docker --context finrl-desktop exec sg1-xauusd \
    ls -la /app/state/ | grep -E "last_bundle|swap_approved"
# If `*last_bundle` exists from the prior live run, skip the touch — the
# swap-detection path will compute the SHA delta and proceed normally.
# If NEITHER exists (e.g. fresh state volume), touch the sentinel:
docker --context finrl-desktop exec sg1-xauusd \
    touch /app/state/finrl_live_sg1_xauusd_kill.swap_approved
# (Per feedback_v23_first_deploy_swap_approved_sentinel.md — FTMO-tagged
# configs require this when prev_path is None.)

# 4b. Image rebuild (picks up new ckpts + new baseline + new bundle via COPY).
cd docker/live
docker --context finrl-desktop compose build --no-cache live-engine
cd ../..

# 4c. Force-recreate sg1-xauusd only (NOT `down` — that's project-wide
# per feedback_manage_strategies_down_is_project_wide).
./scripts/manage_strategies.sh stop sg1-xauusd
docker --context finrl-desktop compose -f docker/live/docker-compose.yaml \
    up -d --force-recreate sg1-xauusd
```

---

## 5. Post-deploy validation (first 60 minutes)

```bash
# 5a. Container health
docker --context finrl-desktop logs -n 80 sg1-xauusd 2>&1 | \
    grep -E "LIVE TRADING STARTED|swap_handshake|baseline.*loaded|ERROR|Traceback"
# Expect: "swap detected v1 -> v2" + "baseline loaded ... vs_v2_phase2_fold_07"
# + "LIVE TRADING STARTED". No tracebacks.

# 5b. WandB run online + clean drift state (give it ~15 bars after start)
# WandB run: live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520
# Check drift/status starts at WARMUP, transitions to OK after min_bars_before_check=500.

# 5c. Watchdog alert sweep (no CRIT after first decision bar)
docker --context finrl-desktop logs watchdog 2>&1 | tail -20 | grep -E "sg1-xauusd|CRIT|WARN"

# 5d. After ~24h soak: re-run live-monitor for fleet sweep
python scripts/monitor_fleet.py
```

Pass criteria for first 24h soak:
- `LIVE TRADING STARTED` clean, no `Traceback` in container logs
- `swap_handshake` logged the bundle transition (v1 → v2 SHA)
- `drift/status=OK` after the 500-bar warmup
- No watchdog CRIT alerts on sg1-xauusd
- WandB run shows `target_position` activity (signal-gate firing), not flat

If any fail → halt and surface; the swap is reversible by reverting the
live config + rebuilding the image (predecessor checkpoints + bundle are
preserved in repo).

---

## Provenance / cross-refs

- Stage 2.5-R verdict: `results/sg1_xauusd_volume_study_v2_phase2_stage25r/verdict.json`
- Stage 2.5-R bundle: `results/sg1_xauusd_volume_study_v2_phase2_stage25r/ensemble_v2.tar.gz`
- Stage 3 WF config: `configs/sg1_xauusd_volume_study_v2_phase2_wf_multiseed.yaml`
- Stage 3 WF gates: `configs/sg1_xauusd_volume_study_v2_phase2_ensemble.gates.yaml`
- Predecessor live config: `configs/live_sg1_xauusd_ctrader.yaml` (head block notes
  the 2026-04-22 deploy + 2026-05-09 rule revert)
- Predecessor baseline: `baselines/sg1_xauusd_wf_fold_07_ens_mean/`
- Sentinel rule: `feedback_v23_first_deploy_swap_approved_sentinel.md`
- Memory bug to remember: `feedback_seed_report_key_scramble_s540.md` (the
  S540 aggregator scrambled batch 1 keys — re-verify any new L1 verdicts
  with the `key == name_seed == ckpt_seed` sanity check before trusting top-3).

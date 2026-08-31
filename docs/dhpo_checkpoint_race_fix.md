# Distributed HPO Checkpoint Race-Condition Fix

**Status:** DRAFT — not yet applied. Blocker for SG-1 BTC Velotrade re-HPO (queue stage 1.5).

## Problem

Study `sg1_xauusd_ftmo_rehpo_20260419` ran 6 workers (3 replicas per GPU × 2 GPUs). All three replicas on a host share the same cwd (`/workspace/DeepScalper`) and the same checkpoint destination:

```python
# sharpen/training/sac_trainer.py:42,140
self.run_name = run_name or "sac_run"
self.ckpt_dir = os.path.join("checkpoints", self.run_name)
```

`make_objective` (`sharpen/hpo/objective.py:348`) constructs `SACTrainer(env, config, device=device, hpo_mode=True)` with **no `run_name`** → every trial in every replica targets `checkpoints/sac_run/checkpoint_final.pth`. Each trial completion overwrites the previous one.

## Impact on S487 SG-1 XAUUSD re-HPO

- Best trial = #56, PF=2.8534 (produced by r1 on gpuhub-1 at 08:22:29 UTC)
- gpuhub-1 last trial = #57 (PF=2.3518) → trial 56 weights overwritten
- gpuhub-2 last trial = #61 (PF=2.6637) → trial 56 not on this host
- **Trial 56 weights are permanently lost.** Only the hyperparameters survive (Postgres + worker log).

## Same bug affects all distributed HPO studies past and future

- SG-1 BTC Velotrade re-HPO (queued stage 1.5) — same 6-worker × 3-replica config
- GMGP1 XAUUSD narrow 30 re-HPO (queued stage 2) — will be affected
- Any prior distributed study: best-trial checkpoint was only recovered if, by coincidence, the best was the last trial on that host

## Proposed fix

Two-part patch, both in `sharpen/hpo/objective.py` (no changes to `SACTrainer` core).

### Part 1 — Trial-unique `run_name` in HPO mode

In `make_objective`, construct the trainer with a `run_name` that encodes `study_name`, `worker_id`, and `trial.number`:

```python
# sharpen/hpo/objective.py, around line 348
study_name = getattr(trial.study, "study_name", "hpo")
worker_id = os.environ.get("DHPO_WORKER_ID", "local")
hpo_run_name = f"{study_name}/worker_{worker_id}/trial_{trial.number:04d}"
trainer = SACTrainer(env, config, device=device, hpo_mode=True, run_name=hpo_run_name)
```

Each trial now writes to `checkpoints/<study>/worker_<id>/trial_<N>/checkpoint_final.pth` — no collisions.

`distributed_hpo_worker.py` should export `DHPO_WORKER_ID` into the subprocess env before running trials (already has the value; just needs the env var).

### Part 2 — Promote-best callback

Register an Optuna callback so the best trial's final checkpoint is atomically copied to a stable path, and spent trial dirs are cleaned:

```python
# In objective.py or the caller
def _promote_best_callback(study, trial):
    if trial.state != TrialState.COMPLETE:
        return
    hpo_dir = Path("checkpoints") / study.study_name
    trial_ckpt = hpo_dir / f"worker_{os.environ.get('DHPO_WORKER_ID','local')}" / f"trial_{trial.number:04d}" / "checkpoint_final.pth"
    # Only this worker's checkpoints are visible on this host; skip if absent
    if not trial_ckpt.exists():
        return
    best_dir = hpo_dir / "best"
    if study.best_trial.number == trial.number:
        best_dir.mkdir(parents=True, exist_ok=True)
        # Atomic replace
        tmp = best_dir / "checkpoint_final.pth.new"
        shutil.copy2(trial_ckpt, tmp)
        os.replace(tmp, best_dir / "checkpoint_final.pth")
        (best_dir / "trial_number.txt").write_text(str(trial.number))
        (best_dir / "params.json").write_text(json.dumps(trial.params, indent=2))
    # Disk hygiene: delete this trial's dir (keep only best copy)
    shutil.rmtree(trial_ckpt.parent, ignore_errors=True)
```

**Important caveat:** in distributed HPO, `study.best_trial` may reference a trial that ran on a *different host* — the checkpoint for the global best will only be promoted by workers that produced a trial matching the current best-number on their local disk. That's correct behavior: each host promotes its own best candidate. After the study finishes, scan both hosts for `checkpoints/<study>/best/` and pick the one whose `trial_number.txt` matches `study.best_trial.number`.

### Part 3 — Post-study collector

Extend `scripts/collect_run.py` (or add a new `scripts/collect_study.py`) with a mode that:

1. Loads the study from Postgres (or local SQLite archive)
2. Determines `best_trial.number`
3. SFTPs `checkpoints/<study>/best/checkpoint_final.pth` from each host; picks the one whose `trial_number.txt` matches.
4. Writes to local `checkpoints/<study>_best/checkpoint_final.pth` + a manifest (trial number, params, PF, host).

## Acceptance criteria

- [ ] SG-1 BTC Velotrade re-HPO runs to completion and its best-trial checkpoint is recoverable (verified by loading weights + forward pass).
- [ ] No cross-trial / cross-replica checkpoint collisions observed on either GPU host (inspect `checkpoints/<study>/` for expected structure).
- [ ] Total disk used by HPO checkpoints stays bounded — only the best-per-host is retained after study completion.
- [ ] `scripts/collect_study.py <study_name>` produces a single final checkpoint + manifest matching Postgres best-trial.

## Known risks / side effects

- **Disk pressure mid-run:** before `_promote_best_callback` cleans up, each worker briefly holds trial N's checkpoint while trial N+1 runs. ~10 MB × 1 retained ≈ negligible.
- **Callback ordering:** the cleanup step must run *after* `best_trial.number` is updated in the study. Optuna callbacks fire post-commit, so this is safe.
- **Backward compat:** non-HPO training (main pipeline, L1 multiseed) passes an explicit `run_name` — unaffected by Part 1.
- **Study ID collisions across workers:** Part 2 keys directories by `DHPO_WORKER_ID`, which `distributed_hpo_coordinator.py` already makes unique (`gpuhub-1-gpu0-r0`, etc.).

## Out of scope (intentionally)

- Retraining the lost SG-1 XAUUSD trial 56 — separate decision (cheap: ~1-2h on one GPU with known HPs).
- Migrating HPO off Neon Postgres — storage works; this patch is filesystem-only.

## Recommendation

Apply **before** launching SG-1 BTC Velotrade stage 1.5. The patch is small (<50 LOC), isolated to HPO code path, and low-risk. Retraining SG-1 XAUUSD trial 56 can happen in parallel with the fix.

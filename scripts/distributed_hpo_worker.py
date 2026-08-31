"""Distributed HPO Worker — runs Optuna trials on a remote GPU instance.

Connects to a shared Optuna PostgreSQL study (created by the coordinator) and
pulls trials dynamically until the global target is reached or SIGTERM is
received.  Uses the **identical** objective function as serial HPO via
``sharpen.hpo.objective.make_objective``.

All project invariants are preserved:
  - LEAK-1: EMA-Z normalization reset at split boundaries (via objective)
  - BUG-01: HPO objective = profit_factor (via objective)
  - BUG-03: hindsight_weight=0.0 during HPO (via objective)
  - HPO-2: 3-seed median PF evaluation (via objective)
  - HPO-3: <30 trades = -999.0 lazy-kill (via objective)

Usage (launched by coordinator via SSH):
    python scripts/distributed_hpo_worker.py \\
        --config configs/gmgp1_sac_gc_15min.yaml \\
        --db_url "postgresql+psycopg://user:pass@host/optuna" \\
        --study_name "gmgp1_hpo_distributed" \\
        --worker_id 3 \\
        --target_trials 50 \\
        --wandb_group "dhpo_gmgp1_20260407"
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import signal
import sys
from pathlib import Path

import yaml

# Ensure project root is importable when launched standalone on remote instances
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna  # noqa: E402
from optuna.storages import RDBStorage  # noqa: E402
from optuna.trial import TrialState  # noqa: E402

import wandb  # noqa: E402

from sharpen.hpo.objective import make_objective  # noqa: E402
from sharpen.hpo.sampler import create_sampler  # noqa: E402
from sharpen.logging import init_wandb, is_consolidated  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FinRL.HPO.Worker")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _sigterm_handler(signum, frame):
    """Handle SIGTERM/SIGINT: set shutdown flag; loop checks it after each trial.

    Note: study.optimize(n_trials=1) is blocking — a long-running trial will run
    to completion before the shutdown flag is observed. Send SIGKILL (kill -9)
    to force-stop a stuck worker; the trial will be marked stale by Optuna's
    heartbeat (grace_period=600s) and retried by another worker.
    """
    global _shutdown
    logger.info("Signal %d received — will exit after current trial completes.", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_worker(
    config_path: str,
    db_url: str,
    study_name: str,
    worker_id: str,
    target_trials: int,
    wandb_group: str,
    device: str = "cuda",
    agent_type: str = "sac",
) -> None:
    """Connect to shared Optuna study and run HPO trials until target reached.

    Args:
        config_path: Path to experiment YAML config.
        db_url: PostgreSQL connection string for Optuna RDBStorage.
        study_name: Name of the Optuna study (must already exist).
        worker_id: Unique worker ID string (e.g. "vastai-12345" or "gpuhub-1-gpu0").
        target_trials: Global target — stop when study has this many completed trials.
        wandb_group: WandB group name for grouping distributed workers.
        device: PyTorch device string ("cuda" or "cpu").
        agent_type: Agent type ("sac", "ppo", "iqn", "bdq").
    """
    global _shutdown

    # S487 race-fix: advertise worker identity to the HPO objective so each
    # trial writes to its own checkpoints/<study>/worker_<id>/trial_<N>/ dir.
    os.environ["DHPO_WORKER_ID"] = worker_id

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = str(Path(config_path).resolve())
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)

    steps_per_trial = config.get("hpo", {}).get("steps_per_trial",
                                                 config.get("training", {}).get("total_timesteps", 500_000))
    logger.info("Steps per trial: %d", steps_per_trial)

    # ------------------------------------------------------------------
    # 2. Connect to shared Optuna storage
    # ------------------------------------------------------------------
    storage = RDBStorage(
        url=db_url,
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    logger.info("Connected to Optuna storage: %s", db_url.split("@")[-1])  # Log host only, not credentials

    # ------------------------------------------------------------------
    # 3. Load existing study (coordinator already created it)
    # ------------------------------------------------------------------
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=create_sampler(config.get("hpo", {}), distributed=True),
    )
    logger.info(
        "Loaded study '%s' — %d trials so far (completed: %d)",
        study_name,
        len(study.trials),
        len([t for t in study.trials if t.state == TrialState.COMPLETE]),
    )

    # ------------------------------------------------------------------
    # 4. Init WandB — attach-only if coordinator set FINRL_WANDB_RUN_ID
    # (S488 round-2 consolidation); else standalone per-worker (legacy).
    # Namespace is NOT set here; objective.py rotates it per-trial so
    # each trial's logs land under `hpo/t<N>/*` on the parent run.
    # ------------------------------------------------------------------
    standalone_name = f"worker_{worker_id}"
    init_wandb(
        config,
        fallback_name=standalone_name,
        tags=["distributed_hpo", study_name, f"worker_{worker_id}"],
        extra_config={
            "worker_id": worker_id,
            "study_name": study_name,
            "target_trials": target_trials,
            "steps_per_trial": steps_per_trial,
            "agent_type": agent_type,
            "device": device,
        },
        # Standalone-mode legacy grouping — consolidated children inherit
        # from the coordinator's parent run and ignore these kwargs.
        group=wandb_group,
        job_type="hpo_worker",
    )
    if is_consolidated():
        logger.info(
            "WandB attached to coordinator run (worker_%s). Trials will "
            "namespace as hpo/t<N>/*.", worker_id,
        )
    else:
        logger.info(
            "WandB initialized standalone — group=%s, worker_%s",
            wandb_group, worker_id,
        )

    # ------------------------------------------------------------------
    # 5. Create objective function
    # ------------------------------------------------------------------
    trial_records: list[dict] = []
    objective = make_objective(
        base_config=copy.deepcopy(config),
        steps_per_trial=steps_per_trial,
        agent_type=agent_type,
        device=device,
        trial_records=trial_records,
    )
    logger.info("Objective function created (agent_type=%s, device=%s)", agent_type, device)

    # ------------------------------------------------------------------
    # 6. Dynamic pull loop — run trials until global target reached
    # ------------------------------------------------------------------
    local_trials_completed = 0
    local_trials_started = 0

    # S487 race-fix: after each committed trial, promote this worker's
    # best-so-far checkpoint to checkpoints/<study>/best/ and delete the
    # spent trial dir to keep disk bounded.
    study_ckpt_root = Path("checkpoints") / study_name
    worker_ckpt_root = study_ckpt_root / f"worker_{worker_id}"
    best_ckpt_root = study_ckpt_root / "best"

    def _promote_best_callback(study, frozen_trial):
        if frozen_trial.state != TrialState.COMPLETE:
            return
        trial_dir = worker_ckpt_root / f"trial_{frozen_trial.number:04d}"
        trial_final = trial_dir / "checkpoint_final.pth"
        try:
            # Compare against best-so-far across the whole study (distributed).
            # If multiple workers' best land on the same trial number, each
            # local copy is valid; last-write-wins is fine because weights are
            # identical (same Optuna trial = same training run on one host).
            try:
                study_best = study.best_trial
            except ValueError:
                study_best = None

            is_global_best = study_best is not None and study_best.number == frozen_trial.number
            if is_global_best and trial_final.exists():
                best_ckpt_root.mkdir(parents=True, exist_ok=True)
                tmp_path = best_ckpt_root / "checkpoint_final.pth.new"
                shutil.copy2(trial_final, tmp_path)
                os.replace(tmp_path, best_ckpt_root / "checkpoint_final.pth")
                manifest = {
                    "study_name": study_name,
                    "trial_number": frozen_trial.number,
                    "value": frozen_trial.value,
                    "params": frozen_trial.params,
                    "promoted_by_worker": worker_id,
                }
                (best_ckpt_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
                logger.info(
                    "Promoted trial %d (PF=%.4f) to %s",
                    frozen_trial.number, frozen_trial.value or 0.0, best_ckpt_root,
                )
            # Clean up this trial's dir regardless (best was already copied)
            if trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)
        except Exception as e:  # noqa: BLE001 — cleanup must never kill the run
            logger.warning("promote-best callback failed for trial %d: %s",
                           frozen_trial.number, e)

    while not _shutdown:
        # S493 fix (DHP-WORKER-OVERRUN): count COMPLETE + RUNNING + WAITING so N
        # concurrent workers don't each pull a fresh trial after observing the
        # same COMPLETE count — previously caused (n_workers - 1) overrun per
        # campaign. FAIL stays un-counted so Optuna's RetryFailedTrialCallback
        # can re-enqueue without the target shrinking.
        in_flight_or_done = len([
            t for t in study.trials
            if t.state in (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)
        ])
        completed_trials = len([
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ])
        if in_flight_or_done >= target_trials:
            logger.info(
                "Global target reached: %d in-flight-or-done / %d target "
                "(completed: %d). Stopping.",
                in_flight_or_done, target_trials, completed_trials,
            )
            break

        logger.info(
            "Worker %s: starting trial (global progress: %d/%d in-flight-or-done, "
            "%d completed, local: %d completed)",
            worker_id,
            in_flight_or_done,
            target_trials,
            completed_trials,
            local_trials_completed,
        )

        # Run exactly 1 trial, then re-check global progress
        local_trials_started += 1
        try:
            study.optimize(
                objective,
                n_trials=1,
                gc_after_trial=True,
                callbacks=[_promote_best_callback],
            )
            local_trials_completed += 1
        except Exception as e:
            logger.error("Trial failed with exception: %s", e)
            # The objective itself handles errors and returns 0.0,
            # so this catches unexpected Optuna-level errors only.
            local_trials_completed += 1  # Count it as done (Optuna recorded the failure)

    # ------------------------------------------------------------------
    # 7. Summary logging
    # ------------------------------------------------------------------
    completed_trials = len([
        t for t in study.trials
        if t.state == TrialState.COMPLETE
    ])

    best_trial = None
    try:
        best_trial = study.best_trial
    except ValueError:
        logger.warning("No completed trials in study — cannot determine best trial.")

    summary: dict = {
        "worker_id": worker_id,
        "local_trials_started": local_trials_started,
        "local_trials_completed": local_trials_completed,
        "global_trials_completed": completed_trials,
        "target_trials": target_trials,
        "shutdown_requested": _shutdown,
    }
    if best_trial is not None:
        summary["best_trial_number"] = best_trial.number
        summary["best_trial_value"] = best_trial.value
        summary["best_trial_params"] = best_trial.params

    wandb.log({"worker_summary": summary})
    logger.info("Worker %s finished — %d local trials, %d global completed",
                worker_id, local_trials_completed, completed_trials)
    if best_trial is not None:
        logger.info("Study best: trial %d, PF=%.4f, params=%s",
                     best_trial.number, best_trial.value, best_trial.params)

    # Log local trial records table
    if trial_records:
        wandb.log({"trial_records": wandb.Table(
            columns=list(trial_records[0].keys()),
            data=[list(r.values()) for r in trial_records],
        )})

    wandb.finish()
    logger.info("Worker %s — WandB run finished. Exiting.", worker_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed HPO Worker — run Optuna trials on a remote GPU instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        required=False,
        default=None,
        help=(
            "PostgreSQL connection string. If omitted, reads DISTRIBUTED_HPO_DB_URL "
            "from env. Prefer the env path on shared hosts — secrets in argv are "
            "readable to any user via `ps -ef`."
        ),
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the Optuna study (must already exist, created by coordinator).",
    )
    parser.add_argument(
        "--worker_id",
        type=str,
        required=True,
        help="Unique worker ID string (e.g. vastai-12345, gpuhub-1-gpu0).",
    )
    parser.add_argument(
        "--target_trials",
        type=int,
        required=True,
        help="Global target — worker stops when study has this many completed trials.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        required=True,
        help="WandB group name for grouping distributed workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda).",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="sac",
        choices=["sac", "ppo", "iqn", "bdq"],
        help="Agent type for HPO (default: sac).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_url = args.db_url or os.environ.get("DISTRIBUTED_HPO_DB_URL")
    if not db_url:
        print(
            "error: --db_url not provided and DISTRIBUTED_HPO_DB_URL env var "
            "not set. The coordinator should set it via the launch script.",
            file=sys.stderr,
        )
        sys.exit(2)
    run_worker(
        config_path=args.config,
        db_url=db_url,
        study_name=args.study_name,
        worker_id=args.worker_id,
        target_trials=args.target_trials,
        wandb_group=args.wandb_group,
        device=args.device,
        agent_type=args.agent_type,
    )

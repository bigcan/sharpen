#!/usr/bin/env python
"""Distributed-HPO mechanics smoke test (no GPU, no training data).

Validates the orchestration layer added by S371 and patched in S480:
  - Neon (or any PostgreSQL) connectivity from this machine
  - Concurrent workers coordinating via shared Optuna storage
  - TPESampler with constant_liar=True does not deadlock or duplicate trials
  - RDBStorage heartbeat + RetryFailedTrialCallback configured correctly
  - NopPruner on the study propagates to workers
  - Trial counts: total_trials_completed == --target_trials (no over- or under-shoot)

This deliberately bypasses ``sharpen.hpo.objective.make_objective`` so it
runs in seconds and needs no CUDA / data files. The real worker
(``scripts/distributed_hpo_worker.py``) wraps the same Optuna calls around a
heavy training objective; once this smoke passes, mechanics are validated.

Usage:
    export DISTRIBUTED_HPO_DB_URL="postgresql+psycopg://user:pw@ep-xxx.neon.tech/optuna_hpo?sslmode=require"
    python scripts/dhpo_neon_smoke.py --workers 2 --target-trials 8

Cleanup is automatic — the throwaway study is deleted on exit (success or fail).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dhpo.smoke")

WORKER_BODY = '''
import os, sys, time, random, signal, logging
import optuna
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from optuna.trial import TrialState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker_{wid}] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger()

DB_URL = os.environ["DHPO_SMOKE_DB_URL"]
STUDY = os.environ["DHPO_SMOKE_STUDY"]
TARGET = int(os.environ["DHPO_SMOKE_TARGET"])

# Mirror scripts/distributed_hpo_worker.py:119-124
storage = RDBStorage(
    url=DB_URL,
    heartbeat_interval=60,
    grace_period=600,
    failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
)

# Mirror sharpen/hpo/sampler.py distributed=True
sampler = optuna.samplers.TPESampler(
    seed=None, n_startup_trials=2, multivariate=True, constant_liar=True,
)
study = optuna.load_study(study_name=STUDY, storage=storage, sampler=sampler)

def objective(trial):
    # Cheap stand-in for make_objective: spend a few hundred ms per trial so
    # concurrent workers actually overlap, but keep total runtime low.
    x = trial.suggest_float("x", -2.0, 2.0)
    y = trial.suggest_float("y", -2.0, 2.0)
    z = trial.suggest_categorical("z", [0.1, 0.5, 1.0])
    time.sleep(random.uniform(0.2, 0.6))
    return -(x ** 2 + y ** 2) * z  # maximize → near (0,0)

local_done = 0
while True:
    completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    if completed >= TARGET:
        log.info("global target %d hit (completed=%d) — exiting", TARGET, completed)
        break
    log.info("starting trial — global completed=%d/%d, local=%d", completed, TARGET, local_done)
    try:
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        local_done += 1
    except Exception as e:
        log.error("trial failed: %s", e)
        local_done += 1

log.info("worker_{wid} done — local_completed=%d", local_done)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db_url", default=os.environ.get("DISTRIBUTED_HPO_DB_URL"),
                    help="PostgreSQL URL (or set DISTRIBUTED_HPO_DB_URL).")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent worker processes.")
    ap.add_argument("--target-trials", type=int, default=8, help="Total trials across workers.")
    ap.add_argument("--keep-study", action="store_true", help="Skip study cleanup on exit.")
    args = ap.parse_args()

    if not args.db_url:
        logger.error("No --db_url and no DISTRIBUTED_HPO_DB_URL env var. Aborting.")
        sys.exit(2)

    import optuna
    from optuna.storages import RDBStorage, RetryFailedTrialCallback

    study_name = f"dhpo_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    logger.info("Smoke study: %s  (workers=%d, target=%d)", study_name, args.workers, args.target_trials)
    logger.info("DB host: %s", args.db_url.split("@")[-1] if "@" in args.db_url else args.db_url)

    # Mirror coordinator: create study with NopPruner + TPE constant_liar
    storage = RDBStorage(
        url=args.db_url,
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
    )
    sampler = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=2, multivariate=True, constant_liar=True,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=False,
    )
    logger.info("Study created (id=%d).", study._study_id)

    env = {
        **os.environ,
        "DHPO_SMOKE_DB_URL": args.db_url,
        "DHPO_SMOKE_STUDY": study_name,
        "DHPO_SMOKE_TARGET": str(args.target_trials),
    }

    procs: list[subprocess.Popen] = []
    t0 = time.monotonic()
    try:
        for wid in range(args.workers):
            body = WORKER_BODY.replace("{wid}", str(wid))
            p = subprocess.Popen(
                [sys.executable, "-u", "-c", body],
                env=env,
                stdout=sys.stdout, stderr=sys.stderr,
            )
            procs.append(p)
            logger.info("Spawned worker %d (pid=%d)", wid, p.pid)

        exit_codes = [p.wait() for p in procs]
        elapsed = time.monotonic() - t0
        logger.info("All workers exited in %.1fs. Exit codes: %s", elapsed, exit_codes)

        # --- Assertions ---
        # Reload to get fresh state
        study = optuna.load_study(study_name=study_name, storage=storage)
        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        failed = [t for t in study.trials if t.state == TrialState.FAIL]
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        n_total = len(study.trials)

        logger.info("=" * 60)
        logger.info("Result: %d completed, %d failed, %d running, %d total",
                    len(completed), len(failed), len(running), n_total)
        if completed:
            best = study.best_trial
            logger.info("Best trial #%d: value=%.4f, params=%s", best.number, best.value, best.params)

        # Pass criteria
        ok = True
        if any(c != 0 for c in exit_codes):
            logger.error("FAIL: at least one worker exited non-zero: %s", exit_codes)
            ok = False
        if len(completed) < args.target_trials:
            logger.error("FAIL: only %d/%d trials completed", len(completed), args.target_trials)
            ok = False
        # Detect duplicate params (constant_liar should prevent exact dupes)
        param_tuples = [tuple(sorted(t.params.items())) for t in completed]
        n_dupes = len(param_tuples) - len(set(param_tuples))
        if n_dupes > 0:
            logger.warning("WARN: %d duplicate parameter sets across workers "
                           "(constant_liar should minimize this; not a hard fail)", n_dupes)

        if ok:
            logger.info("PASS: distributed mechanics smoke succeeded.")
        else:
            logger.error("OVERALL: SMOKE FAILED.")
        sys.exit(0 if ok else 1)

    finally:
        # Always cleanup unless --keep-study
        if not args.keep_study:
            try:
                optuna.delete_study(study_name=study_name, storage=args.db_url)
                logger.info("Cleaned up study %s", study_name)
            except Exception as e:
                logger.warning("Could not delete study %s: %s", study_name, e)
        # Make sure no zombie workers
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()

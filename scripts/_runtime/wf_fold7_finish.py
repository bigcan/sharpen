"""Fold-7-only recovery dispatcher.

Original recovery dispatcher (wf_recovery_finish.py, bg ID `bxu18d4pj`)
died with exit code 4 after firing fold 6 at 10:03 UTC. Folds 5+6 already
in flight / done at the time of death; this script picks up at fold 7.

Behavior:
  1. Wait for fold 6 (all 3 seeds) to reach terminal state on WandB.
  2. Fire fold 7 with `DEPLOY_SETUP_GAP_S=900` 3-seed parallel across
     gpuhub-1:0,gpuhub-1:1,gpuhub-2:0.
  3. Auto-solo-relaunch any seed that fails to register / finish.
  4. Exit clean.

Spawn via Bash run_in_background. Logs to logs/wf_fold7_finish_<ts>.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

RUNTIME_DIR = PROJECT_ROOT / "configs" / "_gmgp1_xauusd_wf_extended_runtime"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"

ENTITY = "bigcan-chiwin-technology"
PROJECT = "FinRL-Pro-DS"
STUDY_ID = "20260527_015652"

SEEDS = [789, 2025, 1024]
PARALLEL_SLOTS = "gpuhub-1:0,gpuhub-1:1,gpuhub-2:0"
SOLO_SLOT = "gpuhub-2:0"
SETUP_GAP_S_PARALLEL = "900.0"
SETUP_GAP_S_SOLO = "60.0"

POLL_S = 120
WANDB_GRACE_S = 600
MAX_FOLD_WAIT_H = 14.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / f"gmgp1_xauusd_wf_extended_fold7_{TS}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = RESULTS_DIR / "manifest.json"

manifest = {
    "timestamp": TS,
    "study_id": STUDY_ID,
    "seeds": SEEDS,
    "parallel_slots": PARALLEL_SLOTS,
    "solo_slot": SOLO_SLOT,
    "setup_gap_parallel_s": float(SETUP_GAP_S_PARALLEL),
    "actions": [],
}


def _save_manifest() -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def wandb_seed_states(fold_idx: int) -> dict[int, list[tuple[str, str]]]:
    api = wandb.Api(timeout=30)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"$and": [
        {"tags": {"$in": ["wf-stage3"]}},
        {"tags": {"$in": ["gmgp1-xauusd"]}},
        {"tags": {"$in": ["extended-window"]}},
        {"tags": {"$in": [f"fold-{fold_idx}"]}},
        {"tags": {"$in": [f"study-id-{STUDY_ID}"]}},
    ]}))
    out: dict[int, list[tuple[str, str]]] = {s: [] for s in SEEDS}
    for r in runs:
        for s in SEEDS:
            if f"-seed{s}_" in r.name:
                out[s].append((r.name, r.state))
                break
    return out


def fire_launcher(fold_idx: int, seeds: list[int], slots: str,
                  setup_gap_s: str, log_path: Path) -> int:
    cfg = (RUNTIME_DIR / f"gmgp1_xauusd_wf_extended_fold_{fold_idx}.yaml") \
        .relative_to(PROJECT_ROOT).as_posix()
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", cfg,
        "--seeds", ",".join(str(s) for s in seeds),
        "--run_name_prefix", f"gmgp1-xauusd-ext-wf-fold{fold_idx}",
        "--separate_runs",
        "--slots", slots,
    ]
    env = os.environ.copy()
    env["DEPLOY_SETUP_GAP_S"] = setup_gap_s
    env["PYTHONIOENCODING"] = "utf-8"
    log(f"  launcher cmd: {' '.join(cmd)}")
    log(f"  setup gap: {setup_gap_s}s  log -> {log_path.relative_to(PROJECT_ROOT)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def all_terminal(states: dict[int, str]) -> bool:
    return all(st in ("finished", "crashed", "failed") for st in states.values())


def poll_until_terminal(fold_idx: int) -> dict[int, str]:
    """Returns {seed: most-recent-state} once all 3 seeds are terminal,
    treating MISSING as terminal (caller decides to relaunch)."""
    deadline = time.time() + MAX_FOLD_WAIT_H * 3600
    last_print: tuple = ()
    while time.time() < deadline:
        try:
            states_by_seed = wandb_seed_states(fold_idx)
        except Exception as e:
            log(f"  wandb poll error: {e}; retrying in {POLL_S}s")
            time.sleep(POLL_S)
            continue
        per_seed: dict[int, str] = {}
        for s in SEEDS:
            runs = states_by_seed.get(s, [])
            if not runs:
                per_seed[s] = "MISSING"
            else:
                fin = [r for r in runs if r[1] == "finished"]
                term = [r for r in runs if r[1] in ("finished", "crashed", "failed")]
                if fin:
                    per_seed[s] = "finished"
                elif term:
                    per_seed[s] = term[0][1]
                else:
                    per_seed[s] = runs[0][1]
        cur = tuple(sorted(per_seed.items()))
        if cur != last_print:
            log(f"  fold {fold_idx} state: {per_seed}")
            last_print = cur
        non_terminal = [s for s, st in per_seed.items()
                        if st not in ("finished", "crashed", "failed", "MISSING")]
        if not non_terminal:
            return per_seed
        time.sleep(POLL_S)
    log(f"  TIMEOUT after {MAX_FOLD_WAIT_H:.1f}h on fold {fold_idx}")
    return per_seed


def run_fold(fold_idx: int) -> bool:
    log(f"=== FOLD {fold_idx} START ===")
    manifest["actions"].append({"fold": fold_idx, "phase": "start", "at": _ts()})
    _save_manifest()

    log_p = LOG_DIR / f"wf_fold7_finish_fold{fold_idx}_p1_{TS}.log"
    rc = fire_launcher(fold_idx, list(SEEDS), PARALLEL_SLOTS,
                       SETUP_GAP_S_PARALLEL, log_p)
    manifest["actions"].append({
        "fold": fold_idx, "phase": "parallel_fire",
        "seeds": list(SEEDS), "rc": rc, "at": _ts(),
    })
    _save_manifest()

    log(f"  sleeping {WANDB_GRACE_S}s for wandb.init grace")
    time.sleep(WANDB_GRACE_S)

    states = wandb_seed_states(fold_idx)
    unregistered = [s for s in SEEDS if not states.get(s)]
    log(f"  registered: {[s for s in SEEDS if states.get(s)]}  unregistered: {unregistered}")

    for s in unregistered:
        log(f"  solo-relaunching seed {s} on {SOLO_SLOT}")
        log_s = LOG_DIR / f"wf_fold7_finish_fold{fold_idx}_solo_seed{s}_{TS}.log"
        rc = fire_launcher(fold_idx, [s], SOLO_SLOT, SETUP_GAP_S_SOLO, log_s)
        manifest["actions"].append({
            "fold": fold_idx, "phase": "solo_relaunch",
            "seed": s, "rc": rc, "at": _ts(),
        })
        _save_manifest()
        time.sleep(WANDB_GRACE_S)

    log(f"  waiting for fold {fold_idx} to reach terminal")
    final = poll_until_terminal(fold_idx)
    log(f"  final: {final}")
    manifest["actions"].append({"fold": fold_idx, "phase": "final",
                                "result": final, "at": _ts()})
    _save_manifest()

    not_finished = [s for s, st in final.items() if st != "finished"]
    if not_finished:
        log(f"  retry round: solo-relaunching {not_finished}")
        for s in not_finished:
            log_s = LOG_DIR / f"wf_fold7_finish_fold{fold_idx}_r2_seed{s}_{TS}.log"
            rc = fire_launcher(fold_idx, [s], SOLO_SLOT, SETUP_GAP_S_SOLO, log_s)
            manifest["actions"].append({
                "fold": fold_idx, "phase": "solo_relaunch_r2",
                "seed": s, "rc": rc, "at": _ts(),
            })
            _save_manifest()
            time.sleep(WANDB_GRACE_S)
        final = poll_until_terminal(fold_idx)
        manifest["actions"].append({"fold": fold_idx, "phase": "final_r2",
                                    "result": final, "at": _ts()})
        _save_manifest()
        if any(st != "finished" for st in final.values()):
            log(f"  FOLD {fold_idx} INCOMPLETE — escalating to operator")
            return False
    log(f"=== FOLD {fold_idx} COMPLETE ===")
    return True


def main() -> int:
    log("GMGP1-XAUUSD WF fold-7 finish dispatcher starting")
    log(f"  study_id: {STUDY_ID}")
    log(f"  manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    _save_manifest()

    log("STEP 1: waiting for fold 6 to reach terminal on wandb")
    fold6 = poll_until_terminal(6)
    log(f"STEP 1 done: {fold6}")
    manifest["actions"].append({"fold": 6, "phase": "wait_running",
                                "result": fold6, "at": _ts()})
    _save_manifest()

    # If fold 6 has any non-finished seed, solo-relaunch + retry
    not_finished_6 = [s for s, st in fold6.items() if st != "finished"]
    if not_finished_6:
        log(f"STEP 1.5: fold 6 has unfinished seeds {not_finished_6} — solo retry")
        for s in not_finished_6:
            log_s = LOG_DIR / f"wf_fold7_finish_fold6_retry_seed{s}_{TS}.log"
            rc = fire_launcher(6, [s], SOLO_SLOT, SETUP_GAP_S_SOLO, log_s)
            manifest["actions"].append({
                "fold": 6, "phase": "solo_relaunch_retry",
                "seed": s, "rc": rc, "at": _ts(),
            })
            _save_manifest()
            time.sleep(WANDB_GRACE_S)
        fold6 = poll_until_terminal(6)
        manifest["actions"].append({"fold": 6, "phase": "final_retry",
                                    "result": fold6, "at": _ts()})
        _save_manifest()
        if any(st != "finished" for st in fold6.values()):
            log("STEP 1.5: fold 6 still incomplete — escalating")
            return 1

    log("STEP 2: fold 7")
    if not run_fold(7):
        return 1

    log("ALL FOLDS COMPLETE")
    manifest["actions"].append({"phase": "all_done", "at": _ts()})
    _save_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())

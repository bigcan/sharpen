"""Recovery dispatcher: drive GMGP1-XAUUSD extended-window Stage 3 WF to completion.

State as of spawn (2026-05-27 ~00:50 UTC):
  - folds 0..4: 3/3 finished cleanly
  - fold 5: seeds 789 + 1024 RUNNING on WandB (step ~341k/500k, ETA ~01:35 UTC),
            seed 2025 silently died at remote python startup with
            `ImportError: cannot import name 'Imports' from
            wandb.proto.wandb_telemetry_pb2` — concurrent pip install race
            with seed 789 on gpuhub-1 (5-min stagger lost the race)
  - folds 6, 7: NOT launched (original wf_folds4567 dispatcher died after
                fold-5 spawn at ~23:25 UTC, parent shell closed)

Recovery plan:
  1. Wait on WandB for fold-5 seeds 789 + 1024 to reach 'finished'.
  2. Relaunch fold-5 seed 2025 SOLO on gpuhub-2:0 (no pip race possible).
  3. Fire fold 6 (3 seeds, 3-slot pool, 900s setup-gap via env override) +
     auto-relaunch missing seeds solo if wandb registration falls short.
  4. Same for fold 7.
  5. Final manifest at results/gmgp1_xauusd_wf_extended_recovery_<ts>/.

Survives parent shell close: spawn via Bash run_in_background.
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
STUDY_ID = "20260527_015652"      # inherited from existing fold tags
EXTENDED_TAG = "extended-window"
WF_TAG = "wf-stage3"
ASSET_TAG = "gmgp1-xauusd"

SEEDS = [789, 2025, 1024]
PARALLEL_SLOTS = "gpuhub-1:0,gpuhub-1:1,gpuhub-2:0"
SOLO_SLOT = "gpuhub-2:0"
SETUP_GAP_S_PARALLEL = "900.0"     # 15-min stagger — defends against pip race
SETUP_GAP_S_SOLO = "60.0"          # single deploy, no contention

POLL_S = 120
WANDB_GRACE_S = 600                # after launcher OK, wait this long for init
MAX_RELAUNCH_PER_SEED = 2
MAX_FOLD_WAIT_H = 14.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / f"gmgp1_xauusd_wf_extended_recovery_{TS}"
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
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)


def wandb_seed_states(fold_idx: int) -> dict[int, list[tuple[str, str]]]:
    """Return {seed: [(run_name, state), ...]} for fold's runs on WandB."""
    api = wandb.Api(timeout=30)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"$and": [
        {"tags": {"$in": [WF_TAG]}},
        {"tags": {"$in": [ASSET_TAG]}},
        {"tags": {"$in": [EXTENDED_TAG]}},
        {"tags": {"$in": [f"fold-{fold_idx}"]}},
        {"tags": {"$in": [f"study-id-{STUDY_ID}"]}},
    ]}))
    out: dict[int, list[tuple[str, str]]] = {s: [] for s in SEEDS}
    for r in runs:
        # run name format: gmgp1-xauusd-ext-wf-fold{F}-seed{S}_<ts>
        nm = r.name
        for s in SEEDS:
            if f"-seed{s}_" in nm:
                out[s].append((nm, r.state))
                break
    return out


def fire_launcher(fold_idx: int, seeds: list[int], slots: str,
                  setup_gap_s: str, log_path: Path) -> int:
    """Spawn launch_l1_multiseed.py for `seeds` on `slots`.

    Returns the launcher's exit code (0 = all deploys succeeded). Note: this
    is a DEPLOY success indicator, not a training/wandb success indicator —
    the caller still has to verify wandb-side state afterwards.
    """
    cfg = (RUNTIME_DIR / f"gmgp1_xauusd_wf_extended_fold_{fold_idx}.yaml") \
        .relative_to(PROJECT_ROOT).as_posix()
    seeds_csv = ",".join(str(s) for s in seeds)
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", cfg,
        "--seeds", seeds_csv,
        "--run_name_prefix", f"gmgp1-xauusd-ext-wf-fold{fold_idx}",
        "--separate_runs",
        "--slots", slots,
    ]
    env = os.environ.copy()
    env["DEPLOY_SETUP_GAP_S"] = setup_gap_s
    log(f"  launcher cmd: {' '.join(cmd)}")
    log(f"  setup gap: {setup_gap_s}s   log -> {log_path.relative_to(PROJECT_ROOT)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def wait_fold_finished(fold_idx: int, expected_seeds: list[int],
                       max_wait_h: float = MAX_FOLD_WAIT_H) -> dict[int, str]:
    """Poll until all expected_seeds are in terminal state on WandB.

    Returns {seed: final_state} where final_state ∈ {finished, crashed,
    failed, MISSING, TIMEOUT}.
    """
    deadline = time.time() + max_wait_h * 3600
    last_print = None
    while time.time() < deadline:
        try:
            states_by_seed = wandb_seed_states(fold_idx)
        except Exception as e:
            log(f"  wandb poll error: {e}; retrying in {POLL_S}s")
            time.sleep(POLL_S); continue

        # Pick the most-recent run state per seed (handle relaunch duplicates)
        per_seed_state: dict[int, str] = {}
        for s in expected_seeds:
            runs = states_by_seed.get(s, [])
            if not runs:
                per_seed_state[s] = "MISSING"
            else:
                # If any 'finished', prefer that; else any terminal; else first
                term = [r for r in runs if r[1] in ("finished", "crashed", "failed")]
                fin = [r for r in term if r[1] == "finished"]
                if fin:
                    per_seed_state[s] = "finished"
                elif term:
                    per_seed_state[s] = term[0][1]
                else:
                    per_seed_state[s] = runs[0][1]

        cur = tuple(sorted(per_seed_state.items()))
        if cur != last_print:
            log(f"  fold {fold_idx} state: {per_seed_state}")
            last_print = cur

        non_terminal = [s for s, st in per_seed_state.items()
                        if st not in ("finished", "crashed", "failed", "MISSING")]
        if not non_terminal:
            return per_seed_state
        time.sleep(POLL_S)

    log(f"  TIMEOUT after {max_wait_h:.1f}h on fold {fold_idx}")
    out = wandb_seed_states(fold_idx)
    return {s: (out[s][0][1] if out[s] else "TIMEOUT") for s in expected_seeds}


def run_fold(fold_idx: int, missing_seeds: list[int]) -> bool:
    """Drive one fold to completion. Returns True on full success."""
    log(f"=== FOLD {fold_idx} START (missing seeds: {missing_seeds}) ===")
    manifest["actions"].append({
        "fold": fold_idx,
        "phase": "start",
        "missing_seeds": list(missing_seeds),
        "at": _ts(),
    })
    _save_manifest()

    # Phase 1: parallel fire if more than 1 seed, else solo
    if len(missing_seeds) >= 2:
        log(f"  PHASE 1: parallel fire {missing_seeds} on {PARALLEL_SLOTS}")
        log_p = LOG_DIR / f"wf_recovery_fold{fold_idx}_p1_{TS}.log"
        rc = fire_launcher(fold_idx, missing_seeds, PARALLEL_SLOTS,
                           SETUP_GAP_S_PARALLEL, log_p)
        manifest["actions"].append({
            "fold": fold_idx, "phase": "parallel_fire",
            "seeds": list(missing_seeds), "rc": rc, "at": _ts(),
        })
        _save_manifest()
        if rc != 0:
            log(f"  PHASE 1 launcher exit={rc} — continuing to check wandb anyway")

        log(f"  PHASE 1: sleeping {WANDB_GRACE_S}s for wandb.init grace")
        time.sleep(WANDB_GRACE_S)
    else:
        log(f"  single-seed work — skipping parallel phase")

    # Phase 2: check wandb registration and identify gaps
    states = wandb_seed_states(fold_idx)
    registered = [s for s in missing_seeds if states.get(s)]
    unregistered = [s for s in missing_seeds if not states.get(s)]
    log(f"  registered on wandb: {registered}   unregistered: {unregistered}")

    # Phase 3: solo-relaunch each unregistered seed
    for s in unregistered:
        log(f"  PHASE 3: solo-relaunching seed {s} on {SOLO_SLOT}")
        log_s = LOG_DIR / f"wf_recovery_fold{fold_idx}_solo_seed{s}_{TS}.log"
        rc = fire_launcher(fold_idx, [s], SOLO_SLOT, SETUP_GAP_S_SOLO, log_s)
        manifest["actions"].append({
            "fold": fold_idx, "phase": "solo_relaunch",
            "seed": s, "rc": rc, "at": _ts(),
        })
        _save_manifest()
        time.sleep(WANDB_GRACE_S)

    # Phase 4: wait for ALL 3 expected seeds to reach terminal state on wandb
    log(f"  PHASE 4: waiting for fold {fold_idx} — all {len(SEEDS)} seeds to reach terminal")
    final = wait_fold_finished(fold_idx, list(SEEDS))
    log(f"  PHASE 4 final: {final}")
    manifest["actions"].append({
        "fold": fold_idx, "phase": "final", "result": final, "at": _ts(),
    })
    _save_manifest()

    # Phase 5: any seed still not 'finished'? Try one more solo relaunch
    not_finished = [s for s, st in final.items() if st != "finished"]
    if not_finished:
        log(f"  PHASE 5: seeds {not_finished} not finished — solo relaunch round 2")
        for s in not_finished:
            log_s = LOG_DIR / f"wf_recovery_fold{fold_idx}_solo_seed{s}_r2_{TS}.log"
            rc = fire_launcher(fold_idx, [s], SOLO_SLOT, SETUP_GAP_S_SOLO, log_s)
            manifest["actions"].append({
                "fold": fold_idx, "phase": "solo_relaunch_r2",
                "seed": s, "rc": rc, "at": _ts(),
            })
            _save_manifest()
            time.sleep(WANDB_GRACE_S)
        final = wait_fold_finished(fold_idx, list(SEEDS))
        manifest["actions"].append({
            "fold": fold_idx, "phase": "final_r2", "result": final, "at": _ts(),
        })
        _save_manifest()
        if any(st != "finished" for st in final.values()):
            log(f"  FOLD {fold_idx} INCOMPLETE — escalating to operator")
            return False

    log(f"=== FOLD {fold_idx} COMPLETE ===")
    return True


def main() -> int:
    log("GMGP1-XAUUSD extended-window WF recovery dispatcher starting")
    log(f"  study_id: {STUDY_ID}")
    log(f"  manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    log(f"  parallel slots: {PARALLEL_SLOTS}  (DEPLOY_SETUP_GAP_S={SETUP_GAP_S_PARALLEL}s)")
    log(f"  solo slot: {SOLO_SLOT}")
    _save_manifest()

    # Step 1: wait for fold-5 seeds 789 + 1024 (already running)
    log("STEP 1: waiting for fold-5 seeds 789 + 1024 to finish on wandb")
    fold5_initial = wait_fold_finished(5, [789, 1024])
    log(f"STEP 1 done: {fold5_initial}")
    manifest["actions"].append({
        "fold": 5, "phase": "wait_running", "result": fold5_initial, "at": _ts(),
    })
    _save_manifest()
    # if 789 or 1024 didn't finish, relaunch them too
    fold5_missing = [s for s in SEEDS if fold5_initial.get(s, "MISSING") != "finished"
                     or s == 2025]
    # seed 2025 always needs relaunching
    if 2025 not in fold5_missing:
        fold5_missing.append(2025)
    fold5_missing = sorted(set(fold5_missing))
    log(f"STEP 2: fold-5 work to do: {fold5_missing}")
    if not run_fold(5, fold5_missing):
        return 1

    # Step 3: fold 6
    log("STEP 3: fold 6")
    if not run_fold(6, list(SEEDS)):
        return 1

    # Step 4: fold 7
    log("STEP 4: fold 7")
    if not run_fold(7, list(SEEDS)):
        return 1

    log("ALL FOLDS COMPLETE")
    manifest["actions"].append({"phase": "all_done", "at": _ts()})
    _save_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())

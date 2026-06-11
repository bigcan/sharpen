"""Wrapper: wait for fold-1 to finish on WandB, then fire Streams A + B in parallel.

Spawned S550-cont-8 after killing the sequential dispatcher (bg7p0es01). Fold 1
training continues on remote GPUs autonomously (gpuhub-1:0/1 + gpuhub-2:0). This
script polls WandB for the 3 fold-1 runs (study-id 20260526_085736) and, once
all 3 are `finished`, fires:

  Stream A: --folds 2,3,4 --slots gpuhub-1:0,gpuhub-1:1,gpuhub-2:0
  Stream B: --folds 5,6,7 --slots gpuhub-1:0,gpuhub-1:1,gpuhub-2:1

These run as parallel subprocesses (this wrapper as their parent). 2-on-1 GPU
sharing on gpuhub-1:0 and gpuhub-1:1 between streams — within validated
envelope per feedback_concurrent_gpu_runs.md + L1 batch 2 precedent.

Wall-time target: ~15h from fold-1 finish, vs ~30h sequential.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

ENTITY = "bigcan-chiwin-technology"
PROJECT = "FinRL-Pro-DS"
STUDY_ID = "20260526_085736"
FOLD_1_TAG = "fold-1"

POLL_S = 120


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fold1_finished() -> tuple[int, int]:
    api = wandb.Api(timeout=30)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"$and": [
        {"tags": {"$in": [FOLD_1_TAG]}},
        {"tags": {"$in": [f"study-id-{STUDY_ID}"]}},
        {"tags": {"$in": ["gmgp1-xauusd"]}},
        {"tags": {"$in": ["extended-window"]}},
    ]}))
    finished = sum(1 for r in runs if r.state == "finished")
    return finished, len(runs)


def main() -> int:
    print(f"[{ts()}] waiting for fold-1 (study-id {STUDY_ID}) to finish on WandB", flush=True)

    while True:
        try:
            finished, total = fold1_finished()
            print(f"[{ts()}] fold-1: {finished}/3 finished ({total} runs)", flush=True)
            if finished >= 3:
                break
        except Exception as e:
            print(f"[{ts()}] poll error (will retry): {e}", flush=True)
        time.sleep(POLL_S)

    print(f"[{ts()}] fold-1 complete — firing Streams A + B in parallel", flush=True)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_a = logs_dir / f"wf_stream_a_{stamp}.log"
    log_b = logs_dir / f"wf_stream_b_{stamp}.log"

    cmd_common = [
        sys.executable,
        "scripts/launch_gmgp1_xauusd_wf_extended.py",
        "--config", "configs/gmgp1_xauusd_wf_extended.yaml",
    ]
    cmd_a = cmd_common + ["--folds", "2,3,4",
                          "--slots", "gpuhub-1:0,gpuhub-1:1,gpuhub-2:0"]
    cmd_b = cmd_common + ["--folds", "5,6,7",
                          "--slots", "gpuhub-1:0,gpuhub-1:1,gpuhub-2:1"]

    print(f"[{ts()}] Stream A log -> {log_a.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[{ts()}] Stream B log -> {log_b.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[{ts()}] Stream A cmd: {' '.join(cmd_a)}", flush=True)
    print(f"[{ts()}] Stream B cmd: {' '.join(cmd_b)}", flush=True)

    fa = log_a.open("w", encoding="utf-8")
    fb = log_b.open("w", encoding="utf-8")
    proc_a = subprocess.Popen(cmd_a, stdout=fa, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    # Stagger B by 30s so wandb-side polling queries don't collide on first cycle.
    time.sleep(30)
    proc_b = subprocess.Popen(cmd_b, stdout=fb, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    print(f"[{ts()}] Stream A pid={proc_a.pid}, Stream B pid={proc_b.pid}", flush=True)

    rc_a = proc_a.wait()
    rc_b = proc_b.wait()
    fa.close(); fb.close()

    print(f"[{ts()}] Stream A exit={rc_a}, Stream B exit={rc_b}", flush=True)
    return 0 if (rc_a == 0 and rc_b == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

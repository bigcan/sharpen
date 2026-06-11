"""Wrapper: wait for fold-0 (cost-corrected DECAY-01 WF) to finish on WandB,
then fire the gpuhub-2 stream for folds 5,6,7.

Context (S552): cost-corrected WF re-run (taker 5.5bps + 5bps slippage). The
original sequential dispatcher was stopped to enable a 2-instance split. Fold 0
keeps training on gpuhub-2:0 autonomously (orphaned, but valid cost-corrected
work). gpuhub-1 runs folds 1-4 in parallel NOW. This wrapper polls WandB for the
3 fold-0 runs; once all are `finished` (gpuhub-2 freed), it fires folds 5,6,7
there. Mirrors scripts/_runtime/wait_fold1_then_fire_parallel_streams.py.
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
POLL_S = 120
# Study-id of the CURRENT cost-corrected fold-0 run (b8wrswf0b, launched
# 2026-05-29 16:49:01). MUST scope by this — the loose [fold-0,decay01,wf-stage3]
# filter also matched the historical cost-FREE S542 fold-0 runs (study
# 20260520_133401, all `finished`) and falsely fired the continuation while fold
# 0 was still training. See conversation S552.
FOLD0_STUDY_ID = "20260529_164901"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fold0_finished() -> tuple[int, int]:
    """Return (n_finished, n_total) for the CURRENT cost-corrected fold-0 runs."""
    api = wandb.Api(timeout=30)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"$and": [
        {"tags": {"$in": ["fold-0"]}},
        {"tags": {"$in": ["decay01"]}},
        {"tags": {"$in": ["wf-stage3"]}},
        {"tags": {"$in": [f"study-id-{FOLD0_STUDY_ID}"]}},
    ]}))
    finished = sum(1 for r in runs if r.state == "finished")
    return finished, len(runs)


def main() -> int:
    print(f"[{ts()}] waiting for fold-0 (decay01 cost-corrected WF) to finish on WandB", flush=True)
    while True:
        try:
            fin, total = fold0_finished()
            print(f"[{ts()}] fold-0: {fin}/3 finished ({total} runs seen)", flush=True)
            if fin >= 3:
                break
        except Exception as e:  # noqa: BLE001
            print(f"[{ts()}] poll error (will retry): {e}", flush=True)
        time.sleep(POLL_S)

    print(f"[{ts()}] fold-0 complete — gpuhub-2 free; firing folds 5,6,7", flush=True)
    cmd = [
        sys.executable,
        "scripts/launch_sg1_btc_decay01_wf_multiseed.py",
        "--config", "configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml",
        "--folds", "5,6,7",
        "--instance", "gpuhub-2", "--gpu", "0", "--concurrent", "3",
    ]
    print(f"[{ts()}] cmd: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    print(f"[{ts()}] gpuhub-2 continuation dispatcher exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

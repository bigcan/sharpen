#!/usr/bin/env python
"""Relaunch GMGP1-BTC L1 seed 456 after the other 9 exit.

Context (S494): 10-seed fan-out launched 2026-04-23 08:01 UTC on gpuhub-1
(5 per GPU). Seed 456 crashed at init with `RuntimeError: can't start new
thread` during the 10-process spawn window. Two immediate retries also
crashed on resource contention (once on cache corruption, once on
`BlockingIOError: Resource temporarily unavailable` inside subprocess
fork_exec — fork EAGAIN, most likely kernel thread/memory pressure from
9 other inductor compile pools concurrently).

Strategy: wait for all 9 current seeds to exit (their Python PIDs leave
`pgrep run_full_pipeline` gmgp1-btc-l1-rehpo), THEN launch seed 456 on
a quiet host.

Run:
    python scripts/relaunch_btc_l1_seed456.py \
        2>&1 | tee C:/tmp/relaunch_btc_l1_seed456.log
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time

SSH_PREFIX = ("ssh", "-p", "<SSH_PORT>", "-o", "StrictHostKeyChecking=no",
              "root@<GPU_HOST>")

CONFIG_PATH = "configs/gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml"
RUN_NAME_PREFIX = "gmgp1-btc-velotrade-l1-rehpo"
LOG_NAME = "run_gmgp1_btc_l1_rehpo_seed456.log"
SEED = 456
GPU = 1            # use gpu1 (less memory consumed than gpu0 in the 9-seed fleet)
POLL_INTERVAL = 300
MAX_POLL_HOURS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] btc_l1_seed456 - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*SSH_PREFIX, cmd], capture_output=True, text=True, timeout=timeout,
    )


def _alive_python_count() -> int:
    # Count only Python processes (not bash wrappers / shell subshells)
    cmd = (
        f"pgrep -af 'run_full_pipeline.*{RUN_NAME_PREFIX}' "
        "| grep -v 'bash -c' | grep python | wc -l"
    )
    try:
        res = ssh(cmd, timeout=60)
    except subprocess.TimeoutExpired:
        return -1
    if res.returncode != 0:
        return -1
    try:
        return int(res.stdout.strip())
    except ValueError:
        return -1


def _wait_quiet() -> bool:
    start = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    while True:
        elapsed = time.time() - start
        if elapsed > max_seconds:
            log.error("poll ceiling %dh exceeded", MAX_POLL_HOURS)
            return False
        n = _alive_python_count()
        if n == 0:
            log.info("all current L1 seeds exited (elapsed %.1fm)",
                     elapsed / 60.0)
            return True
        if n == -1:
            log.warning("poll error; retrying in %ds", POLL_INTERVAL)
        else:
            log.info("L1 python procs alive: %d (elapsed %.1fm)",
                     n, elapsed / 60.0)
        time.sleep(POLL_INTERVAL)


def _launch_solo() -> int:
    cache_dir = f"/tmp/inductor_gmgp1_btc_seed{SEED}_solo"
    remote_cmd = (
        f"rm -rf {cache_dir} && "
        "cd /workspace/DeepScalper && "
        "set -a && source .env && set +a && "
        "TS=$(date +%Y%m%d_%H%M%S) && "
        f"TORCHINDUCTOR_CACHE_DIR={cache_dir} "
        f"CUDA_VISIBLE_DEVICES={GPU} "
        "nohup /root/miniconda3/bin/python -u scripts/run_full_pipeline.py "
        f"--config {CONFIG_PATH} "
        f"--run_name {RUN_NAME_PREFIX}-seed{SEED}_${{TS}} "
        f"--seed {SEED} "
        f"> {LOG_NAME} 2>&1 & "
        f"echo launched seed {SEED} pid $!; "
        "sleep 30; "
        f"pgrep -af 'run_full_pipeline.*seed{SEED}' | grep python | head"
    )
    log.info("launching seed %d solo on gpuhub-1:gpu%d", SEED, GPU)
    try:
        res = ssh(remote_cmd, timeout=120)
    except subprocess.TimeoutExpired:
        log.error("launch ssh timed out")
        return 124
    log.info("stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.error("launch FAILED exit=%d stderr:\n%s", res.returncode, res.stderr)
    return res.returncode


def main() -> int:
    log.info("waiting for current L1 9-seed fleet to exit...")
    if not _wait_quiet():
        return 2
    # Brief cooldown — let inductor subproc pools fully tear down
    log.info("cooldown 30s before solo launch")
    time.sleep(30)
    rc = _launch_solo()
    if rc != 0:
        log.error("solo launch failed rc=%d", rc)
        return rc
    log.info("seed %d launched. Monitor via tail /workspace/DeepScalper/%s",
             SEED, LOG_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
